#include <algorithm>
#include <vector>
#include <cfloat>
#include <math.h>
#include "caffe/layers/ghmc_loss_layer.hpp"

namespace caffe {

  template <typename Dtype>
  void GhmcLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
    // softmax laye setup
    LossLayer<Dtype>::LayerSetUp(bottom, top);
    LayerParameter softmax_param(this->layer_param_);
    softmax_param.set_type("Softmax");
    softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
    softmax_bottom_vec_.clear();
    softmax_bottom_vec_.push_back(bottom[0]);
    softmax_top_vec_.clear();
    softmax_top_vec_.push_back(&prob_);
    softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

    // ignore label
    has_ignore_label_ = this->layer_param_.loss_param().has_ignore_label();
    if (has_ignore_label_) {
        ignore_label_   = this->layer_param_.loss_param().ignore_label();
    }

    // normalization
    if (!this->layer_param_.loss_param().has_normalization() &&
        this->layer_param_.loss_param().has_normalize()) 
    {
        normalization_ = this->layer_param_.loss_param().normalize() ?
                        LossParameter_NormalizationMode_VALID :
                        LossParameter_NormalizationMode_BATCH_SIZE;
    } else {
        normalization_ = this->layer_param_.loss_param().normalization();
    }                                                

    //ghmc loss parameter 
    const GhmcLossParameter& param = this->layer_param_.ghmc_loss_param();
    m_ = param.m();
    alpha = param.alpha();
    LOG(INFO) << "m: " << m_;
    CHECK_GT(m_, 0) << "m must be larger than zero";
    CHECK_GE(alpha, 0) << "alpha must be >= 0";
    CHECK_LT(alpha, 1) << "alpha must be < 1";

    r_num = new float[m_];
    memset(r_num, 0, m_ * sizeof(float));
  }

  template <typename Dtype>
  void GhmcLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
    // softmax laye reshape
    LossLayer<Dtype>::Reshape(bottom, top);
    softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);

    // cross-channels
    softmax_axis_ = bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
    outer_num_    = bottom[0]->count(0, softmax_axis_);
    inner_num_    = bottom[0]->count(softmax_axis_ + 1);
    CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
        << "Number of labels must match number of predictions; "
        << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
        << "label count (number of labels) must be N*H*W, "
        << "with integer values in {0, 1, ..., C-1}.";
  
    // softmax output
    if (top.size() >= 2) {
        top[1]->ReshapeLike(*bottom[0]);
    }

    //ghmc layer

    diff_ce.ReshapeLike(*bottom[0]);
    beta.ReshapeLike(*bottom[0]);

  }

template <typename Dtype>
Dtype GhmcLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, int valid_count) 
{
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = Dtype(outer_num_ * inner_num_);
      } else {
        normalizer = Dtype(valid_count);
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(outer_num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
void GhmcLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                                  const vector<Blob<Dtype>*>& top) {
    // The forward pass computes the softmax prob values.
    softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);

    // compute loss and diff_ce
    const Dtype* label_data = bottom[1]->cpu_data();
    const Dtype* prob_data = prob_.cpu_data();
    Dtype* ce_diff_data = diff_ce.mutable_cpu_data();
    Dtype* beta_data = beta.mutable_cpu_data();

    int channels    = bottom[0]->shape(softmax_axis_);
    int dim         = prob_.count() / outer_num_;

    count = 0;
    Dtype loss = 0;

    caffe_copy(prob_.count(), prob_data, ce_diff_data);
    caffe_set(beta.count(), Dtype(0), beta_data);

    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; ++j) {
        // label
        const int label_value = static_cast<int>(label_data[i * inner_num_ + j]);
        
        // ignore label
        if (has_ignore_label_ && label_value == ignore_label_) {
          for (int c = 0; c < channels; ++c) {
            int sepdim = i * dim + c * inner_num_ + j;
            ce_diff_data[sepdim] = 0;
            beta_data[sepdim] = -1;
          }
          continue;
        }

        int index = i * dim + label_value * inner_num_ + j;
        //bottom_diff
        ce_diff_data[index] -= 1;
        //count
        ++count;
      }
    }

    int count_num = bottom[0]->count();
    int num = bottom[0]->num();
    float epsin = 1.0 / m_;
    dim = count_num / num;

    //compute the r_num
    int *num_in_bin = new int[m_];
    memset(num_in_bin, 0, m_ * sizeof(int));

    for(int k = 0; k < count_num; k++) {
        for(int i = 0; i < m_; i++) {
            float min_g = i * epsin;
            float max_g = (i + 1) * epsin;
            // Don't calculate ignore label
            if( beta_data[k] >= 0) {
                float abs_value = fabs(ce_diff_data[k]);
                if( abs_value < max_g && abs_value >= min_g) {
                    num_in_bin[i] += 1;
                    //record the index of r_num
                    beta_data[k] = i;
                    break;
                }
            }
        }
    }

    int valid = 0;
    for(int i = 0; i < m_; i++)
    {
       //LOG(INFO) << "r_num[ "   << i  << "]:  " << r_num[i];
       if(num_in_bin[i] > 0) {
          r_num[i] = alpha * r_num[i] + (1 - alpha) * num_in_bin[i];
          valid++;
          //LOG(INFO) << alpha << "   ** r_num[ "   << i  << "]:  " << r_num[i];
         }
    }

    delete[] num_in_bin;
    
    //compute beta and loss,   beta = N / GD(g)
    if (valid > 0) {
      for(int i = 0; i < num; i++) {
          int gt = static_cast<int>(label_data[i]);
          //get the index of r_num
          int index = i * dim + gt;
          int id = beta_data[index];
          // Don't calculate ignore label
          if(id >= 0) {
              //compute the beta
              beta_data[index] = count * 1.0 / (r_num[id] * valid);
              //compute loss
              loss += -log(std::max(prob_data[index], Dtype(FLT_MIN))) * beta_data[index];
          }
      }
    }

    top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, count);
    if (top.size() == 2) {
        top[1]->ShareData(prob_);
    }
}

template <typename Dtype>
void GhmcLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                   const vector<bool>& propagate_down,
                                                   const vector<Blob<Dtype>*>& bottom)
{
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* ce_diff_data = diff_ce.cpu_data();
    const Dtype* beta_data = beta.cpu_data();
    const Dtype* label_data = bottom[1]->cpu_data();

    caffe_copy(bottom[0]->count(), ce_diff_data, bottom_diff);

    int count_num = bottom[0]->count();
    int num = bottom[0]->num();
    const int dim = count_num / num;

    for(int i = 0; i < num; i++) {
        int gt = static_cast<int>(label_data[i]);
        int index = i * dim + gt;
        Dtype weight = beta_data[index];
        caffe_scal(dim, weight, bottom_diff + i * dim);
    }
    
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] /
                        get_normalizer(normalization_, count);
    caffe_scal(prob_.count(), loss_weight, bottom_diff);
  }
}

INSTANTIATE_CLASS(GhmcLossLayer);
REGISTER_LAYER_CLASS(GhmcLoss);

}  // namespace caffe
