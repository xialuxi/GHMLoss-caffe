#include <algorithm>
#include <vector>
#include <cfloat>
#include <math.h>
#include "caffe/layers/ghmr_loss_layer.hpp"

namespace caffe {

  template <typename Dtype>
  void GhmrLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
    
    LossLayer<Dtype>::LayerSetUp(bottom, top);
                                                

    //ghmr loss parameter 
    const GhmrLossParameter& param = this->layer_param_.ghmr_loss_param();
    m_ = param.m();
    alpha = param.alpha();
    mu = param.alpha();
    LOG(INFO) << "m: " << m_;
    CHECK_GT(m_, 0) << "m must be larger than zero";
    CHECK_GE(alpha, 0) << "alpha must be >= 0";
    CHECK_LT(alpha, 1) << "alpha must be < 1";
    CHECK_GT(mu, 0) << "mu must be larger than zero";

    r_num = new float[m_];
    memset(r_num, 0, m_ * sizeof(float));
  }

  template <typename Dtype>
  void GhmrLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::Reshape(bottom, top);
    //ghmr layer
    diff_asl.ReshapeLike(*bottom[0]);
    beta.ReshapeLike(*bottom[0]);
    distance.ReshapeLike(*bottom[0]);
    loss_value.ReshapeLike(*bottom[0]);
  }


template <typename Dtype>
void GhmrLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                                  const vector<Blob<Dtype>*>& top) {
    
    // compute loss and diff_ce
    const Dtype* label_data = bottom[1]->cpu_data();
    Dtype* asl_diff_data = diff_asl.mutable_cpu_data();
    Dtype* beta_data = beta.mutable_cpu_data();
    Dtype* distance_data = distance.mutable_cpu_data();
    Dtype* loss_value_data = loss_value.mutable_cpu_data();


    int count = bottom[0]->count();
    float epsin = 1.0 / m_;
    Dtype loss = 0;
    Dtype mu_2 = mu * mu;

    caffe_set(beta.count(), Dtype(0), beta_data);
    caffe_sub(count, bottom[0]->cpu_data(), label_data, distance_data);
    caffe_powx(count, distance_data, Dtype(2), asl_diff_data);
    caffe_add_scalar(count, mu_2, asl_diff_data);
    caffe_sqrt(count, asl_diff_data, asl_diff_data);
    caffe_set(count, Dtype(-mu), loss_value_data);
    caffe_add(count, loss_value_data, asl_diff_data, loss_value_data);
    caffe_div(count, distance_data, asl_diff_data, asl_diff_data);


    //compute the r_num
    int *num_in_bin = new int[m_];
    memset(num_in_bin, 0, m_ * sizeof(int));

    for(int k = 0; k < count; k++) {
        for(int i = 0; i < m_; i++) {
            float min_g = i * epsin;
            float max_g = (i + 1) * epsin;
            float abs_value = fabs(asl_diff_data[k]);
            if( abs_value < max_g && min_g >= min_g) {
                num_in_bin[i] += 1;
                //record the index of r_num
                beta_data[k] = i;
                break;
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
      for(int i = 0; i < count; i++) {
          //get the index of r_num
          int id = beta_data[i];
          //compute the beta
          beta_data[i] = count * 1.0 / (r_num[id] * valid);
          //compute loss
          loss += loss_value_data[i] * beta_data[i];
      }
    }

    top[0]->mutable_cpu_data()[0] = loss / count;
}

template <typename Dtype>
void GhmrLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                   const vector<bool>& propagate_down,
                                                   const vector<Blob<Dtype>*>& bottom)
{
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* asl_diff_data = diff_asl.cpu_data();
    const Dtype* beta_data = beta.cpu_data();

    
    int count = bottom[0]->count();
    caffe_copy(count, asl_diff_data, bottom_diff);
    caffe_mul(count, beta_data, bottom_diff, bottom_diff);
    
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] / count;
    caffe_scal(count, loss_weight, bottom_diff);
  }
}

INSTANTIATE_CLASS(GhmrLossLayer);
REGISTER_LAYER_CLASS(GhmrLoss);

}  // namespace caffe
