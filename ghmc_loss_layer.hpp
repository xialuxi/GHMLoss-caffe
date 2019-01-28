#ifndef CAFFE_GHMC_LOSS_LAYERS_HPP_
#define CAFFE_GHMC_LOSS_LAYERS_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"

namespace caffe {
template <typename Dtype>
class GhmcLossLayer : public LossLayer<Dtype> {
 public:
    explicit GhmcLossLayer(const LayerParameter& param)
        : LossLayer<Dtype>(param){}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "GhmcLoss"; }

    virtual inline int ExactNumBottomBlobs() const { return -1; }
    virtual inline int MinBottomBlobs() const { return 1; }
    virtual inline int MaxBottomBlobs() const { return 2; }

    /**
     * Unlike most loss layers, in the SmoothL1LossLayer we can backpropagate
     * to both inputs -- override to return true and always allow force_backward.
     */
    virtual inline bool AllowForceBackward(const int bottom_index) const {
        return true;
    }

 protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    //virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    //    const vector<Blob<Dtype>*>& top);

    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
    //    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

     virtual Dtype get_normalizer(
      LossParameter_NormalizationMode normalization_mode, int valid_count);

    shared_ptr<Layer<Dtype> > softmax_layer_;
    vector<Blob<Dtype>*> softmax_bottom_vec_;
    vector<Blob<Dtype>*> softmax_top_vec_;
    Blob<Dtype> prob_;        // softmax output
    bool has_ignore_label_;
    int ignore_label_;
    LossParameter_NormalizationMode normalization_;
    int softmax_axis_, outer_num_, inner_num_;

    int m_;
    int count;
    float * r_num;
    Dtype alpha;
    Blob<Dtype> diff_ce;
    Blob<Dtype> beta;  //beta = N / GD(g)
};

}  // namespace caffe

#endif  // CAFFE_GHMC_LOSS_LAYERS_HPP_
