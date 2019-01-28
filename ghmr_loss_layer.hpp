#ifndef CAFFE_GHMR_LOSS_LAYERS_HPP_
#define CAFFE_GHMR_LOSS_LAYERS_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {
template <typename Dtype>
class GhmrLossLayer : public LossLayer<Dtype> {
 public:
    explicit GhmrLossLayer(const LayerParameter& param)
        : LossLayer<Dtype>(param){}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "GhmrLoss"; }
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

    int m_;
    float * r_num;
    Dtype alpha;
    Dtype mu;
    Blob<Dtype> diff_asl;
    Blob<Dtype> beta;  //beta = N / GD(g)
    Blob<Dtype> distance;
    Blob<Dtype> loss_value;
};

}  // namespace caffe

#endif  // CAFFE_GHMR_LOSS_LAYERS_HPP_
