#include <vector>

#include "caffe/layers/tensorrt_layer.hpp"

namespace caffe {

template <typename Dtype>
void TensorRTLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	context_->execute(bottom[0]->num(), buffers_);
}

template <typename Dtype>
void TensorRTLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
}
INSTANTIATE_LAYER_GPU_FUNCS(TensorRTLayer);
}
