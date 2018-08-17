#ifndef CAFFE_TENSORRT_LAYER_HPP_
#define CAFFE_TENSORRT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include <string>
#include "NvInfer.h"
using namespace nvinfer1;

namespace caffe {


template <typename Dtype>
class TensorRTLayer : public Layer<Dtype> {
 public:
  explicit TensorRTLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "TensorRT"; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual ~TensorRTLayer();
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  std::string engine_filename_;
  int engine_data_size_;
  void * engine_data_;
  //
  IRuntime* infer_;
  ICudaEngine* engine_;
  IExecutionContext* context_;
  //
  std::vector<std::string> input_blob_name_;
  std::vector<std::vector<int> > input_dim_;
  std::vector<std::string> output_blob_name_;
  std::vector<std::vector<int> > output_dim_;
  void ** buffers_;

};

}  // namespace caffe

#endif  // CAFFE_INNER_PRODUCT_LAYER_HPP_
