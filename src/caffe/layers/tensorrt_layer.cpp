#include <vector>
#include "caffe/layers/tensorrt_layer.hpp"

namespace caffe {

class Logger : public nvinfer1::ILogger
{
	public:
		void log(nvinfer1::ILogger::Severity severity, const char* msg) override
		{
			if (severity == Severity::kINFO) return;
			        switch (severity)
					{
						case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
						case Severity::kERROR: std::cerr << "ERROR: "; break;
						case Severity::kWARNING: std::cerr << "WARNING: "; break;
						case Severity::kINFO: std::cerr << "INFO: "; break;
						default: std::cerr << "UNKNOWN: "; break;
					}
					std::cerr << msg << std::endl;
		}
};


template <typename Dtype>
void TensorRTLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top) 
{
	// init engine and create context
	engine_filename_ = this->layer_param_.tensorrt_param().engine_filename();
	FILE * file = fopen(engine_filename_.c_str(),"rb");
	int size = fread(&engine_data_size_,sizeof(int),1,file);
	CHECK_EQ(size,1);
	engine_data_ = malloc(engine_data_size_);
	int engine_data_size = fread(engine_data_,1,engine_data_size_,file);
	CHECK_EQ(engine_data_size_,engine_data_size);
	fclose(file);
	Logger gLogger;
	infer_ = createInferRuntime(gLogger);
	engine_ = infer_->deserializeCudaEngine(engine_data_,engine_data_size_, nullptr);
	context_ = engine_->createExecutionContext();

	
	input_blob_name_.resize(bottom.size());
	for(int index = 0;index < bottom.size();index ++)
	{
		input_blob_name_[index] = this->layer_param_.bottom(index);
	}

	output_blob_name_.resize(top.size());
	output_dim_.resize(top.size());
    for(int index = 0;index < top.size();index ++)
	{
		output_blob_name_[index] = this->layer_param_.top(index);
		DimsCHW outputDims = static_cast<DimsCHW&&>(engine_->getBindingDimensions(engine_->getBindingIndex(output_blob_name_[index].c_str())));
		output_dim_[index].resize(4);
		output_dim_[index][1] = outputDims.c();
		output_dim_[index][2] = outputDims.h();
		output_dim_[index][3] = outputDims.w();
	}
	
	buffers_ = (void **)malloc(sizeof(void *)*(bottom.size()+top.size()));
}

template <typename Dtype>
void TensorRTLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
{
	for(int index = 0;index < bottom.size();index ++)
	{
		int buffer_index = engine_->getBindingIndex(input_blob_name_[index].c_str());
		DimsCHW inputDims = static_cast<DimsCHW&&>(engine_->getBindingDimensions(buffer_index));
		CHECK_EQ(inputDims.c(),bottom[index]->channels());
		CHECK_EQ(inputDims.h(),bottom[index]->height());
		CHECK_EQ(inputDims.w(),bottom[index]->width());
		buffers_[buffer_index] = bottom[index]->mutable_gpu_data();
	}


	for(int index = 0;index < top.size();index ++)
	{
		output_dim_[index][0] = bottom[0]->num();
		top[index]->Reshape(output_dim_[index]);
		int buffer_index = engine_->getBindingIndex(output_blob_name_[index].c_str());
		buffers_[buffer_index] = top[index]->mutable_gpu_data();
	}
}

template <typename Dtype>
void TensorRTLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
{
}

template <typename Dtype>
void TensorRTLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
}


template <typename Dtype>
TensorRTLayer<Dtype>::~TensorRTLayer()
{
	free(buffers_);
	free(engine_data_);
	context_->destroy();
	engine_->destroy();
	infer_->destroy();
}

#ifdef CPU_ONLY
STUB_GPU(TensorRTLayer);
#endif

INSTANTIATE_CLASS(TensorRTLayer);
REGISTER_LAYER_CLASS(TensorRT);

}
