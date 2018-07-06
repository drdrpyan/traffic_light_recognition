#include "caffe_wrapper.hpp"

namespace bgm
{





//
//template <typename Dtype>
//void CaffeWrapper<Dtype>::InitNet(const std::string& model_file,
//                                  const std::string& trained_file,
//                                  bool use_gpu) {
//  if (use_gpu) {
//#ifdef CPU_ONLY
//    LOG(WARNING) << "Can't use GPU, CPU is used instead GPU";
//    caffe::Caffe::set_mode(caffe::Caffe::CPU);
//#else
//    caffe::Caffe::set_mode(caffe::Caffe::GPU);
//#endif
//  }
//  else 
//    caffe::Caffe::set_mode(caffe::Caffe::CPU);
//
//  net_.reset(new caffe::Net<float>(net_model_file, caffe::TEST));
//  net_->CopyTrainedLayersFrom(net_trained_file);
//  net_->output_blobs()[0]->set
//}
//
//template <typename Dtype>
//void CaffeWrapper<Dtype>::Process() {
//  net_->Forward();
//}
} // namespace bgm