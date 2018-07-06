#ifndef TLR_IMG_CAFFE_WRAPPER_HPP_
#define TLR_IMG_CAFFE_WRAPPER_HPP_

#include "caffe_wrapper.hpp"

namespace bgm
{

template <typename Dtype>
class ImgCaffeWrapper : public CaffeWrapper<Dtype>
{
 public:
  void SetInput(const std::vector<cv::Mat>& imgs, int input_idx);

 private:
  
}; // class ImgCaffe

} // namespace bgm

#endif // !TLR_IMG_CAFFE_WRAPPER_HPP_s
