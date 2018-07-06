#ifndef BGM_DNN_UNIT_HPP_
#define BGM_DNN_UNIT_HPP_

#ifdef USE_OPENCV
#include <opencv2/core.hpp>
#endif // USE_OPENCV

#ifdef USE_CAFFE
#include "caffe/caffe.hpp"
#endif // USE_CAFFE

#include <memory>

namespace bgm
{

class DNNUnit
{
 public:
  enum ElemType {UCHAR = sizeof(unsigned char),
                 INT32 = sizeof(long),
                 SINGLE = sizeof(float), 
                 DOUBLE = sizeof(double)};

 public:
  DNNUnit();
  DNNUnit(DNNUnit::ElemType elem_type, char* data, bool shared,
          int batch, int channel, int height, int width);
  //DNNUnit& operator=(const DNNUnit& ref);

  DNNUnit Clone() const;

  int Count() const;
  int Volume() const;
  int Area() const;
  int Size() const;
  int Depth() const;

  //void SetShape(int batch, int channel, int height, int width);
  void set_elem_type(ElemType elem_type);
  void set_data(char* data, bool shared = true);
  void set_batch(int batch);
  void set_channel(int channel);
  void set_height(int height);
  void set_width(int width);

  ElemType elem_type() const;
  bool shared() const;
  const char* data() const;
  char* data();
  int batch() const;
  int channel() const;
  int height() const;
  int width() const;

#ifdef USE_OPENCV
  DNNUnit(const cv::Mat& mat);
  DNNUnit(const std::vector<cv::Mat>& mat_list);
  //DNNUnit(const cv::Mat& mat, ElemType type);
  //DNNUnit(const std::vector<cv::Mat>& mat_list, ElemType type);
  void ToMat(std::vector<cv::Mat>* mat_list) const;
#endif // USE_OPENCV

#ifdef USE_CAFFE
  template <typename Dtype>
  DNNUnit(caffe::Blob<Dtype>& blob);
  //template <typename Dtype>
  //void WrapBlob(caffe::Blob<Dtype>& blob);

  template <typename Dtype>
  void ToBlob(caffe::Blob<Dtype>* blob) const;
#endif // USE_CAFFE

 private:
#ifdef USE_OPENCV
  void MatToData(const cv::Mat& mat, void* data_ptr);
#endif // USE_OPENCV

  ElemType elem_type_;
  bool shared_;
  std::shared_ptr<char> shared_data_;
  char* ref_data_;
  int batch_;
  int channel_;
  int height_;
  int width_;
}; // class DNNUnit<T>


// inline functions
inline int DNNUnit::Count() const {
  return batch_ * channel_ * height_ * width_;
}

inline int DNNUnit::Volume() const {
  return channel_ * height_ * width_;
}

inline int DNNUnit::Area() const {
  return height_ * width_;
}

inline int DNNUnit::Size() const {
  return Count() * elem_type_;
}

inline int DNNUnit::Depth() const {
  return elem_type_;
}

inline void DNNUnit::set_elem_type(ElemType elem_type) {
  elem_type_ = elem_type;
}

inline void DNNUnit::set_batch(int batch) {
  assert(batch > 0);
  batch_ = batch;
}

inline void DNNUnit::set_channel(int channel) {
  assert(channel > 0);
  channel_ = channel;
}

inline void DNNUnit::set_height(int height) {
  assert(height > 0);
  height_ = height;
}

inline void DNNUnit::set_width(int width) {
  assert(width > 0);
  width_ = width;
}

inline DNNUnit::ElemType DNNUnit::elem_type() const {
  return elem_type_;
}

inline bool DNNUnit::shared() const {
  return shared_;
}

inline const char* DNNUnit::data() const {
  return shared() ? shared_data_.get() : ref_data_;
}

inline char* DNNUnit::data() {
  return const_cast<char*>(static_cast<const DNNUnit*>(this)->data());
}

inline int DNNUnit::batch() const {
  return batch_;
}

inline int DNNUnit::channel() const {
  return channel_;
}

inline int DNNUnit::height() const {
  return height_;
}

inline int DNNUnit::width() const {
  return width_;
}

// template functions
#ifdef USE_CAFFE

template <typename Dtype>
DNNUnit::DNNUnit(caffe::Blob<Dtype>& blob) {
  set_batch(blob.num());
  set_channel(blob.channels());
  set_height(blob.height());
  set_width(blob.width());
  set_data(reinterpret_cast<char*>(blob.mutable_cpu_data()),
           false);
  if (sizeof(Dtype) == SINGLE)
    set_elem_type(SINGLE);
  else if (sizeof(Dtype) == DOUBLE)
    set_elem_type(DOUBLE);
  else
    assert(("Illegal blob type", 0));
}

template <typename Dtype>
void DNNUnit::ToBlob(caffe::Blob<Dtype>* blob) const {
  CHECK(blob);
  CHECK_EQ(elem_type_, sizeof(Dtype));

  std::vector<int> shape(4);
  shape[0] = batch_;
  shape[1] = channel_;
  shape[2] = height_;
  shape[3] = width_;

  blob->Reshape(shape);

  blob->set_cpu_data(reinterpret_cast<Dtype*>(shared_data_.get()));
}

#endif // USE_CAFFE
} // namespace bgm

#endif // !BGM_DNN_UNIT_HPP_
