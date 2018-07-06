#ifndef TLR_CAFFE_WRAPPER_HPP_
#define TLR_CAFFE_WRAPPER_HPP_

#include "dnn_wrapper.hpp"

#include "caffe/caffe.hpp"

#include <memory>

namespace bgm
{

template <typename Dtype>
class CaffeWrapper : public DNNWrapper
{
 public:
  CaffeWrapper(const std::string& model_file,
               const std::string& trained_file,
               bool use_gpu = true);
  virtual void Process() override;

  //virtual const DNNUnit& input(int idx) const;
  //virtual const std::vector<DNNUnit>& input() const;

  //virtual void SetInput(int idx, const DNNUnit& unit) override;
  //virtual void SetInput(const std::vector<DNNUnit>& unit) override;
  //virtual void GetOutput(int idx, DNNUnit* unit) override;
  //virtual void GetOutput(std::vector<DNNUnit>* unit) override;

 private:
  std::unique_ptr<caffe::Net<float> > net_;
  //std::vector<DNNUnit> input_;
  //std::vector<DNNUnit> output_;
  
};

typedef CaffeWrapper<float> CaffeSingle;
typedef CaffeWrapper<double> CaffeDouble;

// inline functions
//template <typename Dtype>
//inline const DNNUnit& CaffeWrapper<Dtype>::input(int idx) const {
//  return input_[idx];
//}
//
//template <typename Dtype>
//inline const std::vector<DNNUnit>& CaffeWrapper<Dtype>::input() const {
//  return input_;
//}

// template functions
template <typename Dtype>
CaffeWrapper<Dtype>::CaffeWrapper(const std::string& model_file,
                                  const std::string& trained_file,
                                  bool use_gpu) {
    if (use_gpu) {
#ifdef CPU_ONLY
    caffe::Caffe::set_mode(Caffe::CPU);
#else
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
#endif
  }
  else 
    caffe::Caffe::set_mode(caffe::Caffe::CPU);

  net_.reset(new caffe::Net<float>(model_file, caffe::TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  //SetInputSize(net_->num_inputs());
  //SetOutputSize(net_->num_outputs());
  input_.resize(net_->num_inputs());
  //for (int i = 0; i < input_.size(); ++i)
  //  input_[i] = *(net_->input_blobs()[i]);
  output_.resize(net_->num_outputs());
  //for (int i = 0; i < output_.size(); ++i)
  //  output_[i] = *(net_->output_blobs()[i]);
}

template <typename Dtype>
void CaffeWrapper<Dtype>::Process() {
  for (int i = 0; i < input_.size(); ++i) {
    //net_->input_blobs()[i]->set_cpu_data(
    //    reinterpret_cast<Dtype*>(input_[i].data()));
    input_[i].ToBlob(net_->input_blobs()[i]);
  }

  net_->Forward();

  for (int i = 0; i < output_.size(); ++i) {
    // output_[i].set_data(reinterpret_cast<char*>(net_->output_blobs()[i]->mutable_cpu_data()), false);
    output_[i] = DNNUnit(*(net_->output_blobs()[i])).Clone();
  }
}

// inline functions
//template <typename Dtype>
//inline void CaffeWrapper<Dtype>::SetInput(int idx, 
//                                          const DNNUnit& unit) {
//  unit.ToBlob(net_->input_blobs()[idx]);  
//}
//
//template <typename Dtype>
//inline void CaffeWrapper<Dtype>::SetInput(
//    const std::vector<DNNUnit>& unit) {
//  CHECK_EQ(unit.size(), net_->input_blobs().size());
//  for (int i = 0; i < unit.size(); ++i)
//    SetInput(i, unit[i]);
//}

//template <typename Dtype>
//class CaffeWrapper
//{
// public:
//  void InitNet(const std::string& model_file,
//               const std::string& trained_file,
//               bool use_gpu = true);
//  void Process();
//
//  const caffe::Blob<Dtype>& InputBlob(int idx) const;
//  caffe::Blob<Dtype>& InputBlob(int idx);
//  const std::vector<caffe::Blob<Dtype>*>& InputBlob() const;
//  const caffe::Blob<Dtype>& OutputBlob(int idx) const;
//  const std::vector<caffe::Blob<Dtype>*>& OutputBlob() const;
//
// private:
//  std::unique_ptr<caffe::Net<float> > net_;
//}; // class CaffeWrapper
//
//// inline functions
//template <typename Dtype>
//inline const caffe::Blob<Dtype>& CaffeWrapper<Dtype>::InputBlob(
//    int idx) const {
//  return *(net_->input_blobs()[idx]);
//}
//
//template <typename Dtype>
//inline caffe::Blob<Dtype>& CaffeWrapper<Dtype>::InputBlob(
//    int idx) {
//  return const_cast<caffe::Blob<Dtype>&>(static_cast<const CaffeWrapper<Dtype>&>(*this).InputBlob(idx));
//}
//
//template <typename Dtype>
//inline const std::vector<caffe::Blob<Dtype>*>& CaffeWrapper<Dtype>::InputBlob() const {
//  return net_->input_blobs();
//}
//
//template <typename Dtype>
//inline const caffe::Blob<Dtype>& CaffeWrapper<Dtype>::OutputBlob(
//    int idx) const {
//  return *(net_->output_blobs()[idx]);
//}
//
//template <typename Dtype>
//inline const std::vector<caffe::Blob<Dtype>*>& CaffeWrapper<Dtype>::OutputBlob() const {
//  return net_->output_blobs();
//}
} // namespace bgm

#endif // !TLR_CAFFE_WRAPPER_HPP_
