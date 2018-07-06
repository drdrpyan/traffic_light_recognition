#ifndef TLR_SOFTMAX_OUT_ITERATOR_HPP_
#define TLR_SOFTMAX_OUT_ITERATOR_HPP_

#include "caffe/blob.hpp"

#include <vector>

namespace bgm
{

template <typename Dtype>
class SoftmaxOutConstIterator
{
 public:
  SoftmaxOutConstIterator(const caffe::Blob<Dtype>& softmax_out, int idx);
  
  void GetMax(int* label, Dtype* prob);
  int GetMaxLabel();
  Dtype GetMaxProb();
  const std::vector<Dtype>& GetProb();
  
  SoftmaxOutConstIterator& operator++();
  const SoftmaxOutConstIterator& operator++(int);

 private:
  void Next();

  int count_;
  int end_;

  bool max_label_updated_;
  bool probs_updated_;

  int max_label_;
  std::vector<Dtype> probs_;

  std::vector<const Dtype*> const_iter_;
};

// inline functions
template <typename Dtype>
inline int SoftmaxOutConstIterator<Dtype>::GetMaxLabel() {
  int label;
  Dtype prob;
  GetMax(&label, &prob);
  return label;
}

template <typename Dtype>
inline Dtype SoftmaxOutConstIterator<Dtype>::GetMaxProb() {
  int label;
  Dtype prob;
  GetMax(&label, &prob);
  return prob;
}

template <typename Dtype>
inline const std::vector<Dtype>& SoftmaxOutConstIterator<Dtype>::GetProb() {
  if (!probs_updated_) {
    probs_updated_ = true;
    for (int i = 0; i < const_iter_.size(); i++)
      probs_[i] = *(const_iter_[i]);
  }

  return probs_;
}

template <typename Dtype>
inline SoftmaxOutConstIterator<Dtype>& SoftmaxOutConstIterator<Dtype>::operator++() {
  Next();
  return *this;
}

template <typename Dtype>
inline const SoftmaxOutConstIterator<Dtype>& SoftmaxOutConstIterator<Dtype>::operator++(int) {
  SoftmaxOutConstIterator<Dtype> current = *this;
  Next();
  return current;
}

// template functions
template <typename Dtype>
SoftmaxOutConstIterator<Dtype>::SoftmaxOutConstIterator(
    const caffe::Blob<Dtype>& softmax_out, int idx)
  : count_(0), end_(softmax_out.width() * softmax_out.height()), 
    max_label_updated_(false), probs_updated_(false), 
    probs_(softmax_out.channels()) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, softmax_out.num());

  int stride = softmax_out.width() * softmax_out.height();
  const Dtype* ptr = softmax_out.cpu_data() + softmax_out.offset(idx);
  const_iter_.resize(softmax_out.channels());
  for (int i = 0; i < softmax_out.channels(); i++) {
    const_iter_[i] = ptr;
    ptr += stride;
  }
}

template <typename Dtype>
void SoftmaxOutConstIterator<Dtype>::GetMax(int* label, Dtype* prob) {
  CHECK(label);
  CHECK(prob);

  if (!max_label_updated_) {
    max_label_updated_ = true;
    const std::vector<Dtype>& probs = GetProb();
    std::vector<Dtype>::const_iterator max_iter = 
        std::max_element(probs.cbegin(), probs.cend());
    //std::vector<Dtype>::const_iterator max_iter = 
    //    std::min_element(probs.cbegin(), probs.cend());
    max_label_ = std::distance(probs.cbegin(), max_iter);
  }

  const std::vector<Dtype>& probs = GetProb();
  //DLOG(INFO) << "prob : " << probs[0] << ", " << probs[1] << ", " << probs[2];

  *label = max_label_;
  *prob = probs_[max_label_];
}

template <typename Dtype>
void SoftmaxOutConstIterator<Dtype>::Next() {
  CHECK(count_++ < end_) << "Out of range";

  max_label_updated_ = false;
  probs_updated_ = false;

  for (auto iter = const_iter_.begin(); iter != const_iter_.end(); iter++)
    (*iter)++;
}

} // namespace bgm
#endif // !TLR_SOFTMAX_OUT_ITERATOR_HPP_
