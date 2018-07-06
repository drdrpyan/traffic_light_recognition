#include "softmax_out_const_iterator.hpp"

namespace bgm
{

//template <typename Dtype>
//SoftmaxOutConstIterator<Dtype>::SoftmaxOutConstIterator(
//    const caffe::Blob<Dtype>& softmax_out, int idx)
//  : count_(0), end(softmax_out.width() * softmax_out.height()), 
//    max_label_updated_(false), probs_updated_(false), 
//    probs_(softmax_out.channels()) {
//  CHECK_GE(idx, 0);
//  CHECK_LT(idx, softmax_out.num());
//
//  int stride = softmax_out.width() * softmax_out.height();
//  const Dtype* ptr = softmax_out.cpu_data() + softmax_out.offset(idx);
//  const_iter_.resize(softmax_out.channels());
//  for (int i = 0; i < softmax_out.channels(); i++) {
//    const_iter_[i] = ptr;
//    ptr += stride;
//  }
//}
//
//template <typename Dtype>
//void SoftmaxOutConstIterator<Dtype>::GetMax(int* label, Dtype* prob) {
//  CHECK(label);
//  CHECK(prob);
//
//  if (!max_label_updated_) {
//    max_label_updated_ = true;
//    const std::vector<Dtype>& probs = GetProbs();
//    std::vector<Dtype>::const_iterator max_iter = 
//        std::max_element(probs.cbegin(), probs.cend());
//    max_label_ = std::distance(probs.cbegin(), max_iter);
//  }
//
//  *label = max_label_;
//  *prob = probs_[max_label_];
//}
//
//template <typename Dtype>
//void SoftmaxOutConstIterator<Dtype>::Next() {
//  CHECK(count_++ < end_) << "Out of range";
//
//  max_label_updated_ = false;
//  probs_updated_ = false;
//
//  for (auto iter = const_iter_.begin(); iter != const_iter_.end(); iter++)
//    (*iter)++;
//}


} // namespace bgm