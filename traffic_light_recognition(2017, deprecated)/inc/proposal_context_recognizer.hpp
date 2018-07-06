#ifndef BGM_PROPOSAL_CONTEXT_RECOGNIZER_HPP_
#define BGM_PROPOSAL_CONTEXT_RECOGNIZER_HPP_

#include "proposal_dnn_recognizer.hpp"

namespace bgm
{
class ProposalContextRecognizer : public ProposalDNNRecognizer
{
 public:
  ProposalContextRecognizer::ProposalContextRecognizer(
      DNNWrapper* dnn, NetOutHandler* net_out_handler,
      int margin_left, int margin_right,
      int margin_up, int margin_down,
      int batch_size);

  virtual void Recognize(
      const cv::Mat& img, int top_k,
      const std::vector<cv::Rect>& proposal,
      std::vector<std::vector<int> >* label,
      std::vector<std::vector<cv::Rect2f> >* bbox = 0,
      std::vector<std::vector<float> >* conf = 0) override;

 private:
  void GetContext(
      const std::vector<cv::Rect>& proposal,
      std::vector<cv::Rect>* context) const;
  cv::Mat ExtractROI(
      const cv::Mat& img, const cv::Rect& roi) const;
  
  int margin_left_;
  int margin_right_;
  int margin_up_;
  int margin_down_;
  int batch_size_;
}; // class ProposalActiveRecognizer
} // namespace bgm

#endif // !BGM_PROPOSAL_CONTEXT_RECOGNIZER_HPP_
