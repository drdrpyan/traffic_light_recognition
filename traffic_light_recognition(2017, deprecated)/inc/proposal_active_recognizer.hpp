#ifndef BGM_PROPOSAL_ACTIVE_RECOGNIZER_HPP_
#define BGM_PROPOSAL_ACTIVE_RECOGNIZER_HPP_

#include "proposal_dnn_recognizer.hpp"

namespace bgm
{

class ProposalAcitveRecognizer : public ProposalDNNRecognizer
{
 public:
  ProposalAcitveRecognizer::ProposalAcitveRecognizer(
      DNNWrapper* dnn, NetOutHandler* net_out_handler,
      const cv::Size& receptive_field_size,
      const cv::Rect& active_region);

  virtual void Recognize(
      const cv::Mat& img, int top_k,
      const std::vector<cv::Rect>& proposal,
      std::vector<std::vector<int> >* label,
      std::vector<std::vector<cv::Rect2f> >* bbox = 0,
      std::vector<std::vector<float> >* conf = 0) override;

 private:
  cv::Size receptive_field_size_;
  cv::Rect active_region_;
}; // class ProposalActiveRecognizer

} // namespace bgm
#endif // !BGM_PROPOSAL_ACTIVE_RECOGNIZER_HPP_
