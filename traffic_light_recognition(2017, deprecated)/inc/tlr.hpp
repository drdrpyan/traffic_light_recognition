#ifndef TLR_TLR_HPP_
#define TLR_TLR_HPP_

#include "region_proposal.hpp"
#include "proposal_recognizer.hpp"

#include <opencv2/core.hpp>

#include <memory>
#include <vector>

namespace bgm
{

class TLR
{
 public:
  //TLR();
  void Run(const cv::Mat& img, float proposal_threshold);

  void set_sub_wins(const std::vector<cv::Rect>& sub_wins);
  void set_rpn(RegionProposal* rpn);
  void set_recognizer(ProposalRecognizer* recognizer);

  const std::vector<cv::Rect>& proposal() const;
  const std::vector<float>& proposal_confidence() const;
  const std::vector<int>& detected_label() const;
  const std::vector<cv::Rect2f>& detected_bbox() const;

 private:
  void ClearPrevResults();
  void ComputeProposal(const cv::Mat& img, float threshold);
  void DetectFromProposal(const cv::Mat& img);

  std::vector<cv::Rect> sub_wins_;
  std::unique_ptr<RegionProposal> rpn_;
  std::unique_ptr<ProposalRecognizer> recognizer_;

  std::vector<cv::Rect> proposal_;
  std::vector<float> proposal_confidence_;

  std::vector<int> detected_label_;
  std::vector<cv::Rect2f> detected_bbox_;
  std::vector<float> detected_confidence_;
};

// inline functions
//inline TLR::TLR() : rpn_(nullptr), recognizer_(nullptr) {
//
//}

inline void TLR::set_sub_wins(
    const std::vector<cv::Rect>& sub_wins) {
  sub_wins_.assign(sub_wins.begin(), sub_wins.end());
}

inline void TLR::set_rpn(RegionProposal* rpn) {
  rpn_.reset(rpn);
}

inline void TLR::set_recognizer(ProposalRecognizer* recognizer) {
  recognizer_.reset(recognizer);
}

inline const std::vector<cv::Rect>& TLR::proposal() const {
  return proposal_;
}

inline const std::vector<float>& TLR::proposal_confidence() const {
  return proposal_confidence_;
}

inline const std::vector<int>& TLR::detected_label() const {
  return detected_label_;
}

inline const std::vector<cv::Rect2f>& TLR::detected_bbox() const {
  return detected_bbox_;
}

} // namespace tlr

#endif // !TLR_TLR_HPP_
