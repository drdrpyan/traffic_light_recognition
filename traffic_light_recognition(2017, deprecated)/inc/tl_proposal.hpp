//#ifndef TLR_TL_PROPOSAL_HPP_ 
//#define TLR_TL_PROPOSAL_HPP_ 
//
//#include "region_proposal.hpp"
//
//#include <opencv2/core.hpp>
//
//#include <memory>
//
//namespace bgm
//{
//
//class TLProposal
//{
// public:
//  TLProposal(RegionProposal* rpn,
//             const std::vector<cv::Rect>& sub_wins);
//  void Process(const cv::Mat& img, float threshold = 0);
//  void GetProposal(float scale, std::vector<cv::Mat>* proposal);
//
//  void set_sub_windows(const std::vector<cv::Rect>& sub_wins);
//
// private:
//  std::vector<cv::Rect> sub_windows_;
//  std::unique_ptr<RegionProposal> rpn_;
//};
//} // namespace bgm
//
//#endif // !TLR_TL_PROPOSAL_HPP_ 
//
