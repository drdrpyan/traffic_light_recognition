#ifndef BGM_REGION_PROPOSAL_HPP_
#define BGM_REGION_PROPOSAL_HPP_

#include <opencv2/core.hpp>

#include <vector>

namespace bgm
{

class RegionProposal
{
 public:
  virtual void Propose(const cv::Mat& img, float threshold,
                       std::vector<cv::Rect>* region,
                       std::vector<float>* confidence = 0) = 0;
  virtual void Propose(
      const std::vector<cv::Mat>& img, float threshold,
      std::vector<std::vector<cv::Rect> >* region,
      std::vector<std::vector<float> >* confidence = 0) = 0;
}; // class RegionProposal

} // namespace bgm

#endif // !BGM_REGION_PROPOSAL_HPP_
