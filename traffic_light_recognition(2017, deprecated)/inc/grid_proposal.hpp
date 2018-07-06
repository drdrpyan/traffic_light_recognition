#ifndef BGM_GRID_PROPOSAL_HPP_
#define BGM_GRID_PROPOSAL_HPP_

#include "region_proposal.hpp"

#include "dnn_wrapper.hpp"

#include <opencv2/core.hpp>

#include <memory>
#include <vector>

namespace bgm
{

class GridProposal : public RegionProposal
{
 public:
  GridProposal(DNNWrapper* grid_proposal_net,
               int grid_rows, int grid_cols);
  virtual void Propose(const cv::Mat& img, float threshold,
                       std::vector<cv::Rect>* region,
                       std::vector<float>* confidence = 0) override;
  virtual void Propose(
      const std::vector<cv::Mat>& img, float threshold,
      std::vector<std::vector<cv::Rect> >* region,
      std::vector<std::vector<float> >* confidence = 0) override;

  //virtual void Propose(const cv::Mat& input) = 0;
  //virtual void Propose(const std::vector<cv::Mat>& input) = 0;
  //virtual void GetActivationProposal(float threshold, 
  //                                   cv::Mat& proposal) = 0;
  //virtual void GetActivationProposal(
  //    float threshold, std::vector<cv::Mat>* proposal) = 0;
  //virtual void GetDensityProposal(float threshold, 
  //                                cv::Mat& proposal) = 0;
  //virtual void GetDensityProposal(
  //    float threshold, std::vector<cv::Mat>* proposal) = 0;
 private:
  void GridConfToRegionConf(
      const DNNUnit& grid_conf,
      std::vector<std::vector<float> >* region_conf) const;
  cv::Rect ComputeRegion(const cv::Size& img_size,
                         int region_idx) const;

  std::unique_ptr<DNNWrapper> grid_proposal_net_;
  int grid_rows_;
  int grid_cols_;
}; // class GridProposal

// inline functions


}
#endif // !BGM_GRID_PROPOSAL_HPP_
