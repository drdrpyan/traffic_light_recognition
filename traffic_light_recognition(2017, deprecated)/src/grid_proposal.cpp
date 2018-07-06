#include "grid_proposal.hpp"

namespace bgm
{

GridProposal::GridProposal(DNNWrapper* grid_proposal_net,
                           int grid_rows, int grid_cols) 
  : grid_proposal_net_(grid_proposal_net),
    grid_rows_(grid_rows), grid_cols_(grid_cols) {
  assert(grid_proposal_net);
  assert(grid_rows > 0);
  assert(grid_cols > 0);
}

void GridProposal::Propose(const cv::Mat& img, float threshold,
                           std::vector<cv::Rect>* region,
                           std::vector<float>* confidence) {
  std::vector<std::vector<cv::Rect> > regions;
  if (confidence) {
    std::vector<std::vector<float> > confidences;
    Propose(std::vector<cv::Mat>(1, img), threshold,
            &regions, &confidences);
    region->assign(regions[0].begin(), regions[0].end());
    confidence->assign(confidences[0].begin(), confidences[0].end());
  }
  else {
    Propose(std::vector<cv::Mat>(1, img), threshold, &regions);
    region->assign(regions[0].begin(), regions[0].end());
  }
}

void GridProposal::Propose(
    const std::vector<cv::Mat>& img, float threshold,
    std::vector<std::vector<cv::Rect> >* region,
    std::vector<std::vector<float> >* confidence) {
  assert(region);
  region->resize(img.size());
  if (confidence) confidence->resize(img.size());

  grid_proposal_net_->set_input(0, DNNUnit(img));
  grid_proposal_net_->Process();
  const DNNUnit& grid_conf = grid_proposal_net_->output(0);

  std::vector<std::vector<float> > region_conf;
  GridConfToRegionConf(grid_conf, &region_conf);

  for (int i = 0; i < img.size(); ++i) {
    (*region)[i].clear();
    if (confidence) (*confidence)[i].clear();

    for (int j = 0; j < region_conf[i].size(); ++j) {
      if (region_conf[i][j] >= threshold) {
        (*region)[i].push_back(ComputeRegion(img[i].size(), j));
        if(confidence) (*confidence)[i].push_back(region_conf[i][j]);
      }
    }
  }
}

void GridProposal::GridConfToRegionConf(
  const DNNUnit& grid_conf,
  std::vector<std::vector<float> >* region_conf) const {
  assert(region_conf);
  assert(grid_conf.Volume() == grid_rows_ + grid_cols_);
  //region_conf->resize(grid_conf.Size());
  region_conf->resize(grid_conf.batch());

  switch (grid_conf.elem_type()) {
    case DNNUnit::SINGLE:
    {
      const float* row_conf =
          reinterpret_cast<const float*>(grid_conf.data());
      const float* col_conf = row_conf + grid_rows_;

      for (int i = 0; i < grid_conf.batch(); ++i) {
        (*region_conf)[i].resize(grid_rows_ * grid_cols_);

        for (int r = 0; r < grid_rows_; ++r) {
          for (int c = 0; c < grid_cols_; ++c) {
            (*region_conf)[i][r * grid_cols_ + c] =
                row_conf[r] * col_conf[c];
          }
        }

        row_conf += grid_conf.Volume();
        col_conf += grid_conf.Volume();
      }
    }
    break;

    default:
      assert(("Not implemented yet", 0));
  }
}

cv::Rect GridProposal::ComputeRegion(const cv::Size& img_size,
                                     int region_idx) const {
  assert(region_idx >= 0 && region_idx < (grid_rows_ * grid_cols_));

  int col_idx = region_idx % grid_cols_;
  int row_idx = region_idx / grid_cols_;

  float region_width = img_size.width / static_cast<float>(grid_cols_);
  float region_height = img_size.height / static_cast<float>(grid_rows_);
  
  float region_x = col_idx * region_width;
  float region_y = row_idx * region_height;

  return cv::Rect(region_x, region_y, 
                  region_width, region_height);
}

} // namespace bgm