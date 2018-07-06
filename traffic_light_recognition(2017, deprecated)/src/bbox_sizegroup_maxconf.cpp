#include "bbox_sizegroup_maxconf.hpp"

#include <numeric>

namespace bgm
{

void BBoxSizegroupMaxconf::Refine(
    const std::vector<cv::Rect2f>& src_bbox,
    const std::vector<float>& src_conf,
    std::vector<cv::Rect2f>* refined_bbox,
    std::vector<float>* refined_conf) {
  assert(refined_bbox);

  std::vector<float> bbox_size(src_bbox.size());
  for (int i = 0; i < bbox_size.size(); ++i)
    bbox_size[i] = BBoxSize(src_bbox[i]);

  std::vector<int> sorted_idx;
  SortIdx(bbox_size, &sorted_idx);

  std::vector<std::vector<int> > group;
  Group(src_bbox, bbox_size, sorted_idx, &group);

  std::vector<int> picked;
  PickMaxConf(group, src_conf, &picked);

  refined_bbox->resize(picked.size());
  for (int i = 0; i < picked.size(); ++i) {
    (*refined_bbox)[i] = src_bbox[picked[i]];
  }

  if (refined_conf) {
    refined_conf->resize(picked.size());
    for (int i = 0; i < picked.size(); ++i) {
      (*refined_conf)[i] = src_conf[picked[i]];
    }
  }
}


float BBoxSizegroupMaxconf::BBoxDistance(
    const cv::Rect2f& bbox1,
    const cv::Rect2f& bbox2) const {
  float x1 = bbox1.x + bbox1.width / 2;
  float y1 = bbox1.y + bbox1.height / 2;
  float x2 = bbox2.x + bbox2.width / 2;
  float y2 = bbox2.y + bbox2.height / 2;

  return std::sqrtf(std::powf(x1 - x2, 2) + std::powf(y1 - y2, 2));
}

void BBoxSizegroupMaxconf::SortIdx(
    const std::vector<float>& bbox_size,
    std::vector<int>* sorted_idx) const {
  assert(sorted_idx);

  sorted_idx->resize(bbox_size.size());

  std::iota(sorted_idx->begin(), sorted_idx->end(), 0);

  std::sort(sorted_idx->begin(), sorted_idx->end(),
            [&bbox_size](int i1, int i2) {return bbox_size[i1] < bbox_size[i2];});
}

void BBoxSizegroupMaxconf::Group(
    const std::vector<cv::Rect2f>& bbox,
    const std::vector<float>& bbox_size,
    const std::vector<int>& sorted_idx,
    std::vector<std::vector<int> >* group) const {
  assert(group);
  group->clear();

  std::vector<bool> checked(bbox.size(), false);

  for (int i = 0; i < sorted_idx.size(); ++i) {
    if (!checked[i]) {
      group->push_back(std::vector<int>(1, sorted_idx[i]));
      checked[i] = true;

      const cv::Rect2f base_bbox = bbox[sorted_idx[i]];
      float threshold = bbox_size[sorted_idx[i]] * grouping_factor_;

      for (int j = i + 1; j < sorted_idx.size(); ++j) {
        float dist = BBoxDistance(base_bbox, bbox[sorted_idx[j]]);
        if (dist < threshold) {
          group->back().push_back(sorted_idx[j]);
          checked[j] = true;
        }
      }
    }
  }
}

void BBoxSizegroupMaxconf::PickMaxConf(
    const std::vector<std::vector<int> >& group,
    const std::vector<float>& conf,
    std::vector<int>* picked) const {
  assert(picked);
  picked->clear();

  for (int i = 0; i < group.size(); ++i) {
    int best_idx = -1;
    float best_conf = 0;
    for (int j = 0; j < group[i].size(); ++j) {
      if (conf[group[i][j]] > best_conf) {
        best_idx = group[i][j];
        best_conf = conf[group[i][j]];
      }
    }

    picked->push_back(best_idx);
  }
}
} // namespace bgm