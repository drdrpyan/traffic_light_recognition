#ifndef BGM_BBOX_SIZEGROUP_MAXCONF_HPP_
#define BGM_BBOX_SIZEGROUP_MAXCONF_HPP_

#include "bbox_refine.hpp"

namespace bgm
{

class BBoxSizegroupMaxconf : public BBoxRefine
{
 public:
  BBoxSizegroupMaxconf(float grouping_factor = 1.5);

  virtual void Refine(
      const std::vector<cv::Rect2f>& src_bbox,
      const std::vector<float>& src_conf,
      std::vector<cv::Rect2f>* refined_bbox,
      std::vector<float>* refined_conf = 0) override;

  void set_grouping_factor(float factor);

 private:
  float BBoxSize(const cv::Rect2f& bbox) const;
  float BBoxDistance(const cv::Rect2f& bbox1,
                     const cv::Rect2f& bbox2) const;
  void SortIdx(const std::vector<float>& bbox_size,
               std::vector<int>* sorted_idx) const;
  void Group(const std::vector<cv::Rect2f>& bbox,
             const std::vector<float>& bbox_size,
             const std::vector<int>& sorted_idx,
             std::vector<std::vector<int> >* group) const;
  void PickMaxConf(const std::vector<std::vector<int> >& group,
                   const std::vector<float>& conf,
                   std::vector<int>* picked) const;

  float grouping_factor_;

};

// inline functions
inline BBoxSizegroupMaxconf::BBoxSizegroupMaxconf(
    float grouping_factor)
  : grouping_factor_(grouping_factor) {

}

inline void BBoxSizegroupMaxconf::set_grouping_factor(float factor) {
  grouping_factor_ = factor;
}

inline float BBoxSizegroupMaxconf::BBoxSize(const cv::Rect2f& bbox) const {
  return std::sqrtf(std::powf(bbox.width, 2) + std::powf(bbox.height, 2));
}


} // namespace bgm
#endif // !BGM_BBOX_SIZEGROUP_MAXCONF_HPP_