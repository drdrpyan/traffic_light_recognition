#ifndef BGM_BBOX_NMS_HPP_
#define BGM_BBOX_NMS_HPP_

#include <opencv2/core.hpp>

#include <vector>

namespace bgm
{

class BBoxRefine
{
 public:
  virtual void Refine(const std::vector<cv::Rect2f>& src_bbox,
                      const std::vector<float>& src_conf,
                      std::vector<cv::Rect2f>* refined_bbox,
                      std::vector<float>* refined_conf = 0) = 0;
};
} // namespace bgm

#endif // !BGM_BBOX_NMS_HPP_
