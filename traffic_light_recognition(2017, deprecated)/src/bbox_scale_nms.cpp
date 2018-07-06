//#include "bbox_scale_nms.hpp"
//
//namespace bgm
//{
//
//void BBoxScaleNMS::Refine(
//    const std::vector<cv::Rect2f>& src_bbox,
//    const std::vector<float>& src_conf,
//    std::vector<cv::Rect2f>* refined_bbox) {
//  std::vector<cv::Rect2f> scaled_bbox(src_bbox);
//  for (auto iter = scaled_bbox.begin();
//       iter != scaled_bbox.end(); ++iter) {
//    iter->width *= scale_;
//    iter->height *= scale_;
//  }
//
//  std::vector<
//
//
//}
//
//} // namespace bgm