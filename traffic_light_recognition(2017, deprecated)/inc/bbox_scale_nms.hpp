//#ifndef BGM_BBOX_SCALE_NMS_HPP_
//#define BGM_BBOX_SCALE_NMS_HPP_
//
//#include "bbox_refine.hpp"
//
//namespace bgm
//{
//
//class BBoxScaleNMS : public BBoxRefine
//{
// public:
//  BBoxScaleNMS(float scale = 1, float overlap_threshold = 0);
//  virtual void Refine(const std::vector<cv::Rect2f>& src_bbox,
//                      const std::vector<float>& src_conf,
//                      std::vector<cv::Rect2f>* refined_bbox) override;
// 
//  void set_scale(float scale);
//  void set_overlap_threshold(float threshold);
//
// private:
//  float scale_;
//  float overlap_threshold_;
//}; // class BBoxScaleNMS
//
//// inline functions
//inline BBoxScaleNMS::BBoxScaleNMS(float scale, 
//                                  float overlap_threshold) 
//  : scale_(scale), overlap_threshold_(overlap_threshold) {
//
//}
//
//inline void BBoxScaleNMS::set_scale(float scale) {
//  scale_ = scale;
//}
//
//inline void BBoxScaleNMS::set_overlap_threshold(float threshold) {
//  overlap_threshold_ = threshold;
//}
//} // namespace bgm
//#endif // !BGM_BBOX_SCALE_NMS_HPP_
