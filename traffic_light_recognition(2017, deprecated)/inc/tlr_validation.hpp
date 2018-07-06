#ifndef BGM_TLR_VALIDATION_HPP_
#define BGM_TLR_VALIDATION_HPP_

#include "db_define.hpp"
#include "tlr.hpp"

#include <memory>
#include <string>
#include <vector>

namespace bgm
{

class TLRValidation
{
 public:
  //virtual void LoadGT(const std::string& gt_file) = 0;
  void RunTLR(float proposal_threshold);
  void ValidateDetection(int gt_min_width = 0, float iou_threshold = 0.3);
  void ValidateRecognition(int gt_min_width = 0, float iou_threshold = 0.3);
 
  void set_tlr(std::shared_ptr<TLR> tlr);
  void set_gt(const std::vector<ImgBBoxAnno>& gt);

 private:
  int Match(const BB& gt, const std::vector<BB>& detected,
            const std::vector<bool>& checked,
            float iou_threshold = 0.f);
  int Match(const BB& gt_box, int gt_label,
            const std::vector<BB>& detected_box,
            const std::vector<int>& detected_label,
            const std::vector<bool>& checked,
            float iou_threshold = 0.f);
  float ComputeIOU(const BB& box1, const BB& box2) const;

  std::shared_ptr<TLR> tlr_;
  std::vector<ImgBBoxAnno> gt_;
  std::vector<ImgBBoxAnno> result_;
}; // class TLRValidation

// inline functions
inline void TLRValidation::set_tlr(std::shared_ptr<TLR> tlr) {
  tlr_ = tlr;
}

inline void TLRValidation::set_gt(
    const std::vector<ImgBBoxAnno>& gt) {
  gt_.assign(gt.begin(), gt.end());
}

} // namespace bgm
#endif // !BGM_TLR_VALIDATION_HPP_
