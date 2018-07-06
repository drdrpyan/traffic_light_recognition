#ifndef BGM_TLR_DRAW_HPP_
#define BGM_TLR_DRAW_HPP_

#include "tlr.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <memory>

namespace bgm
{

class TLRDraw
{
 public:
  TLRDraw(
      std::shared_ptr<TLR> tlr = nullptr, 
      const cv::Scalar& default_rect_color = cv::Scalar(255, 255, 255),
      const cv::Scalar& default_font_color = cv::Scalar(255, 255, 255));

  cv::Mat DrawHeatmap(const cv::Mat& src, float blend_alpha,
                      int color_map = cv::COLORMAP_JET) const;
  //cv::Mat DrawProposal(const cv::Mat& src) const;
  //cv::Mat DrawProposal(
  //    const cv::Mat& src,
  //    const cv::Scalar& rect_color,
  //    const cv::Scalar& font_color) const;
  cv::Mat DrawResult(
      const cv::Mat& src,
      const std::vector<std::string>& label_str = std::vector<std::string>(),
      const std::vector<cv::Scalar>& rect_color_for_label = std::vector<cv::Scalar>(),
      const std::vector<cv::Scalar>& font_color_for_label = std::vector<cv::Scalar>()) const;

  void set_tlr(std::shared_ptr<TLR>& tlr);
  void set_default_rect_color(const cv::Scalar& color);
  void set_default_font_color(const cv::Scalar& color);

 private:
  std::shared_ptr<TLR> tlr_;
  cv::Scalar default_rect_color_;
  cv::Scalar default_font_color_;
};

// inline functions
inline TLRDraw::TLRDraw(std::shared_ptr<TLR> tlr,
                 const cv::Scalar& default_rect_color,
                 const cv::Scalar& default_font_color) 
  : tlr_(tlr),
    default_rect_color_(default_rect_color),
    default_font_color_(default_font_color) {

}

//inline cv::Mat TLRDraw::DrawProposal(const cv::Mat& src) const {
//  DrawProposal(src, default_rect_color_, default_font_color_);
//}

inline void TLRDraw::set_tlr(std::shared_ptr<TLR>& tlr) {
  tlr_ = tlr;
}

inline void TLRDraw::set_default_rect_color(const cv::Scalar& color) {
  default_rect_color_ = color;
}
inline void TLRDraw::set_default_font_color(const cv::Scalar& color) {
  default_font_color_ = color;
}
} // namespace bgm

#endif // !BGM_TLR_DRAW_HPP_
