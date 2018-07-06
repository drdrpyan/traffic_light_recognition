#include "tlr_draw.hpp"

#include <opencv2/imgproc.hpp>

namespace bgm
{



cv::Mat TLRDraw::DrawHeatmap(const cv::Mat& src,
                             float blend_alpha,
                             int color_map) const {
  assert(blend_alpha >= 0 && blend_alpha <= 1);

  cv::Mat heatmap(src.size(), CV_32FC1);
  cv::Mat color_heatmap;

  const std::vector<cv::Rect>& proposal = tlr_->proposal();
  const std::vector<float>& proposal_conf = tlr_->proposal_confidence();

  for (int i = 0; i < proposal.size(); ++i)
    heatmap(proposal[i]) = proposal_conf[i];
  heatmap.convertTo(color_heatmap, CV_8UC3, 255);
  cv::applyColorMap(color_heatmap, color_heatmap, color_map);

  if (blend_alpha == 0)
    return heatmap;
  else {
    cv::Mat blended;
    cv::addWeighted(src, 1 - blend_alpha, color_heatmap, blend_alpha,
                    0, blended);
    return blended;
  }
}

//cv::Mat TLRDraw::DrawProposal(const cv::Mat& src,
//                              const cv::Scalar& rect_color,
//                              const cv::Scalar& font_color) const {
//  std::vector<cv::Rect> proposal = tlr_->proposal();
//  std::vector<float> proposal_conf = tlr_->proposal_confidence();
//
//  cv::Mat result = src.clone();
//  for (int i = 0; i < proposal.size(); ++i) {
//    cv::rectangle(result, proposal[i], rect_color, 2);
//    cv::putText(result, std::to_string(proposal_conf[i]),
//                proposal[i].tl(), 1, 4, font_color);
//  }
//  return result;
//}
//
cv::Mat TLRDraw::DrawResult(
    const cv::Mat& src,
    const std::vector<std::string>& label_str,
    const std::vector<cv::Scalar>& rect_color_for_label,
    const std::vector<cv::Scalar>& font_color_for_label) const {
  std::vector<int> detected_label = tlr_->detected_label();
  std::vector<cv::Rect2f> detected_bbox = tlr_->detected_bbox();
  std::vector<float> detected_conf = tlr_->proposal_confidence();

  cv::Mat result = src.clone();
  for (int i = 0; i < detected_label.size(); ++i) {
    int label = detected_label[i];
    if (label) {
      //if (detected_bbox[i].width < 5) continue;

      cv::Scalar rect_color =
        (label > rect_color_for_label.size()) ? default_rect_color_ : rect_color_for_label[label-1];
      cv::Scalar font_color = 
        (label > font_color_for_label.size()) ? default_font_color_ : font_color_for_label[label-1];
    
      cv::rectangle(result, detected_bbox[i], rect_color, 2);

      std::string txt = label_str[label - 1] + ", " + std::to_string(detected_conf[i]);
      cv::putText(result, txt,
                  cv::Point(detected_bbox[i].x,
                            detected_bbox[i].y - 10),
                  1, 1, font_color, 2);

    }

    
  }

  return result;
}

} // namespace bgm