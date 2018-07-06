#include "tlr_validation.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

namespace bgm
{

void TLRValidation::RunTLR(float proposal_threshold) {
  result_.resize(gt_.size());
  for (int i = 0; i < gt_.size(); ++i) {
    cv::Mat src_img = cv::imread(gt_[i].img_name);
    tlr_->Run(src_img, proposal_threshold);

    const std::vector<cv::Rect2f>& detected_box = tlr_->detected_bbox();
    result_[i].bboxes.resize(detected_box.size());
    for (int j = 0; j < detected_box.size(); ++j) {
      result_[i].bboxes[j].set_x_min(detected_box[j].x);
      result_[i].bboxes[j].set_x_max(detected_box[j].x + detected_box[j].width - 1);
      result_[i].bboxes[j].set_y_min(detected_box[j].y);
      result_[i].bboxes[j].set_y_max(detected_box[j].y + detected_box[j].height- 1);
    }
    const std::vector<int>& detected_label = tlr_->detected_label();
    result_[i].labels.assign(detected_label.begin(), detected_label.end());
  }
}

void TLRValidation::ValidateDetection(int gt_min_width,
                                      float iou_threshold) {
  int tp = 0;
  int fp = 0;
  int tn = 0;
  int fn = 0;

  for (int i = 0; i < gt_.size(); ++i) {
    std::vector<bool> checked(gt_[i].bboxes.size(), false);

    for (int j = 0; j < gt_[i].bboxes.size(); ++j) {
      float gt_width = gt_[i].bboxes[j].x_max() - gt_[i].bboxes[j].x_min();

      int match_idx = Match(gt_[i].bboxes[j],result_[i].bboxes,
                            checked, iou_threshold);
      if (match_idx >= 0) {
        if(gt_width > gt_min_width) 
          tp++;
        checked[match_idx] = true;
      }
      else
        fn++;
    }

    for (int j = 0; j < checked.size(); ++j)
      if (!checked[j]) fp++;
  }
}

void TLRValidation::ValidateRecognition(int gt_min_width,
                                        float iou_threshold) {
  int tp = 0;
  int fp = 0;
  int tn = 0;
  int fn = 0;

  for (int i = 0; i < gt_.size(); ++i) {
    std::vector<bool> checked(gt_[i].bboxes.size(), false);

    for (int j = 0; j < gt_[i].bboxes.size(); ++j) {
      float gt_width = gt_[i].bboxes[j].x_max() - gt_[i].bboxes[j].x_min();

      int match_idx = Match(gt_[i].bboxes[j], gt_[i].labels[j],
                            result_[i].bboxes, result_[i].labels,
                            checked, iou_threshold);
      if (match_idx >= 0) {
        if(gt_width > gt_min_width) 
          tp++;
        checked[match_idx] = true;
      }
      else
        fn++;
    }

    for (int j = 0; j < checked.size(); ++j)
      if (!checked[j]) fp++;
  }
}

int TLRValidation::Match(const BB& gt, 
                         const std::vector<BB>& detected,
                         const std::vector<bool>& checked,
                         float iou_threshold) {
  int match_idx = -1;

  float best_iou = 0;
  for (int i = 0; i < detected.size(); ++i) {
    if (!checked[i]) {
      float iou = ComputeIOU(gt, detected[i]);
      if (iou > best_iou) {
        match_idx = i;
        best_iou = iou;
      }
    }
  }

  return (best_iou >= iou_threshold) ? match_idx : -1;
}

int TLRValidation::Match(
    const BB& gt_box, int gt_label,
    const std::vector<BB>& detected_box,
    const std::vector<int>& detected_label,
    const std::vector<bool>& checked,
    float iou_threshold) {
  int match_idx = Match(gt_box, detected_box, checked,
                        iou_threshold);

  if (match_idx >= 0 && gt_label == detected_label[match_idx])
    return match_idx;
  else
    return -1;
}

float TLRValidation::ComputeIOU(const BB& box1, const BB& box2) const {
  float inter_left = std::max(box1.x_min(), box2.x_min());
  float inter_right = std::min(box1.x_max(), box2.x_max());
  float inter_width = inter_right - inter_left;

  float inter_top = std::max(box1.y_min(), box2.y_min());
  float inter_bottom = std::min(box1.y_max(), box2.y_max());
  float inter_height = inter_bottom - inter_top;

  if (inter_width < 0 || inter_height < 0)
    return 0;
  else {
    float inter_area = inter_width * inter_height;
    float box1_area = (box1.x_max() - box1.x_max()) * (box1.y_max() - box1.y_min());
    float box2_area = (box2.x_max() - box2.x_max()) * (box2.y_max() - box2.y_min());
    float union_area = box1_area + box2_area - inter_area;

    return inter_area / union_area;
  }
}

} // namespace bgm