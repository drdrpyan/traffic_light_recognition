#include "proposal_context_recognizer.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace bgm
{

ProposalContextRecognizer::ProposalContextRecognizer(
    DNNWrapper* dnn, NetOutHandler* net_out_handler,
    int margin_left, int margin_right,
    int margin_up, int margin_down,
    int batch_size) 
  : ProposalDNNRecognizer(dnn, net_out_handler),
    margin_left_(margin_left), margin_right_(margin_right),
    margin_up_(margin_up), margin_down_(margin_down),
    batch_size_(batch_size) {

}

void ProposalContextRecognizer::Recognize(
    const cv::Mat& img, int top_k,
    const std::vector<cv::Rect>& proposal,
    std::vector<std::vector<int> >* label,
    std::vector<std::vector<cv::Rect2f> >* bbox,
    std::vector<std::vector<float> >* conf) {
  label->clear();
  if (bbox)
    bbox->clear();
  if (conf)
    conf->clear();

  std::vector<cv::Rect> context;
  GetContext(proposal, &context);  

  std::vector<cv::Mat> roi(batch_size_ * 2);
  std::vector<std::vector<int> > temp_label;
  std::vector<std::vector<cv::Rect2f> > temp_bbox;
  std::vector<std::vector<float> > temp_conf;

  int i = 0;
  while (proposal.size() - i >= batch_size_) {
    for (int b = 0; b < batch_size_; ++b) {
      roi[b] = ExtractROI(img, context[i + b]);
      cv::resize(ExtractROI(img, proposal[i + b]),
                 roi[batch_size_ + b],
                 context[i + b].size());

      //// debug
      //cv::Mat context_img = roi[b];
      //cv::Mat proposal_img = roi[batch_size_ + b];

    }
    // 계층 구조 오류 유발 가능성 있음
    ProposalDNNRecognizer::Recognize(roi, top_k, &temp_label,
                                     bbox ? &temp_bbox : NULL,
                                     conf ? &temp_conf : NULL);
    label->insert(label->end(),
                  temp_label.begin(), temp_label.end());
    if(bbox)
      bbox->insert(bbox->end(),
                  temp_bbox.begin(), temp_bbox.end());
    if(conf)
      conf->insert(conf->end(),
                  temp_conf.begin(), temp_conf.end());

    temp_label.clear();
    temp_bbox.clear();
    temp_conf.clear();
    i += batch_size_;
  }

  if (proposal.size() - i > 0) {
    assert(("Not verified yet", 0)); // 아직 이부분 미검증됨

    int remain = proposal.size() - i;
    std::vector<cv::Mat> roi(batch_size_ * 2);
    for (int b = 0; b < batch_size_; ++b) {
      if (b < remain) {
        roi[b] = ExtractROI(img, context[i + b]);
        cv::resize(ExtractROI(img, proposal[i + b]),
                   roi[batch_size_ + b],
                   context[i + b].size());
      }
      else {
        roi[b] = cv::Mat(context[0].size(), img.type());
        roi[batch_size_ + b] = cv::Mat(context[0].size(), img.type());
      }
    }

    // 계층 구조 오류 유발 가능성 있음
    ProposalDNNRecognizer::Recognize(roi, top_k, &temp_label,
                                     bbox ? &temp_bbox : NULL,
                                     conf ? &temp_conf : NULL);
    label->insert(label->end(),
                  temp_label.begin(), temp_label.end());
    if(bbox)
      bbox->insert(bbox->end(),
                  temp_bbox.begin(), temp_bbox.end());
    if(conf)
      conf->insert(conf->end(),
                  temp_conf.begin(), temp_conf.end());
  }

  if (bbox) {
    for (int i = 0; i < bbox->size(); ++i) {
      for (int j = 0; j < (*bbox)[i].size(); ++j) {
        //(*bbox)[i][j].x += (active_region_.x + proposal[i].x);
        //(*bbox)[i][j].y += (active_region_.y + proposal[i].y);
        (*bbox)[i][j].x += (context[i].x);
        (*bbox)[i][j].y += (context[i].y);
      }
    }
  }
}

void ProposalContextRecognizer::GetContext(
    const std::vector<cv::Rect>& proposal,
    std::vector<cv::Rect>* context) const {
  assert(context);

  context->assign(proposal.begin(), proposal.end());

  for (auto iter = context->begin();
       iter != context->end(); ++iter) {
    iter->x -= margin_left_;
    iter->y -= margin_up_;
    iter->width += (margin_left_ + margin_right_);
    iter->height += (margin_up_ + margin_down_);
  }
}

cv::Mat ProposalContextRecognizer::ExtractROI(
    const cv::Mat& img, const cv::Rect& roi) const {
  cv::Mat base(cv::Size(roi.width, roi.height),
               img.type(), cv::Scalar(0));

  int left = std::max(0, roi.x);
  //int right = std::min(img.cols - 1, 
  //                     proposal.x + proposal.width - 1);
  int right = std::min(img.cols - 1, 
                       roi.x + roi.width);
  int top = std::max(0, roi.y);
  //int bottom = std::min(img.rows - 1,
  //                      proposal.y + proposal.height - 1);
  int bottom = std::min(img.rows - 1,
                        roi.y + roi.height);
  cv::Rect src_roi(cv::Point(left, top), 
                   cv::Point(right, bottom));

  int offset_x = left - roi.x;
  int offset_y = top - roi.y;
  cv::Rect dst_roi(offset_x, offset_y,
                   src_roi.width, src_roi.height);

  img(src_roi).copyTo(base(dst_roi));

  //ofs << img_cnt++ << ' ' << left << ' ' << top << std::endl;
  //cv::imwrite("f:/tlr_neg/" + std::to_string(img_cnt++) + ".jpg", base);

  return base;
}

} // namespace bgm