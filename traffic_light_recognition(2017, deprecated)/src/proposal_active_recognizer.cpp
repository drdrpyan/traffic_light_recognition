#include "proposal_active_recognizer.hpp"

namespace bgm
{

ProposalAcitveRecognizer::ProposalAcitveRecognizer(
    DNNWrapper* dnn, NetOutHandler* net_out_handler,
    const cv::Size& receptive_field_size,
    const cv::Rect& active_region) 
  : ProposalDNNRecognizer(dnn, net_out_handler),
    receptive_field_size_(receptive_field_size),
    active_region_(active_region) {

}

void ProposalAcitveRecognizer::Recognize(
    const cv::Mat& img, int top_k,
    const std::vector<cv::Rect>& proposal,
    std::vector<std::vector<int> >* label,
    std::vector<std::vector<cv::Rect2f> >* bbox,
    std::vector<std::vector<float> >* conf) {
  std::vector<cv::Rect> receptive_fields(proposal);

  for (auto iter = receptive_fields.begin();
       iter != receptive_fields.end(); ++iter) {
    iter->x -= active_region_.x;
    iter->y -= active_region_.y;
    iter->width = receptive_field_size_.width;
    iter->height = receptive_field_size_.height;
  }

  // 계층 구조 오류 유발 가능성 있음
  ProposalRecognizer::Recognize(img, top_k, receptive_fields,
                                label, bbox, conf);

  if (bbox) {
    for (int i = 0; i < bbox->size(); ++i) {
      for (int j = 0; j < (*bbox)[i].size(); ++j) {
        //(*bbox)[i][j].x += (active_region_.x + proposal[i].x);
        //(*bbox)[i][j].y += (active_region_.y + proposal[i].y);
        (*bbox)[i][j].x += (proposal[i].x);
        (*bbox)[i][j].y += (proposal[i].y);
      }
    }
  }
}

} // namespace bgm