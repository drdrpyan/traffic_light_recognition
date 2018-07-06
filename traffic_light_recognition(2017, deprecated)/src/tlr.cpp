#include "tlr.hpp"

#include "dnn_unit.hpp"

namespace bgm
{

void TLR::Run(const cv::Mat& img, float proposal_threshold) {
  ClearPrevResults();

  cv::Mat single_img;
  if (img.channels() == 1)
    img.convertTo(single_img, CV_32FC1);
  else if (img.channels() == 3)
    img.convertTo(single_img, CV_32FC3);

  //ComputeProposal(img, proposal_threshold);
  //DetectFromProposal(img);
  ComputeProposal(single_img, proposal_threshold);
  DetectFromProposal(single_img);
}

void TLR::ClearPrevResults() {
  proposal_.clear();
  proposal_confidence_.clear();
  detected_label_.clear();
  detected_bbox_.clear();
  detected_confidence_.clear();
}

void TLR::ComputeProposal(const cv::Mat& img,
                          float threshold) {
  assert(img.channels() == 1 || img.channels() == 3);

  //cv::Mat single_img;
  //if (img.channels() == 1)
  //  img.convertTo(single_img, CV_32FC1);
  //else if (img.channels() == 3)
  //  img.convertTo(single_img, CV_32FC3);


  std::vector<cv::Mat> net_inputs(sub_wins_.size());
  for (int i = 0; i < net_inputs.size(); ++i) {
    //net_inputs[i] = single_img(sub_wins_[i]).clone();
    net_inputs[i] = img(sub_wins_[i]).clone();
  }
  //DNNUnit net_input(net_inputs);

  std::vector<std::vector<cv::Rect> > proposal;
  std::vector<std::vector<float> > proposal_conf;
  rpn_->Propose(net_inputs, threshold,
                &proposal, &proposal_conf);

  proposal_.clear();
  proposal_confidence_.clear();
  for (int i = 0; i < proposal.size(); ++i) {
    int offset_x = sub_wins_[i].x;
    int offset_y = sub_wins_[i].y;
    std::vector<cv::Rect>& sub_win_proposal = proposal[i];
    std::vector<float>& sub_win_proposal_conf = proposal_conf[i];
    for (int j = 0; j < proposal[i].size(); ++j) {
      sub_win_proposal[j].x += offset_x;
      sub_win_proposal[j].y += offset_y;

      proposal_.push_back(sub_win_proposal[j]);
      proposal_confidence_.push_back(sub_win_proposal_conf[j]);
    }
  }
}

void TLR::DetectFromProposal(const cv::Mat& img) {
  detected_label_.clear();
  detected_bbox_.clear();
  detected_confidence_.clear();

  if (!proposal_.empty()) {
    recognizer_->Recognize(img, proposal_,
                           &detected_label_,
                           &detected_bbox_,
                           &detected_confidence_);
  }
}


}