#include "proposal_recognizer.hpp"

#include <opencv2/highgui.hpp>

#include <fstream>
static int img_cnt = 0;
static std::ofstream ofs("f:/tlr_neg/offset.txt");

namespace bgm
{

void ProposalRecognizer::Recognize(
    const cv::Mat& proposal, 
    int* label, cv::Rect2f* bbox, float* conf) {
  assert(label);

  std::vector<int> labels;
  std::vector<cv::Rect2f> bboxes;
  std::vector<float> confs;

  if (bbox && conf) {  
    Recognize(proposal, 1, &labels, &bboxes, &confs);
    *label = labels[0];
    *bbox = bboxes[0];
    *conf = confs[0];
  }
  else if (bbox) {
    Recognize(proposal, 1, &labels, &bboxes);
    *label = labels[0];
    *bbox = bboxes[0];
  }
  else {
    Recognize(proposal, 1, &labels);
    *label = labels[0];
  }
}

void ProposalRecognizer::Recognize(
    const cv::Mat& proposal, int top_k,
    std::vector<int>* label,
    std::vector<cv::Rect2f>* bbox,
    std::vector<float>* conf) {
  assert(label);

  std::vector<cv::Mat> proposals(1, proposal);
  std::vector<std::vector<int> > labels;
  std::vector<std::vector<cv::Rect2f> > bboxes;
  std::vector<std::vector<float> > confs;

  if (bbox && conf) {  
    Recognize(proposals, top_k, &labels, &bboxes, &confs);
    *label = labels[0];
    *bbox = bboxes[0];
    *conf = confs[0];
  }
  else if (bbox) {
    Recognize(proposals, top_k, &labels, &bboxes);
    *label = labels[0];
    *bbox = bboxes[0];
  }
  else {
    Recognize(proposals, top_k, &labels);
    *label = labels[0];
  }
}

void ProposalRecognizer::Recognize(
    const std::vector<cv::Mat>& proposal,
    std::vector<int>* label,
    std::vector<cv::Rect2f>* bbox,
    std::vector<float>* conf) {
  std::vector<std::vector<int> > labels(proposal.size());
  std::vector<std::vector<cv::Rect2f> > bboxes(proposal.size());
  std::vector<std::vector<float> > confs(proposal.size());

  if (bbox && conf) {  
    Recognize(proposal, 1, &labels, &bboxes, &confs);
    for (int i = 0; i < proposal.size(); ++i) {
      (*label)[i] = labels[i][0];
      (*bbox)[i] = bboxes[i][0];
      (*conf)[i] = confs[i][0];
    }
  }
  else if (bbox) {
    Recognize(proposal, 1, &labels, &bboxes);
    for (int i = 0; i < proposal.size(); ++i) {
      (*label)[i] = labels[i][0];
      (*bbox)[i] = bboxes[i][0];
    }
  }
  else {
    Recognize(proposal, 1, &labels);
    for (int i = 0; i < proposal.size(); ++i)
      (*label)[i] = labels[i][0];
  }
}

void ProposalRecognizer::Recognize(
    const cv::Mat& img,
    const std::vector<cv::Rect>& proposal,
    std::vector<int>* label,
    std::vector<cv::Rect2f>* bbox,
    std::vector<float>* conf) {
  std::vector<std::vector<int> > labels(proposal.size());
  std::vector<std::vector<cv::Rect2f> > bboxes(proposal.size());
  std::vector<std::vector<float> > confs(proposal.size());

  label->resize(proposal.size());

  if (bbox && conf) {  
    bbox->resize(proposal.size());
    conf->resize(proposal.size());

    Recognize(img, 1, proposal, &labels, &bboxes, &confs);
    for (int i = 0; i < proposal.size(); ++i) {
      (*label)[i] = labels[i][0];
      (*bbox)[i] = bboxes[i][0];
      (*conf)[i] = confs[i][0];
    }
  }
  else if (bbox) {
    bbox->resize(proposal.size());

    Recognize(img, 1, proposal, &labels, &bboxes);
    for (int i = 0; i < proposal.size(); ++i) {
      (*label)[i] = labels[i][0];
      (*bbox)[i] = bboxes[i][0];
    }
  }
  else {
    Recognize(img, 1, proposal, &labels);
    for (int i = 0; i < proposal.size(); ++i)
      (*label)[i] = labels[i][0];
  }
}

void ProposalRecognizer::Recognize(
    const cv::Mat& img, int top_k,
    const std::vector<cv::Rect>& proposal,
    std::vector<std::vector<int> >* label,
    std::vector<std::vector<cv::Rect2f> >* bbox,
    std::vector<std::vector<float> >* conf) {
  std::vector<cv::Mat> proposals(proposal.size());

  for (int i = 0; i < proposal.size(); ++i)
    proposals[i] = ExtractProposal(img, proposal[i]);
  
  Recognize(proposals, top_k, label, bbox, conf);
}

cv::Mat ProposalRecognizer::ExtractProposal(
    const cv::Mat& img, const cv::Rect& proposal) const {
  cv::Mat base(cv::Size(proposal.width, proposal.height),
               img.type(), cv::Scalar(0));

  int left = std::max(0, proposal.x);
  //int right = std::min(img.cols - 1, 
  //                     proposal.x + proposal.width - 1);
  int right = std::min(img.cols - 1, 
                       proposal.x + proposal.width);
  int top = std::max(0, proposal.y);
  //int bottom = std::min(img.rows - 1,
  //                      proposal.y + proposal.height - 1);
  int bottom = std::min(img.rows - 1,
                        proposal.y + proposal.height);
  cv::Rect src_roi(cv::Point(left, top), 
                   cv::Point(right, bottom));

  int offset_x = left - proposal.x;
  int offset_y = top - proposal.y;
  cv::Rect dst_roi(offset_x, offset_y,
                   src_roi.width, src_roi.height);

  img(src_roi).copyTo(base(dst_roi));

  //ofs << img_cnt++ << ' ' << left << ' ' << top << std::endl;
  //cv::imwrite("f:/tlr_neg/" + std::to_string(img_cnt++) + ".jpg", base);

  return base;
}

}