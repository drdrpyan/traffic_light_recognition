#include "caffe_wrapper.hpp"

#include "tlr.hpp"
#include "proposal_active_recognizer.hpp"
#include "proposal_context_recognizer.hpp"
#include "grid_proposal.hpp"
#include "caffe_wrapper.hpp"
#include "D:\workspace\TLR\caffe\include\caffe\include_symbols.hpp"

#include "tlr_draw.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <memory>
#include <vector>

#include <algorithm>
#include <stdio.h>
#include <io.h>
#include <conio.h>

//std::string rp_net_model = "D:/workspace/TLR/model/deploy_tlr_grid_proposal_2.prototxt";
std::string rp_net_model = "D:/workspace/TLR/model/deploy_tlr_grid_proposal_resnet50.prototxt";
//std::string rp_net_weight = "D:/workspace/TLR/model/tlr_grid_proposal_iter_100000.caffemodel";
std::string rp_net_weight = "D:/workspace/TLR/model/tlr_grid_proposal_resnet50_iter_114000.caffemodel";
//std::string rp_net_weight = "F:/tlr_snapshot/grid_resnet50_2/tlr__iter_48000.caffemodel";
//std::string recog_net_model = "D:/workspace/TLR/model/deploy_tlr_class_bbox.prototxt";
//std::string recog_net_model = "D:/workspace/TLR/model/deploy_tlr_class_bbox_resnet50_bosch.prototxt";
std::string recog_net_model = "D:/workspace/TLR/model/deploy_tlr_class_bbox_context_squeeze_bn_lisabosch.prototxt";
//std::string recog_net_model = "D:/workspace/TLR/model/deploy_class_bbox_bosch.prototxt";
//std::string recog_net_weight = "D:/workspace/TLR/model/tlr_class_bbox_iter_300000.caffemodel";
//std::string recog_net_weight = "f:/tlr_snapshot/class_bbox_bosch/tlr__iter_186000.caffemodel";
//std::string recog_net_weight = "D:/workspace/TLR/model/tlr_class_bbox_resnet50_bosch_iter_213160.caffemodel";
std::string recog_net_weight = "D:/workspace/TLR/model/tlr_class_bbox_context_squeeze_bn_lisabosch_iter_71000.caffemodel";


static std::shared_ptr<bgm::TLR> tlr;
static std::shared_ptr<bgm::TLRDraw> tlr_draw;

void Init() {
  bgm::DNNWrapper* rp_net = new bgm::CaffeWrapper<float>(rp_net_model, rp_net_weight);
  bgm::RegionProposal* region_proposal = new bgm::GridProposal(rp_net, 7, 7);
  
  bgm::DNNWrapper* recog_net = new bgm::CaffeWrapper<float>(recog_net_model, recog_net_weight);
  //bgm::ProposalRecognizer* recognizer = 
  //    new bgm::ProposalAcitveRecognizer(
  //        recog_net, 
  //        new bgm::ProposalAcitveRecognizer::SortAndReturn(1, 0, 32, 32),
  //        cv::Size(127, 127), cv::Rect(48, 48, 32, 32));
  bgm::ProposalRecognizer* recognizer = 
      new bgm::ProposalContextRecognizer(
          recog_net, 
          new bgm::ProposalAcitveRecognizer::SortAndReturn(1, 0),
          48, 47, 48, 47, 1);

  std::vector<cv::Rect> sub_wins;
  int x = 0;
  int y = 0;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 6; ++j) {
      sub_wins.push_back(cv::Rect(x, y, 224, 224));
      x += 211;
    }
    x = 0;
    y += 211;
  }
  
  tlr.reset(new bgm::TLR);
  tlr->set_rpn(region_proposal);
  tlr->set_recognizer(recognizer);
  tlr->set_sub_wins(sub_wins);

  tlr_draw.reset(new bgm::TLRDraw(tlr));
}

void GetImgList(const std::string& path,
                std::vector<std::string>* list) {
  assert(list);
  list->clear();

  std::string ext[3] = {"png", "jpg", "bmp"};

  std::vector<std::string> partial_list;
  for (int j = 0; j < 3; ++j) {
    _finddata_t fd;
    intptr_t handle;
    int result = 1;
    handle = _findfirst((path + "\\*." + ext[j]).c_str(), &fd);

    if (handle == -1)
      continue;

    while (result != -1) {
      partial_list.push_back(fd.name);
      result = _findnext(handle, &fd);
    }

    _findclose(handle);
  }

  for (int j = 0; j < partial_list.size(); ++j)
    list->push_back(path + "\\" + partial_list[j]);
}
void TestLISA() {
   //std::string img_path = "D:/DB/traffic_light/tl_LISA/dayTest/daySequence1/frames";
  //std::string img_path = "D:/DB/traffic_light/tl_LISA/dayTrain/dayClip1/frames";
  std::string img_path = "D:/DB/traffic_light/Bosh_Small_Traffic_Lights_dataset/dataset_test_rgb/rgb/test";
  //std::string img_path = "D:/DB/traffic_light/tl_Lara/Lara3D_UrbanSeq1_JPG";
  //std::string img_path = "D:/DB/traffic_light/Bosh_Small_Traffic_Lights_dataset/dataset_train_rgb/rgb/train/2015-10-05-10-55-33_bag";
  std::vector<std::string> img_list;
  GetImgList(img_path, &img_list);

  for (int i = 400; i < img_list.size(); ++i) {
    std::cout << "frame : " << i << std::endl;
    
    cv::Mat img = cv::imread(img_list[i]);
    //cv::resize(img, img, cv::Size(1280, 960));
    
    tlr->Run(img, 0.45f);

    cv::Mat proposal_heatmap = tlr_draw->DrawHeatmap(img, 0.5);
    
    std::vector<std::string> label_str(4);
    label_str[0] = "RED";
    label_str[1] = "YELLOW";
    label_str[2] = "GREEN";
    label_str[3] = "OFF";
    std::vector<cv::Scalar> detection_color(4);
    detection_color[0] = cv::Scalar(0, 0, 255);
    detection_color[1] = cv::Scalar(0, 255, 255);
    detection_color[2] = cv::Scalar(0, 255, 0);
    detection_color[3] = cv::Scalar(0, 0, 0);
    cv::Mat detection_result = tlr_draw->DrawResult(img, 
                                                    label_str,
                                                    detection_color,
                                                    detection_color);

    cv::imshow("src image", img);
    cv::imshow("proposal heatmap", proposal_heatmap);
    cv::imshow("detection result", detection_result);

    //cv::imwrite("./proposal heatmap/" + std::to_string(i) + ".jpg", proposal_heatmap);
    cv::imwrite("f:/detection result/" + std::to_string(i) + ".jpg", detection_result);

    if (cv::waitKey(1) == 27)
      break;
  }
}

int main() {
  Init();

  TestLISA();


  return 0;
}