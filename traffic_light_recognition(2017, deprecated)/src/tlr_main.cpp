//#include "caffe_sw_detector_factory.hpp"
//
//#include "layer_instantiation.hpp"
//
//#include <opencv2/highgui.hpp>
//#include <opencv2/imgproc.hpp>
//
//#include <stdio.h>
//#include <io.h>
//#include <conio.h>
//
////const std::string MODEL_FILE = "D:/workspace/TLR/model/deploy_redgreen_sizeunit.prototxt";
//const std::string MODEL_FILE = "D:/workspace/TLR/model/deploy_redgreen_sizeunit_offsetfc.prototxt";
////const std::string TRAINED_FILE = "D:/workspace/TLR/model/tlr_redgreen_sizeunit_map_iter_4317.caffemodel";;
//const std::string TRAINED_FILE = "D:/workspace/TLR/model/tlr_mid_activation_9x9patch6_iter_500000.caffemodel";;
////const std::string IMG_PATH = "D:/DB/traffic_light/tl_LISA/dayTrain/dayClip1/frames";
//const std::string IMG_PATH = "D:/DB/traffic_light/tl_Lara/Lara3D_UrbanSeq1_JPG/";
//const std::string IMG_EXT = "jpg"; // lisa : png, lara : jpg
//const bool USE_GPU = true;
//
//std::shared_ptr<bgm::CaffeSWDetector> detector;
//
//void InitDetector();
//void DrawResults(const std::vector<bgm::Detection>& result,
//                 cv::Mat* dst);
//void DrawDensityMap(const cv::Mat& img, std::vector<cv::Mat>* results);
//void GetImgList(const std::string& path, const std::string& ext,
//                std::vector<std::string>* list);
//
//int main() {
//  InitDetector();
//  std::vector<std::string> img_list;
//  GetImgList(IMG_PATH, IMG_EXT, &img_list);
//  
//  for (int i = 6676; i < img_list.size(); i++) {
//    LOG(INFO) << "frame : " << i << ", " << img_list[i];
//    std::string img_file = IMG_PATH + '/' + img_list[i];
//    cv::Mat img = cv::imread(img_file);
//    cv::Mat padded;
//    cv::resize(img, img, cv::Size(1280, 960));
//    //cv::copyMakeBorder(img, img, 1, 2, 1, 2, IPL_BORDER_REPLICATE);
//    cv::copyMakeBorder(img, padded, 105, 106, 105, 106, IPL_BORDER_CONSTANT);
//
//    detector->Detect(padded);
//    auto raw_result = detector->GetRawResults(0);
//
//    cv::Mat result_img = padded.clone();
//    DrawResults(raw_result, &result_img);
//    
//    std::vector<cv::Mat> density_map;
//    DrawDensityMap(img, &density_map);
//
//    cv::imshow("detection raw result", result_img);
//
//    cv::imshow("bg density map", density_map[0]);
//    cv::imshow("red density map", density_map[1]);
//    cv::imshow("green density map", density_map[2]);
//
//    while (1) {
//      if (cv::waitKey(0) == 27)
//        break;
//    }
//  }
//
//
//  return 0;
//}
//
//void InitDetector() {
//  detector = bgm::CaffeSWDetectorFactory::GetMidActiveUnitSizeDetector(
//      227, 227, 16, 1.5, 16, 16);
//  detector->InitNet(MODEL_FILE, TRAINED_FILE, USE_GPU);
//}
//
//void DrawResults(const std::vector<bgm::Detection>& result,
//                 cv::Mat* dst) {
//  CHECK(dst);
//
//  LOG(INFO) << "Result";
//
//  for (int i = 0; i < result.size(); i++) {
//    cv::Scalar rect_color;
//    switch (result[i].label) {
//      case 1:
//        rect_color = cv::Scalar(0, 0, 255); // red
//        break;
//      case 2:
//        rect_color = cv::Scalar(0, 255, 0); // green
//        break;
//      case 0:
//        rect_color = cv::Scalar(0, 0, 0);
//        break;
//    }
//
//    const bgm::BBox<int>& bbox = result[i].bbox;
//    int x_min = std::min(std::max(0, bbox.x_min()), dst->cols - 1);
//    int y_min = std::min(std::max(0, bbox.y_min()), dst->rows - 1);
//    int x_max = std::max(std::min(dst->cols - 1, bbox.x_max()), 0);
//    int y_max = std::max(std::min(dst->rows - 1, bbox.y_max()), 0);
//    cv::rectangle(*dst, cv::Point(x_min, y_min), cv::Point(x_max, y_max),
//                  rect_color, 2);
//    LOG(INFO) << "\tlabel=" << result[i].label
//    << " bbox=(" << x_min << ", " << y_min << ", " << x_max << ", " << y_max << ")"
//    << " confidence=" << result[i].confidence;
//  }
//}
//
//void DrawDensityMap(const cv::Mat& img, std::vector<cv::Mat>* results) {
//  std::vector<cv::Mat> maps;
//  detector->GetLabelDensityMap(&maps);
//  cv::Mat debug = maps[0];
//  for (int i = 0; i < maps.size(); ++i)
//    maps[i].convertTo(maps[i], CV_8UC1, 255.0);
//  debug = maps[0];
//  
//
//  cv::Mat black = cv::Mat::zeros(maps[0].rows, maps[0].cols, CV_8UC1);
//
//  std::vector<cv::Mat> bg_channels(3);
//  bg_channels[0] = maps[0];
//  bg_channels[1] = black;
//  bg_channels[2] = black;
//  cv::Mat bg;
//  cv::merge(bg_channels, bg);
//  cv::resize(bg, bg, img.size(), 0, 0, CV_INTER_NN);
//
//  std::vector<cv::Mat> red_channels(3);
//  red_channels[0] = black;
//  red_channels[1] = black;
//  red_channels[2] = maps[1];
//  cv::Mat red;
//  cv::merge(red_channels, red);
//  cv::resize(red, red, img.size(), 0, 0, CV_INTER_NN);
//
//  std::vector<cv::Mat> green_channels(3);
//  green_channels[0] = black;
//  green_channels[1] = maps[2];
//  green_channels[2] = black;
//  cv::Mat green;
//  cv::merge(green_channels, green);
//  cv::resize(green, green, img.size(), 0, 0, CV_INTER_NN);
//
//  results->resize(3);
//  cv::addWeighted(img, 0.5, bg, 0.5, 0.0, (*results)[0]);
//  cv::addWeighted(img, 0.5, red, 0.5, 0.0, (*results)[1]);
//  cv::addWeighted(img, 0.5, green, 0.5, 0.0, (*results)[2]);
//}
//
//void GetImgList(const std::string& path, const std::string& ext,
//                std::vector<std::string>* list) {
//  CHECK(list);
//  list->clear();
//
//  _finddata_t fd;
//  intptr_t handle;
//  int result = 1;
//  handle = _findfirst((path + "\\*." + ext).c_str(), &fd);
//
//  while (result != -1) {
//    list->push_back(fd.name);
//    result = _findnext(handle, &fd);
//  }
//
//  _findclose(handle);
//}