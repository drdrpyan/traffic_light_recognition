//#include "caffe_sw_detector_factory.hpp"
//
//#include "layer_instantiation.hpp"
//#include "caffe/proto/caffe.pb.h"
//
//#include <opencv2/highgui.hpp>
//#include <opencv2/imgproc.hpp>
//
//const std::string MODEL_FILE = "D:/workspace/TLR/model/deploy_redgreen_sizeunit_offsetfc.prototxt";
// //const std::string TRAINED_FILE = "D:/workspace/TLR/model/tlr_redgreen_sizeunit_negative2_iter_1000000.caffemodel";;
//const std::string TRAINED_FILE = "D:/workspace/TLR/model/tlr_redgreen_sizeunit_3x3patch_iter_85202.caffemodel";
//const bool USE_GPU = true;
//
//std::shared_ptr<bgm::CaffeSWDetector> detector;
//
//void InitDetector();
//void DrawGT(const caffe::PatchDatum& datum, cv::Mat* dst);
//void DrawResults(const std::vector<bgm::Detection>& result,
//                 cv::Mat* dst);
//
//int main() {
//  InitDetector();
//  // std::string db_path = "D:/DB/traffic_light/tl_LISA/LMDB/lisa_patch_227_color_jpg_all_shuffle/";
//  std::string db_path = "D:/DB/traffic_light/tl_LISA/LMDB/lisa_patch_227_color_jpg_all_shuffle_active16/";
//
//  std::unique_ptr<caffe::db::DB> db(caffe::db::GetDB("lmdb"));
//  db->Open(db_path, caffe::db::READ);
//  std::unique_ptr<caffe::db::Cursor> cursor(db->NewCursor());
//
//  caffe::TransformationParameter transform_param;
//  caffe::DataTransformer<float> transformer(transform_param, caffe::Phase::TEST);
//  std::unique_ptr<caffe::DataTransformer<float> > data_transformer;
//  caffe::Blob<float> img_blob;
//  std::vector<int> img_blob_shape(4);
//  img_blob_shape[0] = 1;
//  img_blob_shape[1] = 3;
//  img_blob_shape[2] = 227;
//  img_blob_shape[3] = 227;
//  img_blob.Reshape(img_blob_shape);
//
//  caffe::PatchDatum patch_datum;
//
//  while (1) {
//    patch_datum.ParseFromString(cursor->value());
//
//    cv::Mat img = caffe::DecodeDatumToCVMatNative(patch_datum.patch_img());
//    
//    cv::Mat gt_img = img.clone();
//    DrawGT(patch_datum, &gt_img);
//    cv::imshow("gt", gt_img);
//
//    detector->Detect(img);
//    const std::vector<bgm::Detection>& detection1 = detector->GetRawResults(0);
//    cv::Mat result1 = img.clone();
//    DrawResults(detection1, &result1);
//    cv::imshow("result1", result1);
//
//    //caffe::Datum img_datum(patch_datum.patch_img());
//    //caffe::DecodeDatumNative(&img_datum);
//    //transformer.Transform(img_datum, &img_blob);
//    //detector->Detect(img_blob);
//    //const std::vector<bgm::Detection>& detection2 = detector->GetRawResults(0);
//    //cv::Mat result2 = img.clone();
//    //DrawResults(detection2, &result2);
//    //cv::imshow("result2", result2);
//
//    //detector->Detect(img);
//    //const std::vector<bgm::Detection>& detection1 = detector->GetRawResults(0);
//    //cv::Mat result1 = img.clone();
//    //DrawResults(detection1, &result1);
//    //cv::imshow("result1", result1);
//
//    cursor->Next();
//
//    if (cv::waitKey(0) == 27)
//      break;
//    else
//      continue;
//  }
//
//
////  std::string img_file = "D:/DB/traffic_light/tl_LISA/dayTrain/dayClip1/frames/dayClip1--00085.png";
////  cv::Mat img = cv::imread(img_file);
////  cv::copyMakeBorder(img, img, 1, 2, 1, 2, IPL_BORDER_REPLICATE);
////
////  detector->Detect(img);
////  auto raw_result = detector->GetRawResults(0);
////
////  cv::Mat result_img = img.clone();
////  DrawResults(raw_result, &result_img);
////
////  cv::imshow("detection raw result", result_img);
////
////  while (1) {
////    if (cv::waitKey(0) == 27)
////      break;
////  }
//
//  return 0;
//}
//
//void InitDetector() {
//  //detector = bgm::CaffeSWDetectorFactory::Get4DRegBBoxDetector(227, 227, 16, 16);
//  detector = bgm::CaffeSWDetectorFactory::GetMidActiveUnitSizeDetector(
//      227, 227, 16, 1.5, 16, 16);
//  detector->InitNet(MODEL_FILE, TRAINED_FILE, USE_GPU);
//}
//
//void DrawGT(const caffe::PatchDatum& datum, cv::Mat* dst) {
//  CHECK(dst);
//  //int x_min = std::min(std::max(0.0f, datum.bbox_xmin() * 227), 226.0f);
//  //int y_min = std::min(std::max(0.0f, datum.bbox_ymin() * 227), 226.0f);
//  //int x_max = std::min(std::max(0.0f, datum.bbox_xmax() * 227), 226.0f);
//  //int y_max = std::min(std::max(0.0f, datum.bbox_ymax() * 227), 226.0f);
//  int x_min = datum.bbox_xmin() - datum.patch_offset_xmin();
//  int y_min = datum.bbox_ymin() - datum.patch_offset_ymin();
//  int x_max = datum.bbox_xmax() - datum.patch_offset_xmin();
//  int y_max = datum.bbox_ymax() - datum.patch_offset_ymin();
//
//  cv::rectangle(*dst, cv::Point(x_min, y_min), cv::Point(x_max, y_max),
//                cv::Scalar(255, 0, 0), 2);
//  LOG(INFO) << "GT : label=" << datum.label()
//    << " bbox=(" << x_min << ", " << y_min << ", " << x_max << ", " << y_max << ")";
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