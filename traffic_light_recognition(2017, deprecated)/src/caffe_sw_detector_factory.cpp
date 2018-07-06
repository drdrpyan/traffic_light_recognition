#include "caffe_sw_detector_factory.hpp"

//#include "net_out_4d_regression_parser.hpp"
#include "net_out_unit_size_parser.hpp"

namespace bgm
{

//std::shared_ptr<CaffeSWDetector> CaffeSWDetectorFactory::Get4DRegBBoxDetector(
//      int receptive_field_width, int receptive_field_height,
//      int horizontal_stride, int vertical_stride, bool bbox_normalized) {
//  CaffeSWDetector* detector = new CaffeSWDetector;
//  detector->set_input_resizing(false);
//
//  NetOut4DRegressionParser* parser = new NetOut4DRegressionParser(
//      receptive_field_width, receptive_field_height, horizontal_stride,
//      vertical_stride, bbox_normalized);
//  //parser->AddIgnoredLabel(0);
//  detector->set_net_out_parser(parser);
//
//  return std::shared_ptr<CaffeSWDetector>(detector);
//}

std::shared_ptr<CaffeSWDetector> CaffeSWDetectorFactory::GetMidActiveUnitSizeDetector(
  int receptive_field_width, int receptive_field_height,
  float unit, float aspect_ratio, int horizontal_stride, int vertical_stride,
  bool vertical, bool normalized) {
  CaffeSWDetector* detector = new CaffeSWDetector;
  detector->set_input_resizing(false);
  NetOutUnitSizeParser* parser = new NetOutUnitSizeParser(
      receptive_field_width, receptive_field_height, unit, aspect_ratio,
      horizontal_stride, vertical_stride, vertical, normalized);
  parser->AddIgnoredLabel(0);
  detector->set_net_out_parser(parser);

  return std::shared_ptr<CaffeSWDetector>(detector);
}

}