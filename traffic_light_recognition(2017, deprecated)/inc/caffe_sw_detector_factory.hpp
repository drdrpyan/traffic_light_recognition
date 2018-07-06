#ifndef TLR_CAFFE_SW_DETECTOR_FACTORY_HPP_
#define TLR_CAFFE_SW_DETECTOR_FACTORY_HPP_

#include "caffe_sw_detector.hpp"

#include <memory>

namespace bgm
{

class CaffeSWDetectorFactory
{
 public:
   //static std::shared_ptr<CaffeSWDetector> Get4DRegBBoxDetector(
   //   int receptive_field_width, int receptive_field_height,
   //   int horizontal_stride, int vertical_stride, bool bbox_normalized = true);
  static std::shared_ptr<CaffeSWDetector> GetMidActiveUnitSizeDetector(
     int receptive_field_width, int receptive_field_height,
     float unit, float aspect_ratio, int horizontal_stride, int vertical_stride,
     bool vertical = true, bool normalized = true);
}; // class CaffeSWDetectorFactory

} // namespace bgm

#endif // !TLR_CAFFE_SW_DETECTOR_FACTORY_HPP_
