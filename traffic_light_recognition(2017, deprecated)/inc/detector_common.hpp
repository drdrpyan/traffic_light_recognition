#ifndef TLR_DETECTOR_COMMON_HPP_
#define TLR_DETECTOR_COMMON_HPP_

#include "bbox.hpp"

namespace bgm
{
  //typedef std::pair<int, BBox<int> > Detection;
  typedef struct _Detection
  {
    int label;
    BBox<int> bbox;
    float confidence;
  } Detection;
} // namespace bgm

#endif // !TLR_DETECTOR_COMMON_HPP_
