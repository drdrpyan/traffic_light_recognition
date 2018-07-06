#include "net_out_unit_size_parser.hpp"

#include "softmax_out_const_iterator.hpp"

namespace bgm
{

NetOutUnitSizeParser::NetOutUnitSizeParser(int receptive_field_width,
                                           int receptive_field_height,
                                           float unit, float aspect_ratio,
                                           int horizontal_stride, 
                                           int vertical_stride,
                                           bool vertical, bool normalized)
  : unit_(unit), aspect_ratio_(aspect_ratio), 
    horizontal_stride_(horizontal_stride), vertical_stride_(vertical_stride),
    vertical_(vertical), normalized_(normalized) {
  CHECK_GT(unit, 0);
  CHECK_GT(aspect_ratio, 0);
  CHECK_GE(horizontal_stride, 0);
  CHECK_GE(vertical_stride, 0);

  set_receptive_field_width(receptive_field_width);
  set_receptive_field_height(receptive_field_height);
}

void NetOutUnitSizeParser::Parse(
    const std::vector<caffe::Blob<float>*>& net_out,
    std::vector<std::vector<Detection> >* detection) {
  CHECK(detection);

  const caffe::Blob<float>& label = *(net_out[1]);
  const caffe::Blob<float>& size = *(net_out[0]);
  //const caffe::Blob<float>& label = *(net_out[0]);
  //const caffe::Blob<float>& size = *(net_out[1]);

  CheckNetOut(label, size);

  detection->resize(label.num());
  for (int n = 0; n < label.num(); n++)
    Parse(label, size, n, &(*detection)[n]);
}

bool NetOutUnitSizeParser::CheckNetOut(const caffe::Blob<float>& label,
                                       const caffe::Blob<float>& size) const {
  CHECK_EQ(label.num(), size.num());
  CHECK_EQ(label.width(), size.width());
  CHECK_EQ(label.height(), size.height());
  CHECK_EQ(size.channels(), 1);
  if (horizontal_stride_ == 0)
    CHECK_EQ(label.width(), 1);
  if (vertical_stride_ == 0)
    CHECK_EQ(label.height(), 1);

  return true;
}

void NetOutUnitSizeParser::Parse(
    const caffe::Blob<float>& label_result,
    const caffe::Blob<float>& size_result,
    int item_idx, std::vector<Detection>* detection) const {
  CHECK(detection);

  detection->clear();

  SoftmaxOutConstIterator<float> label_iter(label_result, item_idx);
  const float* size_iter = size_result.cpu_data() + size_result.offset(item_idx);
  int offset_x = 0;
  int offset_y = 0;
  for (int i = label_result.height(); i--; ) {
    for (int j = label_result.width(); j--; ) {
      int label;
      float confidence;
      label_iter.GetMax(&label, &confidence);
      float size = *size_iter;

      if (!IsIgnoredLabel(label)) {
        // LOG(INFO) << "size : " << size;

        BBox<int> bbox;
        GetBBox(size, &bbox);
        bbox.Shift(offset_x, offset_y);

        Detection obj;
        obj.label = label;
        obj.confidence = confidence;
        obj.bbox = bbox;
        detection->push_back(obj);
      }

      ++label_iter;
      ++size_iter;
      offset_x += horizontal_stride_;
    }
    offset_x = 0;
    offset_y += vertical_stride_;
  }
}



} // namespace bgm