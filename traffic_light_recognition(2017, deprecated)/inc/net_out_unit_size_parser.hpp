#ifndef TLR_NET_OUT_UNIT_SIZE_PARSER_HPP_
#define TLR_NET_OUT_UNIT_SIZE_PARSER_HPP_

#include "net_out_parser.hpp"

namespace bgm
{

class NetOutUnitSizeParser : public NetOutParser
{
 public:
  NetOutUnitSizeParser();
  NetOutUnitSizeParser(int receptive_field_width, int receptive_field_height,
                       float unit, float aspect_ratio,
                       int horizontal_stride = 0, int vertical_stride = 0,
                       bool vertical = true, bool normalized = true);
  virtual void Parse(const std::vector<caffe::Blob<float>*>& net_out,
                     std::vector<std::vector<Detection> >* detection) override;

  void set_receptive_field_width(int width);
  void set_receptive_field_height(int height);
  void set_unit(float unit);
  void set_aspect_ratio(float aspect_ratio);
  void set_horizontal_stride(int stride);
  void set_vertical_stride(int stride);
  void set_vertical(bool is_vertical);
  void set_normalized(bool is_normalized);
  
 private:
   bool CheckNetOut(const caffe::Blob<float>& label,
                    const caffe::Blob<float>& size) const;
   void Parse(const caffe::Blob<float>& net_label_out,
              const caffe::Blob<float>& net_bbox_out,
              int item_idx, std::vector<Detection>* detection) const;
   void GetBBox(float size, BBox<int>* bbox) const;

  int receptive_field_width_;
  int receptive_field_height_;
  float unit_;
  float aspect_ratio_;
  int horizontal_stride_;
  int vertical_stride_;
  bool vertical_;
  bool normalized_;

  float center_x_;
  float center_y_;
};

// inline functions
inline NetOutUnitSizeParser::NetOutUnitSizeParser() {
}

inline void NetOutUnitSizeParser::set_receptive_field_width(int width) {
  CHECK_GT(width, 0);
  receptive_field_width_ = width;
  center_x_ = width / 2.0f;
}

inline void NetOutUnitSizeParser::set_receptive_field_height(int height) {
  CHECK_GT(height, 0);
  receptive_field_height_ = height;
  center_y_ = height / 2.0f;
}

inline void NetOutUnitSizeParser::set_unit(float unit) {
  CHECK_GT(unit, 0);
  unit_ = unit;
}

inline void NetOutUnitSizeParser::set_aspect_ratio(float aspect_ratio) {
  CHECK_GT(aspect_ratio, 0);
  aspect_ratio_ = aspect_ratio;
}

inline void NetOutUnitSizeParser::set_horizontal_stride(int stride) {
  CHECK_GE(stride, 0);
  horizontal_stride_ = stride;
}

inline void NetOutUnitSizeParser::set_vertical_stride(int stride) {
  CHECK_GE(stride, 0);
  vertical_stride_ = stride;
}

inline void NetOutUnitSizeParser::set_vertical(bool is_vertical) {
  vertical_ = is_vertical;
}

inline void NetOutUnitSizeParser::set_normalized(bool is_normalized) {
  normalized_ = is_normalized;
}

inline void NetOutUnitSizeParser::GetBBox(float size, BBox<int>* bbox) const {
  CHECK(bbox);
  float bbox_size = std::max(1.0f, std::roundf(size)) * unit_;
  float half_width = (vertical_ ? bbox_size / aspect_ratio_ : bbox_size) / 2.0f;
  float half_height = (vertical_ ? bbox_size : bbox_size / aspect_ratio_) / 2.0f;
  bbox->Set(center_x_ - half_width, center_y_ - half_height,
            center_x_ + half_width, center_y_ + half_height);
}

} // namespace bgm

#endif // !TLR_NET_OUT_UNIT_SIZE_PARSER_HPP_
