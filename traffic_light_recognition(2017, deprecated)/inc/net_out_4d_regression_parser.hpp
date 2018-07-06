//#ifndef TLR_NET_OUT_4D_REGRESSION_PARSER_HPP_
//#define TLR_NET_OUT_4D_REGRESSION_PARSER_HPP_
//
//#include "net_out_parser.hpp"
//
//namespace bgm
//{
//
//class NetOut4DRegressionParser : public NetOutParser
//{
// public:
//   NetOut4DRegressionParser(int receptive_field_width,
//                            int receptive_field_height,
//                            int horizontal_stride = 0,
//                            int vertical_stride = 0,
//                            bool bbox_normalized = true);
//
//  virtual void Parse(const std::vector<const caffe::Blob<float>*>& net_out,
//                     std::vector<std::vector<Detection> >* detection) override;
//
//  void Parse(const caffe::Blob<float>& net_label_out,
//             std::vector<std::vector<Detection> >* detection);
//  void Parse(const caffe::Blob<float>& net_label_out,
//             const caffe::Blob<float>& net_bbox_out,
//             std::vector<std::vector<Detection> >* detection);
//
// private:
//  void Parse(const caffe::Blob<float>& net_label_out, int item_idx,
//             std::vector<Detection>* detection) const;
//  void Parse(const caffe::Blob<float>& net_label_out, 
//             const caffe::Blob<float>& net_bbox_out,
//             int item_idx, std::vector<Detection>* detection) const;
//
//  int receptive_field_width_;
//  int receptive_field_height_;
//  int horizontal_stride_;
//  int vertical_stride_;
//  bool bbox_normalized_;
//};
//
//// inline functions
//inline NetOut4DRegressionParser::NetOut4DRegressionParser(
//    int receptive_field_width, int receptive_field_height,
//    int horizontal_stride, int vertical_stride, bool bbox_normalized) 
//  : receptive_field_width_(receptive_field_width),
//    receptive_field_height_(receptive_field_height), 
//    horizontal_stride_(horizontal_stride), vertical_stride_(vertical_stride),
//    bbox_normalized_(bbox_normalized) {
//  CHECK_GT(receptive_field_width_, 0);
//  CHECK_GT(receptive_field_height_, 0);
//  CHECK_GE(horizontal_stride_, 0);
//  CHECK_GE(vertical_stride_, 0);
//}
//
//
//} // namespace bgm
//
//#endif // !TLR_NET_OUT_4D_REGRESSION_PARSER_HPP_