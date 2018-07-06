//#ifndef TLR_NET_OUT_GRID_SIZE_PARSER_HPP_
//#define TLR_NET_OUT_GRID_SIZE_PARSER_HPP_
//
//#include "net_out_4d_regression_parser.hpp"
//
//namespace bgm
//{
//
//class NetOutGridSizeParser : public NetOutParser
//{
// public:
//  NetOutGridSizeParser(bool vertical, float ratio,
//                       int receptive_field_width, int receptive_field_height,
//                       int horizontal_stride = 0, int vertical_stride = 0,
//                       bool bbox_normalized = true,
//                       const std::vector<int>& center_x = std::vector<int>(), 
//                       const std::vector<int>& center_y = std::vector<int>(),
//                       const std::vector<int>& size  = std::vector<int>());
//
//  virtual void Parse(const std::vector<caffe::Blob<float>*>& net_out,
//                     std::vector<std::vector<Detection> >* detection) override;
//
// private:
//  bool vertical_;
//  float ratio_;
//  int receptive_field_width_;
//  int receptive_field_height_;
//  int horizontal_stride_;
//  int vertical_stride_;
//  bool bbox_normalized_;
//  std::vector<int> center_x_;
//  std::vector<int> center_y_;
//  std::vector<int> size_;
//  
//
//}; // class NetOutGridSizeParser
//
//} // namespace bgm
//#endif // !TLR_NET_OUT_GRID_SIZE_PARSER_HPP_
