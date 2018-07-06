//#include "net_out_grid_size_parser.hpp"
//
//namespace bgm
//{
//NetOutGridSizeParser::NetOutGridSizeParser(bool vertical, float ratio,
//                       int receptive_field_width, int receptive_field_height,
//                       int horizontal_stride, int vertical_stride,
//                       bool bbox_normalized,
//                       const std::vector<int>& center_x, 
//                       const std::vector<int>& center_y,
//                       const std::vector<int>& size) 
//  : vertical_(vertical), ratio_(ratio), 
//    receptive_field_width_(receptive_field_width),
//    receptive_field_height_(receptive_field_height), 
//    horizontal_stride_(horizontal_stride), vertical_stride_(vertical_stride),
//    bbox_normalized_(bbox_normalized),
//    center_x_(center_x), center_y_(center_y), size_(size) {
//  CHECK_GT(ratio, 0);
//  CHECK_GT(receptive_field_width, 0);
//  CHECK_GT(receptive_field_height, 0);
//  CHECK_GE(horizontal_stride, 0);
//  CHECK_GE(vertical_stride, 0);
//}
//
//void NetOutGridSizeParser::Parse(const std::vector<caffe::Blob<float>*>& net_out,
//                                 std::vector<std::vector<Detection> >* detection) {
//
//}
//} // namespace bgm