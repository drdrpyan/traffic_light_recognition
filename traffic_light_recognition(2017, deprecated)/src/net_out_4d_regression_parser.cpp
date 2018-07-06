//#include "net_out_4d_regression_parser.hpp"
//
//#include "softmax_out_const_iterator.hpp"
//
//namespace bgm
//{
//
//void NetOut4DRegressionParser::Parse(
//    const std::vector<const caffe::Blob<float>*>& net_out,
//    std::vector<std::vector<Detection> >* detection) {
//  if (net_out.size() == 1)
//    Parse(*(net_out[0]), detection);
//  else if (net_out.size() == 2)
//    Parse(*(net_out[0]), *(net_out[1]), detection);
//  else
//    LOG(ERROR) << "Illegal net_out";
//}
//
//void NetOut4DRegressionParser::Parse(
//    const caffe::Blob<float>& net_label_out,
//    std::vector<std::vector<Detection> >* detection) {
//  //CHECK_EQ(net_label_out.channels(), 1);
//  CHECK(detection);
//
//  detection->resize(net_label_out.num());
//
//  for (int n = 0; n < detection->size(); n++)
//    Parse(net_label_out, n, &(*detection)[n]);
//}
//
//void NetOut4DRegressionParser::Parse(const caffe::Blob<float>& net_label_out,
//                                     const caffe::Blob<float>& net_bbox_out,
//                                     std::vector<std::vector<Detection> >* detection) {
//  CHECK(detection);
//  CHECK_EQ(net_label_out.num(), net_bbox_out.num());
//  CHECK_EQ(net_label_out.height(), net_bbox_out.height());
//  CHECK_EQ(net_label_out.width(), net_bbox_out.width());
//  //CHECK_EQ(net_label_out.channels(), 1);
//  CHECK_EQ(net_bbox_out.channels(), 4);
//
//  detection->resize(net_label_out.num());
//  for (int n = 0; n < detection->size(); n++)
//    Parse(net_label_out, net_bbox_out, n, &(*detection)[n]);
//}
//
//void NetOut4DRegressionParser::Parse(
//    const caffe::Blob<float>& net_label_out, int item_idx,
//    std::vector<Detection>* detection) const {
//  CHECK(detection);  
//
//  detection->clear();
//  
//  SoftmaxOutConstIterator<float> label_iter(net_label_out, item_idx);
//  int offset_x = 0;
//  int offset_y = 0;
//  for (int i = net_label_out.height(); i--; ) {
//    for (int j = net_label_out.width(); j--; ) {
//      int label = label_iter.GetMaxLabel();
//      if (!IsIgnoredLabel(label)) {
//        BBox<int> bbox(offset_x, offset_y,
//                       offset_x + receptive_field_width_ - 1,
//                       offset_y + receptive_field_height_ - 1);
//        detection->push_back(Detection(label, bbox));
//      }
//
//      ++label_iter;
//      offset_x += horizontal_stride_;
//    }
//
//    offset_x = 0;
//    offset_y += vertical_stride_;
//  }
//}
//
//void NetOut4DRegressionParser::Parse(
//    const caffe::Blob<float>& net_label_out,
//    const caffe::Blob<float>& net_bbox_out,
//    int item_idx, std::vector<Detection>* detection) const {
//  CHECK(detection);
//
//  detection->clear();
//
//  //const float* label_iter = net_label_out.cpu_data() + net_label_out.offset(item_idx);
//  SoftmaxOutConstIterator<float> label_iter(net_label_out, item_idx);
//  const float* x_iter = net_bbox_out.cpu_data() + net_bbox_out.offset(item_idx, 0);
//  const float* y_iter = net_bbox_out.cpu_data() + net_bbox_out.offset(item_idx, 1);
//  const float* w_iter = net_bbox_out.cpu_data() + net_bbox_out.offset(item_idx, 2);
//  const float* h_iter = net_bbox_out.cpu_data() + net_bbox_out.offset(item_idx, 3);
//  int offset_x = 0;
//  int offset_y = 0;
//  for (int i = net_label_out.height(); i--; ) {
//    for (int j = net_label_out.width(); j--; ) {
//      //int label = *label_iter++;
//      int label = label_iter.GetMaxLabel();
//      float x = std::max(0.0f, *x_iter++);
//      float y = std::max(0.0f, *y_iter++);
//      float w = std::max(0.0f, *w_iter++);
//      float h = std::max(0.0f, *h_iter++);
//
//      DLOG(INFO) << "net output : (label, x, y, w, h) = ("
//        << label << ", " << x << ", " << y << ", " << w << ", " << h << ")";
//
//      if (!IsIgnoredLabel(label)) {
//        if (bbox_normalized_) {
//          x *= receptive_field_width_;
//          y *= receptive_field_height_;
//          w *= receptive_field_width_;
//          h *= receptive_field_height_;
//        }
//
//        BBox<int> bbox(offset_x + x, offset_y + y,
//                       offset_x + x + w - 1, offset_y + y + h - 1);
//        detection->push_back(Detection(label, bbox));
//
//        DLOG(INFO) << "parsing result : (label, xmin, ymin, xmax, ymax) = ("
//            << label << ", " << offset_x + x << ", " << offset_y + y << ", " 
//              << offset_x + x + w - 1 << ", " << offset_y + y + h - 1 << ")";
//      }
//      ++label_iter;
//      offset_x += horizontal_stride_;
//    }
//    offset_x = 0;
//    offset_y += vertical_stride_;
//  }
//}
//
//} // namespace bgm