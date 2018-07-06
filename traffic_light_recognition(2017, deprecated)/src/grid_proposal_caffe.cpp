//#include "grid_proposal_caffe.hpp"
//
//namespace bgm
//{
//
//void GridProposalCaffe::Init(
//    const std::string& net_model_file, 
//    const std::string& net_trained_file,
//    const cv::Size& grid_size, bool use_gpu = true) {
//  if (use_gpu) {
//#ifdef CPU_ONLY
//    LOG(WARNING) << "Can't use GPU, CPU is used instead GPU";
//    caffe::Caffe::set_mode(caffe::Caffe::CPU);
//#else
//    caffe::Caffe::set_mode(caffe::Caffe::GPU);
//#endif
//  }
//  else 
//    caffe::Caffe::set_mode(caffe::Caffe::CPU);
//
//  net_.reset(new caffe::Net<float>(net_model_file, caffe::TEST));
//  net_->CopyTrainedLayersFrom(net_trained_file);
//
//  net_input_ = net_->input_blobs()[0];
//  CHECK_GT(net_input_->num(), 0);
//  CHECK(net_input_->channels() == 1 || net_input_->channels() == 3);
//  use_color_input_ = (net_input_->channels() == 3);
//
//  CHECK_GT(grid_size.width, 0);
//  CHECK_GT(grid_size.height, 0);
//  grid_size_ = grid_size;
//}
//
//void GridProposalCaffe::Propose(const std::vector<cv::Mat>& input) {
//
//}
//
//} // namespace bgm