//#ifndef TLR_GRID_PROPOSAL_CAFFE_HPP_
//#define TLR_GRID_PROPOSAL_CAFFE_HPP_
//
//#include "caffe/caffe.hpp"
//
//#include "grid_proposal.hpp"
//
//#include <memory>
//#include <string>
//
//namespace bgm
//{
//
//class GridProposalCaffe : public GridProposal
//{
// public:
//  void Init(const std::string& net_model_file,
//            const std::string& net_trained_file,
//            const cv::Size& grid_size,
//            bool use_gpu = true);
//  virtual void Propose(const cv::Mat& input) override;
//  virtual void Propose(const std::vector<cv::Mat>& input) override;
//  virtual void GetActivationProposal(float threshold, 
//                                     cv::Mat& proposal) override;
//  virtual void GetActivationProposal(
//      float threshold, std::vector<cv::Mat>* proposal) override;
//  virtual void GetDensityProposal(float threshold, 
//                                  cv::Mat& proposal) override;
//  virtual void GetDensityProposal(
//      float threshold, std::vector<cv::Mat>* proposal) override;
//
// private:
//  std::unique_ptr<caffe::Net<float> > net_;
//  caffe::Blob<float>* net_input_;
//  caffe::Blob<float>* net_putput_;
//
//  bool use_color_input_;
//
//  cv::Size grid_size_;
//
//  std::vector<std::vector<float> > raw_net_out_;
//}; // class GridProposalCaffe
//
//// inline functions
//inline void GridProposalCaffe::Propose(const cv::Mat& input) {
//  std::vector<cv::Mat> input_vec(1, input);
//  Propose(input_vec);
//}
//} // namespace bgm
//
//#endif // !TLR_GRID_PROPOSAL_CAFFE_HPP_
