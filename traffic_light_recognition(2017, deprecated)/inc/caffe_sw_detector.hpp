#ifndef TLR_CAFFE_SW_DETECTOR_HPP_
#define TLR_CAFFE_SW_DETECTOR_HPP_

#include "detector_common.hpp"
#include "net_out_parser.hpp"

#include <opencv2/core.hpp>

#include "caffe/caffe.hpp"

#include <memory>
#include <set>
#include <utility>
#include <vector>


namespace bgm
{

class CaffeSWDetector
{
 public:
  void InitNet(const std::string& model_file,
               const std::string& trained_file,
               bool use_gpu = true);
  //void InitGeometry(int base_win_width, int base_win_height,
  //                  int base_horizontal_stride, int base_vertical_stride);
  void Detect(const std::vector<cv::Mat>& imgs);
  void Detect(const cv::Mat& img);
  void Detect(const cv::Mat& img, 
              float offset_x, float offset_y);
  // debug
  //void Detect(const caffe::Blob<float>& img);

  const std::vector<std::vector<Detection> >& GetRawResults();
  const std::vector<Detection>& GetRawResults(int item_idx);
  //const std::vector<std::vector<Detection> >& GetMergedResults();
  //const std::vector<Detection>& GetMergedResults(int item_idx);
  void GetLabelDensityMap(std::vector<std::vector<cv::Mat> >* maps) const;
  void GetLabelDensityMap(std::vector<cv::Mat>* maps) const;

  void set_net_out_parser(NetOutParser* net_out_parser);
  void set_input_resizing(bool resizing);

 private:
  void ClearPrevResult();
  cv::Mat PreprocessInputImg(const cv::Mat& img) const;
  void MatsToBlob(const std::vector<cv::Mat>& mats,
                  caffe::Blob<float>* blob, bool blob_resize = false) const;

  void GetLabelDensityMapImpl(bool single_item,
                              std::vector<std::vector<cv::Mat> >* maps) const;

  std::unique_ptr<caffe::Net<float> > net_;
  caffe::Blob<float>* net_input_;
  caffe::Blob<float>* net_bbox_output_;
  caffe::Blob<float>* net_label_output_;
  std::vector<const caffe::Blob<float>*> net_output_;
  
  bool color_img_required_;

  bool input_resizing_;

  int net_input_width_;
  int net_input_height_;
  int net_input_channel_;
  int net_max_input_;

  std::unique_ptr<NetOutParser> net_out_parser_;

  std::vector<std::vector<Detection> > raw_results_;
  std::vector<std::vector<Detection> > merged_results_;

  bool detected_;
  bool merged_;  
}; // class CaffeSWDetector


// inline functions
inline void CaffeSWDetector::Detect(const cv::Mat& img) {
  std::vector<cv::Mat> temp_vec(1, img);
  Detect(temp_vec);
}

inline const std::vector<std::vector<Detection> >& CaffeSWDetector::GetRawResults() {
  CHECK(detected_) << "There's no result.";
  return raw_results_;
}

inline const std::vector<Detection>& CaffeSWDetector::GetRawResults(int item_idx) {
  CHECK_GE(item_idx, 0) << "Illegal item_idx";
  CHECK_LT(item_idx, raw_results_.size()) << "There's no " << item_idx << "th item.";
  return GetRawResults()[item_idx];
}

inline void CaffeSWDetector::GetLabelDensityMap(
    std::vector<std::vector<cv::Mat> >* maps) const {  
  GetLabelDensityMapImpl(true, maps);
}

inline void CaffeSWDetector::GetLabelDensityMap(std::vector<cv::Mat>* maps) const {
  std::vector<std::vector<cv::Mat> > temp_maps;
  GetLabelDensityMapImpl(false, &temp_maps);
  *maps = temp_maps[0];
}

inline void CaffeSWDetector::set_net_out_parser(NetOutParser* net_out_parser) {
  net_out_parser_.reset(net_out_parser);
}
  
inline void CaffeSWDetector::set_input_resizing(bool resizing) {
  input_resizing_ = resizing;
}

inline void CaffeSWDetector::ClearPrevResult() {
  detected_ = false;
  merged_ = false;

  raw_results_.clear();
  merged_results_.clear();
}
} // namespace bgm

#endif // !TLR_CAFFE_SW_DETECTOR_HPP_
