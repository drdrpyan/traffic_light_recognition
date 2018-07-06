#include "caffe_sw_detector.hpp"

#include <opencv2/imgproc.hpp>

#include <corecrt_io.h>

namespace bgm
{

void CaffeSWDetector::InitNet(const std::string& model_file,
                              const std::string& trained_file,
                              bool use_gpu) {
  if (use_gpu) {
#ifdef CPU_ONLY
    caffe::Caffe::set_mode(Caffe::CPU);
#else
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
#endif
  }
  else 
    caffe::Caffe::set_mode(caffe::Caffe::CPU);

  net_.reset(new caffe::Net<float>(model_file, caffe::TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  net_input_ = net_->input_blobs()[0];
 
  //// 순서 확인할 것
  //net_bbox_output_ = net_->output_blobs()[0];
  //net_label_output_ = net_->output_blobs()[1];
  //net_output_.clear();
  //net_output_.push_back(net_label_output_);
  //net_output_.push_back(net_bbox_output_);

  const caffe::Blob<float>* input_layer = net_->input_blobs()[0];
  net_input_width_ = input_layer->width();
  net_input_height_ = input_layer->height();
  net_input_channel_ = input_layer->channels();
  CHECK(net_input_channel_ == 3 || net_input_channel_ == 1)
    << "Input layer should have 1 or 3 channels.";
  net_max_input_ = input_layer->num();
  color_img_required_ = (net_input_channel_ == 3) ? true : false;
}

//void CaffeSWDetector::Detect(const std::vector<cv::Mat>& imgs) {
//  CHECK_LE(imgs.size(), net_max_input_);
//
//  ClearPrevResult();
//
//  std::vector<cv::Mat> preprocessed_imgs(imgs.size());
//  for (int i = 0; i < imgs.size(); i++)
//    preprocessed_imgs[i] = PreprocessInputImg(imgs[i]);
//
//}

void CaffeSWDetector::Detect(const std::vector<cv::Mat>& imgs) {
  ClearPrevResult();

  std::vector<cv::Mat> preprocessed(imgs.size());
  for (int i = 0; i < imgs.size(); i++)
    preprocessed[i] = PreprocessInputImg(imgs[i]);

  MatsToBlob(preprocessed, net_input_, false);

  //cv::Mat debug(cv::Size(227, 227), CV_32FC3, net_input_->mutable_cpu_data());
  //cv::FileStorage fs("img.yml", cv::FileStorage::WRITE);
  //fs << "Img" << debug;
  //fs.release();

  net_->Forward();

  //net_out_parser_->Parse(*net_label_output_, *net_bbox_output_, &raw_results_);
  //net_out_parser_->Parse(*net_label_output_, &raw_results_);
  //net_out_parser_->Parse(net_output_, &raw_results_);
  net_out_parser_->Parse(net_->output_blobs(), &raw_results_);

  detected_ = true;
}

void CaffeSWDetector::Detect(const cv::Mat& img,
                             float offset_x, float offset_y) {
  ClearPrevResult();
  std::vector<cv::Mat> preprocessed(1);
  preprocessed[0] = PreprocessInputImg(img);
  MatsToBlob(preprocessed, net_->input_blobs()[0], false);

  caffe::Blob<float> offset_input;
  std::vector<int> offset_shape(4, 1);
  offset_shape[1] = 4;
  offset_input.Reshape(offset_shape);
  float* offset_data = offset_input.mutable_cpu_data();
  offset_data[0] = offset_x;
  offset_data[1] = offset_y;
  offset_data[2] = 227.0f / 1491.0f;
  offset_data[3] = 227.0f / 1171.0f;
  net_->input_blobs()[1]->CopyFrom(offset_input);

  net_->Forward();

  net_out_parser_->Parse(net_->output_blobs(), &raw_results_);

  detected_ = true;
}


//void CaffeSWDetector::Detect(const caffe::Blob<float>& img) {
//  ClearPrevResult();
//
//  net_input_->CopyFrom(img);
//
//  //cv::Mat debug(cv::Size(227, 227), CV_32FC3, net_input_->mutable_cpu_data());
//  //cv::FileStorage fs("blob.yml", cv::FileStorage::WRITE);
//  //fs << "Blob" << debug;
//  //fs.release();
//
//  net_->Forward();
//  //net_out_parser_->Parse(*net_label_output_, *net_bbox_output_, &raw_results_);
//  net_out_parser_->Parse(net_output_, &raw_results_);
//  detected_ = true;
//}

cv::Mat CaffeSWDetector::PreprocessInputImg(const cv::Mat& img) const {
  //CHECK(color_img_required_ && img.channels() == 3)
  //    << "The net requires a input images as color image";

  cv::Mat net_input_mat;

  // 여기 나중에 주의해서 검증할 것
  if (!color_img_required_ && img.channels() == 3)
    cv::cvtColor(img, net_input_mat, cv::ColorConversionCodes::COLOR_BGR2GRAY);
  else
    net_input_mat = img;

  if (input_resizing_)
    cv::resize(net_input_mat, net_input_mat,
               cv::Size(net_input_width_, net_input_height_));

  //if(color_img_required_ && net_input_mat.type() != CV_32FC3)
  //  net_input_mat.convertTo(net_input_mat, CV_32FC3);
  //else if(!color_img_required_ && net_input_mat.type() != CV_32FC1)
  //  net_input_mat.convertTo(net_input_mat, CV_32FC1);

  return net_input_mat;
}


void CaffeSWDetector::MatsToBlob(const std::vector<cv::Mat>& mats,
                                 caffe::Blob<float>* blob,
                                 bool blob_resize) const {
  CHECK_GT(mats.size(), 0);

  if (blob_resize) {
    std::vector<int> shape(4);
    shape[0] = mats.size();
    shape[1] = mats[0].channels();
    shape[2] = mats[0].rows;
    shape[3] = mats[0].cols;
    if (blob->shape() != shape)
      blob->Reshape(shape);
  }
  else {
    CHECK_LE(mats.size(), blob->num());
    CHECK_LE(mats[0].channels(), blob->channels());
    CHECK_LE(mats[0].rows, blob->height());
    CHECK_LE(mats[0].cols, blob->width());
  }

  for (int n = 0; n < mats.size(); n++) {
    cv::Mat mat = mats[n];

    if(mat.channels() == 3 && mat.type() != CV_32FC3)
      mat.convertTo(mat, CV_32FC3);
    else if(mat.channels() == 1 && mat.type() != CV_32FC1)
      mat.convertTo(mat, CV_32FC1);

    std::vector<cv::Mat> split_vec(3);
    for (int c = 0; c < mat.channels(); c++) {
      float* data_ptr = blob->mutable_cpu_data() + blob->offset(n, c);
      split_vec[c] = cv::Mat(mat.size(), CV_32FC1, data_ptr);
    }

    cv::split(mat, split_vec);

    // debug
    //cv::Mat blue = split_vec[0];
    //cv::Mat green = split_vec[1];
    //cv::Mat red = split_vec[2];
  }
}

void CaffeSWDetector::GetLabelDensityMapImpl(
    bool single_item, std::vector<std::vector<cv::Mat> >* maps) const {
  if (!detected_) {
    LOG(WARNING) << "Call Detect() first.";
    return;
  }

  CHECK(maps);

  const caffe::Blob<float>& label = *(net_->output_blobs()[1]);

  const int NUM_ITEM = single_item ? 1 : label.num();
  const int NUM_LABEL = label.channels();
  const cv::Size MAP_SIZE(label.width(), label.height());
  const int MAP_ELEMS = MAP_SIZE.area();

  maps->resize(NUM_ITEM);
  for (int n = 0; n < NUM_ITEM; ++n) {
    (*maps)[n].resize(NUM_LABEL);
    for (int l = 0; l < NUM_LABEL; ++l) {
      cv::Mat map(MAP_SIZE, CV_32FC1);
      caffe::caffe_copy(MAP_ELEMS, label.cpu_data() + label.offset(n, l),
                        reinterpret_cast<float*>(map.data));

      (*maps)[n][l] = map;
    }
  }
}

} // namespace bgm