#include "dnn_unit.hpp"

namespace bgm
{

DNNUnit::DNNUnit() 
  : shared_(false), shared_data_(nullptr), ref_data_(nullptr) {

}

DNNUnit::DNNUnit(DNNUnit::ElemType elem_type, char* data, bool shared,
                 int batch, int channel, int height, int width)
  : shared_(shared), batch_(batch), channel_(channel),
  height_(height), width_(width) {
  assert(data);
  assert(batch > 0);
  assert(channel > 0);
  assert(height > 0);
  assert(width > 0);
  if (shared)
    set_data(data);
  else
    set_data(data, false);
}

//DNNUnit& DNNUnit::operator=(const DNNUnit& ref) {
//  elem_type_ = ref.elem_type_;
//  batch_ = ref.batch_;
//  channel_ = ref.channel_;
//  height_ = ref.height_;
//  width_ = ref.width_;
//
//  shared_ = shared_;
//
//  set_batch(ref.batch());
//  set_channel(ref.channel());
//  set_height(ref.height());
//}

DNNUnit DNNUnit::Clone() const {
  DNNUnit clone = *this;
  
  if (shared() && shared_data_) {
    const int SIZE = Size();
    char* clone_data = new char[SIZE];

    const char* src_data = shared_data_.get();
    std::copy(src_data, src_data + SIZE, clone_data);

    clone.set_data(clone_data);
  }
  else if (!shared() && ref_data_) {
    const int SIZE = Size();
    char* clone_data = new char[SIZE];

    std::copy(ref_data_, ref_data_ + SIZE, clone_data);
    clone.set_data(clone_data);
  }

  return clone;
}

void DNNUnit::set_data(char* data, bool shared) {
  assert(data);

  shared_ = shared;

  if (shared) {
    shared_data_.reset(data, [](char* ptr) {delete[] ptr; });
    ref_data_ = nullptr;
  }
  else {
    shared_data_.reset();
    ref_data_ = data;
  }
}

//template <typename T>
//void DNNUnit<T>::SetShape(int batch, int channel,
//                          int height, int width) {
//  set_batch(batch);
//  set_channel(channel);
//  set_height(height);
//  set_width(width);
//}

#ifdef USE_OPENCV
DNNUnit::DNNUnit(const cv::Mat& mat)
  : DNNUnit(std::vector<cv::Mat>(1, mat)) {

}

DNNUnit::DNNUnit(const std::vector<cv::Mat>& mat_list)
  : batch_(mat_list.size()) {
  //assert(mat_list.size() > 1);
  assert(!mat_list.empty());
  assert(!mat_list[0].empty());
  set_channel(mat_list[0].channels());
  set_height(mat_list[0].rows);
  set_width(mat_list[0].cols);

  int mat_depth = mat_list[0].depth();
  switch (mat_depth) {
    case CV_8U:
      set_elem_type(UCHAR);
      break;
    case CV_32F:
      set_elem_type(SINGLE);
      break;
    case CV_64F:
      set_elem_type(DOUBLE);
      break;
    default:
      assert(("Not implemented yet", 0));
  }

  set_data(new char[Size()]);
  char* data_iter = data();
  const int BATCH_STEP = Volume() * static_cast<int>(elem_type());

  for (int i = 0; i < batch_; ++i) {
    assert(mat_list[i].channels() == channel());
    assert(mat_list[i].rows == height());
    assert(mat_list[i].cols == width());
    assert(mat_list[i].depth() == mat_depth);

    MatToData(mat_list[i], data_iter);

    data_iter += BATCH_STEP;
  }
}

//DNNUnit::DNNUnit(const cv::Mat& mat, ElemType type) 
//  : DNNUnit(std::vector<cv::Mat>(1, mat), type) {
//  
//}

//DNNUnit::DNNUnit(const std::vector<cv::Mat>& mat_list, 
//                 ElemType type) 
//  : batch_(mat_list.size()), elem_type_(type) {
//  assert(mat_list.size() > 1);
//  assert(!mat_list[0].empty());
//  set_channel(mat_list[0].channels());
//  set_height(mat_list[0].rows);
//  set_width(mat_list[0].cols);
//
//  set_data(new char[Size()]);
//  char* data_iter = data();
//  const int BATCH_STEP = Volume() * static_cast<int>(elem_type());
//
//  for (int i = 0; i < batch_; ++i) {
//    assert(mat_list[i].channels() == channel());
//    assert(mat_list[i].rows == height());
//    assert(mat_list[i].cols == width());
//    
//    cv::Mat converted;
//    switch (elem_type()) {
//      case SINGLE:
//        if (mat_list[i].depth() == CV_32F)
//          converted = mat_list[i];
//        else
//          mat_list[i].convertTo(converted, cv_32fc)
//    }
//
//    MatToData(mat_list[i], data_iter);
//
//    data_iter += BATCH_STEP;
//  }
//}

void DNNUnit::ToMat(std::vector<cv::Mat>* mat_list) const {
  assert(mat_list);
  mat_list->resize(batch());

  for (int n = 0; n < batch(); ++n) {
    switch (elem_type()) {
      case UCHAR:
      {
        std::vector<std::vector<unsigned char> > channels;
        channels.reserve(channel());

        const unsigned char* data_iter = reinterpret_cast<const unsigned char*>(data());
        int step = Area();
        for (int i = channel(); i--; ) {
          channels.push_back(
              std::vector<unsigned char>(data_iter, data_iter + step));
          data_iter += step;
        }

        cv::merge(channels, (*mat_list)[n]);
      }
        break;

      case SINGLE:
      {
        assert(0); // 여기 체크할 것
        std::vector<std::vector<float> > channels;
        channels.reserve(channel());

        const float* data_iter = reinterpret_cast<const float*>(data());
        int step = Area();
        for (int i = channel(); i--; ) {
          channels.push_back(
              std::vector<float>(data_iter, data_iter + step));
          data_iter += step;
        }

        cv::merge(channels, (*mat_list)[n]);
      }
        break;

      default:
        assert(("Not implemented yet", 0));
    }
  }
}

void DNNUnit::MatToData(const cv::Mat& mat, void* data_ptr) {
  assert(mat.rows == height());
  assert(mat.cols == width());

  switch (elem_type()) {
    case SINGLE:
    {
      cv::Mat src;
      if (mat.depth() == CV_32F)
        src = mat;
      else
        mat.convertTo(src, CV_32F);

      std::vector<cv::Mat> split_vec(mat.channels());

      float* data_iter = static_cast<float*>(data_ptr);
      for (int c = 0; c < mat.channels(); ++c) {
        split_vec[c] = cv::Mat(mat.size(), CV_32FC1, data_iter);
        data_iter += Area();
      }

      cv::split(mat, split_vec);
      break;
    }
    default:
      assert(("not implemented yet", 0));
  }
}

#endif // USE_OPENCV
}