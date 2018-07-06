#include "layer_instantiation.hpp"

#include "caffe/caffe.hpp"

int main() {
  caffe::Net<float> net(
      "D:/workspace/TLR/tlr/model/tlr_train.prototxt",
      caffe::Phase::TRAIN);
  //cublasHandle_t handle = NULL;
  //int result = cublasCreate(&handle);
  //caffe::Caffe();
  
  return 0;
}