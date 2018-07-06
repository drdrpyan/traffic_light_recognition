#ifndef TLR_LAYER_INSTANTIATION_HPP_
#define TLR_LAYER_INSTANTIATION_HPP_

#include "bbox_anno_map_layer.hpp"
#include "heatmap_concat_layer.hpp"
#include "img_bbox_anno_layer.hpp"
#include "sliding_window_input_layer.hpp"
#include "vectorization_layer.hpp"


#include "caffe/layers/concat_layer.hpp"
#include "caffe/layers/cudnn_conv_layer.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/layers/dropout_layer.hpp"
#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/layers/input_layer.hpp"
#include "caffe/layers/softmax_loss_layer.hpp"



#include "caffe/solver.hpp"
#include "caffe/sgd_solvers.hpp"

#pragma comment(linker, "/include:?gInstantiationGuardSlidingWindowInputLayer@caffe@@3DA")
#pragma comment(linker, "/include:?gInstantiationGuardConcatLayer@caffe@@3DA")
namespace caffe
{

template class ImgBBoxAnnoLayer<float>;
template class ImgBBoxAnnoLayer<double>;

template class BBoxAnnoMapLayer<float>;
template class BBoxAnnoMapLayer<double>;

template class HeatmapConcatLayer<float>;
template class HeatmapConcatLayer<double>;

template class SlidingWindowInputLayer<float>;
template class SlidingWindowInputLayer<double>;

//template class VectorizationLayer<float>
//template class VectorizationLayer<double>

//template class ConvolutionLayer<float>;
//template class ConvolutionLayer<double>;

//template class CuDNNConvolutionLayer<float>;
//template class CuDNNConvolutionLayer<double>;

template class DropoutLayer<float>;
template class DropoutLayer<double>;

template class EuclideanLossLayer<float>;
template class EuclideanLossLayer<double>;

template class SoftmaxWithLossLayer<float>;
template class SoftmaxWithLossLayer<double>;

template class SGDSolver<float>;
template class SGDSolver<double>;

template class InputLayer<float>;
template class InputLayer<double>;
}

#endif // !TLR_LAYER_INSTANTIATION_HPP_
