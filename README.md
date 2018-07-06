# Traffic Light Recognition (TLR)
This TLR system designed as a module of Advanced Driver Assistance Systems (ADAS).
It detects and recognizes traffic lights from driving-view image sequences.

## Hyundai Contest (2017)
TLR system designed for Hyundai Contest (2017).
TLs are detected by using ACF and HUV-Histogram.

## Traffic Light Recognition (2017, deprecated)
(It is no longer used.)
TLR system using deep learning approach including my proposed "Grid Proposal Network".
I had trained CNN model with Caffe framework, then attached the trained model to this system by using "DNN Wrapper".


DNNWrapper : https://github.com/drdrpyan/dnn_wrapper
NMS : https://github.com/martinkersner/non-maximum-suppression-cpp