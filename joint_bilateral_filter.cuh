#ifndef OUTLIER_CUH
#define OUTLIER_CUH 

// opencv
#include <opencv2/opencv.hpp>

// cuda
#include <opencv2/cudev/common.hpp>// CV_CUDEV_SAFE_CALL


void applyJointBilateralCaller(float*src, float* dst, float4* joint, int rows, int cols, int imageStep_src, int imageStep_joint,
  float sigma_color, float sigma_spatial, int radius);

#endif