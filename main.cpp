#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc/edge_filter.hpp>// https://docs.opencv.org/3.4.11/da/d17/group__ximgproc__filters.html#ga80b9b58fb85dd069691b709285ab985c

#include "joint_bilateral_filter.cuh"

int main (int argc, char* argv[])
{
	if (argc < 2){
		std::cout << "Usage Example: ./build/main inv_depth7.png cur_rgb1_ref.png\n";
		std::cerr << "argc: " << argc << "should be 2\n";
		return 1;
	}	

	cv::Mat src, joint;
	try {
		const std::string imgname = argv[1];
		const std::string imgname_joint = argv[2];
		src = cv::imread(imgname, cv::IMREAD_ANYDEPTH);
		joint = cv::imread(imgname_joint, cv::IMREAD_ANYCOLOR);
		if (src.empty() || joint.empty()) {
			std::cerr << "failed to load image. check path: " << imgname << std::endl;
			std::cerr << "failed to load image. check path: " << imgname_joint << std::endl;
			return 1;
		}
	}
	catch(const cv::Exception& ex)
	{
		std::cout << "cv::Exception Error: " << ex.what() << std::endl;
	}

	/*
	cv::Mat depth_map;
	depth_map.create(src.rows, src.cols, CV_32FC1);
	// const int minv = 4, maxv = 14;
	const float minv = 1./14., maxv = 1./4.;
	for(int v=0; v<src.rows; v++) {
		for(int u=0; u<src.cols; u++) {
			 const int dst = src.at<unsigned char>(v,u);
			 float val = (dst * (maxv - minv))/255. + minv;
			 depth_map.at<float>(v,u) = val;
		}
	}
	cv::Mat visu;
	cv::normalize(depth_map, visu, 0, 255, CV_MINMAX, CV_8U);
	cv::imwrite("debug.png", visu);
	*/

	CV_Assert(src.type() == CV_8UC1);
	// src.convertTo(src, CV_32FC1);// 8UC1 -> 32FC1
	CV_Assert(joint.type() == CV_8UC3);
	cv::Mat dst;
	dst.create(src.rows, src.cols, CV_32FC1);
	float sigma_color = 1.0, sigma_spatial = 2.0;// 2/255
	
	int d = -1, radius;
	const int rows = src.rows, cols = src.cols;
	if(d < 0) {
		radius = cvRound(sigma_spatial*1.5);
		std::cout << "radius: " << radius << std::endl;
		radius = MAX(radius, 1);
		d = radius*2 + 1;
		// https://github.com/norishigefukushima/WeightedJointBilateralFilter/blob/master/DepthMapRefinement/jointBilateralFilter.cpp
	}
	// cv::ximgproc::jointBilateralFilter(joint, src, dst, d, sigma_color, sigma_spatial);
	
	cv::cuda::GpuMat src_gpu, dst_gpu, joint_gpu;
	cv::cuda::createContinuous(rows, cols, CV_32FC1, src_gpu);
	cv::cuda::createContinuous(rows, cols, CV_32FC1, dst_gpu);
	cv::cuda::createContinuous(rows, cols, CV_32FC4, joint_gpu);
	src.convertTo(src, CV_32FC1);// 8UC1 -> 32FC1
	src_gpu.upload(src);
	dst_gpu.upload(dst);
	joint.convertTo(joint, CV_32FC3);// CV_8UC3 -> CV_32FC3
	cvtColor(joint, joint, CV_BGR2BGRA);// CV_32FC3 -> CV_32FC4 
	joint_gpu.upload(joint);
	applyJointBilateralCaller((float*)src_gpu.data, (float*)dst_gpu.data, (float4*)joint_gpu.data, rows, cols, src_gpu.step, joint_gpu.step,
  		sigma_color, sigma_spatial, radius);
	dst_gpu.download(dst);

	
	cv::Mat visu;
	cv::normalize(dst, visu, 0, 255, CV_MINMAX, CV_8U);
	cv::imwrite("joint_filter_cpu.png", visu);
	
	return 0;
}

