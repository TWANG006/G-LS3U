#include <iostream>
#include <vector>

#include "Utils.h"
#include "WFT.h"
#include "WFT2_CPU.h"
#include "gtest\gtest.h"
#include "opencv2\opencv.hpp"
#include "aia_cudaf.h"

using namespace std;


int main(int argc, char** argv)
{
	std::vector<cv::Mat> f;

	cv::Mat img = cv::imread("1.bmp");
	cv::cvtColor(img,
				 img,
				 CV_BGR2GRAY);
	f.push_back(img);

	img = cv::imread("2.bmp");
	cv::cvtColor(img,
				 img,
				 CV_BGR2GRAY);
	f.push_back(img);

	img = cv::imread("3.bmp");
	cv::cvtColor(img,
				 img,
				 CV_BGR2GRAY);
	f.push_back(img);
	

	// computation
	std::vector<float> phi;
	std::vector<float> delta{  
		 0.8622f,
    0.3188f,
   -1.3077f};
	double time = 0;
	float err = 0;
	int iter = 0;

	AIA::AIA_CUDAF aia(f);
	aia(phi, delta, time, iter, err, f, 20, 1e-4, 6);

	::testing::InitGoogleTest(&argc, argv);
	RUN_ALL_TESTS();

	return 0;
}