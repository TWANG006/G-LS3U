#include <gtest\gtest.h>

#include <vector>
#include "aia_cudaf.h"
#include "opencv2\opencv.hpp"
#include "opencv2\highgui.hpp"

TEST(AIA_GPU_3_Frames_Single, AIA_GPU_Test)
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


	

	std::cout << "Running Time: " << time << std::endl;
	std::cout << "Error is: " << err << std::endl;
	std::cout << "Iteration is: " << iter << std::endl;

	std::cout << "Delta is: " << delta[0] << "," << delta[1] << "," << delta[2] << std::endl;
	
}