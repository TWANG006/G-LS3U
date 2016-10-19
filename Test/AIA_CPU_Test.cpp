#include <gtest\gtest.h>

#include <vector>

#include "aia_cpu.h"
#include "opencv2\opencv.hpp"
#include "opencv2\highgui.hpp"

TEST(AIA_CPU_3_Frames, AIA_CPU_Test)
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
	std::vector<double> phi;
	std::vector<double> delta{  
		-0.7965,
   -0.3522,
    1.0094};
	double time = 0;
	double err = 0;
	int iter = 0;

	AIA::AIA_CPU_Dn aia;
	aia(phi, time, iter, err, f, delta);

	std::cout << "Running Time: " << time << std::endl;
	std::cout << "Error is: " << err << std::endl;
	std::cout << "Iteration is: " << iter << std::endl;

	std::cout << "Delta is: " << aia.m_v_delta[0] << "," << aia.m_v_delta[1] << "," << aia.m_v_delta[2] << std::endl;
	
}