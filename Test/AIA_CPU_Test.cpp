//#include <gtest\gtest.h>
//
//#include <vector>
//
//#include "aia_cpu.h"
//#include "aia_cpuf.h"
//#include "opencv2\opencv.hpp"
//#include "opencv2\highgui.hpp"
//
//TEST(AIA_CPU_3_Frames_Double, AIA_CPU_Test)
//{
//	std::vector<cv::Mat> f;
//
//	cv::Mat img = cv::imread("1.bmp");
//	cv::cvtColor(img,
//				 img,
//				 CV_BGR2GRAY);
//	f.push_back(img);
//
//	img = cv::imread("2.bmp");
//	cv::cvtColor(img,
//				 img,
//				 CV_BGR2GRAY);
//	f.push_back(img);
//
//	img = cv::imread("3.bmp");
//	cv::cvtColor(img,
//				 img,
//				 CV_BGR2GRAY);
//	f.push_back(img);
//	
//
//	// computation
//	std::vector<double> phi;
//	std::vector<double> delta{0.8622,
//    0.3188,
//   -1.3077};
//	double time = 0;
//	double err = 0;
//	int iter = 0;
//
//	AIA::AIA_CPU_Dn aia;
//	aia(phi, delta, time, iter, err, f, 20, 1e-4, 12);
//
//	std::cout << "Running Time: " << time << std::endl;
//	std::cout << "Error is: " << err << std::endl;
//	std::cout << "Iteration is: " << iter << std::endl;
//
//	std::cout << "Delta is: " << delta[0] << "," << delta[1] << "," << delta[2] << std::endl;
//	
//}
//
//TEST(AIA_CPU_3_Frames_Single, AIA_CPU_Test)
//{
//	std::vector<cv::Mat> f;
//
//	cv::Mat img = cv::imread("1.bmp");
//	cv::cvtColor(img,
//				 img,
//				 CV_BGR2GRAY);
//	f.push_back(img);
//
//	img = cv::imread("2.bmp");
//	cv::cvtColor(img,
//				 img,
//				 CV_BGR2GRAY);
//	f.push_back(img);
//
//	img = cv::imread("3.bmp");
//	cv::cvtColor(img,
//				 img,
//				 CV_BGR2GRAY);
//	f.push_back(img);
//	
//
//	// computation
//	std::vector<float> phi;
//	std::vector<float> delta{  
//		 0.8622f,
//    0.3188f,
//   -1.3077f};
//	double time = 0;
//	float err = 0;
//	int iter = 0;
//
//	AIA::AIA_CPU_DnF aia;
//	aia(phi, delta, time, iter, err, f, 20, 1e-4, 12);
//
//	std::cout << "Running Time: " << time << std::endl;
//	std::cout << "Error is: " << err << std::endl;
//	std::cout << "Iteration is: " << iter << std::endl;
//
//	std::cout << "Delta is: " << delta[0] << "," << delta[1] << "," << delta[2] << std::endl;
//	
//}