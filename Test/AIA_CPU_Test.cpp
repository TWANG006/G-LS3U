#include <gtest\gtest.h>

#include <vector>

#include "aia_cpu.h"
#include "aia_cpuf.h"
#include "opencv2\opencv.hpp"
#include "opencv2\highgui.hpp"

//TEST(AIA_CPU_3_Frames_Double, AIA_CPU_Test)
//{
//
//	std::vector<cv::Mat> f;
//
//	cv::Mat img = cv::imread("../Test_image/choped_f1.tiff");
//	f.push_back(img);
//
//	img = cv::imread("../Test_image/choped_f2.tiff");
//	f.push_back(img);
//
//	img = cv::imread("../Test_image/choped_f3.tiff");
//	f.push_back(img);
//	/*std::vector<cv::Mat> f;
//
//	cv::Mat img = cv::imread("1.bmp");
//	cv::cvtColor(img,
//		img,
//		CV_BGR2GRAY);
//	f.push_back(img);
//
//	img = cv::imread("2.bmp");
//	cv::cvtColor(img,
//		img,
//		CV_BGR2GRAY);
//	f.push_back(img);
//
//	img = cv::imread("3.bmp");
//	cv::cvtColor(img,
//		img,
//		CV_BGR2GRAY);
//	f.push_back(img);*/
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

TEST(AIA_CPU_3_Frames_Single, AIA_CPU_Test)
{
	std::vector<cv::Mat> f;

	cv::Mat img = cv::imread("../Test_image/t2/1.tiff");
	cv::cvtColor(img,
		img,
		CV_BGR2GRAY);
	f.push_back(img);
	img = cv::imread("../Test_image/t2/2.tiff");
	cv::cvtColor(img,
		img,
		CV_BGR2GRAY);
	f.push_back(img);
	img = cv::imread("../Test_image/t2/3.tiff");
	cv::cvtColor(img,
		img,
		CV_BGR2GRAY);
	f.push_back(img);
	//img = cv::imread("../Test_image/t2/4.tiff");
	//cv::cvtColor(img,
	//	img,
	//	CV_BGR2GRAY);
	//f.push_back(img);
	//img = cv::imread("../Test_image/t2/5.tiff");
	//cv::cvtColor(img,
	//	img,
	//	CV_BGR2GRAY);
	//f.push_back(img);
	//img = cv::imread("../Test_image/t2/6.tiff");
	//cv::cvtColor(img,
	//	img,
	//	CV_BGR2GRAY);
	//f.push_back(img);
	//img = cv::imread("../Test_image/t2/7.tiff");
	//cv::cvtColor(img,
	//	img,
	//	CV_BGR2GRAY);
	//f.push_back(img);
	//img = cv::imread("../Test_image/t2/8.tiff");
	//cv::cvtColor(img,
	//	img,
	//	CV_BGR2GRAY);
	//f.push_back(img);
	//img = cv::imread("../Test_image/t2/9.tiff");
	//cv::cvtColor(img,
	//	img,
	//	CV_BGR2GRAY);
	//f.push_back(img);
	//img = cv::imread("../Test_image/t2/10.tiff");
	//cv::cvtColor(img,
	//	img,
	//	CV_BGR2GRAY);
	//f.push_back(img);
	//img = cv::imread("../Test_image/t2/11.tiff");
	//cv::cvtColor(img,
	//	img,
	//	CV_BGR2GRAY);
	//f.push_back(img);

	/*std::vector<cv::Mat> f;

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
	f.push_back(img);*/
	// computation
	std::vector<float> phi;
	//std::vector<float> delta{
	//	 0.8622f,
	//0.3188f,
 //  -1.3077f ,
 //  0.8622f,
	//0.3188f,
 //  -1.3077f,
	// 0.8622f,
	//0.3188f,
 //  -1.3077f ,
	// 0.8622f };
	std::vector<float> delta{
	 0.8622f,
0.3188f,
-1.3077f };
	double time = 0;
	float err = 0;
	int iter = 0;

	AIA::AIA_CPU_DnF aia;
	aia(phi, delta, time, iter, err, f, 55, 1e-4, 12);

	std::cout << "Running Time: " << time << std::endl;
	std::cout << "Error is: " << err << std::endl;
	std::cout << "Iteration is: " << iter << std::endl;
	std::cout << "delta: ";

	for (int i = 0; i < delta.size(); i++)
	{

		std::cout << delta[i] << ",";

	}
	std::cout << std::endl;
	
}

//TEST(AIA_CPU_4_Frames_Single, AIA_CPU_Test)
//{
//	std::vector<cv::Mat> f;
//
//	cv::Mat img = cv::imread("00.bmp");
//	cv::cvtColor(img,
//				 img,
//				 CV_BGR2GRAY);
//	f.push_back(img);
//
//	img = cv::imread("01.bmp");
//	cv::cvtColor(img,
//				 img,
//				 CV_BGR2GRAY);
//	f.push_back(img);
//
//	img = cv::imread("02.bmp");
//	cv::cvtColor(img,
//				 img,
//				 CV_BGR2GRAY);
//	f.push_back(img);
//
//	img = cv::imread("03.bmp");
//	cv::cvtColor(img,
//				 img,
//				 CV_BGR2GRAY);
//	f.push_back(img);
//	
//
//	// computation
//	std::vector<float> phi;
//	std::vector<float> delta{  
//		  0.5055f,
//    0.8790f,
//    0.6663f,
//    0.3342f};
//	double time = 0;
//	float err = 0;
//	int iter = 0;
//
//	AIA::AIA_CPU_DnF aia;
//	aia(phi, delta, time, iter, err, f, 1000, 1e-4, 12);
//
//	std::cout << "Running Time: " << time << std::endl;
//	std::cout << "Error is: " << err << std::endl;
//	std::cout << "Iteration is: " << iter << std::endl;
//
//	std::cout << "Delta is: " << delta[0] << "," << delta[1] << "," << delta[2] << "," << delta[3] << std::endl;
//	
//}