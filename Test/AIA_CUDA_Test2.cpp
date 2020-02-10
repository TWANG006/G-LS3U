#include <gtest\gtest.h>

#include <vector>
#include "aia_cudaf_yctest.h"
#include "aia_cudaf_v2.h"
#include "opencv2\opencv.hpp"
#include "opencv2\highgui.hpp"

TEST(AIA_GPU2_3_Frames_Single, AIA_GPU_Test2)
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
	/*img = cv::imread("../Test_image/t2/4.tiff");
	cv::cvtColor(img,
		img,
		CV_BGR2GRAY);
	f.push_back(img);
	img = cv::imread("../Test_image/t2/5.tiff");
	cv::cvtColor(img,
		img,
		CV_BGR2GRAY);
	f.push_back(img);
	img = cv::imread("../Test_image/t2/6.tiff");
	cv::cvtColor(img,
		img,
		CV_BGR2GRAY);
	f.push_back(img);
	img = cv::imread("../Test_image/t2/7.tiff");
	cv::cvtColor(img,
		img,
		CV_BGR2GRAY);
	f.push_back(img);
	img = cv::imread("../Test_image/t2/8.tiff");
	cv::cvtColor(img,
		img,
		CV_BGR2GRAY);
	f.push_back(img);
	img = cv::imread("../Test_image/t2/9.tiff");
	cv::cvtColor(img,
		img,
		CV_BGR2GRAY);
	f.push_back(img);
	img = cv::imread("../Test_image/t2/10.tiff");
	cv::cvtColor(img,
		img,
		CV_BGR2GRAY);
	f.push_back(img);*/
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
-1.3077f  };
	double time = 0;
	float err = 0;
	int iter = 0;



	AIA::AIA_CUDAF_YC2 aia(f);
	aia(phi, delta, time, iter, err, f, 120, 1e-4, 12);


	std::cout << "Running Time: " << time << std::endl;
	std::cout << "Error is: " << err << std::endl;
	std::cout << "Iteration is: " << iter << std::endl;

	std::cout << "delta: ";

	for (int i = 0; i < delta.size(); i++)
	{
		if (i != delta.size() - 1)
		{
			std::cout << delta[i] << ",";
		}
		else
		{
			std::cout << delta[i];
		}
	}
	std::cout << std::endl;
}