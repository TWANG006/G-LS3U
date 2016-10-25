#include "gtest\gtest.h"

#include "aia_cpu.h"
#include "aia_cpuf.h"
#include "dpra_cpu.h"
#include "dpra_cpuf.h"

TEST(dpra_single, DPRA_Test)
{
	/* AIA to get the initial phi */
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
	std::vector<float> delta{-1.4577f,
   -0.8285f,
    1.3368f};
	double time = 0;
	float err = 0;
	int iter = 0;

	AIA::AIA_CPU_DnF aia;
	aia(phi, delta, time, iter, err, f, 20, 1e-4, 12);

	std::cout << "AIA Running Time: " << time << std::endl;
	std::cout << "AIA Error is: " << err << std::endl;
	std::cout << "AIA Iteration is: " << iter << std::endl;

	std::cout << "Delta is: " << delta[0] << "," << delta[1] << "," << delta[2] << std::endl;

	/* DPRA Results */
	
	DPRA::DPRA_CPUF dpra(phi.data(), f[0].cols, f[0].rows, 1, 12);

	std::vector<float> delta_phi(f[0].cols*f[0].rows,0);
	double timeDpra = 0;

	dpra.dpra_per_frame(f[0], delta_phi, timeDpra);

	std::cout << "DPRA Running Time is: " << timeDpra << std::endl;
}

TEST(dpra_double, DPRA_Test)
{
	/* AIA to get the initial phi */
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
	std::vector<double> delta{-1.4577,
   -0.8285,
    1.3368};
	double time = 0;
	double err = 0;
	int iter = 0;

	AIA::AIA_CPU_Dn aia;
	aia(phi, delta, time, iter, err, f, 20, 1e-4, 12);

	std::cout << "AIA Running Time: " << time << std::endl;
	std::cout << "AIA Error is: " << err << std::endl;
	std::cout << "AIA Iteration is: " << iter << std::endl;

	std::cout << "Delta is: " << delta[0] << "," << delta[1] << "," << delta[2] << std::endl;

	/* DPRA Results */
	
	DPRA::DPRA_CPU dpra(phi.data(), f[0].cols, f[0].rows, 1, 12);

	std::vector<double> delta_phi(f[0].cols*f[0].rows,0);
	double timeDpra = 0;

	dpra.dpra_per_frame(f[0], delta_phi, timeDpra);

	std::cout << "DPRA Running Time is: " << timeDpra << std::endl;
}