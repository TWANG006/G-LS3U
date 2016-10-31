//#include "gtest\gtest.h"
//
//#include "aia_cpu.h"
//#include "aia_cpuf.h"
//#include "dpra_cpu.h"
//#include "dpra_cpuf.h"
//
////TEST(dpra_single4, DPRA_Test)
////{
////	/* AIA to get the initial phi */
////	std::vector<cv::Mat> f;
////
////	cv::Mat img = cv::imread("00.bmp");
////	cv::cvtColor(img,
////				 img,
////				 CV_BGR2GRAY);
////	f.push_back(img);
////
////	img = cv::imread("01.bmp");
////	cv::cvtColor(img,
////				 img,
////				 CV_BGR2GRAY);
////	f.push_back(img);
////
////	img = cv::imread("02.bmp");
////	cv::cvtColor(img,
////				 img,
////				 CV_BGR2GRAY);
////	f.push_back(img);
////
////	img = cv::imread("03.bmp");
////	cv::cvtColor(img,
////				 img,
////				 CV_BGR2GRAY);
////	f.push_back(img);
////	
////
////	// computation
////	std::vector<float> phi;
////	std::vector<float> delta{
////	 -0.1806f,
////   -0.8725f,
////    0.0501f,
////    0.1354f
////	};
////	double time = 0;
////	float err = 0;
////	int iter = 0;
////
////	AIA::AIA_CPU_DnF aia;
////	aia(phi, delta, time, iter, err, f, 20, 1e-4, 12);
////
////	int iWidth = f[0].cols;
////	int iHeight = f[0].rows;
////
////	std::cout << "AIA Running Time: " << time << std::endl;
////	std::cout << "AIA Error is: " << err << std::endl;
////	std::cout << "AIA Iteration is: " << iter << std::endl;
////
////	std::cout << "Delta is: " << delta[0] << "," << delta[1] << "," << delta[2] << std::endl;
////
////	/* DPRA Results */
////	
////	DPRA::DPRA_CPUF dpra(phi.data(), f[0].cols, f[0].rows, 1, 12);
////
////	std::vector<float> delta_phi(f[0].cols*f[0].rows,0);
////	double timeDpra = 0;
////
////	cv::Mat dpra_f = cv::imread("1000.bmp");
////	cv::cvtColor(dpra_f,
////				 dpra_f,
////				 CV_BGR2GRAY);
////	dpra.dpra_per_frame(dpra_f, delta_phi, timeDpra);
////
////
////	std::ofstream out("deltaPhiSum_CPU.csv", std::ios::out | std::ios::trunc);
////
////	for (int i = 0; i < iHeight; i++)
////	{
////		for (int j = 0; j < iWidth; j++)
////		{
////			
////				out << delta_phi[i * iWidth + j]<<",";
////		}
////		out<<"\n";
////	}
////	out.close();
////
////	// check m_d_z.filtered
////
////	//std::ofstream out("cpu_z_filtered4.csv", std::ios::out | std::ios::trunc);
////
////	//for (int i = 0; i < dpra.m_iHeight; i++)
////	//{
////	//	for (int j = 0; j < dpra.m_iWidth; j++)
////	//	{
////	//		out << dpra.m_z.m_filtered[i * dpra.m_iWidth + j][0] << "+"<< dpra.m_z.m_filtered[i * dpra.m_iWidth + j][1]<<"i,";
////	//		
////	//	}
////	//	out<<"\n";
////	//}
////
////	//out.close();
////
////
////	//// Check before wff
////	//out.open("m_dPhiWFT_CPU4.csv", std::ios::out | std::ios::trunc);
////
////	//for (int i = 0; i < dpra.m_iHeight; i++)
////	//{
////	//	for (int j = 0; j < dpra.m_iWidth; j++)
////	//	{
////	//		out << dpra.m_dPhiWFT[i * dpra.m_iWidth + j][0] << "+"<< dpra.m_dPhiWFT[i * dpra.m_iWidth + j][1]<<"i,";
////	//		
////	//	}
////	//	out<<"\n";
////	//}
////
////	//out.close();
////
////	std::cout << "DPRA Running Time is: " << timeDpra << std::endl;
////}
//
//TEST(dpra_double4, DPRA_Test)
//{
//	/* AIA to get the initial phi */
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
//	std::vector<double> phi;
//	std::vector<double> delta{};
//	double time = 0;
//	double err = 0;
//	int iter = 0;
//
//	AIA::AIA_CPU_Dn aia;
//	aia(phi, delta, time, iter, err, f, 20, 1e-4, 12);
//
//	int iWidth = f[0].cols;
//	int iHeight = f[0].rows;
//
//	std::cout << "AIA Running Time: " << time << std::endl;
//	std::cout << "AIA Error is: " << err << std::endl;
//	std::cout << "AIA Iteration is: " << iter << std::endl;
//
//	std::cout << "Delta is: " << delta[0] << "," << delta[1] << "," << delta[2] << "," << delta[3] << std::endl;
//
//	/* DPRA Results */
//	
//	DPRA::DPRA_CPU dpra(phi.data(), f[0].cols, f[0].rows, 1, 12);
//
//	std::vector<double> delta_phi(f[0].cols*f[0].rows,0);
//	double timeDpra = 0;
//
//	cv::Mat dpra_f = cv::imread("1000.bmp");
//	cv::cvtColor(dpra_f,
//				 dpra_f,
//				 CV_BGR2GRAY);
//	dpra.dpra_per_frame(dpra_f, delta_phi, timeDpra);
//
//
//	std::ofstream out("deltaPhiSum_CPU_double.csv", std::ios::out | std::ios::trunc);
//
//	for (int i = 0; i < iHeight; i++)
//	{
//		for (int j = 0; j < iWidth; j++)
//		{
//			
//				out << delta_phi[i * iWidth + j]<<",";
//		}
//		out<<"\n";
//	}
//	out.close();
//
//	// check m_d_z.filtered
//
//	//std::ofstream out("cpu_z_filtered4d.csv", std::ios::out | std::ios::trunc);
//
//	//for (int i = 0; i < dpra.m_iHeight; i++)
//	//{
//	//	for (int j = 0; j < dpra.m_iWidth; j++)
//	//	{
//	//		out << dpra.m_z.m_filtered[i * dpra.m_iWidth + j][0] << "+"<< dpra.m_z.m_filtered[i * dpra.m_iWidth + j][1]<<"i,";
//	//		
//	//	}
//	//	out<<"\n";
//	//}
//
//	//out.close();
//
//
//	//// Check before wff
//	//out.open("m_dPhiWFT_CPU4d.csv", std::ios::out | std::ios::trunc);
//
//	//for (int i = 0; i < dpra.m_iHeight; i++)
//	//{
//	//	for (int j = 0; j < dpra.m_iWidth; j++)
//	//	{
//	//		out << dpra.m_dPhiWFT[i * dpra.m_iWidth + j][0] << "+"<< dpra.m_dPhiWFT[i * dpra.m_iWidth + j][1]<<"i,";
//	//		
//	//	}
//	//	out<<"\n";
//	//}
//
//	//out.close();
//
//	//std::cout << "DPRA Running Time is: " << timeDpra << std::endl;
//}
//
////TEST(dpra_single_4, DPRA_Test)
////{
////	/* AIA to get the initial phi */
////	std::vector<cv::Mat> f;
////
////	cv::Mat img = cv::imread("1.bmp");
////	cv::cvtColor(img,
////				 img,
////				 CV_BGR2GRAY);
////	f.push_back(img);
////
////	img = cv::imread("2.bmp");
////	cv::cvtColor(img,
////				 img,
////				 CV_BGR2GRAY);
////	f.push_back(img);
////
////	img = cv::imread("3.bmp");
////	cv::cvtColor(img,
////				 img,
////				 CV_BGR2GRAY);
////	f.push_back(img);
////	
////
////	// computation
////	std::vector<float> phi;
////	std::vector<float> delta{
////	0.8622f,
////    0.3188f,
////   -1.3077f
////	};
////	double time = 0;
////	float err = 0;
////	int iter = 0;
////
////	AIA::AIA_CPU_DnF aia;
////	aia(phi, delta, time, iter, err, f, 20, 1e-4, 12);
////
////	std::cout << "AIA Running Time: " << time << std::endl;
////	std::cout << "AIA Error is: " << err << std::endl;
////	std::cout << "AIA Iteration is: " << iter << std::endl;
////
////	std::cout << "Delta is: " << delta[0] << "," << delta[1] << "," << delta[2] << std::endl;
////
////	/* DPRA Results */
////	
////	DPRA::DPRA_CPUF dpra(phi.data(), f[0].cols, f[0].rows, 1, 12);
////
////	std::vector<float> delta_phi(f[0].cols*f[0].rows,0);
////	double timeDpra = 0;
////
////	dpra.dpra_per_frame(f[0], delta_phi, timeDpra);
////
////		// check m_d_z.filtered
////
////	std::ofstream out("cpu_z_filtered3.csv", std::ios::out | std::ios::trunc);
////
////	for (int i = 0; i < dpra.m_iHeight; i++)
////	{
////		for (int j = 0; j < dpra.m_iWidth; j++)
////		{
////			out << dpra.m_z.m_filtered[i * dpra.m_iWidth + j][0] << "+"<< dpra.m_z.m_filtered[i * dpra.m_iWidth + j][1]<<"i,";
////			
////		}
////		out<<"\n";
////	}
////
////	out.close();
////
////	std::cout << "DPRA Running Time is: " << timeDpra << std::endl;
////}
////
////TEST(dpra_double, DPRA_Test)
////{
////	/* AIA to get the initial phi */
////	std::vector<cv::Mat> f;
////
////	cv::Mat img = cv::imread("1.bmp");
////	cv::cvtColor(img,
////				 img,
////				 CV_BGR2GRAY);
////	f.push_back(img);
////
////	img = cv::imread("2.bmp");
////	cv::cvtColor(img,
////				 img,
////				 CV_BGR2GRAY);
////	f.push_back(img);
////
////	img = cv::imread("3.bmp");
////	cv::cvtColor(img,
////				 img,
////				 CV_BGR2GRAY);
////	f.push_back(img);
////	
////
////	// computation
////	std::vector<double> phi;
////	std::vector<double> delta{-1.4577,
////   -0.8285,
////    1.3368};
////	double time = 0;
////	double err = 0;
////	int iter = 0;
////
////	AIA::AIA_CPU_Dn aia;
////	aia(phi, delta, time, iter, err, f, 20, 1e-4, 12);
////
////	std::cout << "AIA Running Time: " << time << std::endl;
////	std::cout << "AIA Error is: " << err << std::endl;
////	std::cout << "AIA Iteration is: " << iter << std::endl;
////
////	std::cout << "Delta is: " << delta[0] << "," << delta[1] << "," << delta[2] << std::endl;
////
////	/* DPRA Results */
////	
////	DPRA::DPRA_CPU dpra(phi.data(), f[0].cols, f[0].rows, 1, 12);
////
////	std::vector<double> delta_phi(f[0].cols*f[0].rows,0);
////	double timeDpra = 0;
////
////	dpra.dpra_per_frame(f[0], delta_phi, timeDpra);
////
////	std::cout << "DPRA Running Time is: " << timeDpra << std::endl;
////}