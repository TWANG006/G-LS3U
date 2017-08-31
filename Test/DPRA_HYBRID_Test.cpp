//#include "gtest\gtest.h"
//
//#include "dpra_hybridf.h"
//#include "dpra_hybrid.h"
//#include "WFT2_CPUf.h"
//#include "aia_cpuf.h"
//#include "matrixIO.h"
//#include <opencv2\opencv.hpp>
//#include <fstream>
//
//using namespace std;
//
//TEST(DPRA_Hybrid_Single_4, DPRA_HYBRID_Test)
//{
//	float *phi = nullptr;
//
//	int rows = 0, cols = 0;
//
//	WFT_FPA::Utils::ReadMatrixFromDisk("2_phi.bin", &rows, &cols, &phi);
//	
//	cv::Mat dpra_f = cv::imread("1000.bmp");
//	cv::cvtColor(dpra_f,
//				 dpra_f,
//				 CV_BGR2GRAY);
//	int iWidth = dpra_f.cols;
//	int iHeight = dpra_f.rows;
//
//	DPRA::DPRA_HYBRIDF dpra_hybrid(phi, iWidth, iHeight, 1, 12);
//	
//	vector<float> dPHi(iWidth*iHeight, 0);
//	double ddtime = 0;
//
//	dpra_hybrid.dpra_per_frame(dpra_f, dPHi, ddtime);
//
//	ofstream out("deltaPhiSum.csv", std::ios::out | std::ios::trunc);
//
//	for (int i = 0; i < iHeight; i++)
//	{
//		for (int j = 0; j < iWidth; j++)
//		{
//
//			out << dPHi[i * iWidth + j] << ",";
//		}
//		out << "\n";
//	}
//	out.close();
//
//	free(phi);
//	std::cout << "DPRA hybrid Running Time is: " << ddtime << "ms" << std::endl;
//}
//
////TEST(DPRA_Hybrid_Double_4, DPRA_HYBRID_Test)
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
////	/*std::vector<float> phi;
////	std::vector<float> delta{};
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
////	std::cout << "Delta is: " << delta[0] << "," << delta[1] << "," << delta[2] << "," << delta[3] << std::endl;*/
////
////	//WFT_FPA::WFT::WFT2_HostResultsF z;
////	//WFT_FPA::WFT::WFT2_cpuF wft(iWidth, iHeight, WFT_FPA::WFT::WFT_TYPE::WFF, 
////	//		20, -0.2f, 0.2f, 0.1f, 20, -0.2f, 0.2f, 0.1f, 15,
////	//		z, 12);
////
////	double *phi = nullptr;
////
////	int rows = 0, cols = 0;
////
////	WFT_FPA::Utils::ReadMatrixFromDisk("2_phi_double.bin", &rows, &cols, &phi);
////	
////	std::cout << rows << ", " << cols << std::endl;
////
////	int iWidth = f[0].cols;
////	int iHeight = f[0].rows;
////	DPRA::DPRA_HYBRID dpra_hybrid(phi, iWidth, iHeight, 1, 12);
////	
////	vector<double> dPHi(iWidth*iHeight, 0);
////	double ddtime = 0;
////
////	cv::Mat dpra_f = cv::imread("1000.bmp");
////	cv::cvtColor(dpra_f,
////				 dpra_f,
////				 CV_BGR2GRAY);
////
////	dpra_hybrid.dpra_per_frame(dpra_f, dPHi, ddtime);
////
////	ofstream out("deltaPhiSum_double.csv", std::ios::out | std::ios::trunc);
////
////	for (int i = 0; i < iHeight; i++)
////	{
////		for (int j = 0; j < iWidth; j++)
////		{
////			
////				out << dPHi[i * iWidth + j]<<",";
////		}
////		out<<"\n";
////	}
////	out.close();
////
////	WFT_FPA::Utils::WriteMatrixToDisk("2_deltaPhi.bin", iHeight, iWidth, dPHi.data());
////
////	free(phi);
////
////	// Check A and b
////	//ofstream out("csrA_b_hybrid4.csv", std::ios::out | std::ios::trunc);
////
////	//for (int i = 0; i < dpra_hybrid.m_iHeight*dpra_hybrid.m_iWidth; i++)
////	//{
////	//	for (int j = 0; j < 3; j++)
////	//	{
////	//		for (int k = 0; k < 3; k++)
////	//		{
////	//			out << dpra_hybrid.m_h_A[i * 9 + j * 3 + k]<<",";
////	//		}
////	//		out << dpra_hybrid.m_h_b[i * 3 + j] << "\n";
////	//	}
////	//	out<<"\n";
////	//}
////
////	//out.close();
////
////	// check before filtered
////	//out.open("hybrid_dphi_wft4.csv", std::ios::out | std::ios::trunc);
////
////	//for (int i = 0; i < dpra_hybrid.m_iHeight; i++)
////	//{
////	//	for (int j = 0; j < dpra_hybrid.m_iWidth; j++)
////	//	{
////	//		out << dpra_hybrid.m_h_deltaPhiWFT[i * dpra_hybrid.m_iWidth + j].x << "+"<<dpra_hybrid.m_h_deltaPhiWFT[i * dpra_hybrid.m_iWidth + j].y<<"i,";
////	//		
////	//	}
////	//	out<<"\n";
////	//}
////
////	//out.close();
////
////	////out.open("371-366.fp");
////	////WFT_FPA::Utils::cufftComplexMatWrite2D(out, dpra_hybrid.m_h_deltaPhiWFT, iHeight, iWidth);
////	////out.close();
////
////	//// check m_d_z.filtered
////	//cufftComplex *h_z = (cufftComplex*)malloc(sizeof(cufftComplex)*iWidth*iHeight);
////	//cudaMemcpy(h_z, dpra_hybrid.m_d_z.m_d_filtered, sizeof(cufftComplex)*iWidth*iHeight, cudaMemcpyDeviceToHost);
////
////	//out.open("hybrid_d_z_filtered4.csv", std::ios::out | std::ios::trunc);
////
////	//for (int i = 0; i < dpra_hybrid.m_iHeight; i++)
////	//{
////	//	for (int j = 0; j < dpra_hybrid.m_iWidth; j++)
////	//	{
////	//		out << h_z[i * dpra_hybrid.m_iWidth + j].x << "+"<<h_z[i * dpra_hybrid.m_iWidth + j].y<<"i,";
////	//		
////	//	}
////	//	out<<"\n";
////	//}
////
////	//out.close();
////
////	//free(h_z);
////	std::cout << "DPRA hybrid Running Time is: " << ddtime << "ms" << std::endl;
////
////	//dpra_hybrid.dpra_per_frame(f[1], dPHi, ddtime);
////
////	//std::cout << "DPRA hybrid Running Time is: " << ddtime << "ms" << std::endl;
////}
//
////TEST(DPRA_Hybrid_Single_3, DPRA_HYBRID_Test)
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
////	 -0.0299f,
////   -0.4918f,
////    2.1298f
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
////	//WFT_FPA::WFT::WFT2_HostResultsF z;
////	//WFT_FPA::WFT::WFT2_cpuF wft(iWidth, iHeight, WFT_FPA::WFT::WFT_TYPE::WFF, 
////	//		20, -0.2f, 0.2f, 0.1f, 20, -0.2f, 0.2f, 0.1f, 15,
////	//		z, 12);
////
////	DPRA::DPRA_HYBRIDF dpra_hybrid(phi.data(), iWidth, iHeight, 1, 1);
////	
////	vector<float> dPHi(iWidth*iHeight, 0);
////	double ddtime = 0;
////
////	cv::Mat dpra_f = cv::imread("1.bmp");
////	cv::cvtColor(dpra_f,
////				 dpra_f,
////				 CV_BGR2GRAY);
////
////	dpra_hybrid.dpra_per_frame(dpra_f, dPHi, ddtime);
////
////	// Check A and b
////	ofstream out("csrA_b_hybrid.csv", std::ios::out | std::ios::trunc);
////
////	for (int i = 0; i < dpra_hybrid.m_iHeight*dpra_hybrid.m_iWidth; i++)
////	{
////		for (int j = 0; j < 3; j++)
////		{
////			for (int k = 0; k < 3; k++)
////			{
////				out << dpra_hybrid.m_h_A[i * 9 + j * 3 + k]<<",";
////			}
////			out << dpra_hybrid.m_h_b[i * 3 + j] << "\n";
////		}
////		out<<"\n";
////	}
////
////	out.close();
////
////	// check before filtered
////	out.open("hybrid_dphi_wft.csv", std::ios::out | std::ios::trunc);
////
////	for (int i = 0; i < dpra_hybrid.m_iHeight; i++)
////	{
////		for (int j = 0; j < dpra_hybrid.m_iWidth; j++)
////		{
////			out << dpra_hybrid.m_h_deltaPhiWFT[i * dpra_hybrid.m_iWidth + j].x << "+"<<dpra_hybrid.m_h_deltaPhiWFT[i * dpra_hybrid.m_iWidth + j].y<<"i,";
////			
////		}
////		out<<"\n";
////	}
////
////	out.close();
////
////	// check m_d_z.filtered
////	cufftComplex *h_z = (cufftComplex*)malloc(sizeof(cufftComplex)*iWidth*iHeight);
////	cudaMemcpy(h_z, dpra_hybrid.m_d_z.m_d_filtered, sizeof(cufftComplex)*iWidth*iHeight, cudaMemcpyDeviceToHost);
////
////	out.open("hybrid_d_z_filtered.csv", std::ios::out | std::ios::trunc);
////
////	for (int i = 0; i < dpra_hybrid.m_iHeight; i++)
////	{
////		for (int j = 0; j < dpra_hybrid.m_iWidth; j++)
////		{
////			out << h_z[i * dpra_hybrid.m_iWidth + j].x << "+"<<h_z[i * dpra_hybrid.m_iWidth + j].y<<"i,";
////			
////		}
////		out<<"\n";
////	}
////
////	out.close();
////
////	free(h_z);
////	std::cout << "DPRA hybrid Running Time is: " << ddtime << "ms" << std::endl;
////
////	//dpra_hybrid.dpra_per_frame(f[1], dPHi, ddtime);
////
////	//std::cout << "DPRA hybrid Running Time is: " << ddtime << "ms" << std::endl;
////}