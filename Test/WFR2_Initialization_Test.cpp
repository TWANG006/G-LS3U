#include "gtest\gtest.h"
#include "WFT2_CPU.h"
#include "Utils.h"
#include <fstream>
#include <omp.h>

//TEST(WFR_Init_xgg_ygg, WFR_Init)
//{
//	/* Load the FP image f */
//	fftw_complex *f = nullptr;
//	std::ifstream in("132.fp");
//	int rows, cols;
//
//	if (!WFT_FPA::fftwComplexMatRead2D(in, f, rows, cols))
//		std::cout << "load error" << std::endl;
//	std::cout << rows << ", " << cols << std::endl;
//
//	in.close();
//
//	// Test constructor
//	WFT_FPA::WFT::WFT2_HostResults z;
//	WFT_FPA::WFT::WFT2_cpu wfr(cols,rows,WFT_FPA::WFT::WFT_TYPE::WFR, z, 6);
//
//	// Test xg, yg
//	std::ofstream out("xg.csv", std::ios::out | std::ios::trunc);
//	for (int i = 0; i < wfr.m_iPaddedHeightCurvature; i++)
//	{
//		for (int j = 0; j < wfr.m_iPaddedWidthCurvature; j++)
//		{
//			out << wfr.im_xgPadded[i*wfr.m_iPaddedWidthCurvature + j][0] << "+" << "i" << wfr.im_xgPadded[i*wfr.m_iPaddedWidthCurvature + j][1] << ", ";
//		}
//		out << "\n";
//	}
//
//	out.close();
//
//	out.open("yg.csv", std::ios::out | std::ios::trunc);
//	for (int i = 0; i < wfr.m_iPaddedHeightCurvature; i++)
//	{
//		for (int j = 0; j < wfr.m_iPaddedWidthCurvature; j++)
//		{
//			out << wfr.im_ygPadded[i*wfr.m_iPaddedWidthCurvature + j][0] << "+" << "i" << wfr.im_ygPadded[i*wfr.m_iPaddedWidthCurvature + j][1] << ", ";
//		}
//		out << "\n";
//	}
//
//	out.close();
//
//	out.open("g.csv", std::ios::out | std::ios::trunc);
//	for (int i = 0; i < wfr.m_iPaddedHeight; i++)
//	{
//		for (int j = 0; j < wfr.m_iPaddedWidth; j++)
//		{
//			out << wfr.m_gPadded[i*wfr.m_iPaddedWidth + j][0] << "+" << "i" << wfr.m_gPadded[i*wfr.m_iPaddedWidth + j][1] << ", ";
//		}
//		out << "\n";
//	}
//
//	out.close();
//
//	fftw_free(f);
//}