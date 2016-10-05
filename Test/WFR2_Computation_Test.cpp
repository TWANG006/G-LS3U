#include "gtest\gtest.h"
#include "WFT2_CPU.h"
#include "Utils.h"
#include <fstream>
#include <omp.h>

TEST(WFR_Copmute, WFR_Compute)
{
	/* Load the FP image f */
	fftw_complex *f = nullptr;
	std::ifstream in("132.fp");
	int rows, cols;

	if (!WFT_FPA::fftwComplexMatRead2D(in, f, rows, cols))
		std::cout << "load error" << std::endl;
	std::cout << rows << ", " << cols << std::endl;

	in.close();

	// Test constructor
	WFT_FPA::WFT::WFT2_HostResults z;
	WFT_FPA::WFT::WFT2_cpu wfr(cols,rows,WFT_FPA::WFT::WFT_TYPE::WFR, z, 6);

	double time = 0;

	wfr(f, z, time);
	wfr(f, z, time);

	// Test xg, yg
	std::ofstream out("m_r.csv", std::ios::out | std::ios::trunc);
	for (int i = 0; i < wfr.m_iHeight; i++)
	{
		for (int j = 0; j < wfr.m_iWidth; j++)
		{
			out << z.m_r[i*wfr.m_iWidth + j] << ",";
		}
		out << "\n";
	}

	out.close();

	out.open("wx.csv", std::ios::out | std::ios::trunc);
	for (int i = 0; i < wfr.m_iHeight; i++)
	{
		for (int j = 0; j < wfr.m_iWidth; j++)
		{
			out << z.m_wx[i*wfr.m_iWidth + j]<< ",";
		}
		out << "\n";
	}

	out.close();

	out.open("wy.csv", std::ios::out | std::ios::trunc);
	for (int i = 0; i < wfr.m_iHeight; i++)
	{
		for (int j = 0; j < wfr.m_iWidth; j++)
		{
			out << z.m_wy[i*wfr.m_iWidth + j] << ",";
		}
		out << "\n";
	}

	out.close();

	out.open("phase.csv", std::ios::out | std::ios::trunc);
	for (int i = 0; i < wfr.m_iHeight; i++)
	{
		for (int j = 0; j < wfr.m_iWidth; j++)
		{
			out << z.m_phase[i*wfr.m_iWidth + j] << ",";
		}
		out << "\n";
	}

	out.close();

	out.open("cxx.csv", std::ios::out | std::ios::trunc);
	for (int i = 0; i < wfr.m_iPaddedHeightCurvature; i++)
	{
		for (int j = 0; j < wfr.m_iPaddedWidthCurvature; j++)
		{
			out << wfr.im_cxxPadded[i*wfr.m_iPaddedWidthCurvature + j][0] << ",";
		}
		out << "\n";
	}

	out.close();

	fftw_free(f);
}