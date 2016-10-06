#include "gtest\gtest.h"
#include "WFT2_CPU.h"
#include "Utils.h"
#include <fstream>
#include <omp.h>

TEST(WFR_Copmute, WFR_Compute)
{
	/* Load the FP image f */
	fftw_complex *f = nullptr;
	std::ifstream in("1024.fp");
	int rows, cols;

	if (!WFT_FPA::fftwComplexMatRead2D(in, f, rows, cols))
		std::cout << "load error" << std::endl;
	std::cout << rows << ", " << cols << std::endl;

	in.close();

	// Test constructor
	WFT_FPA::WFT::WFT2_HostResults z;
	WFT_FPA::WFT::WFT2_cpu wfr(cols,rows,WFT_FPA::WFT::WFT_TYPE::WFR,
		10, -2, 2, 0.025,
		10, -2, 2, 0.025,
		-1, z, 6);

	double time = 0;

	wfr(f, z, time);
	
	std::cout << "WFR Time: " << time << std::endl;

	wfr(f, z, time);

	std::cout << "WFR Time: " << time << std::endl;

	// Test xg, yg
	std::ofstream out("r.csv", std::ios::out | std::ios::trunc);
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

	out.open("phase_comp.csv", std::ios::out | std::ios::trunc);
	for (int i = 0; i < wfr.m_iHeight; i++)
	{
		for (int j = 0; j < wfr.m_iWidth; j++)
		{
			out << z.m_phase_comp[i*wfr.m_iWidth + j] << ",";
		}
		out << "\n";
	}

	out.close();

	out.open("cxx.csv", std::ios::out | std::ios::trunc);
	for (int i = 0; i < wfr.m_iHeight; i++)
	{
		for (int j = 0; j < wfr.m_iWidth; j++)
		{
			out << z.m_cxx[i*wfr.m_iWidth + j] << ",";
		}
		out << "\n";
	}

	out.close();

	out.open("cyy.csv", std::ios::out | std::ios::trunc);
	for (int i = 0; i < wfr.m_iHeight; i++)
	{
		for (int j = 0; j < wfr.m_iWidth; j++)
		{
			out << z.m_cyy[i*wfr.m_iWidth + j] << ",";
		}
		out << "\n";
	}

	out.close();

	out.open("b.csv", std::ios::out | std::ios::trunc);
	for (int i = 0; i < wfr.m_iHeight; i++)
	{
		for (int j = 0; j < wfr.m_iWidth; j++)
		{
			out << z.m_b[i*wfr.m_iWidth + j] << ",";
		}
		out << "\n";
	}

	out.close();


	fftw_free(f);
}