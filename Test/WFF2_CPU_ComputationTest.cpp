#include "gtest\gtest.h"
#include "WFT2_CPU.h"
#include "WFT2_CPUf.h"
#include "Utils.h"
#include <fstream>
#include <omp.h>

TEST(WFF_Double, WFT2_CPU)
{
	/* Need to be revised*/

	double time = 0;

	/* Load the FP image f */
	fftw_complex *f = nullptr;
	std::ifstream in("1024.fp");
	int rows, cols;

	if(!WFT_FPA::fftwComplexMatRead2D(in, f, rows, cols))
		std::cout<<"load error"<<std::endl;
	std::cout<<rows<<", "<<cols<<std::endl;

	in.close();

	/* Multi-core initialization */
	WFT_FPA::WFT::WFT2_HostResults z2;
	WFT_FPA::WFT::WFT2_cpu wft(cols, rows, WFT_FPA::WFT::WFT_TYPE::WFF,
		10, -2, 2, 0.1, 10, -2, 2, 0.1, 6, z2, 6);
	wft(f, z2, time);

	std::cout<<"Multicore-time:  "<<time<<std::endl;

	/* Single-core initialization */
	/*WFT_FPA::WFT::WFT2_HostResults z1;	
	WFT_FPA::WFT::WFT2_cpu wft1(cols, rows, WFT_FPA::WFT::WFT_TYPE::WFF,z1,1);
	wft1(f, z1, time);
	std::cout<<"Singlecore-time:  "<<time<<std::endl;*/
	
	fftw_free(f);



	std::ofstream out("zfiltered.csv", std::ios::out | std::ios::trunc);

	for (int i = 0; i < wft.m_iHeight; i++)
	{
		for (int j = 0; j < wft.m_iWidth; j++)
		{
			out << z2.m_filtered[i*wft.m_iWidth + j][0] << "+" << "i" << z2.m_filtered[i*wft.m_iWidth + j][1] << ", ";
		}
		out << "\n";
	}

	out.close();
}

TEST(WFF_Single, WFT2_CPU)
{
	/* Need to be revised*/

	double time = 0;

	/* Load the FP image f */
	fftwf_complex *f = nullptr;
	std::ifstream in("1024.fp");
	int rows, cols;

	if(!WFT_FPA::fftwComplexMatRead2D(in, f, rows, cols))
		std::cout<<"load error"<<std::endl;
	std::cout<<rows<<", "<<cols<<std::endl;

	in.close();

	/* Multi-core initialization */
	WFT_FPA::WFT::WFT2_HostResultsF z2;
	WFT_FPA::WFT::WFT2_cpuF wft(cols, rows, WFT_FPA::WFT::WFT_TYPE::WFF,
		10, -2, 2, 0.1f, 10, -2, 2, 0.1f, 6, z2, 6);	
	wft(f, z2, time);

	std::cout<<"Multicore-time:  "<<time<<std::endl;

	wft(f, z2, time);
	std::cout<<"Multicore-time:  "<<time<<std::endl;


	/* Single-core initialization */
	/*WFT_FPA::WFT::WFT2_HostResultsF z1;	
	WFT_FPA::WFT::WFT2_cpuF wft1(cols, rows, WFT_FPA::WFT::WFT_TYPE::WFF,z1,1);
	wft1(f, z1, time);
	std::cout<<"Singlecore-time:  "<<time<<std::endl;*/
	
	fftwf_free(f);

	std::ofstream out("zfilteredF.csv", std::ios::out | std::ios::trunc);

	for (int i = 0; i < wft.m_iHeight; i++)
	{
		for (int j = 0; j < wft.m_iWidth; j++)
		{
			out << z2.m_filtered[i*wft.m_iWidth + j][0] << "+" << "i" << z2.m_filtered[i*wft.m_iWidth + j][1] << ", ";
		}
		out << "\n";
	}

	out.close();
}