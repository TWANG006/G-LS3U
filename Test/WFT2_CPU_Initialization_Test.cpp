#include "gtest\gtest.h"
//#define WFT_FPA_DOUBLE
#include "WFT2_CPU.h"
#include "Utils.h"
#include <fstream>
#include <omp.h>

TEST(WFT2_CPU_Init, WFT2_CPU)
{
	fftwf_complex *f = nullptr;

	std::ifstream in("ff.fp");
	int rows, cols;

	if(!WFT_FPA::fftwComplexMatRead2D(in, f, rows, cols))
		std::cout<<"load error"<<std::endl;

	std::cout<<rows<<", "<<cols<<std::endl;


	WFT_FPA::WFT::WFT2_HostResults z1;
	
	WFT_FPA::WFT::WFT2_cpu wft1(cols, rows, WFT_FPA::WFT::WFT_TYPE::WFF,z1,1);
	double start = omp_get_wtime();
	wft1(f,z1);
	double end = omp_get_wtime();
	std::cout<<"Single time = "<<1000 * (end - start)<<std::endl;

	WFT_FPA::WFT::WFT2_HostResults z2;

	WFT_FPA::WFT::WFT2_cpu wft(cols, rows, WFT_FPA::WFT::WFT_TYPE::WFF,z2,6);
	start = omp_get_wtime();
	wft(f,z2);
	end = omp_get_wtime();
	std::cout<<"Multi-core time = "<<1000 * (end - start)<<std::endl;


	ASSERT_TRUE(wft1.m_gwavePadded[9][0] - wft.m_gwavePadded[9][0]<1e-6);
}