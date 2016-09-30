#include "gtest\gtest.h"
#include <iostream>
#include <omp.h>

#include "Utils.h"

#include "WFT2_CPU.h"

TEST(WFT2_CPU_Constructor, WF2_CPU)
{
	double time = 0;

	/* Load the FP image f */
	fftwf_complex *f = nullptr;
	std::ifstream in("f.fp");
	int rows, cols;

	if (!WFT_FPA::fftwComplexMatRead2D(in, f, rows, cols))
		std::cout << "load error" << std::endl;
	std::cout << rows << ", " << cols << std::endl;

	in.close();

	/* Multi-core initialization */
	WFT_FPA::WFT::WFT2_HostResults z2;
	WFT_FPA::WFT::WFT2_cpu wft(cols, rows, WFT_FPA::WFT::WFT_TYPE::WFF, z2, 1);
	
	wft(f, z2, time);

	WFT_FPA::fftwComplexPrint(wft.m_gwavePadded[64]);
	WFT_FPA::fftwComplexPrint(wft.m_gwavePaddedFq[0]);
	WFT_FPA::fftwComplexPrint(wft.m_fPaddedFq[0]);

	std::ofstream out("gwave.csv", std::ios::out | std::ios::trunc);
	for(int i=0; i<1120; i++)
	{
		for(int j=0; j<1120; j++)
		{
			out<<wft.m_gwavePaddedFq[i*1120+j][0]<<"+"<<wft.m_gwavePaddedFq[i*1120+j][1]<<", ";
		}
		out<<"\n";
	}
	out.close();


	std::cout << "Multi-core time = " << time << std::endl;

	free(f);
}