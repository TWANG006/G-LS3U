#include "gtest\gtest.h"
#include "WFT2_CPU.h"
#include "Utils.h"
#include <fstream>
#include <omp.h>

TEST(WFT2_CPU_Init_Double, WFT2_CPU)
{
	/* Need to be revised*/

	double time = 0;

	/* Load the FP image f */
	fftw_complex *f = nullptr;
	std::ifstream in("132.fp");
	int rows, cols;

	if(!WFT_FPA::fftwComplexMatRead2D(in, f, rows, cols))
		std::cout<<"load error"<<std::endl;
	std::cout<<rows<<", "<<cols<<std::endl;

	in.close();

	/* Multi-core initialization */
	WFT_FPA::WFT::WFT2_HostResults z2;
	WFT_FPA::WFT::WFT2_cpu wft(cols, rows, WFT_FPA::WFT::WFT_TYPE::WFF,
		10, -2, 2, 0.1, 10, -2, 2, 0.1, 6, z2, 6);

	/* Single-core initialization */
	WFT_FPA::WFT::WFT2_HostResults z1;	
	WFT_FPA::WFT::WFT2_cpu wft1(cols, rows, WFT_FPA::WFT::WFT_TYPE::WFF,
		10, -2, 2, 0.1f, 10, -2, 2, 0.1f, 6, z1, 6);	

	/* Assert the results */
	std::cout << wft1.m_FfPadded[0][0] << ", " << wft1.m_FfPadded[0][1] << std::endl;
	std::cout << wft1.m_FfPadded[19][1] << ", " << wft.m_FfPadded[19][1] << std::endl;
	//ASSERT_TRUE(wft1.im_Fgwave[cols*rows + 9][0] - wft.im_Fgwave[cols*rows + 9][0]<1e-6);

	fftw_free(f);
}