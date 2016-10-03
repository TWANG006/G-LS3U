#include "gtest\gtest.h"
#include "WFT2_CPU.h"
#include "Utils.h"
#include <fstream>
#include <omp.h>

TEST(WFT2_CPU_Init, WFT2_CPU)
{
	/* Need to be revised*/

	double time = 0;

	/* Load the FP image f */
	fftw_complex *f = nullptr;
	std::ifstream in("124.fp");
	int rows, cols;

	if(!WFT_FPA::fftwComplexMatRead2D(in, f, rows, cols))
		std::cout<<"load error"<<std::endl;
	std::cout<<rows<<", "<<cols<<std::endl;

	in.close();

	/* Multi-core initialization */
	WFT_FPA::WFT::WFT2_HostResults z2;
	WFT_FPA::WFT::WFT2_cpu wft(cols, rows, WFT_FPA::WFT::WFT_TYPE::WFF,z2,6);
	wft(f, z2, time);

	std::cout<<"Multicore-time:  "<<time<<std::endl;

	/* Single-core initialization */
	WFT_FPA::WFT::WFT2_HostResults z1;	
	WFT_FPA::WFT::WFT2_cpu wft1(cols, rows, WFT_FPA::WFT::WFT_TYPE::WFF,z1,1);
	wft1(f, z1, time);
	std::cout<<"Singlecore-time:  "<<time<<std::endl;
	
	

	/* Assert the results */
	std::cout<<wft1.m_FfPadded[0][0]<<", "<<wft1.m_FfPadded[0][1]<<std::endl;
	std::cout<<wft.im_Fgwave[wft1.m_iPaddedWidth*wft.m_iPaddedHeight][0]<<", "<<wft1.im_Fgwave[wft1.m_iPaddedWidth*wft1.m_iPaddedHeight][1]<<std::endl;

	std::cout << wft1.m_FfPadded[19][1] << ", " << wft.m_FfPadded[19][1] << std::endl;
	ASSERT_TRUE(wft1.im_Fgwave[cols*rows + 9][0] - wft.im_Fgwave[cols*rows + 9][0]<1e-6);

	fftw_free(f);


	std::ofstream out("gwavePre.csv", std::ios::out | std::ios::trunc);
   
	for(int k = 0; k<wft.m_iNumberThreads; k++)
	{
		for (int i = 0; i < wft.m_iPaddedHeight; i++)
		{
			for (int j = 0; j < wft.m_iPaddedWidth; j++)
			{
				out << wft.im_Fgwave[k*wft.m_iPaddedWidth*wft.m_iPaddedHeight+i*wft.m_iPaddedWidth + j][0] << "+" 
					<< "i" << wft.im_Fgwave[k*wft.m_iPaddedWidth*wft.m_iPaddedHeight+i*wft.m_iPaddedWidth + j][1] << ", ";
			}
			out << "\n";
		}
		out << "\n";
	}
    out.close();

	out.open("gwavePre2.csv", std::ios::out | std::ios::trunc);
	for(int k = 0; k<wft1.m_iNumberThreads; k++)
	{
		for (int i = 0; i < wft1.m_iPaddedHeight; i++)
		{
			for (int j = 0; j < wft1.m_iPaddedWidth; j++)
			{
				out << wft1.im_Fgwave[k*wft1.m_iPaddedWidth*wft1.m_iPaddedHeight+i*wft1.m_iPaddedWidth + j][0] << "+" << "i" << wft1.im_Fgwave[k*wft1.m_iPaddedWidth*wft1.m_iPaddedHeight+i*wft1.m_iPaddedWidth + j][1] << ", ";
			}
			out << "\n";
		}
		out<<"\n";
	}
    out.close();

}