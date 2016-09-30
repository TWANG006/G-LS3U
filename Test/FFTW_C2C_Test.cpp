#include "gtest\gtest.h"
#include <vector>
#include <iostream>
#include <fftw\fftw3.h>
#include <omp.h>

#include "Utils.h"

#include "WFT-FPA.h"

using namespace std;

TEST(FFTW3_C2C_In_place, FFTW3_C2C)
{
	//vector<float> B{
	//	0.8147f, 0.6324f, 0.9575f, 0.9572f,
	//	0.9058f, 0.0975f, 0.9649f, 0.4854f,
	//	0.1270f, 0.2785f, 0.1576f, 0.8003f,
	//	0.9134f, 0.5469f, 0.9706f, 0.1419f};

	//fftw_complex *A = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * 16);
	//fftw_complex *m_freqDom1 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * 16);
	//fftw_complex *m_freqDom2 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * 4 * (4 / 2 + 1));

	//A[0][0] = 0.8147; A[0][1] = 0;
	//A[1][0] = 0.6324; A[1][1] = 0;
	//A[2][0] = 0.9575; A[2][1] = 0;
	//A[3][0] = 0.9572; A[3][1] = 0;
	//A[4][0] = 0.9058; A[4][1] = 0;
	//A[5][0] = 0.0975; A[5][1] = 0;
	//A[6][0] = 0.9649; A[6][1] = 0;
	//A[7][0] = 0.4854; A[7][1] = 0;
	//A[8][0] = 0.1270; A[8][1] = 0;
	//A[9][0] = 0.2785; A[9][1] = 0;
	//A[10][0] = 0.1576; A[10][1] = 0;
	//A[11][0] = 0.8003; A[11][1] = 0;
	//A[12][0] = 0.9134; A[12][1] = 0;
	//A[13][0] = 0.5469; A[13][1] = 0;
	//A[14][0] = 0.9706; A[14][1] = 0;
	//A[15][0] = 0.1419; A[15][1] = 0;

	//fftw_plan plan1 = fftw_plan_dft_2d(4, 4, A, m_freqDom1, FFTW_FORWARD, FFTW_ESTIMATE);
	//fftw_plan plan2 = fftw_plan_dft_r2c_2d(4, 4, B.data(), m_freqDom2, FFTW_ESTIMATE);
	//fftw_plan plan3 = fftw_plan_dft_2d(4,4, m_freqDom1, A, FFTW_BACKWARD, FFTW_ESTIMATE);

	//fftw_execute(plan1);
	//fftw_execute(plan2);

	//ASSERT_TRUE(m_freqDom1[0][0] - m_freqDom2[0][0] <= 1e-6 && 
	//			m_freqDom1[0][1] - m_freqDom2[0][1] <= 1e-6);

	//// Do scaling
	//for (int i = 0; i < 4; i++)
	//{
	//	for (int j = 0; j < 4; j++)
	//	{
	//		
	//		WFT_FPA::fftwComplexPrint(m_freqDom1[i*4+j]);
	//		WFT_FPA::fftwComplexScale(m_freqDom1[i * 4 + j], 1 / 16.0f);
	//		std::cout<<", ";
	//	}
	//	std::cout<<endl;
	//}
	//fftw_execute(plan3);


	//ASSERT_TRUE(B[0] - A[0][0] <= 1E-6);


	//fftw_destroy_plan(plan1);
	//fftw_destroy_plan(plan2);
	//fftw_destroy_plan(plan3);

	//fftw_free(A);
	//fftw_free(m_freqDom1);
	//fftw_free(m_freqDom2);


}

TEST(FFTW3_Matlab, FFTW3)
{
	/* Load the FP image f */
	fftw_complex *m_gwavePadded = nullptr;
	std::ifstream in("ftw.fp");
	int rows, cols;

	if(!WFT_FPA::fftwComplexMatRead2D(in, m_gwavePadded, rows, cols))
		std::cout<<"load error"<<std::endl;
	std::cout<<rows<<", "<<cols<<std::endl;

	in.close();

    fftw_complex *m_gwavePaddedFq = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*rows*cols);

    fftw_plan m_planForwardgwave = fftw_plan_dft_2d(cols, rows, m_gwavePadded, m_gwavePaddedFq, FFTW_FORWARD, FFTW_ESTIMATE);

    using real_t = double;
    int m_iPaddedHeight = rows;

    int m_iPaddedWidth  = cols;

    real_t rNorm2Factor = 0;

    #pragma omp parallel num_threads(6)
    {
       real_t rNorm2FactorLocal = 0;

       /* Get thread params */
       int tid = omp_get_thread_num();
       int inthread = omp_get_num_threads();

       int id = 0;
       int x = 0, y = 0;

       for (int i = tid; i < m_iPaddedHeight; i += inthread)
       {
           for (int j = 0; j < m_iPaddedWidth; j++)
           {
              id = i * m_iPaddedWidth + j;   // 1D index of 2D array elems      

               
              m_gwavePadded[id][0] = 0;
              m_gwavePadded[id][1] = 0;

           
               if (i < 61 && j < 61)
              {
                  y = i - (61 - 1) / 2;
                  x = j - (61 - 1) / 2;

                  m_gwavePadded[id][0] = exp(-real_t(x*x) / 2 / 10 / 10
                     - real_t(y*y) / 2 / 10 / 10);

                  rNorm2FactorLocal += m_gwavePadded[id][0] * m_gwavePadded[id][0];
              }             
           }
       }
       #pragma omp atomic
           rNorm2Factor += rNorm2FactorLocal;
    }

    rNorm2Factor = sqrt(rNorm2Factor);


    for (int i = 0; i < 61; i++)
    {
       for (int j = 0; j < 61; j++)
       {
           int id = i*m_iPaddedWidth + j;
           m_gwavePadded[id][0] /= rNorm2Factor;
       }
    }   

    real_t wyt = -2.3;
    real_t wxt = -2.3;
    fftw_complex temp;

    for (int i = 0; i < 61; i++)
    {
       for (int j = 0; j < 61; j++)
       {
           int idPadedgwave = i * m_iPaddedWidth + j;   // index of padded gwave

           real_t yy = i - (61 - 1) / 2;
           real_t xx = j - (61 - 1) / 2;

           temp[0] = cos(wxt*xx + wyt*yy);
           temp[1] = sin(wxt*xx + wyt*yy);


           real_t real = m_gwavePadded[idPadedgwave][0] * temp[0];
           real_t imag = m_gwavePadded[idPadedgwave][0] * temp[1];

           m_gwavePadded[idPadedgwave][0] = real;
           m_gwavePadded[idPadedgwave][1] = imag;
       }
    }

    fftw_execute(m_planForwardgwave);

    std::ofstream out("gwave.csv", std::ios::out | std::ios::trunc);
    for(int i=0; i<rows; i++)
    {
       for(int j=0; j<cols; j++)
       {
           out<<m_gwavePadded[i*cols+j][0]<<"+"<<"i"<<m_gwavePadded[i*cols+j][1]<<", ";
       }
       out<<"\n";
    }
    out.close();

	out.open("gwavefq.csv", std::ios::out | std::ios::trunc);
    for(int i=0; i<rows; i++)
    {
       for(int j=0; j<cols; j++)
       {
           out<<m_gwavePaddedFq[i*cols+j][0]<<"+"<<"i"<<m_gwavePaddedFq[i*cols+j][1]<<", ";
       }
       out<<"\n";
    }
    out.close();


    std::cout<<m_gwavePadded[1][0]<<", "<<m_gwavePadded[1][1]<<endl;
	std::cout<<m_gwavePadded[61][0]<<", "<<m_gwavePadded[61][1]<<endl;

    fftw_free(m_gwavePadded);
    fftw_free(m_gwavePaddedFq);

}