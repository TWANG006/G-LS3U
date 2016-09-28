#include "gtest\gtest.h"
#include <vector>
#include <iostream>
#include <fftw3.h>

#include "Utils.h"

#include "WFT-FPA.h"

using namespace std;

TEST(FFTW3_C2C_In_place, FFTW3_C2C)
{
	vector<float> B{
		0.8147f, 0.6324f, 0.9575f, 0.9572f,
		0.9058f, 0.0975f, 0.9649f, 0.4854f,
		0.1270f, 0.2785f, 0.1576f, 0.8003f,
		0.9134f, 0.5469f, 0.9706f, 0.1419f};

	fftwf_complex *A = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * 16);
	fftwf_complex *m_freqDom1 = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * 16);
	fftwf_complex *m_freqDom2 = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * 4 * (4 / 2 + 1));

	A[0][0] = 0.8147; A[0][1] = 0;
	A[1][0] = 0.6324; A[1][1] = 0;
	A[2][0] = 0.9575; A[2][1] = 0;
	A[3][0] = 0.9572; A[3][1] = 0;
	A[4][0] = 0.9058; A[4][1] = 0;
	A[5][0] = 0.0975; A[5][1] = 0;
	A[6][0] = 0.9649; A[6][1] = 0;
	A[7][0] = 0.4854; A[7][1] = 0;
	A[8][0] = 0.1270; A[8][1] = 0;
	A[9][0] = 0.2785; A[9][1] = 0;
	A[10][0] = 0.1576; A[10][1] = 0;
	A[11][0] = 0.8003; A[11][1] = 0;
	A[12][0] = 0.9134; A[12][1] = 0;
	A[13][0] = 0.5469; A[13][1] = 0;
	A[14][0] = 0.9706; A[14][1] = 0;
	A[15][0] = 0.1419; A[15][1] = 0;

	fftwf_plan plan1 = fftwf_plan_dft_2d(4, 4, A, m_freqDom1, FFTW_FORWARD, FFTW_ESTIMATE);
	fftwf_plan plan2 = fftwf_plan_dft_r2c_2d(4, 4, B.data(), m_freqDom2, FFTW_ESTIMATE);
	fftwf_plan plan3 = fftwf_plan_dft_2d(4,4, m_freqDom1, A, FFTW_BACKWARD, FFTW_ESTIMATE);

	fftwf_execute(plan1);
	fftwf_execute(plan2);

	ASSERT_TRUE(m_freqDom1[0][0] - m_freqDom2[0][0] <= 1e-6 && 
				m_freqDom1[0][1] - m_freqDom2[0][1] <= 1e-6);

	// Do scaling
	for(int i=0; i<16; i++)
	{
		WFT_FPA::fftwComplexScale<fftwf_complex, float>(m_freqDom1[i], 1/16.0f);
	}
		
	fftwf_execute(plan3);


	ASSERT_TRUE(B[0] - A[0][0] <= 1E-6);


	fftwf_destroy_plan(plan1);
	fftwf_destroy_plan(plan2);
	fftwf_destroy_plan(plan3);

	fftwf_free(A);
	fftwf_free(m_freqDom1);
	fftwf_free(m_freqDom2);


}