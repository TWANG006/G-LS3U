#include <iostream>
#include <vector>

#include <fftw3.h>

#include "WFT-Utils.h"

using namespace std;

int main()
{
	fftwf_complex *A = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * 16);
	fftwf_complex *m_freqDom1 = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * 16);

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
	fftwf_execute(plan1);

	for (int y = 0; y < 4; y++)
	{
		for (int x = 0; x < 4; x++)
		{
			int gid = y * 4 + x;
			cout << gid << "real" << m_freqDom1[gid][0] << "imag" << m_freqDom1[gid][1] << "\n";
		}
	}
	
	cout<<add(1,2)<<endl;

	return 0;
}