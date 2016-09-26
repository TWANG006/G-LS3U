#include <iostream>
#include <vector>

#include "Utils.h"
#include "WFT.h"

using namespace std;

class Test
{
public:



	void operator()(int a, int b, int c, int d)
	{
		cout<<a+b+c+d<<endl;
	}

private:
	int x;
};

int main()
{
	WFT_FPA::WFT::WFT2_HostResults *h = new WFT_FPA::WFT::WFT2_HostResults();

	delete h;

	int n = WFT_FPA::WFT::getFirstGreater(1);
	cout<<n<<", "<<WFT_FPA::WFT::OPT_FFT_SIZE[n]<< endl;

	/*vector<float> B{
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
	fftwf_plan plan2 = fftwf_plan_dft_r2c_2d(4,4, B.data(), m_freqDom2, FFTW_ESTIMATE);
	fftwf_execute(plan1);
	fftwf_execute(plan2);

	for (int y = 0; y < 4; y++)
	{
		for (int x = 0; x < 4; x++)
		{
			int gid = y * 4 + x;
			cout << gid << "real" << m_freqDom1[gid][0] << "imag" << m_freqDom1[gid][1] << "\n";
		}
	}
	
	cout<<WFT_FPA::add(1,2)<<endl<<endl;

	for(int i=0; i<12; i++)
	{
		cout << i << "real" << m_freqDom1[i][0] << "imag" << m_freqDom1[i][1] << "\n";
	}

	cout<<A[0][0]<<endl;
	cufftComplex *fromFFTW = reinterpret_cast<cufftComplex*>(A);
	cout<<fromFFTW[0].x<<endl;


	fftwf_free(A);
	fftwf_free(m_freqDom1);
	fftwf_free(m_freqDom2);

	fftwf_destroy_plan(plan1);
	fftwf_destroy_plan(plan2);*/

	return 0;
}