#ifndef WFT2_CPUF_H
#define WFT2_CPUF_H

#include "WFT-FPA.h"
#include "WFT.h"

namespace WFT_FPA{
namespace WFT{

/*
*PURPOSE
	Functor to perform WFT algorithm on CPU sequentially.
*/

class WFT_FPA_DLL_EXPORTS WFT2_cpuF
{
public:

	WFT2_cpuF() = delete;
	WFT2_cpuF(const WFT2_cpuF&) = delete;
	WFT2_cpuF &operator=(const WFT2_cpuF&) = delete;

	// Default parameters are used based on the WFT_TYPE
	WFT2_cpuF(int iWidth, int iHeight,
			  WFT_TYPE type,
			  WFT2_HostResultsF &z, 
			  int iNumberThreads = 1);

	// Parameters are set by the input parameters
	WFT2_cpuF(int iWidth, int iHeight,
			  WFT_TYPE type,
			  float rSigmaX, float rWxl, float rWxh, float rWxi,
			  float rSigmaY, float rWyl, float rWyh, float rWyi,
			  float rThr,
			  WFT2_HostResultsF &z,
			  int iNumberThreads = 1);

	~WFT2_cpuF();

	// Make this class a callable object (functor)
	void operator() (fftwf_complex *f, 
					 WFT2_HostResultsF &z,
					 double &time);

private:
	/* Initilaize the WFT2 algorithm 
		1. Calculate the padding size.
		2. Allocate&initialize arrays with the padded size
		3. Make FFTW3 plans								   */
	int WFT2_Initialize(WFT2_HostResultsF &z);
	void WFF2_Init(WFT2_HostResultsF &z);
	int  WFR2_Init(WFT2_HostResultsF &z);

	/* feed the f into its padded m_fPadded */
	void WFT2_feed_fPadded(fftwf_complex *f);

	// Set the threashold of the WFF2 algorithm if the initial value of m_rThr = -1;
	void WFF2_SetThreashold(fftwf_complex *f);	

	/* Sequential & Multi-threaded Implementations of the WFF2&WFR2
	   algorithm												    */
	void WFF2(fftwf_complex *f, WFT2_HostResultsF &z, double &time);
	void WFR2(fftwf_complex *f, WFT2_HostResultsF &z, double &time);
	

private:
	/* Internal arrays */
	fftwf_complex	*m_fPadded;			// Padded f 
	fftwf_complex	*m_FfPadded;		// FFT of padded f
	fftwf_complex	*m_gPadded;			// padded g
	
	fftwf_plan		m_planForwardf;		// FFTW fwd plan of f
	fftwf_plan		*m_planForwardgwave;// FFTW fwd plan of gwave
	fftwf_plan		*m_planForwardSf;	// FFTW fwd plan of Sf
	fftwf_plan		*m_planInverseSf;	// FFTW inv plan of Sf
	
	// Plans for calculating curvature cxx&cyy using FFT-based convolution
	fftwf_plan		m_planForwardcxx;
	fftwf_plan		m_planInversecxx;
	fftwf_plan		m_planForwardcyy;
	fftwf_plan		m_planInversecyy;
	fftwf_plan		m_planForwardxg;
	fftwf_plan		m_planForwardyg;

	// threadprivate intermediate results for WFF & WFR
	fftwf_complex	*im_Fgwave;
	fftwf_complex	*im_gwave;
	fftwf_complex	*im_Sf;					// Sf after wft 

	// WFF partial results
	fftwf_complex	*im_filtered;			// partial filtered image

	// WFR partial results
	float			*im_r;
	float			*im_p;
	float			*im_wx;
	float			*im_wy;
	fftwf_complex	*im_cxxPadded;			// Padded wx for computation of cxx
	fftwf_complex	*im_cyyPadded;			// Padded wy for computation of cyy
	fftwf_complex	*im_xgPadded;			// padded x.*g
	fftwf_complex	*im_ygPadded;			// padded y.*g	
	float			m_rSumxxg;				// 1 / x.*x.*g
	float			m_rSumyyg;				// 1 / y.*y.*g
	

	/* Internal Parameters */
	int				m_iWidth;			// width of the fringe pattern
	int				m_iHeight;			// height of the fringe pattern	

	/* Initially, size(A) + size(B) - 1, search the lookup table for * 
	 * Optimized size for the FFT									 */
	int				m_iPaddedWidth;		// width after padding for optimized FFT
	int				m_iPaddedHeight;	// height after padding for optimized FFT

	int				m_iPaddedWidthCurvature;	// width after padding for optimized FFT for curvature computation
	int				m_iPaddedHeightCurvature;	// height after padding for optimized FFT for curvature computation


	WFT_TYPE		m_type;				// 'WFF' or 'WFR'
	int				m_iSx;				// half Windows size along x
	int				m_iSy;				// half Windows size along y
	int				m_iWinWidth;		// Gaussian Window width
	int				m_iWinHeight;		// Gaussian Window height
	float			m_rSigmaX;			// sigma of the window in x-axis
	float			m_rWxl;				// lower bound of frequency in x-axis
	float			m_rWxi;				// step size of frequency in x-axis
	float			m_rWxh;				// upper bound of frequency in x-axis
	float			m_rSigmaY;			// sigma of the window in x-axis
	float			m_rWyl;				// lower bound of frequency in y-axis
	float			m_rWyh;				// upper bound of frequency in y-axis
	float			m_rWyi;				// step size of frequency in y-axis	

	/* threshold for 'wff', no needed for 'wfr' *
	 * NOTE: if m_rThr < 0, it is calculated as *
	 * m_rThr = 6 * sqrt(mean2(abs(f).^2)/3)    */
	float			m_rThr;		

	/* Parameters for Thread control */
	int m_iNumberThreads;

	/* Low-level members for multi-threading implementation */
};

}	// namepsace WFT
}	// namespace WFT_FPA

#endif // !WFT2_CPUF_H
