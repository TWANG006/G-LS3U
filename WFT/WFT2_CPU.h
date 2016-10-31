#ifndef WFT2_CPU_H
#define WFT2_CPU_H

#include "WFT-FPA.h"
#include "WFT.h"

namespace WFT_FPA{
namespace WFT{

/*
*PURPOSE
	Functor to perform WFT algorithm on CPU sequentially.
*/

class WFT_FPA_DLL_EXPORTS WFT2_cpu
{
public:

	WFT2_cpu() = delete;
	WFT2_cpu(const WFT2_cpu&) = delete;
	WFT2_cpu &operator=(const WFT2_cpu&) = delete;

	// Default parameters are used based on the WFT_TYPE
	WFT2_cpu(int iWidth, int iHeight,
			 WFT_TYPE type,
			 WFT2_HostResults &z, 
			 int iNumberThreads = 1);

	// Parameters are set by the input parameters
	WFT2_cpu(int iWidth, int iHeight,
			 WFT_TYPE type,
			 double rSigmaX, double rWxl, double rWxh, double rWxi,
			 double rSigmaY, double rWyl, double rWyh, double rWyi,
			 double rThr,
			 WFT2_HostResults &z,
			 int iNumberThreads = 1);

	~WFT2_cpu();

	// Make this class a callable object (functor)
	void operator() (fftw_complex *f, 
					 WFT2_HostResults &z,
					 double &time);

private:
	/* Initilaize the WFT2 algorithm 
		1. Calculate the padding size.
		2. Allocate&initialize arrays with the padded size
		3. Make FFTW3 plans								   */
	int WFT2_Initialize(WFT2_HostResults &z);
	void WFF2_Init(WFT2_HostResults &z);
	int  WFR2_Init(WFT2_HostResults &z);

	/* feed the f into its padded m_fPadded */
	void WFT2_feed_fPadded(fftw_complex *f);

	// Set the threashold of the WFF2 algorithm if the initial value of m_rThr = -1;
	void WFF2_SetThreashold(fftw_complex *f);	

	/* Sequential & Multi-threaded Implementations of the WFF2&WFR2
	   algorithm												    */
	void WFF2(fftw_complex *f, WFT2_HostResults &z, double &time);
	void WFR2(fftw_complex *f, WFT2_HostResults &z, double &time);
	

private:
	/* Internal arrays */
	fftw_complex	*m_fPadded;			// Padded f 
	fftw_complex	*m_FfPadded;		// FFT of padded f
	fftw_complex	*m_gPadded;			// padded g
	
	fftw_plan		m_planForwardf;		// FFTW fwd plan of f
	fftw_plan		*m_planForwardgwave;// FFTW fwd plan of gwave
	fftw_plan		*m_planForwardSf;	// FFTW fwd plan of Sf
	fftw_plan		*m_planInverseSf;	// FFTW inv plan of Sf

	// Plans for calculating curvature cxx&cyy using FFT-based convolution
	fftw_plan		m_planForwardcxx;
	fftw_plan		m_planInversecxx;
	fftw_plan		m_planForwardcyy;
	fftw_plan		m_planInversecyy;
	fftw_plan		m_planForwardxg;
	fftw_plan		m_planForwardyg;
	
	// threadprivate intermediate results for WFF & WFR
	fftw_complex	*im_Fgwave;
	fftw_complex	*im_gwave;
	fftw_complex	*im_Sf;					// Sf after wft 

	// WFF partial results
	fftw_complex	*im_filtered;			// partial filtered image

	// WFR partial results				
	double			*im_r;					// partial ridges
	double			*im_p;					// partial phases
	double			*im_wx;					// partial local frequency in x
	double			*im_wy;					// partial local frequency in y
	fftw_complex	*im_cxxPadded;			// Padded wx for computation of cxx
	fftw_complex	*im_cyyPadded;			// Padded wy for computation of cyy
	fftw_complex	*im_xgPadded;			// padded x.*g
	fftw_complex	*im_ygPadded;			// padded y.*g	
	double			m_rSumxxg;				// 1 / x.*x.*g
	double			m_rSumyyg;				// 1 / y.*y.*g

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
	double			m_rSigmaX;			// sigma of the window in x-axis
	double			m_rWxl;				// lower bound of frequency in x-axis
	double			m_rWxi;				// step size of frequency in x-axis
	double			m_rWxh;				// upper bound of frequency in x-axis
	double			m_rSigmaY;			// sigma of the window in x-axis
	double			m_rWyl;				// lower bound of frequency in y-axis
	double			m_rWyh;				// upper bound of frequency in y-axis
	double			m_rWyi;				// step size of frequency in y-axis	

	/* threshold for 'wff', no needed for 'wfr' *
	 * NOTE: if m_rThr < 0, it is calculated as *
	 * m_rThr = 6 * sqrt(mean2(abs(f).^2)/3)    */
	double			m_rThr;		

	/* Parameters for Thread control */
	int m_iNumberThreads;

	/* Low-level members for multi-threading implementation */
};

}	// namespace WFT_FPA
}	// namespace WFT

#endif // !WFT2_CPU_H
