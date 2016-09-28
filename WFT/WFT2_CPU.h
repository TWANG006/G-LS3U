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

	// Default parameters are used based on the WFT_TYPE
	WFT2_cpu(
		int iWidth, int iHeight,
		WFT_TYPE type,
		WFT2_HostResults &z, 
		int iNumberThreads = 1);

	// Parameters are set by the input parameters
	WFT2_cpu(
		int iWidth, int iHeight,
		WFT_TYPE type,
		real_t rSigmaX,	real_t rWxl, real_t rWxh, real_t rWxi,
		real_t rSigmaY, real_t rWyl, real_t rWyh, real_t rWyi,
		real_t rThr,
		WFT2_HostResults &z,
		int iNumberThreads = 1);

	~WFT2_cpu();

	// Make this class a callable object (functor)
	void operator() (
		fftw3Complex *f, 
		WFT2_HostResults &z);

private:
	/* Initilaize the WFT2 algorithm 
		1. Calculate the padding size.
		2. Allocate&initialize arrays with the padded size
		3. Make FFTW3 plans								   */
	int WFT2_Initialize(WFT2_HostResults &z);

	/* feed the f into its padded m_fPadded */
	void WFT2_feed_fPadded(fftw3Complex *f);

	/* Sequential & Multi-threaded Implementations of the WFF2&WFR2
	   algorithm												    */
	void WFF2_Common(fftw3Complex *f);	
	void WFF2_Seq(fftw3Complex *f, WFT2_HostResults &z);
	void WFF2_Mul(fftw3Complex *f, WFT2_HostResults &z);	
	void WFR2_Seq(fftw3Complex *f, WFT2_HostResults &z);
	void WFR2_Mul(fftw3Complex *f, WFT2_HostResults &z);
	

private:
	/* Internal arrays */
	fftw3Complex	*m_fPadded;			// Padded f 
	fftw3Complex	*m_gwavePadded;		// Padded gwave
	fftw3Complex	*m_fPaddedFq;		// FFT of padded f
	fftw3Complex	*m_gwavePaddedFq;	// FFT of padded gwave
	fftw3Complex	*m_Sf;				// Sf after wft 

	/* FFTW plans: all the plans can be made once and used across the algorithm */
	fftw3Plan		m_planForwardf;		// FFTW fwd plan of f
	fftw3Plan		m_planForwardgwave; // FFTW fwd plan of gwave
	fftw3Plan		m_planInverseSf;	// FFTW inv plan of Sf

	/* Internal Parameters */
	int				m_iWidth;			// width of the fringe pattern
	int				m_iHeight;			// height of the fringe pattern	

	/* Initially, size(A) + size(B) - 1, search the lookup table for * 
	 * Optimized size for the FFT									 */
	int				m_iPaddedWidth;		// width after padding for optimized FFT
	int				m_iPaddedHeight;	// height after padding for optimized FFT
	WFT_TYPE		m_type;				// 'WFF' or 'WFR'
	int				m_iSx;				// half Windows size along x
	int				m_iSy;				// half Windows size along y
	real_t			m_rSigmaX;			// sigma of the window in x-axis
	real_t			m_rWxl;				// lower bound of frequency in x-axis
	real_t			m_rWxi;				// step size of frequency in x-axis
	real_t			m_rWxh;				// upper bound of frequency in x-axis
	real_t			m_rSigmaY;			// sigma of the window in x-axis
	real_t			m_rWyl;				// lower bound of frequency in y-axis
	real_t			m_rWyh;				// upper bound of frequency in y-axis
	real_t			m_rWyi;				// step size of frequency in y-axis	

	/* threshold for 'wff', no needed for 'wfr' *
	 * NOTE: if m_rThr < 0, it is calculated as *
	 * m_rThr = 6 * sqrt(mean2(abs(f).^2)/3)    */
	real_t			m_rThr;		

	/* Parameters for Thread control */
	int m_iNumberThreads;
};

}	// namespace WFT_FPA
}	// namespace WFT

#endif // !WFT2_CPU_H
