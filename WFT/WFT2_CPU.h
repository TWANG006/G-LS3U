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
		float rSigmaX, float rWxl, float rWxh, float rWxi,
		float rSigmaY, float rWyl, float rWyh, float rWyi,
		float rThr,
		WFT2_HostResults &z,
		int iNumberThreads = 1);

	~WFT2_cpu();

	// Make this class a callable object (functor)
	void operator() (
		fftwf_complex *f, 
		WFT2_HostResults &z,
		double &time);

private:
	/* Initilaize the WFT2 algorithm 
		1. Calculate the padding size.
		2. Allocate&initialize arrays with the padded size
		3. Make FFTW3 plans								   */
	int WFT2_Initialize(WFT2_HostResults &z);

	/* feed the f into its padded m_fPadded */
	void WFT2_feed_fPadded(fftwf_complex *f);

	/* Sequential & Multi-threaded Implementations of the WFF2&WFR2
	   algorithm												    */
	void WFF2_SetThreashold(fftwf_complex *f);	
	void WFF2(fftwf_complex *f, WFT2_HostResults &z, double &time);
	void WFR2(fftwf_complex *f, WFT2_HostResults &z, double &time);
	

public:
	/* Internal arrays */
	fftwf_complex	*m_fPadded;			// Padded f 
	fftwf_complex	*m_gwavePadded;		// Padded gwave
	fftwf_complex	*m_fPaddedFq;		// FFT of padded f
	fftwf_complex	*m_gwavePaddedFq;	// FFT of padded gwave
	fftwf_complex	*m_Sf;				// Sf after wft 

	/* FFTW plans: all the plans can be made once and used across the algorithm */
	fftwf_plan		m_planForwardf;		// FFTW fwd plan of f
	fftwf_plan		m_planForwardgwave; // FFTW fwd plan of gwave
	fftwf_plan		m_planInverseSf;	// FFTW inv plan of Sf

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

}	// namespace WFT_FPA
}	// namespace WFT

#endif // !WFT2_CPU_H
