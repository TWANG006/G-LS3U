#ifndef WFT2_CUDA_H
#define WFT2_CUDA_H

#include "WFT.h"
#include "Utils.h"


namespace WFT_FPA{
namespace WFT{

class WFT_FPA_DLL_EXPORTS WFT2_CUDA
{
public:

	// Default parameters are used based on the WFT_TYPE
	WFT2_CUDA(
		int iWidth, int iHeight,
		WFT_TYPE type,
		WFT2_DeviceResults& z,
		int iNumStreams = 1);

	// Parameters are set by the input parameters
	WFT2_CUDA(
		int iWidth, int iHeight,
		WFT_TYPE type,
		double rSigmaX, double rWxl, double rWxh, double rWxi,
		double rSigmaY, double rWyl, double rWyh, double rWyi,
		double rThr,
		WFT2_DeviceResults &z,
		int iNumStreams = 1);

	~WFT2_CUDA();

	// Make this class a callable object (functor)
	void operator() (
		cufftDoubleComplex *d_f, 
		WFT2_DeviceResults &d_z,
		double &time);

private:
	/* Initilaize the WFT2 algorithm 
		1. Calculate the padding size.
		2. Allocate&initialize arrays with the padded size
		3. Make cufft plans								   */
	int cuWFT2_Initialize(WFT2_DeviceResults &d_z);
	void cuWFF2_Init(WFT2_DeviceResults &d_z);
	void cuWFR2_Init(WFT2_DeviceResults &d_z);

	/* Feed the f into its padded m_d_fPadded */
	void cuWFT2_feed_fPadded(cufftDoubleComplex *d_f);

	// Set the threashold of the WFF2 algorithm if the initial value of m_rThr = -1;
	void cuWFF2_SetThreashold(cufftDoubleComplex *d_f);

	/* CUDA WFF & WFR Algorithms */
	void cuWFF2(cufftDoubleComplex *d_f, WFT2_DeviceResults &d_z, double &time);
	void cuWFR2(cufftDoubleComplex *d_f, WFT2_DeviceResults &d_z, double &time);

public:
	/* Internal Arrays */
	cufftDoubleComplex	*m_d_fPadded;				// Padded f
	cufftDoubleReal		*m_d_xf;					// Explicit Freq in x for Gaussian Window
	cufftDoubleReal		*m_d_yf;					// Explicit Freq in y for Gaussian Window

	cufftHandle		m_planPadded;			

	cufftHandle		*m_planStreams;
	
	/* Intermediate Results */
	cufftDoubleComplex	**im_d_Fg;					// Explicitly computed Fg in Fourier Domain
	cufftDoubleComplex	**im_d_Sf;

	/* WFF Intermediate Results for each CUDA Stream */

	cufftDoubleComplex	**im_d_filtered;

	/* WFR Intermediate Results for each CUDA stream */
	cufftDoubleReal		**im_d_wx;
	cufftDoubleReal		**im_d_wy;
	cufftDoubleReal		**im_d_p;
	cufftDoubleReal		**im_d_r;
	cufftDoubleReal		*im_d_g;						// g
	cufftDoubleComplex	*im_d_cxxPadded;			// Padded wx for computation of cxx
	cufftDoubleComplex	*im_d_cyyPadded;			// Padded wy for computation of cyy
	cufftDoubleComplex	*im_d_xgPadded;				// padded x.*g
	cufftDoubleComplex	*im_d_ygPadded;				// padded y.*g	

	/* Internal Parameters */
	int				m_iWidth;					// width of the fringe pattern
	int				m_iHeight;					// height of the fringe pattern	

	/* Initially, size(A) + size(B) - 1, search the lookup table for * 
	 * Optimized size for the FFT									 */
	int				m_iPaddedWidth;				// width after padding for optimized FFT
	int				m_iPaddedHeight;			// height after padding for optimized FFT

	int				m_iPaddedWidthCurvature;	// width after padding for optimized FFT for curvature computation
	int				m_iPaddedHeightCurvature;	// height after padding for optimized FFT for curvature computation

	WFT_TYPE		m_type;						// 'WFF' or 'WFR'
	int				m_iSx;						// half Windows size along x
	int				m_iSy;						// half Windows size along y
	int				m_iWinWidth;				// Gaussian Window width
	int				m_iWinHeight;				// Gaussian Window height
	double			m_rSigmaX;					// sigma of the window in x-axis
	double			m_rWxl;						// lower bound of frequency in x-axis
	double			m_rWxi;						// step size of frequency in x-axis
	double			m_rWxh;						// upper bound of frequency in x-axis
	double			m_rSigmaY;					// sigma of the window in x-axis
	double			m_rWyl;						// lower bound of frequency in y-axis
	double			m_rWyh;						// upper bound of frequency in y-axis
	double			m_rWyi;						// step size of frequency in y-axis	
	double			m_rGaussianNorm2;			// L2-norm normalization of Gaussian Window
	double			m_rSumxxg;					// 1 / x.*x.*g
	double			m_rSumyyg;					// 1 / y.*y.*g

	/* threshold for 'wff', no needed for 'wfr' *
	 * NOTE: if m_rThr < 0, it is calculated as *
	 * m_rThr = 6 * sqrt(mean2(abs(f).^2)/3)    */
	double			m_rThr;		
	double			*m_d_rThr;

	/* WFR Intermediate normalization params */
	double			*m_d_rg_norm2;
	double			m_rxxg_norm2;
	double			*m_d_rxxg_norm2;

	double			m_ryyg_norm2;
	double			*m_d_ryyg_norm2;

	/* Parameters for Thread control */
	int m_iNumStreams;
	cudaStream_t *m_cudaStreams;
	int m_iSMs;							// Number of SMs of the device 0
};

}
}
#endif // !WFT2_CUDA_H
