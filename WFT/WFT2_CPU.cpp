#include "WFT2_CPU.h"
#include <iostream>

namespace WFT_FPA{
namespace WFT{

WFT2_cpu::WFT2_cpu(
	int iWidth, int iHeight,
	WFT_TYPE type,
	WFT2_HostResults &z)
	: m_iWidth(iWidth)
	, m_iHeight(iHeight)
	, m_type(type)
	, m_rThr(-1)
{
	/* Type specific parameter initializations*/
	if(WFT_TYPE::WFF == m_type)
	{
		m_rSigmaX = real_t(10.0); 
		m_rWxl = -2 - real_t(3.0) / m_rSigmaX;
		m_rWxi = real_t(1.0) / m_rSigmaX;
		m_rWxh = 2 + real_t(3.0) / m_rSigmaX;

		m_rSigmaY = real_t(10.0);
		m_rWyl = -2 - real_t(3.0) / m_rSigmaY;
		m_rWyi = real_t(1.0) / m_rSigmaY;
		m_rWyh = 2 + real_t(3.0) / m_rSigmaY;
	}
	else if(WFT_TYPE::WFR == m_type)
	{
		m_rSigmaX = real_t(10.0); 
		m_rWxl = real_t(-2);
		m_rWxi = real_t(0.025);
		m_rWxh = real_t(2);

		m_rSigmaY = real_t(10.0);
		m_rWyl = real_t(-2.0);
		m_rWyi = real_t(0.025);
		m_rWyh = real_t(2.0);
	}

	/* Do the Initialization */
	if(-1 == WFT2_Initialize(z))
	{
		std::cout<<"FFT padding is out of range [4096]. Shrink the size of either the image or the Gaussian Window!"<<std::endl;
		throw -1;
	}
	
}

WFT2_cpu::WFT2_cpu(
	int iWidth, int iHeight,
	WFT_TYPE type,
	real_t rSigmaX,	real_t rWxl, real_t rWxh, real_t rWxi,
	real_t rSigmaY, real_t rWyl, real_t rWyh, real_t rWyi,
	real_t rThr,
	WFT2_HostResults &z)
	: m_iWidth(iWidth)
	, m_iHeight(iHeight)
	, m_type(type)
	, m_rSigmaX(rSigmaX)
	, m_rSigmaY(rSigmaY)
	, m_rWxl(rWxl)
	, m_rWxi(rWxi)
	, m_rWxh(rWxh)
	, m_rWyl(rWxl)
	, m_rWyi(rWyi)
	, m_rWyh(rWyh)
	, m_rThr(rThr)
{
	WFT2_Initialize(z);	
}

WFT2_cpu::~WFT2_cpu()
{
#ifdef WFT_FPA_DOUBLE
	fftw_free(m_fPadded); m_fPadded = nullptr;
	fftw_free(m_gwavePadded); m_gwavePadded = nullptr;
#else
	fftwf_free(m_fPadded); m_fPadded = nullptr;
	fftwf_free(m_gwavePadded); m_gwavePadded = nullptr;
#endif // WFT_FPA_DOUBLE
}

void WFT2_cpu::operator() (fftw3Complex *f, WFT2_HostResults &z)
{

}


/* Private functions */
int WFT2_cpu::WFT2_Initialize(WFT2_HostResults &z)
{
	/* General parameters intitialization */
	m_iSx = int(round(m_rSigmaX));
	m_iSy = int(round(m_rSigmaY));

	// Do the first padding in order to perform the cyclic convolution using FFT
	// The padding size is size(A) + size(B) - 1;
	m_iPaddedHeight = m_iHeight + 2 * m_iSy;
	m_iPaddedWidth = m_iWidth + 2 * m_iSx;
	
	// Do the second padding in order to fit the optimized size for FFT
	int iH = getFirstGreater(m_iPaddedHeight);
	int iW = getFirstGreater(m_iPaddedWidth);
	// Check whether the preferred size is within 4096
	if(-1 == iH || -1 == iW)
	{
		// Out of range
		return -1;
	}
	else
	{
		m_iPaddedHeight = OPT_FFT_SIZE[iH];
		m_iPaddedWidth	= OPT_FFT_SIZE[iW];
	}

	/* Memory Allocation (Already padded) */
#ifdef WFT_FPA_DOUBLE
	// Allocate memory for padded arrays
	m_fPadded = (fftw3Complex*)fftw_malloc(sizeof(fftw3Complex)*m_iPaddedHeight*m_iPaddedWidth);
	m_gwavePadded = (fftw3Complex*)fftw_malloc(sizeof(fftw3Complex)*m_iPaddedHeight*m_iPaddedWidth);

	// Allocate memory for the output z
	if(WFT_TYPE::WFF == m_type)
	{
		z.m_filtered = (fftw3Complex*)fftw_malloc(sizeof(fftw3Complex)*m_iWidth*m_iHeight);
	}
	else if(WFT_TYPE::WFR == m_type)
	{
		z.m_wx = (real_t*)malloc(sizeof(real_t)*m_iWidth*m_iHeight);
		z.m_wy = (real_t*)malloc(sizeof(real_t)*m_iWidth*m_iHeight);
		z.m_phase = (real_t*)malloc(sizeof(real_t)*m_iWidth*m_iHeight);
		z.m_phase_comp = (real_t*)malloc(sizeof(real_t)*m_iWidth*m_iHeight);
		z.m_b = (real_t*)malloc(sizeof(real_t)*m_iWidth*m_iHeight);
		z.m_r = (real_t*)malloc(sizeof(real_t)*m_iWidth*m_iHeight);
		z.m_cx =(real_t*)malloc(sizeof(real_t)*m_iWidth*m_iHeight);
		z.m_cy = (real_t*)malloc(sizeof(real_t)*m_iWidth*m_iHeight);
	}
#else
	// Allocate memory for padded arrays
	m_fPadded = (fftw3Complex*)fftwf_malloc(sizeof(fftw3Complex)*m_iPaddedHeight*m_iPaddedWidth);
	m_gwavePadded = (fftw3Complex*)fftwf_malloc(sizeof(fftw3Complex)*m_iPaddedHeight*m_iPaddedWidth);
	// Allocate memory for the output z
	if(WFT_TYPE::WFF == m_type)
	{
		z.m_filtered = (fftw3Complex*)fftwf_malloc(sizeof(fftw3Complex)*m_iWidth*m_iHeight);
	}
	else if(WFT_TYPE::WFR == m_type)
	{
		z.m_wx = (real_t*)malloc(sizeof(real_t)*m_iWidth*m_iHeight);
		z.m_wy = (real_t*)malloc(sizeof(real_t)*m_iWidth*m_iHeight);
		z.m_phase = (real_t*)malloc(sizeof(real_t)*m_iWidth*m_iHeight);
		z.m_phase_comp = (real_t*)malloc(sizeof(real_t)*m_iWidth*m_iHeight);
		z.m_b = (real_t*)malloc(sizeof(real_t)*m_iWidth*m_iHeight);
		z.m_r = (real_t*)malloc(sizeof(real_t)*m_iWidth*m_iHeight);
		z.m_cx =(real_t*)malloc(sizeof(real_t)*m_iWidth*m_iHeight);
		z.m_cy = (real_t*)malloc(sizeof(real_t)*m_iWidth*m_iHeight);
	}
#endif // WFT_FPA_DOUBLE
	
	/* Generate the windows g (g is the same across the calculation) *
	 * g = exp(-x.*x /2/sigmax/sigmax - y.*y /2/sigmay/sigmay)	     * 
	 * And set padded region of both m_fPadded and m_gwavePadded to  *
	 * zeros.						                                 */
	real_t rNorm2Factor = 0;					// Factor used for normalization
	for (auto i= 0; i < m_iPaddedHeight; i++)
	{
		for (auto j = 0; j < m_iPaddedWidth; j++)
		{
			int id = i * m_iPaddedWidth + j;	// 1D index of 2D array elems
			int iWinWidth = 2 * m_iSx + 1;		// Gaussian Window width
			int iWinHeight = 2 * m_iSy + 1;		// Gaussian Window height

			// Construct m_gwavePadded matrix
			// Except the first 2*sx+1 by 2*sy+1 elements, all are 0's. Also, all imags are 0's
			if (i < iWinHeight && j < iWinWidth)
			{
				int y = i - (iWinHeight - 1) / 2;
				int x = j - (iWinWidth - 1) / 2;

				m_gwavePadded[id][0] = exp(-real_t(x*x)/2/m_rSigmaX/m_rSigmaX 
					- real_t(y*y)/2/m_rSigmaY/m_rSigmaY);

				rNorm2Factor += m_gwavePadded[id][0] * m_gwavePadded[id][0];
			}
			m_gwavePadded[id][1] = 0;

			// Set m_fPadded to zeros, because it will be filled later when execute
			// the functor
			m_fPadded[id][0] = m_fPadded[id][1] = 0;	// Both real & imag are set to 0
		}
	}
	// Do the normalization: g = g/sqrt(sum(sum(g.*g));
	rNorm2Factor = sqrt(rNorm2Factor);
	for (auto i = 0; i < 2 * m_iSy + 1; i++)
	{
		for (auto j = 0; j < 2 * m_iSx + 1; j++)
		{
			int id = i*m_iPaddedWidth + j;
			m_gwavePadded[id][0] /= rNorm2Factor;
		}
	}

	return 0;
}


}	// namespace WFT_FPA
}	// namespace WFT

