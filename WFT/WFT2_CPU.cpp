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
		std::cout<<"FFT padding is out of range [4096]. Shrink the size of either the image or the Gaussian Window!"<<std::endl;
		throw -1;
	}
	else
	{
		m_iPaddedHeight = OPT_FFT_SIZE[iH];
		m_iPaddedWidth	= OPT_FFT_SIZE[iW];
	}

	/* Allocate required memory */
	
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
void WFT2_cpu::WFT2_Initialize(WFT2_HostResults &z)
{
	// Memory Allocation (Already padded)

	// Generate the windows g (g is the same across the calculation)

	// Do the first Fourier Transform on 

}


}	// namespace WFT_FPA
}	// namespace WFT

