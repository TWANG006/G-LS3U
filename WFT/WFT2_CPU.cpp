#include "WFT2_CPU.h"
#include <iostream>

#include "Utils.h"

namespace WFT_FPA{
namespace WFT{

WFT2_cpu::WFT2_cpu(
	int iWidth, int iHeight,
	WFT_TYPE type,
	WFT2_HostResults &z,
	int iNumberThreads)
	: m_iWidth(iWidth)
	, m_iHeight(iHeight)
	, m_type(type)
	, m_rThr(-1)
	, m_iNumberThreads(1)
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
	WFT2_HostResults &z,
	int iNumberThreads)
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
	, m_iNumberThreads(1)
{
	/* Do the Initialization */
	if(-1 == WFT2_Initialize(z))
	{
		std::cout<<"FFT padding is out of range [4096]. Shrink the size of either the image or the Gaussian Window!"<<std::endl;
		throw -1;
	}
}

WFT2_cpu::~WFT2_cpu()
{
#ifdef WFT_FPA_DOUBLE
	fftw_destroy_plan(m_planForwardf);
	fftw_destroy_plan(m_planForwardgwave);
	fftw_destroy_plan(m_planInverseSf);

	fftw_free(m_fPadded);		m_fPadded = nullptr;
	fftw_free(m_gwavePadded);	m_gwavePadded = nullptr;
	fftwf_free(m_fPaddedFq);	m_fPaddedFq = nullptr;
	fftwf_free(m_gwavePaddedFq);m_gwavePaddedFq = nullptr;
	fftwf_free(m_Sf);			m_Sf = nullptr;
#else
	fftwf_destroy_plan(m_planForwardf);
	fftwf_destroy_plan(m_planForwardgwave);
	fftwf_destroy_plan(m_planInverseSf);

	fftwf_free(m_fPadded);		m_fPadded = nullptr;
	fftwf_free(m_gwavePadded);	m_gwavePadded = nullptr;
	fftwf_free(m_fPaddedFq);	m_fPaddedFq = nullptr;
	fftwf_free(m_gwavePaddedFq);m_gwavePaddedFq = nullptr;
	fftwf_free(m_Sf);			m_Sf = nullptr;
#endif // WFT_FPA_DOUBLE
}

void WFT2_cpu::operator() (
	fftw3Complex *f, 
	WFT2_HostResults &z)
{

}


/* Private functions */
int WFT2_cpu::WFT2_Initialize(WFT2_HostResults &z)
{
	/* General parameters intitialization */
	m_iSx = int(round(3 * m_rSigmaX));
	m_iSy = int(round(3 * m_rSigmaY));

	// Calculate the first padding in order to perform the cyclic convolution using FFT
	// The padding size is size(A) + size(B) - 1;
	m_iPaddedHeight = m_iHeight + 2 * m_iSy;
	m_iPaddedWidth = m_iWidth + 2 * m_iSx;
	
	// Calculate the second padding in order to fit the optimized size for FFT
	int iH = getFirstGreater(m_iPaddedHeight);
	int iW = getFirstGreater(m_iPaddedWidth);
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
	/* Allocate memory for padded arrays */
#ifdef WFT_FPA_DOUBLE	
	m_fPadded = (fftw3Complex*)fftw_malloc(sizeof(fftw3Complex)*m_iPaddedHeight*m_iPaddedWidth);
	m_gwavePadded = (fftw3Complex*)fftw_malloc(sizeof(fftw3Complex)*m_iPaddedHeight*m_iPaddedWidth);
	m_fPaddedFq = (fftw3Complex*)fftw_malloc(sizeof(fftw3Complex)*m_iPaddedHeight*m_iPaddedWidth);
	m_gwavePaddedFq = (fftw3Complex*)fftw_malloc(sizeof(fftw3Complex)*m_iPaddedHeight*m_iPaddedWidth);
	m_Sf = (fftw3Complex*)fftw_malloc(sizeof(fftw3Complex)*m_iPaddedHeight*m_iPaddedWidth);
	
	/* Make the FFTW plans */
	m_planForwardf = fftw_plan_dft_2d(m_iPaddedWidth, m_iPaddedHeight, m_fPadded, m_fPaddedFq, FFTW_FORWARD, FFTW_ESTIMATE);
	m_planForwardgwave = fftw_plan_dft_2d(m_iPaddedWidth, m_iPaddedHeight, m_gwavePadded, m_gwavePaddedFq, FFTW_FORWARD, FFTW_ESTIMATE);
	m_planInverseSf = fftw_plan_dft_2d(m_iPaddedWidth, m_iPaddedWidth, m_Sf, m_Sf, FFTW_BACKWARD, FFTW_ESTIMATE)
#else
	/* Allocate memory for padded arrays */
	m_fPadded = (fftw3Complex*)fftwf_malloc(sizeof(fftw3Complex)*m_iPaddedHeight*m_iPaddedWidth);
	m_gwavePadded = (fftw3Complex*)fftwf_malloc(sizeof(fftw3Complex)*m_iPaddedHeight*m_iPaddedWidth);
	m_fPaddedFq = (fftw3Complex*)fftwf_malloc(sizeof(fftw3Complex)*m_iPaddedHeight*m_iPaddedWidth);
	m_gwavePaddedFq = (fftw3Complex*)fftwf_malloc(sizeof(fftw3Complex)*m_iPaddedHeight*m_iPaddedWidth);
	m_Sf = (fftw3Complex*)fftwf_malloc(sizeof(fftw3Complex)*m_iPaddedHeight*m_iPaddedWidth);

	/* Make the FFTW plans */
	m_planForwardf = fftwf_plan_dft_2d(m_iPaddedWidth, m_iPaddedHeight, m_fPadded, m_fPaddedFq, FFTW_FORWARD, FFTW_ESTIMATE);
	m_planForwardgwave = fftwf_plan_dft_2d(m_iPaddedWidth, m_iPaddedHeight, m_gwavePadded, m_gwavePaddedFq, FFTW_FORWARD, FFTW_ESTIMATE);
	m_planInverseSf = fftwf_plan_dft_2d(m_iPaddedWidth, m_iPaddedWidth, m_Sf, m_Sf, FFTW_BACKWARD, FFTW_ESTIMATE);
#endif

	/* Allocate memory for the output z */
	if(WFT_TYPE::WFF == m_type)
	{
#ifdef WFT_FPA_DOUBLE
		z.m_filtered = (fftw3Complex*)fftw_malloc(sizeof(fftw3Complex)*m_iWidth*m_iHeight);
#else
		z.m_filtered = (fftw3Complex*)fftwf_malloc(sizeof(fftw3Complex)*m_iWidth*m_iHeight);
#endif
		for (int i = 0; i < m_iWidth*m_iHeight; i++)
		{
			z.m_filtered[i][0] = z.m_filtered[i][1] = 0;
		}
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

		for(int i=0; i<m_iWidth*m_iHeight; i++)
		{
			z.m_wx[i] = 0;
			z.m_wy[i] = 0;
			z.m_phase[i] = 0;
			z.m_phase_comp[i] = 0;
			z.m_b[i] = 0;
			z.m_r[i] = 0;
			z.m_cx[i] = 0;
			z.m_cy[i] = 0;
		}
	}
	
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
			// the functor. Both real & imag are set to 0
			m_fPadded[id][0] = m_fPadded[id][1] = 0;	
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
	std::cout<<m_gwavePadded[0][0]<<", "<< m_gwavePadded[0][1]<<std::endl;
	std::cout<<m_gwavePadded[60*m_iPaddedWidth+60][0]<<std::endl;

	return 0;
}

void WFT2_cpu::WFT2_feed_fPadded(fftw3Complex *f)
{
	for (int i = 0; i < m_iHeight; i++)
	{
		for (int j = 0; j < m_iWidth; j++)
		{
			int idf = i*m_iWidth + j;
			int idfPadded = i*m_iPaddedWidth + j;

			m_fPadded[idfPadded][0] = f[idf][0];
			m_fPadded[idfPadded][1] = f[idf][1];
		}
	}
}

void WFT2_cpu::WFF2_Common(fftw3Complex *f)
{
	/* If m_rThr < 0, use the default values for the Threashold in WFF2 algorithm 
		m_rThr = 6 * sqrt( mean2( abs(f).^2 ) / 3)				
	   As well as feed the f into its padded m_fPadded							  */
	if(m_rThr < 0)
	{
		real_t rTemp = 0;
		for (int i = 0; i < m_iHeight; i++)
		{
			for (int j = 0; j < m_iWidth; j++)
			{
				int idf = i*m_iWidth + j;
				int idfPadded = i*m_iPaddedWidth + j;

				real_t abs = fftwComplexAbs<fftw3Complex, real_t>(f[idf]);
				abs = abs*abs;
				rTemp += abs;

				// Feed the array
				m_fPadded[idfPadded][0] = f[idf][0];
				m_fPadded[idfPadded][1] = f[idf][1];
			}
		}
		rTemp = rTemp / (m_iWidth * m_iHeight) / 3;
		m_rThr = 6 * sqrt(rTemp);
	}
	else
	{
		/* Feed the input f into its padded m_fPadded */
		WFT2_feed_fPadded(f);
	}

	/* Pre-compute the FFT of m_fPadded */
#ifdef WFT_FPA_DOUBLE
	fftw_execute(m_planForwardf);
#else
	fftwf_execute(m_planForwardf);
#endif // WFT_FPA_DOUBLE
}

void WFT2_cpu::WFF2_Seq(fftw3Complex *f, WFT2_HostResults &z)
{
	WFF2_Common(f);
}

void WFT2_cpu::WFF2_Mul(fftw3Complex *f, WFT2_HostResults &z)
{
	WFF2_Common(f);
}



void WFT2_cpu::WFR2_Seq(fftw3Complex *f, WFT2_HostResults &z)
{
	WFT2_feed_fPadded(f);
}

void WFT2_cpu::WFR2_Mul(fftw3Complex *f, WFT2_HostResults &z)
{
	WFT2_feed_fPadded(f);
}

}	// namespace WFT_FPA
}	// namespace WFT

