#include "WFT2_CPUf.h"
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <omp.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include "Utils.h"


namespace WFT_FPA{
namespace WFT{

WFT2_cpuF::WFT2_cpuF(int iWidth, int iHeight,
				     WFT_TYPE type,
					 WFT2_HostResultsF &z,
					 int iNumberThreads)
	: m_iWidth(iWidth)
	, m_iHeight(iHeight)
	, m_type(type)
	, m_rThr(-1)
	, m_iNumberThreads(iNumberThreads)
	, m_fPadded(nullptr)
	, m_FfPadded(nullptr)
	, m_gPadded(nullptr)
	, im_gwave(nullptr)
	, im_Fgwave(nullptr)
	, im_Sf(nullptr)
	, m_planForwardgwave(nullptr)
	, m_planInverseSf(nullptr)
	, im_p(nullptr)
	, im_r(nullptr)
	, im_wx(nullptr)
	, im_wy(nullptr)
	, im_cxxPadded(nullptr)
	, im_cyyPadded(nullptr)
	, im_xgPadded(nullptr)
	, im_ygPadded(nullptr)
	, m_rSumxxg(0)
	, m_rSumyyg(0)
{
	/* Type specific parameter initializations*/
	if (WFT_TYPE::WFF == m_type)
	{
		m_rSigmaX = 10.0f;
		m_rWxl = -2.0f - 3.0f / m_rSigmaX;
		m_rWxi = 1.0f / m_rSigmaX;
		m_rWxh = 2.0f + 3.0f / m_rSigmaX;

		m_rSigmaY = 10.0f;
		m_rWyl = -2.0f - 3.0f / m_rSigmaY;
		m_rWyi = 1.0f / m_rSigmaY;
		m_rWyh = 2.0f + 3.0f / m_rSigmaY;
	}
	else if (WFT_TYPE::WFR == m_type)
	{
		m_rSigmaX = 10.0f;
		m_rWxl = -2.0f;
		m_rWxi = 0.025f;
		m_rWxh = 2.0f;

		m_rSigmaY = 10.0f;
		m_rWyl = -2.0f;
		m_rWyi = 0.025f;
		m_rWyh = 2.0f;
	}

	/* Do the Initialization */
	if(-1 == WFT2_Initialize(z))
	{
		std::cout<<"FFT padding is out of range [4096]. Shrink the size of either the image or the Gaussian Window!"<<std::endl;
		throw -1;
	}
}

WFT2_cpuF::WFT2_cpuF(int iWidth, int iHeight,
				     WFT_TYPE type,
				     float rSigmaX,	float rWxl, float rWxh, float rWxi,
				     float rSigmaY, float rWyl, float rWyh, float rWyi,
				     float rThr,
				     WFT2_HostResultsF &z,
					 int iNumberThreads)
	: m_iWidth(iWidth)
	, m_iHeight(iHeight)
	, m_type(type)
	, m_rSigmaX(rSigmaX)
	, m_rSigmaY(rSigmaY)
	, m_rWxl(rWxl)
	, m_rWxi(rWxi)
	, m_rWxh(rWxh)
	, m_rWyl(rWyl)
	, m_rWyi(rWyi)
	, m_rWyh(rWyh)
	, m_rThr(rThr)
	, m_iNumberThreads(iNumberThreads)
	, m_fPadded(nullptr)
	, m_FfPadded(nullptr)
	, m_gPadded(nullptr)
	, im_gwave(nullptr)
	, im_Fgwave(nullptr)
	, im_Sf(nullptr)
	, m_planForwardgwave(nullptr)
	, m_planInverseSf(nullptr)
	, im_p(nullptr)
	, im_r(nullptr)
	, im_wx(nullptr)
	, im_wy(nullptr)
	, im_cxxPadded(nullptr)
	, im_cyyPadded(nullptr)
	, im_xgPadded(nullptr)
	, im_ygPadded(nullptr)
	, m_rSumxxg(0)
	, m_rSumyyg(0)
{
	/* Do the Initialization */
	if(-1 == WFT2_Initialize(z))
	{
		std::cout<<"FFT padding is out of range [4096]. Shrink the size of either the image or the Gaussian Window!"<<std::endl;
		throw -1;
	}
}

WFT2_cpuF::~WFT2_cpuF()
{	
	fftwf_destroy_plan(m_planForwardf);	
	
	if(WFT::WFT_TYPE::WFF == m_type)
	{
		for (int i = 0; i < m_iNumberThreads; i++)
		{
			fftwf_destroy_plan(m_planForwardgwave[i]);
			fftwf_destroy_plan(m_planInverseSf[i]);
			fftwf_destroy_plan(m_planForwardSf[i]);
		}
		delete[] m_planForwardgwave;	m_planForwardgwave = nullptr;
		delete[] m_planInverseSf;		m_planInverseSf = nullptr;
		delete[] m_planForwardSf;		m_planForwardSf = nullptr;

		fftwf_free(im_filtered);		im_filtered = nullptr;
	}
	if(WFT::WFT_TYPE::WFR == m_type)
	{
		for (int i = 0; i < m_iNumberThreads; i++)
		{
			fftwf_destroy_plan(m_planForwardgwave[i]);
			fftwf_destroy_plan(m_planInverseSf[i]);
		}
		delete[] m_planForwardgwave;	m_planForwardgwave = nullptr;
		delete[] m_planInverseSf;		m_planInverseSf = nullptr;

		// Destroy cxx&cyy plans
		fftwf_destroy_plan(m_planForwardcxx);
		fftwf_destroy_plan(m_planInversecxx);
		fftwf_destroy_plan(m_planForwardcyy);
		fftwf_destroy_plan(m_planInversecyy);
		fftwf_destroy_plan(m_planForwardxg);
		fftwf_destroy_plan(m_planForwardyg);

		fftwf_free(im_cxxPadded);	im_cxxPadded = nullptr;
		fftwf_free(im_cyyPadded);	im_cyyPadded = nullptr;
		fftwf_free(im_xgPadded);		im_xgPadded = nullptr;
		fftwf_free(im_ygPadded);		im_ygPadded = nullptr;

		free(im_r);		im_r = nullptr;
		free(im_wx);	im_wx = nullptr;
		free(im_wy);	im_wy = nullptr;
		free(im_p);		im_p = nullptr;
	}

	fftwf_free(m_fPadded);		m_fPadded = nullptr;
	fftwf_free(m_gPadded);		m_gPadded = nullptr;
	fftwf_free(im_Fgwave);		im_Fgwave = nullptr;
	fftwf_free(m_FfPadded);		m_FfPadded = nullptr;
	fftwf_free(im_Fgwave);		im_Fgwave = nullptr;
	fftwf_free(im_Sf);			im_Sf = nullptr;
}

void WFT2_cpuF::operator() (fftwf_complex *f, 
						   WFT2_HostResultsF &z,
						   double &time)
{
	if(WFT_FPA::WFT::WFT_TYPE::WFF == m_type)
		WFF2(f, z, time);
	else if(WFT_FPA::WFT::WFT_TYPE::WFR == m_type)
		WFR2(f, z, time);
}


/* Private functions */
void WFT2_cpuF::WFF2(fftwf_complex *f, WFT2_HostResultsF &z, double &time)
{
	omp_set_num_threads(m_iNumberThreads);

	/* Set the threshold m_rThr if it's not specified by the client */
	WFF2_SetThreashold(f);
	
	/* Pre-compute the FFT of m_fPadded */
	fftwf_execute(m_planForwardf);

	/* Clear the results if they already contain last results */
	#pragma omp parallel for
	for (int i = 0; i < m_iNumberThreads; i++)
	{

		for (int j = 0; j < m_iWidth*m_iHeight; j++)
		{
			int id = i*m_iWidth*m_iHeight + j;
			if (0 == i)
			{
				z.m_filtered[j][0] = 0;
				z.m_filtered[j][1] = 0;
			}
			im_filtered[id][0] = 0;
			im_filtered[id][1] = 0;
		}
	}

	/* map the wl: wi : wh interval to integers from  0 to 
	   size = (wyh - wyl)/wyi + 1 in order to divide the 
	   copmutations across threads, since threads indices are 
	   more conviniently controlled by integers				  */
	int iwx = int((m_rWxh - m_rWxl)*(1/m_rWxi)) + 1;
	int iwy = int((m_rWyh - m_rWyl)*(1/m_rWyi)) + 1;

	/* The core WFF algorithm */
	double start = omp_get_wtime();
	#pragma omp parallel num_threads(m_iNumberThreads)
	{
		// Get thread-spec parameters
		int tid = omp_get_thread_num();
		int nthread = omp_get_num_threads();

		// Thread-based global indices
		int idItm = tid * m_iPaddedWidth * m_iPaddedHeight;	// Intermidiate id
		int idItmResult = tid * m_iHeight * m_iWidth;		// Intermidiate id of filtered

		for (int y = tid; y < iwy; y += nthread)
		{
			for (int x = 0; x < iwx; x++)
			{
				// Recover the wyt and wxt from integer indices for computation
				float wyt = m_rWyl + float(y) * m_rWyi;
				float wxt = m_rWxl + float(x) * m_rWxi;

				// Construct the gwave: gwave=g.*exp(j*wxt*x+j*wyt*y);
				// Euler's Eq: expj(ux+vy) = cos(ux+vy)+jsin(ux+vy)
				fftwf_complex temp;
				for (int i = 0; i < m_iWinHeight; i++)
				{
					for (int j = 0; j < m_iWinWidth; j++)
					{
						int idPadded = i * m_iPaddedWidth + j;

						float yy = i - (m_iWinHeight - 1) / 2;
						float xx = j - (m_iWinWidth - 1) / 2;

						// exp(jj*(wxt*xx + wyt*yy))
						temp[0] = cos(wxt*xx + wyt*yy);	// real
						temp[1] = sin(wxt*xx + wyt*yy);	// imag

						WFT_FPA::Utils::fftwComplexMul(
							im_gwave[idItm + idPadded],
							m_gPadded[idPadded],
							temp);
					}
				}
				// compute Fg = fft2(gwave);
				fftwf_execute(m_planForwardgwave[tid]);

				// compute sf=ifft2(Ff.*Fg)
				for (int i = 0; i < m_iPaddedHeight; i++)
				{
					for (int j = 0; j < m_iPaddedWidth; j++)
					{
						int idPadded = i * m_iPaddedWidth + j;
						WFT_FPA::Utils::fftwComplexMul(
							im_Sf[idItm + idPadded],
							m_FfPadded[idPadded],
							im_Fgwave[idItm + idPadded]);
						WFT_FPA::Utils::fftwComplexScale(
							im_Sf[idItm + idPadded],
							1.0 / float(m_iPaddedHeight*m_iPaddedWidth));
					}
				}
				fftwf_execute(m_planInverseSf[tid]);

				// Compute sf = sf(1+sx: m+sx, 1+sy:n+sy) 
				//		   sf = sf.*(abs(sf)>=thr);
				for (int i = 0; i < m_iPaddedHeight; i++)
				{
					for (int j = 0; j < m_iPaddedWidth; j++)
					{
						int idPadded = i*m_iPaddedWidth + j;					
						int iTransId = (i + m_iSy)*m_iPaddedWidth + j + m_iSx;

						if (i < m_iHeight && j < m_iWidth)
						{
							float abs = WFT_FPA::Utils::fftwComplexAbs(im_Sf[idItm + iTransId]);

							if (abs >= m_rThr)
							{
								im_Sf[idItm + idPadded][0] = im_Sf[idItm + iTransId][0];
								im_Sf[idItm + idPadded][1] = im_Sf[idItm + iTransId][1];
							}
							else
							{
								im_Sf[idItm + idPadded][0] = 0;
								im_Sf[idItm + idPadded][1] = 0;
							}
							im_Sf[idItm + iTransId][0] = 0;
							im_Sf[idItm + iTransId][1] = 0;
						}
						else
						{
							im_Sf[idItm + idPadded][0] = 0;
							im_Sf[idItm + idPadded][1] = 0;
						}
					}
				}

				// Compute z.filtered's intermediate results im_filtered
				fftwf_execute(m_planForwardSf[tid]);

				for (int i = 0; i < m_iPaddedHeight; i++)
				{
					for (int j = 0; j < m_iPaddedWidth; j++)
					{
						int idPadded = i * m_iPaddedWidth + j;
						WFT_FPA::Utils::fftwComplexMul(
							im_Sf[idItm + idPadded],
							im_Sf[idItm + idPadded],
							im_Fgwave[idItm + idPadded]);
						WFT_FPA::Utils::fftwComplexScale(
							im_Sf[idItm + idPadded],
							1.0 / float(m_iPaddedHeight*m_iPaddedWidth));
					}
				}

				fftwf_execute(m_planInverseSf[tid]);

				for (int i = 0; i < m_iHeight; i++)
				{
					for (int j = 0; j < m_iWidth; j++)
					{
						
						int idfiltered = i*m_iWidth + j;
						int idSf = (i + m_iSy) * m_iPaddedWidth + (j + m_iSx);

						im_filtered[idItmResult + idfiltered][0] += im_Sf[idItm + idSf][0];
						im_filtered[idItmResult + idfiltered][1] += im_Sf[idItm + idSf][1];
					}
				}
			}
		}		
	}

	// Compute z.filtered=z.filtered+filteredt(1+sx:m+sx,1+sy:n+sy); 
	for (int k = 0; k < m_iNumberThreads; k++)
	{
		int idItmResult = k * m_iHeight*m_iWidth;

		for (int i = 0; i < m_iHeight; i++)
		{
			for (int j = 0; j < m_iWidth; j++)
			{
				int idfiltered = i*m_iWidth + j;

				z.m_filtered[idfiltered][0] += im_filtered[idItmResult + idfiltered][0];
				z.m_filtered[idfiltered][1] += im_filtered[idItmResult + idfiltered][1];
			}
		}
	}
	double end = omp_get_wtime();
	time = end - start;

	// Compute z.filtered=z.filtered/4/pi/pi*wxi*wyi; 
	float oneOverPi2 = 1.0 / (M_PI*M_PI);

	#pragma omp parallel for
	for (int i = 0; i < m_iHeight; i++)
	{
		for (int j = 0; j < m_iWidth; j++)
		{
			int idfiltered = i*m_iWidth + j;

			z.m_filtered[idfiltered][0] *= 0.25*oneOverPi2*m_rWxi*m_rWyi;
			z.m_filtered[idfiltered][1] *= 0.25*oneOverPi2*m_rWxi*m_rWyi;
		}
	}
}
void WFT2_cpuF::WFR2(fftwf_complex *f, WFT2_HostResultsF &z, double &time)
{
	omp_set_num_threads(m_iNumberThreads);

	/* Pad the f to be the preferred size of the FFT */
	WFT2_feed_fPadded(f);

	/* Pre-compute the FFT of m_fPadded */
	fftwf_execute(m_planForwardf);

	/* Clear the results if they already contain last results */
	#pragma omp parallel for
	for (int i = 0; i < m_iNumberThreads; i++)
	{
		for (int j = 0; j < m_iWidth*m_iHeight; j++)
		{
			int id = i*m_iWidth*m_iHeight + j;

			if (0 == i)
			{
				z.m_wx[j] = 0;
				z.m_wy[j] = 0;
				z.m_phase[j] = 0;
				z.m_phase_comp[j] = 0;
				z.m_b[j] = 0;
				z.m_r[j] = 0;
				z.m_cxx[j] = 0;
				z.m_cyy[j] = 0;
			}
			im_r[id] = 0; im_p[id] = 0; im_wx[id] = 0; im_wy[id] = 0;
		}
	}

	/* map the wl: wi : wh interval to integers from  0 to 
	   size = (wyh - wyl)/wyi + 1 in order to divide the 
	   copmutations across threads, since threads indices are 
	   more conviniently controlled by integers				  */
	int iwx = int((m_rWxh - m_rWxl)*(1/m_rWxi)) + 1;
	int iwy = int((m_rWyh - m_rWyl)*(1/m_rWyi)) + 1;

	/* The core WFR algorithm */
	double start = omp_get_wtime();
	#pragma omp parallel num_threads(m_iNumberThreads)
	{
		// Get thread-spec parameters
		int tid = omp_get_thread_num();
		int nthread = omp_get_num_threads();

		// Thread-based global indices
		int idItm = tid * m_iPaddedWidth * m_iPaddedHeight;	// Intermidiate id
		int idItmResult = tid * m_iHeight * m_iWidth;		// Intermidiate id of filtered

		for (int y = tid; y < iwy; y += nthread)
		{
			for (int x = 0; x < iwx; x++)
			{
				// Recover the wyt and wxt from integer indices for computation
				float wyt = m_rWyl + float(y) * m_rWyi;
				float wxt = m_rWxl + float(x) * m_rWxi;

				// Construct the gwave: gwave=g.*exp(j*wxt*x+j*wyt*y);
				// Euler's Eq: expj(ux+vy) = cos(ux+vy)+jsin(ux+vy)
				fftwf_complex temp;
				for (int i = 0; i < m_iWinHeight; i++)
				{
					for (int j = 0; j < m_iWinWidth; j++)
					{
						int idPadded = i * m_iPaddedWidth + j;

						float yy = i - (m_iWinHeight - 1) / 2;
						float xx = j - (m_iWinWidth - 1) / 2;

						// exp(jj*(wxt*xx + wyt*yy))
						temp[0] = cos(wxt*xx + wyt*yy);	// real
						temp[1] = sin(wxt*xx + wyt*yy);	// imag

						WFT_FPA::Utils::fftwComplexMul(
							im_gwave[idItm + idPadded],
							m_gPadded[idPadded],
							temp);
					}
				}
				// compute Fg = fft2(gwave);
				fftwf_execute(m_planForwardgwave[tid]);

				// compute sf=ifft2(Ff.*Fg)
				for (int i = 0; i < m_iPaddedHeight; i++)
				{
					for (int j = 0; j < m_iPaddedWidth; j++)
					{
						int idPadded = i * m_iPaddedWidth + j;
						WFT_FPA::Utils::fftwComplexMul(
							im_Sf[idItm + idPadded],
							m_FfPadded[idPadded],
							im_Fgwave[idItm + idPadded]);
						WFT_FPA::Utils::fftwComplexScale(
							im_Sf[idItm + idPadded],
							1.0 / float(m_iPaddedHeight*m_iPaddedWidth));
					}
				}
				fftwf_execute(m_planInverseSf[tid]);

				/*	sf=sf(1+sx:m+sx,1+sy:n+sy);
				 *	%indicate where to update
				 *	t=(abs(sf)>z.r); 
				 *	%update r
				 *	z.r=z.r.*(1-t)+abs(sf).*t; 
				 *	%update wx
				 *	z.wx=z.wx.*(1-t)+wxt*t; 
				 *	%update wy
				 *	z.wy=z.wy.*(1-t)+wyt*t; 
				 *	%update phase
				 *	z.p=z.p.*(1-t)+angle(sf).*t; 		*/
				for (int i = 0; i < m_iHeight; i++)
				{
					for (int j = 0; j < m_iWidth; j++)
					{
						int iTransId = (i + m_iSy)*m_iPaddedWidth + j + m_iSx;

						int idImage = i * m_iWidth + j;
						float abs = WFT_FPA::Utils::fftwComplexAbs(im_Sf[idItm + iTransId]);

						if (abs >= im_r[idItmResult + idImage])
						{
							im_r[idItmResult + idImage] = abs;
							im_wx[idItmResult + idImage] = wxt;
							im_wy[idItmResult + idImage] = wyt;
							im_p[idItmResult + idImage] = WFT_FPA::Utils::fftwComplexAngle(im_Sf[idItm + iTransId]);
						}

					}
				}
			}
		}
	}

	// Merge the partial results to get the final results
	for (int k = 0; k < m_iNumberThreads; k++)
	{
		int idItmResult = k * m_iHeight*m_iWidth;

		for (int i = 0; i < m_iHeight; i++)
		{
			for (int j = 0; j < m_iWidth; j++)
			{
				int idImage = i * m_iWidth + j;

				if(im_r[idItmResult + idImage] > z.m_r[idImage])
				{
					z.m_r[idImage] = im_r[idItmResult + idImage];
					z.m_wx[idImage] = im_wx[idItmResult + idImage];
					z.m_wy[idImage] = im_wy[idItmResult + idImage];
					z.m_phase[idImage] = im_p[idItmResult + idImage];
				}
			}
		}
	}
	double end = omp_get_wtime();
	time = end - start;

	/* Do the Least squre fitting to get cx and cy with data padding replicating the 
	   border pixel																	*/
	#pragma omp parallel num_threads(m_iNumberThreads)
	{
		auto tid = omp_get_thread_num();
		auto nthread = omp_get_num_threads();

		for (auto i = tid; i < m_iPaddedHeightCurvature; i += nthread)
		{
			for (auto j = 0; j < m_iPaddedWidthCurvature; j++)
			{
				auto idPC = i*m_iPaddedWidthCurvature + j;

				// All imag are 0's
				im_cxxPadded[idPC][1] = 0;
				im_cyyPadded[idPC][1] = 0;

				/* Padding based on different regions */
				if (i < m_iHeight + 2 * m_iSy && j < m_iWidth + 2 * m_iSx)
				{
					int idImg = -1;

					// The Image itself
					if (i >= m_iSy && j >= m_iSx && i < m_iSy + m_iHeight && j < m_iSx + m_iWidth)
					{
						idImg = (i - m_iSy) * m_iWidth + j - m_iSx;
					}
					// Top-left region
					else if (i < m_iSy && j < m_iSx)
					{
						idImg = 0;
					}
					// Top region
					else if (i < m_iSy && j >= m_iSx && j < m_iSx + m_iWidth)
					{
						idImg = j - m_iSx;
					}
					// Top-right region
					else if (i < m_iSy && j >= m_iSx + m_iWidth)
					{
						idImg = m_iWidth - 1;
					}
					// Right region
					else if (i >= m_iSy && i < m_iSy + m_iHeight && j >= m_iSx + m_iWidth)
					{
						idImg = (i - m_iSy) * m_iWidth + m_iWidth - 1;
					}
					// Bottom-right region
					else if (i >= m_iSy + m_iHeight && j >= m_iSx + m_iWidth)
					{
						idImg = (m_iHeight - 1)*m_iWidth + m_iWidth - 1;

					}
					// Bottom region
					else if (i >= m_iSy + m_iHeight&& j >= m_iSx && j < m_iSx + m_iWidth)
					{
						idImg = (m_iHeight - 1) * m_iWidth + j - m_iSx;
					}
					// Bottom-left region
					else if (i >= m_iSy + m_iHeight && j < m_iSx)
					{
						idImg = (m_iHeight - 1)*m_iWidth;
					}
					// Left region
					else
					{
						idImg = (i - m_iSy)*m_iWidth;
					}

					im_cxxPadded[idPC][0] = z.m_wx[idImg];
					im_cyyPadded[idPC][0] = z.m_wy[idImg];
				}
				else
				{
					im_cxxPadded[idPC][0] = 0;
					im_cyyPadded[idPC][0] = 0;
				}
			}
		}
	}

	// compute least squares fitting to get cxx and cyy
	fftwf_execute(m_planForwardcxx);
	fftwf_execute(m_planForwardcyy);
	// Do the convolution, i.e. pointwise multiplication
	#pragma omp parallel num_threads(m_iNumberThreads)
	{
		// TODO: 		
		int tid = omp_get_thread_num();
		int nthread = omp_get_num_threads();

		for (int i = tid; i < m_iPaddedHeightCurvature; i += nthread)
		{
			for (int j = 0; j < m_iPaddedWidthCurvature; j++)
			{
				int idP = i*m_iPaddedWidthCurvature + j;
				WFT_FPA::Utils::fftwComplexMul(
					im_cxxPadded[idP], 
					im_cxxPadded[idP],
					im_xgPadded[idP]);
				WFT_FPA::Utils::fftwComplexScale(
					im_cxxPadded[idP],
					1.0 / float(m_iPaddedHeightCurvature*m_iPaddedWidthCurvature));
				WFT_FPA::Utils::fftwComplexMul(
					im_cyyPadded[idP],
					im_cyyPadded[idP],
					im_ygPadded[idP]);
				WFT_FPA::Utils::fftwComplexScale(
					im_cyyPadded[idP],
					1.0 / float(m_iPaddedHeightCurvature*m_iPaddedWidthCurvature));
			}
		}
	}
	fftwf_execute(m_planInversecxx);
	fftwf_execute(m_planInversecyy);

	// Get the final results
	#pragma omp parallel num_threads(m_iNumberThreads)
	{
		int tid = omp_get_thread_num();
		int nthread = omp_get_num_threads();

		for (int i = tid; i < m_iHeight; i += nthread)
		{
			for (int j = 0; j < m_iWidth; j++)
			{
				int idImg = i*m_iWidth + j;
				int idc = (i + 2 * m_iSy)*m_iPaddedWidthCurvature + j + 2 * m_iSx;

				z.m_cxx[idImg] = -im_cxxPadded[idc][0] * m_rSumxxg;
				z.m_cyy[idImg] = -im_cyyPadded[idc][0] * m_rSumyyg;

				z.m_phase_comp[idImg] = z.m_phase[idImg]
					- 0.5*atan(m_rSigmaX*m_rSigmaX*z.m_cxx[idImg])
					- 0.5*atan(m_rSigmaY*m_rSigmaY*z.m_cyy[idImg]);
				z.m_phase_comp[idImg] = atan2(sin(z.m_phase_comp[idImg]), cos(z.m_phase_comp[idImg]));

				z.m_b[idImg] = z.m_r[idImg] 
					* pow((1 + m_rSigmaX*m_rSigmaX*m_rSigmaX*m_rSigmaX*z.m_cxx[idImg] * z.m_cxx[idImg])*0.25 / M_PI / (m_rSigmaX*m_rSigmaX), 0.25)
					* pow((1 + m_rSigmaY*m_rSigmaY*m_rSigmaY*m_rSigmaY*z.m_cyy[idImg] * z.m_cyy[idImg])*0.25 / M_PI / (m_rSigmaY*m_rSigmaY), 0.25);
			}
		}
	}
}


int WFT2_cpuF::WFT2_Initialize(WFT2_HostResultsF &z)
{
	/*----------------------------------General parameters intitialization----------------------------------*/
	// Half the Gaussian window size
	m_iSx = int(round(3 * m_rSigmaX));
	m_iSy = int(round(3 * m_rSigmaY));

	// Gaussian Window size
	m_iWinWidth = 2 * m_iSx +1;
	m_iWinHeight = 2 * m_iSy + 1;

	// Calculate the initial padding in order to perform the cyclic convolution using FFT
	// The padding size is size(A) + size(B) - 1;
	m_iPaddedHeight = m_iHeight + m_iWinHeight - 1;
	m_iPaddedWidth = m_iWidth + m_iWinWidth - 1;
	
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
		m_iPaddedWidth = OPT_FFT_SIZE[iW];

		int iPaddedSize = m_iPaddedHeight * m_iPaddedWidth;

		/* Memory Allocation (Already padded) based on m_iNumberThreads*/
		// Allocate memory for input padded f which is pre-copmuted and remain unchanged among threads
		m_fPadded = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex)*iPaddedSize);
		m_FfPadded = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex)*iPaddedSize);
		m_gPadded = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex)*iPaddedSize);

		// Allocate thread-private intermediate memory
		im_gwave = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex)*m_iNumberThreads*iPaddedSize);
		im_Fgwave = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex)*m_iNumberThreads*iPaddedSize);
		im_Sf = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex)*m_iNumberThreads*iPaddedSize);

		// Allocate FFTW plans for each thread
		m_planForwardgwave = new fftwf_plan[m_iNumberThreads];
		m_planForwardSf = new fftwf_plan[m_iNumberThreads];
		m_planInverseSf = new fftwf_plan[m_iNumberThreads];

		/* Make the FFTW plan for the precomputation of Ff = fft2(f) */
		m_planForwardf = fftwf_plan_dft_2d(m_iPaddedWidth, m_iPaddedHeight, m_fPadded, m_FfPadded, FFTW_FORWARD, FFTW_ESTIMATE);

		/* Generate the windows g (g is the same across the calculation) *
		 * g = exp(-x.*x /2/sigmax/sigmax - y.*y /2/sigmay/sigmay)	     *
		 * And set padded region of both m_fPadded and m_gwavePadded to  *
		 * zeros.						                                 *
		 * Compute onece for thread 0 then do the memcpy				 */
		float rNorm2Factor = 0;			// Factor used for normalization
		#pragma omp parallel num_threads(m_iNumberThreads)
		{
			float rNorm2FactorLocal = 0;

			/* Get thread params */
			int tid = omp_get_thread_num();
			int nthread = omp_get_num_threads();

			for (int i = tid; i < m_iPaddedHeight; i += nthread)
			{
				for (int j = 0; j < m_iPaddedWidth; j++)
				{
					int id = i * m_iPaddedWidth + j;	// 1D index of 2D array elems		

					// Set m_gwavePadded to 0's
					m_gPadded[id][0] = 0;
					m_gPadded[id][1] = 0;

					// Set m_fPadded to zeros, because it will be filled later when execute
					// the functor. Both real & imag are set to 0
					m_fPadded[id][0] = 0;
					m_fPadded[id][1] = 0;

					// Construct m_gwavePadded matrix
					// Except the first 2*sx+1 by 2*sy+1 elements, all are 0's. Also, all imags are 0's
					if (i < m_iWinHeight && j < m_iWinWidth)
					{
						float y = float(i - (m_iWinHeight - 1) / 2);
						float x = float(j - (m_iWinWidth - 1) / 2);

						m_gPadded[id][0] = exp(-float(x*x) / 2 / m_rSigmaX / m_rSigmaX
							- float(y*y) / 2 / m_rSigmaY / m_rSigmaY);

						rNorm2FactorLocal += m_gPadded[id][0] * m_gPadded[id][0];
					}
				}
			}
			// Accumulate the per-thread results one-by-one
			#pragma omp atomic
			rNorm2Factor += rNorm2FactorLocal;
		}

		rNorm2Factor = sqrt(rNorm2Factor);

		// Do the normalization for gwave	
		for (int i = 0; i < 2 * m_iSy + 1; i++)
		{
			for (int j = 0; j < 2 * m_iSx + 1; j++)
			{
				int id = i*m_iPaddedWidth + j;
				m_gPadded[id][0] *= 1 / rNorm2Factor;
			}
		}

		// Do the memcpy to initialize the im_gwave
		for (int i = 0; i < m_iNumberThreads; i++)
			std::memcpy(&im_gwave[i*iPaddedSize],
			&m_gPadded[0],
			sizeof(fftwf_complex)*iPaddedSize);


		/*----------------------------------Specific Inititialization for WFF2&WFR2--------------------------------*/
		if (WFT_TYPE::WFF == m_type)
		{
			WFF2_Init(z);
		}
		else if (WFT_TYPE::WFR == m_type)
		{
			// If the padding inside WFR2_Init fail
			if(-1 == WFR2_Init(z))
				return -1;
		}
	}

	return 0;
}
void WFT2_cpuF::WFF2_Init(WFT2_HostResultsF &z)
{
	int iImageSize = m_iWidth * m_iHeight;
	int iPaddedSize = m_iPaddedHeight * m_iPaddedWidth;

	// Only z.filtered is used in WFF2 algorithm
	z.m_filtered = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex)*iImageSize);
	im_filtered = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex)*m_iNumberThreads*iImageSize);

	// make FFTW plans for each thread
	for (int i = 0; i < m_iNumberThreads; i++)
	{
		m_planForwardgwave[i] = fftwf_plan_dft_2d(m_iPaddedWidth, m_iPaddedHeight, &im_gwave[i*iPaddedSize], &im_Fgwave[i*iPaddedSize], FFTW_FORWARD, FFTW_ESTIMATE);
		m_planInverseSf[i] = fftwf_plan_dft_2d(m_iPaddedWidth, m_iPaddedHeight, &im_Sf[i*iPaddedSize], &im_Sf[i*iPaddedSize], FFTW_BACKWARD, FFTW_ESTIMATE);
		m_planForwardSf[i] = fftwf_plan_dft_2d(m_iPaddedWidth, m_iPaddedHeight, &im_Sf[i*iPaddedSize], &im_Sf[i*iPaddedSize], FFTW_FORWARD, FFTW_ESTIMATE);
	}
}
int WFT2_cpuF::WFR2_Init(WFT2_HostResultsF &z)
{
	// Get the preferred size of the padded curvature cxx and cyy
	m_iPaddedHeightCurvature = m_iHeight + m_iSy + m_iWinHeight - 1;
	m_iPaddedWidthCurvature = m_iWidth + m_iSx + m_iWinWidth - 1;

	std::cout << m_iPaddedHeightCurvature << ", " << m_iPaddedWidthCurvature << std::endl;

	int iH = getFirstGreater(m_iPaddedHeightCurvature);
	int iW = getFirstGreater(m_iPaddedWidthCurvature);
	if(-1 == iH || -1 == iW)
	{
		// Out of range
		return -1;
	}
	else
	{
		m_iPaddedHeightCurvature = OPT_FFT_SIZE[iH];
		m_iPaddedWidthCurvature = OPT_FFT_SIZE[iW];

		std::cout << m_iPaddedHeightCurvature << ", " << m_iPaddedWidthCurvature << std::endl;

		int curvatureSize = m_iPaddedHeightCurvature * m_iPaddedWidthCurvature;

		// If the size is found, allocate the memory for corresponding arrays
		im_cyyPadded = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex)*curvatureSize);
		im_cxxPadded = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex)*curvatureSize);
		im_xgPadded = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex)*curvatureSize);
		im_ygPadded = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex)*curvatureSize);

		// Calculate x.*g and y.*g
		#pragma omp parallel num_threads(m_iNumberThreads)
		{
			float rPartialxxg = 0;
			float rPartialyyg = 0;

			int tid = omp_get_thread_num();
			int nthread = omp_get_num_threads();

			for (int i = tid; i < m_iPaddedHeightCurvature; i += nthread)
			{
				for (int j = 0; j < m_iPaddedWidthCurvature; j++)
				{
					int idPadded = i * m_iPaddedWidthCurvature + j;

					im_xgPadded[idPadded][0] = 0;	im_xgPadded[idPadded][1] = 0;
					im_ygPadded[idPadded][0] = 0;	im_ygPadded[idPadded][1] = 0;

					if (i < m_iWinHeight && j < m_iWinWidth)
					{
						int idgPadded = i * m_iPaddedWidth + j;

						float y = float(i - (m_iWinHeight - 1) / 2);
						float x = float(j - (m_iWinWidth - 1) / 2);

						im_xgPadded[idPadded][0] = m_gPadded[idgPadded][0] * x;
						im_ygPadded[idPadded][0] = m_gPadded[idgPadded][0] * y;

						rPartialxxg += x * x * m_gPadded[idgPadded][0];
						rPartialyyg += y * y * m_gPadded[idgPadded][0];
					}
				}
			}
			#pragma omp atomic
				m_rSumxxg += rPartialxxg;
				m_rSumyyg += rPartialyyg;
		}
		
		std::cout << m_rSumxxg << ", " << m_rSumyyg << std::endl;

		m_rSumxxg = 1.0 / m_rSumxxg;
		m_rSumyyg = 1.0 / m_rSumyyg;

		int iImageSize = m_iWidth * m_iHeight;

		// Allocate memory for final & intermediate results
		z.m_wx = (float*)malloc(sizeof(float)*iImageSize);
		z.m_wy = (float*)malloc(sizeof(float)*iImageSize);
		z.m_phase = (float*)malloc(sizeof(float)*iImageSize);
		z.m_r = (float*)malloc(sizeof(float)*iImageSize);
		z.m_b = (float*)malloc(sizeof(float)*iImageSize);
		z.m_cxx = (float*)malloc(sizeof(float)*iImageSize);
		z.m_cyy = (float*)malloc(sizeof(float)*iImageSize);
		z.m_phase_comp = (float*)malloc(sizeof(float)*iImageSize);

		im_r = (float*)malloc(sizeof(float)*m_iNumberThreads*iImageSize);
		im_p = (float*)malloc(sizeof(float)*m_iNumberThreads*iImageSize);
		im_wx = (float*)malloc(sizeof(float)*m_iNumberThreads*iImageSize);
		im_wy = (float*)malloc(sizeof(float)*m_iNumberThreads*iImageSize);

		// make FFTW plans for each thread
		int iPaddedSize = m_iPaddedHeight * m_iPaddedWidth;		
		for (int i = 0; i < m_iNumberThreads; i++)
		{
			m_planForwardgwave[i] = fftwf_plan_dft_2d(m_iPaddedWidth, m_iPaddedHeight, &im_gwave[i*iPaddedSize], &im_Fgwave[i*iPaddedSize], FFTW_FORWARD, FFTW_ESTIMATE);
			m_planInverseSf[i] = fftwf_plan_dft_2d(m_iPaddedWidth, m_iPaddedWidth, &im_Sf[i*iPaddedSize], &im_Sf[i*iPaddedSize], FFTW_BACKWARD, FFTW_ESTIMATE);
		}
		
		// make the plans for cxx and cyy computation
		m_planForwardxg = fftwf_plan_dft_2d(m_iPaddedWidthCurvature, m_iPaddedHeightCurvature, im_xgPadded, im_xgPadded, FFTW_FORWARD, FFTW_ESTIMATE);
		m_planForwardyg = fftwf_plan_dft_2d(m_iPaddedWidthCurvature, m_iPaddedHeightCurvature, im_ygPadded, im_ygPadded, FFTW_FORWARD, FFTW_ESTIMATE);
		m_planForwardcxx = fftwf_plan_dft_2d(m_iPaddedWidthCurvature, m_iPaddedHeightCurvature, im_cxxPadded, im_cxxPadded, FFTW_FORWARD, FFTW_ESTIMATE);
		m_planInversecxx = fftwf_plan_dft_2d(m_iPaddedWidthCurvature, m_iPaddedHeightCurvature, im_cxxPadded, im_cxxPadded, FFTW_BACKWARD, FFTW_ESTIMATE);
		m_planForwardcyy = fftwf_plan_dft_2d(m_iPaddedWidthCurvature, m_iPaddedHeightCurvature, im_cyyPadded, im_cyyPadded, FFTW_FORWARD, FFTW_ESTIMATE);
		m_planInversecyy = fftwf_plan_dft_2d(m_iPaddedWidthCurvature, m_iPaddedHeightCurvature, im_cyyPadded, im_cyyPadded, FFTW_BACKWARD, FFTW_ESTIMATE);

		// Do the FFT for xg and yg once and use them 
		fftwf_execute(m_planForwardxg);
		fftwf_execute(m_planForwardyg);
	}

	return 0;
}
void WFT2_cpuF::WFT2_feed_fPadded(fftwf_complex *f)
{
	#pragma omp parallel num_threads(m_iNumberThreads)
	{
		#pragma omp for
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
}
void WFT2_cpuF::WFF2_SetThreashold(fftwf_complex *f)
{
	/* If m_rThr < 0, use the default values for the Threashold in WFF2 algorithm 
		m_rThr = 6 * sqrt( mean2( abs(f).^2 ) / 3)				
	   As well as feed the f into its padded m_fPadded							  */
	if(m_rThr < 0)
	{
		float rTemp = 0;

		#pragma omp parallel num_threads(m_iNumberThreads)
		{
			float rTempLocal = 0;

			/* Get thread params */
			int tid = omp_get_thread_num();
			int inthread = omp_get_num_threads();
			int idf, idfPadded;
			float abs;

			for (int i = tid; i < m_iHeight; i += inthread)
			{
				for (int j = 0; j < m_iWidth; j++)
				{
					idf = i*m_iWidth + j;
					idfPadded = i*m_iPaddedWidth + j;

					abs = WFT_FPA::Utils::fftwComplexAbs(f[idf]);
					rTempLocal += abs*abs;

					// Feed the array
					m_fPadded[idfPadded][0] = f[idf][0];
					m_fPadded[idfPadded][1] = f[idf][1];
				}
			}
			#pragma omp atomic
				rTemp += rTempLocal;
		}		
		rTemp = rTemp / (m_iWidth * m_iHeight) / 3;
		m_rThr = 6 * sqrt(rTemp);
	}
	else
	{
		/* Feed the input f into its padded m_fPadded */
		WFT2_feed_fPadded(f);
	}
}

}	// namespace WFT_FPA
}	// namespace WFT

