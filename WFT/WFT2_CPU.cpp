#include "WFT2_CPU.h"
#include <iostream>
#include <omp.h>
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
	, m_iNumberThreads(iNumberThreads)
	, m_fPadded(nullptr)
	, m_FfPadded(nullptr)
	, im_Fgwave(nullptr)
	, im_Sf(nullptr)
	, m_planForwardgwave(nullptr)
	, m_planInverseSf(nullptr)
	, im_p(nullptr)
	, im_r(nullptr)
	, im_wx(nullptr)
	, im_wy(nullptr)
{
	/* Type specific parameter initializations*/
	if (WFT_TYPE::WFF == m_type)
	{
		m_rSigmaX = 10.0;
		m_rWxl = -2 - 3.0 / m_rSigmaX;
		m_rWxi = 1.0 / m_rSigmaX;
		m_rWxh = 2 + 3.0 / m_rSigmaX;

		m_rSigmaY = 10.0;
		m_rWyl = -2 - 3.0 / m_rSigmaY;
		m_rWyi = 1.0 / m_rSigmaY;
		m_rWyh = 2 + 3.0 / m_rSigmaY;
	}
	else if (WFT_TYPE::WFR == m_type)
	{
		m_rSigmaX = 10.0;
		m_rWxl = -2;
		m_rWxi = 0.025;
		m_rWxh = 2;

		m_rSigmaY = 10.0;
		m_rWyl = -2.0;
		m_rWyi = 0.025;
		m_rWyh = 2.0;
	}

	//std::cout<<m_rWxl<<", "<<m_rWxi<<", "<<m_rWxh<<std::endl;
	//std::cout<<m_rWyl<<", "<<m_rWyi<<", "<<m_rWyh<<std::endl;

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
	double rSigmaX,	double rWxl, double rWxh, double rWxi,
	double rSigmaY, double rWyl, double rWyh, double rWyi,
	double rThr,
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
	, m_iNumberThreads(iNumberThreads)
	, m_fPadded(nullptr)
	, m_FfPadded(nullptr)
	, im_Fgwave(nullptr)
	, im_Sf(nullptr)
	, m_planForwardgwave(nullptr)
	, m_planInverseSf(nullptr)
	, im_p(nullptr)
	, im_r(nullptr)
	, im_wx(nullptr)
	, im_wy(nullptr)
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
	for(int i=0; i<m_iNumberThreads; i++)
	{
		fftw_destroy_plan(m_planForwardgwave[i]);
		fftw_destroy_plan(m_planInverseSf[i]);
	}

	delete [] m_planForwardgwave;	m_planForwardgwave = nullptr;
	delete [] m_planInverseSf;		m_planInverseSf = nullptr;

	fftw_destroy_plan(m_planForwardf);

	fftw_free(m_fPadded);		m_fPadded = nullptr;
	fftw_free(im_Fgwave);		im_Fgwave = nullptr;
	fftw_free(m_FfPadded);		m_FfPadded = nullptr;
	fftw_free(im_Fgwave);		im_Fgwave = nullptr;
	fftw_free(im_Sf);			im_Sf = nullptr;
	
	if(WFT::WFT_TYPE::WFF == m_type)
	{
		fftw_free(im_filtered);		im_filtered = nullptr;
	}
	if(WFT::WFT_TYPE::WFR == m_type)
	{
		free(im_r);		im_r = nullptr;
		free(im_wx);	im_wx = nullptr;
		free(im_wy);	im_wy = nullptr;
		free(im_p);		im_p = nullptr;
	}
}

void WFT2_cpu::operator() (
	fftw_complex *f, 
	WFT2_HostResults &z,
	double &time)
{
	if(WFT_FPA::WFT::WFT_TYPE::WFF == m_type)
		WFF2(f, z, time);
	else if(WFT_FPA::WFT::WFT_TYPE::WFR == m_type)
		WFR2(f, z, time);
	/*std::cout << m_iPaddedHeight << ", " << m_iPaddedWidth << std::endl;
	std::cout << "The threashold is = " << m_rThr << std::endl;*/
}


/* Private functions */
int WFT2_cpu::WFT2_Initialize(WFT2_HostResults &z)
{
	/* General parameters intitialization */
	// Half the Gaussian window size
	m_iSx = int(round(3 * m_rSigmaX));
	m_iSy = int(round(3 * m_rSigmaY));

	// Gaussian Window size
	m_iWinWidth = 2 * m_iSx +1;
	m_iWinHeight = 2 * m_iSy + 1;

	// Calculate the initial padding in order to perform the cyclic convolution using FFT
	// The padding size is size(A) + size(B) - 1;
	m_iPaddedHeight = m_iHeight + m_iWinHeight - 1;
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

	/* Memory Allocation (Already padded) based on m_iNumberThreads*/
	// Allocate memory for input padded f which is pre-copmuted and remain unchanged among threads
	m_fPadded = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*m_iPaddedHeight*m_iPaddedWidth);
	m_FfPadded = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*m_iPaddedHeight*m_iPaddedWidth);
	
	// Allocate thread-private intermediate memory
	im_Fgwave = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*m_iNumberThreads*m_iPaddedHeight*m_iPaddedWidth);
	im_Sf = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*m_iNumberThreads*m_iPaddedHeight*m_iPaddedWidth);
	
	// Allocate FFTW plans for each thread
	m_planForwardgwave = new fftw_plan[m_iNumberThreads];
	m_planInverseSf = new fftw_plan[m_iNumberThreads];

	/* Make the FFTW plan for the precomputation of Ff = fft2(f) */
	m_planForwardf = fftw_plan_dft_2d(m_iPaddedWidth, m_iPaddedHeight, m_fPadded, m_FfPadded, FFTW_FORWARD, FFTW_ESTIMATE);

	// Allocate memory for the output z and WFF&WFR specific intermediate arrays
	if(WFT_TYPE::WFF == m_type)
	{	
		z.m_filtered = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*m_iWidth*m_iHeight);
		im_filtered = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*m_iNumberThreads*m_iWidth*m_iHeight);

		for (int i = 0; i < m_iNumberThreads; i++)
		{
			// make FFTW plans for each thread
			m_planForwardgwave[i] = fftw_plan_dft_2d(m_iPaddedWidth, m_iPaddedHeight, &im_Fgwave[i*m_iPaddedHeight*m_iPaddedWidth], &im_Fgwave[i*m_iPaddedHeight*m_iPaddedWidth], FFTW_FORWARD, FFTW_ESTIMATE);
			m_planInverseSf[i] = fftw_plan_dft_2d(m_iPaddedWidth, m_iPaddedWidth, &im_Sf[i*m_iPaddedHeight*m_iPaddedWidth], &im_Sf[i*m_iPaddedHeight*m_iPaddedWidth], FFTW_BACKWARD, FFTW_ESTIMATE);
		
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
	}
	else if (WFT_TYPE::WFR == m_type)
	{
		z.m_wx = (double*)malloc(sizeof(double)*m_iWidth*m_iHeight);
		z.m_wy = (double*)malloc(sizeof(double)*m_iWidth*m_iHeight);
		z.m_phase = (double*)malloc(sizeof(double)*m_iWidth*m_iHeight);
		z.m_phase_comp = (double*)malloc(sizeof(double)*m_iWidth*m_iHeight);
		z.m_b = (double*)malloc(sizeof(double)*m_iWidth*m_iHeight);
		z.m_r = (double*)malloc(sizeof(double)*m_iWidth*m_iHeight);
		z.m_cx = (double*)malloc(sizeof(double)*m_iWidth*m_iHeight);
		z.m_cy = (double*)malloc(sizeof(double)*m_iWidth*m_iHeight);
		im_r = (double*)malloc(sizeof(double)*m_iNumberThreads*m_iWidth*m_iHeight);
		im_p = (double*)malloc(sizeof(double)*m_iNumberThreads*m_iWidth*m_iHeight);
		im_wx = (double*)malloc(sizeof(double)*m_iNumberThreads*m_iWidth*m_iHeight);
		im_wy = (double*)malloc(sizeof(double)*m_iNumberThreads*m_iWidth*m_iHeight);

		for (int i = 0; i < m_iNumberThreads; i++)
		{
			// make FFTW plans for each thread
			m_planForwardgwave[i] = fftw_plan_dft_2d(m_iPaddedWidth, m_iPaddedHeight, &im_Fgwave[i*m_iPaddedHeight*m_iPaddedWidth], &im_Fgwave[i*m_iPaddedHeight*m_iPaddedWidth], FFTW_FORWARD, FFTW_ESTIMATE);
			m_planInverseSf[i] = fftw_plan_dft_2d(m_iPaddedWidth, m_iPaddedWidth, &im_Sf[i*m_iPaddedHeight*m_iPaddedWidth], &im_Sf[i*m_iPaddedHeight*m_iPaddedWidth], FFTW_BACKWARD, FFTW_ESTIMATE);

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
					z.m_cx[j] = 0;
					z.m_cy[j] = 0;
				}
				im_r[id] = 0; im_p[id] = 0; im_wx[id] = 0; im_wy[id] = 0;
			}
		}
	}

	/* Generate the windows g (g is the same across the calculation) *
	 * g = exp(-x.*x /2/sigmax/sigmax - y.*y /2/sigmay/sigmay)	     * 
	 * And set padded region of both m_fPadded and m_gwavePadded to  *
	 * zeros.						                                 *
	 * Compute onece for thread 0 then do the memcpy				 */
	double rNorm2Factor = 0;			// Factor used for normalization

	#pragma omp parallel num_threads(m_iNumberThreads)
	{
		double rNorm2FactorLocal = 0;

		/* Get thread params */
		int tid = omp_get_thread_num();
		int nthread = omp_get_num_threads();

		for (int i = tid; i < m_iPaddedHeight; i += nthread)
		{
			for (int j = 0; j < m_iPaddedWidth; j++)
			{
				int id = i * m_iPaddedWidth + j;	// 1D index of 2D array elems		

				// Set m_gwavePadded to 0's
				im_Fgwave[id][0] = 0;
				im_Fgwave[id][1] = 0;

				// Set m_fPadded to zeros, because it will be filled later when execute
				// the functor. Both real & imag are set to 0
				m_fPadded[id][0] = 0;
				m_fPadded[id][1] = 0;

				// Construct m_gwavePadded matrix
				// Except the first 2*sx+1 by 2*sy+1 elements, all are 0's. Also, all imags are 0's
				if (i < m_iWinHeight && j < m_iWinWidth)
				{
					double y = i - (m_iWinHeight - 1) / 2;
					double x = j - (m_iWinWidth - 1) / 2;

					im_Fgwave[id][0] = exp(-double(x*x) / 2 / m_rSigmaX / m_rSigmaX
						- double(y*y) / 2 / m_rSigmaY / m_rSigmaY);

					rNorm2FactorLocal += im_Fgwave[id][0] * im_Fgwave[id][0];
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
			im_Fgwave[id][0] /= rNorm2Factor;
		}
	}	
	
	// Do the memcpy
	for (int i = 1; i < m_iNumberThreads; i++)
		std::memcpy(&im_Fgwave[i*m_iPaddedHeight*m_iPaddedWidth], 
					&im_Fgwave[0],
					sizeof(fftw_complex)*m_iPaddedHeight*m_iPaddedWidth);
	

	//std::cout<<m_gwavePadded[9][0]<<", "<< m_gwavePadded[9][1]<<std::endl;
	//std::cout<<m_gwavePadded[60*m_iPaddedWidth+60][0]<<std::endl;
	
	return 0;
}

void WFT2_cpu::WFT2_feed_fPadded(fftw_complex *f)
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

void WFT2_cpu::WFF2_SetThreashold(fftw_complex *f)
{
	/* If m_rThr < 0, use the default values for the Threashold in WFF2 algorithm 
		m_rThr = 6 * sqrt( mean2( abs(f).^2 ) / 3)				
	   As well as feed the f into its padded m_fPadded							  */
	if(m_rThr < 0)
	{
		double rTemp = 0;

		#pragma omp parallel num_threads(m_iNumberThreads)
		{
			double rTempLocal = 0;

			/* Get thread params */
			int tid = omp_get_thread_num();
			int inthread = omp_get_num_threads();
			int idf, idfPadded;
			double abs;

			for (int i = tid; i < m_iHeight; i += inthread)
			{
				for (int j = 0; j < m_iWidth; j++)
				{
					idf = i*m_iWidth + j;
					idfPadded = i*m_iPaddedWidth + j;

					abs = fftwComplexAbs(f[idf]);
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

void WFT2_cpu::WFF2(fftw_complex *f, WFT2_HostResults &z, double &time)
{
	/* Set the threshold m_rThr if it's not specified by the client */
	WFF2_SetThreashold(f);
	
	/* Pre-compute the FFT of m_fPadded */
	fftw_execute(m_planForwardf);


	///* map the wl: wi : wh interval to integers from  0 to 
	//   size = (wyh - wyl)/wyi + 1 in order to divide the 
	//   copmutations across threads, since threads indices are 
	//   more conviniently controlled by integers				  */
	//int iwx = int(floor((m_rWxh - m_rWxl)/m_rWxi)) + 1;
	//int iwy = int(floor((m_rWyh - m_rWyl)/m_rWyi)) + 1;

	//std::cout<<m_iPaddedHeight<<", "<<m_iPaddedWidth<<std::endl;

	///* The core WFF algorithm */
	//static fftwf_complex *filteredLocal = nullptr;	// shared across threads
	//#pragma omp threadprivate(filteredLocal)

	///* Allocate per-thread local memory holding the partial results of the 
	//	   final z.m_filtered */
	//#pragma omp parallel num_threads(m_iNumberThreads)
	//{		
	//	#ifdef WFT_FPA_DOUBLE
	//		fftwf_complex *filteredLocal = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex)*m_iHeight*m_iWidth);
	//	#else
	//		fftwf_complex *filteredLocal = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)*m_iHeight*m_iWidth);
	//	#endif // WFT_FPA_DOUBLE
	//}


	//double start = omp_get_wtime();
	///* Begin the computation */
	//#pragma omp parallel num_threads(m_iNumberThreads)
	//{
	//	/* Get thread-spec parameters */
	//	int tid = omp_get_thread_num();
	//	int nthread = omp_get_num_threads();

	//	/* strided for-loop: evenly distribute tasks to m_iNumberThreads */
	//	for (int y = tid; y < 1; y += nthread)
	//	{
	//		for (int x = 0; x < 1; x++)
	//		{
	//			// Recover the wyt and wxt from integer indices for computation
	//			float wyt = m_rWyl + float(y) * m_rWyi;
	//			float wxt = m_rWxl + float(x) * m_rWxi;
	//			fftwf_complex temp;
	//	
	//			// Construct the gwave: gwave=g.*exp(j*wxt*x+j*wyt*y);
	//			// Euler's Eq: expj(ux+vy) = cos(ux+vy)+jsin(ux+vy)
	//			for (int i = 0; i <m_iWinHeight; i++)
	//			{
	//				for (int j = 0; j <m_iWinWidth; j++)
	//				{
	//					int idPadedgwave = i * m_iPaddedWidth + j;	// index of padded gwave

	//					float yy = i - (m_iWinHeight - 1) / 2;
	//					float xx = j - (m_iWinWidth - 1) / 2;
	//					
	//					temp[0] = cos(wxt*xx + wyt*yy);
	//					temp[1] = sin(wxt*xx + wyt*yy);
	//					
	//					if(i==0&&j==0)
	//					{
	//						std::cout<< m_gwavePadded[idPadedgwave][0] <<std::endl;
	//						std::cout<< m_gwavePadded[idPadedgwave][1] <<std::endl;
	//					}

	//					float real = m_gwavePadded[idPadedgwave][0] * temp[0];
	//					float imag = m_gwavePadded[idPadedgwave][0] * temp[1];

	//					m_gwavePadded[idPadedgwave][0] = real;
	//					m_gwavePadded[idPadedgwave][1] = imag;
	//				}
	//			}
	//			
	//			// Execute the fft plan: m_gwavePaddedFq = FFT2(m_gwavePadded)
	//			#ifdef WFT_FPA_DOUBLE
	//				fftw_execute(m_planForwardgwave);
	//			#else
	//				fftwf_execute(m_planForwardgwave);
	//			#endif // WFT_FPA_DOUBLE

	//			// sf = m_fPaddedFq .* m_gwavePaddedFq;
	//			//for (int i = 0; i < m_iPaddedHeight; i++)
	//			//{
	//			//	for (int j = 0; j < m_iPaddedWidth; j++)
	//			//	{
	//			//		int idSf = i * m_iPaddedWidth + j;
	//			//		fftwComplexMul(m_Sf[idSf], m_fPaddedFq[idSf], m_gwavePaddedFq[idSf]);
	//			//		fftwComplexScale(m_Sf[idSf], 1/float(m_iPaddedHeight*m_iPaddedWidth));
	//			//	}
	//			//}
	//			//// sf = ifft2(sf)
	//			//#ifdef WFT_FPA_DOUBLE
	//			//	fftw_execute(m_planInverseSf);
	//			//#else
	//			//	fftwf_execute(m_planInverseSf);
	//			//#endif // WFT_FPA_DOUBLE


	//		}
	//	}
	//	std::cout<<m_gwavePadded[1][0]<<", "<<m_gwavePadded[1][1]<<std::endl;
	//}
	//double end = omp_get_wtime();
	//time = end - start;

	//#pragma omp parallel num_threads(m_iNumberThreads)
	//{
	//	/* Free per-thread local memory holding the partial results of the 
	//	   final z.m_filtered */
	//	#ifdef WFT_FPA_DOUBLE
	//		fftw_free(filteredLocal);
	//	#else
	//		fftwf_free(filteredLocal);
	//	#endif // WFT_FPA_DOUBLE
	//}
}


void WFT2_cpu::WFR2(fftw_complex *f, WFT2_HostResults &z, double &time)
{
	WFT2_feed_fPadded(f);

	/* Pre-compute the FFT of m_fPadded */
	fftw_execute(m_planForwardf);
}

}	// namespace WFT_FPA
}	// namespace WFT

