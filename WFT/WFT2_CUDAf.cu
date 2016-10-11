#include "WFT2_CUDAf.h"
#include "device_launch_parameters.h"
#include "mem_manager.h"

#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>

namespace WFT_FPA{
namespace WFT{

/*---------------------------------------------CUDA Kernels-------------------------------------------------*/
/* 
 PURPOSE: 
	1. Generate the xf and yf for analytically computation of the Gaussian Window in Fourier Domain 
	[yf xf]=meshgrid(-fix(nn/2):nn-fix(nn/2)-1,-fix(mm/2):mm-fix(mm/2)-1); mm, nn are padded height&width
 NOTE: 
	Currently only even size in each dimension is supported
 INPUTS:
	xf, yf: meshgrid in frequency domain
	width, height: width and height of the xf and yf matrices
 OUTPUTS:
	xf, yf: Generated meshgrid
  */
__global__ void gen_xf_yf_Kernel(cufftReal *xf, cufftReal *yf, int iWidth, int iHeight)
{
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	int j = threadIdx.x + blockIdx.x * blockDim.x;

	int id = i*iWidth + j;

	int iHalfy = iHeight / 2;
	int iHalfx = iWidth / 2;

	if (i < iHeight && j < iWidth)
	{
		xf[id] = j - iHalfx;
		yf[id] = i - iHalfy;
	}
}
/*
 PURPOSE:
	2. Do the fftshift on xf and yf to be coincide with the CUFFT's results
 NOTE:
	Currently only even size in each dimension is supported 
 INPUTS:
	xf, yf: meshgrid in frequency domian
	width, height: width and height of the xf and yf matrices
 OUTPUTS:
	xf, yf: In-place fft-shifted xf, yf

*/
__global__ void fftshift_xf_yf_kernel(cufftReal *xf, cufftReal *yf, int iWidth, int iHeight)
{
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	int j = threadIdx.x + blockIdx.x * blockDim.x;

	int id = i*iWidth + j;

	int iHalfx = iWidth / 2;
	int iHalfy = iHeight / 2;
	int iSlice = iWidth * iHeight;

	int idQ13 = iSlice / 2 + iHalfx;
	int idQ24 = iSlice / 2 - iHalfx;

	cufftReal Tempx, Tempy;

	if (j < iHalfx)
	{
		if(i < iHalfy)
		{
			Tempx = xf[id];
			Tempy = yf[id];

			// First Quadrant
			xf[id] = xf[id + idQ13];
			yf[id] = yf[id + idQ13];

			// Third Quadrant
			xf[id + idQ13] = Tempx;
			yf[id + idQ13] = Tempy;
		}
	}
	else
	{
		if (i < iHalfy)
		{
			Tempx = xf[id];
			Tempy = yf[id];

			// Second Quadrant
			xf[id] = xf[id + idQ24];
			yf[id] = yf[id + idQ24];

			// Fourth Quadrant
			xf[id + idQ24] = Tempx;
			yf[id + idQ24] = Tempy;
		}
	}
}

/*-------------------------------------------WFT2 Implementations-------------------------------------------*/
WFT2_CUDAF::WFT2_CUDAF(
	int iWidth, int iHeight,
	WFT_TYPE type,
	WFT2_DeviceResultsF& z,
	int iNumStreams)
	: m_iWidth(iWidth)
	, m_iHeight(iHeight)
	, m_type(type)
	, m_rThr(-1)
	, m_d_fPadded(nullptr)
	, m_d_xf(nullptr)
	, m_d_yf(nullptr)
{
	// Check the input image size
	if (iWidth % 2 != 0 || iHeight % 2 != 0)
	{
		std::cout << "GPU implementation of WFT curretly only suppports even image size!" << std::endl;
		throw -2;

	}
	
	// Get the number of SMs on GPU 
	cudaDeviceGetAttribute(&m_iSMs, cudaDevAttrMultiProcessorCount, 0);

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

	// scale for window so that norm2 of the window is 1. 
	m_rGaussianNorm2 = sqrt(4 * float(M_PI)*m_rSigmaX*m_rSigmaY);

	/* Do the Initialization */
	if(-1 == cuWFT2_Initialize(z))
	{
		std::cout<<"FFT padding is out of range [4096]. Shrink the size of either the image or the Gaussian Window!"<<std::endl;
		throw -1;
	}
}

WFT2_CUDAF::WFT2_CUDAF(
	int iWidth, int iHeight,
	WFT_TYPE type,
	float rSigmaX, float rWxl, float rWxh, float rWxi,
	float rSigmaY, float rWyl, float rWyh, float rWyi,
	float rThr,
	WFT2_DeviceResultsF &z,
	int iNumStreams)
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
	, m_d_fPadded(nullptr)
	, m_d_xf(nullptr)
	, m_d_yf(nullptr)
{
	// Check the input image size
	if (iWidth % 2 != 0 || iHeight % 2 != 0)
	{
		std::cout << "GPU implementation of WFT curretly only suppports even image size!" << std::endl;
		throw -2;

	}
	
	// Get the number of SMs on GPU 
	cudaDeviceGetAttribute(&m_iSMs, cudaDevAttrMultiProcessorCount, 0);

	// scale for window so that norm2 of the window is 1. 
	m_rGaussianNorm2 = sqrt(4 * float(M_PI)*m_rSigmaX*m_rSigmaY);
	/* Do the Initialization */
	if (-1 == cuWFT2_Initialize(z))
	{
		std::cout << "FFT padding is out of range [4096]. Shrink the size of either the image or the Gaussian Window!" << std::endl;
		throw - 1;
	}
}

WFT2_CUDAF::~WFT2_CUDAF()
{
	WFT_FPA::Utils::cudaSafeFree(m_d_fPadded);
	WFT_FPA::Utils::cudaSafeFree(m_d_xf);
	WFT_FPA::Utils::cudaSafeFree(m_d_yf);

	cufftDestroy(m_planForwardf);
}

void WFT2_CUDAF::operator()(
	cufftComplex *f,
	WFT2_DeviceResultsF &z,
	double &time)
{
	
}


/* Private functions */

void WFT2_CUDAF:: cuWFF2(cufftComplex *f, WFT2_DeviceResultsF &z, double &time)
{
}
void WFT2_CUDAF::cuWFR2(cufftComplex *f, WFT2_DeviceResultsF &z, double &time)
{
}

int WFT2_CUDAF::cuWFT2_Initialize(WFT2_DeviceResultsF &z)
{
	/*----------------------------WFF&WFR Common parameters initialization-----------------------------*/
	// Half of the Gaussian Window size
	m_iSx = int(round(3 * m_rSigmaX));
	m_iSy = int(round(3 * m_rSigmaY));
	// Guassian Window Size
	m_iWinHeight = 2 * m_iSy + 1;
	m_iWinWidth = 2 * m_iSx + 1;

	// Calculate the initial padding in order to perform the cyclic convolution using FFT
	// The padding size is size(A) + size(B) - 1;
	m_iPaddedHeight = m_iHeight + m_iWinHeight - 1;
	m_iPaddedWidth = m_iWidth + m_iWinWidth - 1;

	// Calculate the second padding in order to fit the optimized size for FFT
	int iH = getFirstGreater(m_iPaddedHeight);
	int iW = getFirstGreater(m_iPaddedWidth);
	if (-1 == iH || -1 == iW)
	{
		// Out of range
		return -1;
	}
	else
	{
		m_iPaddedHeight = OPT_FFT_SIZE[iH];
		m_iPaddedWidth = OPT_FFT_SIZE[iW];

		int iPaddedSize = m_iPaddedHeight * m_iPaddedWidth;

		/* Memory Preallocation on Device */
		// Allocate memory for input padded f which is pre-copmuted and remain unchanged
		checkCudaErrors(cudaMalloc((void**)&m_d_fPadded, sizeof(cufftComplex)*iPaddedSize));
		checkCudaErrors(cudaMalloc((void**)&m_d_xf, sizeof(cufftReal)*iPaddedSize));
		checkCudaErrors(cudaMalloc((void**)&m_d_yf, sizeof(cufftReal)*iPaddedSize));

		/* Make the CUFFT plan for the precomputation of Ff = fft2(f) */
		checkCudaErrors(cufftPlan2d(&m_planForwardf, m_iPaddedWidth, m_iPaddedHeight, CUFFT_C2C));
		
		/* Construct the xf & yf */
		dim3 threads(BLOCK_SIZE_16, BLOCK_SIZE_16);
		dim3 blocks((m_iPaddedWidth + BLOCK_SIZE_16 - 1) / BLOCK_SIZE_16, (m_iPaddedHeight + BLOCK_SIZE_16 - 1) / BLOCK_SIZE_16);
		// Generate xf, yf
		gen_xf_yf_Kernel<<<blocks, threads>>>(m_d_xf, m_d_yf, m_iPaddedWidth, m_iPaddedHeight);
		getLastCudaError("gen_xf_yf_Kernel Launch Failed!");
		// Shift xf, yf to match the FFT's results
		fftshift_xf_yf_kernel<<<blocks, threads>>>(m_d_xf, m_d_yf, m_iPaddedWidth, m_iPaddedHeight);
		getLastCudaError("fftshift_xf_yf_kernel Launch Failed!");
	}

	return 0;
}

void WFT2_CUDAF::cuWFF2_Init(WFT2_DeviceResultsF &z)
{
}

int WFT2_CUDAF:: cuWFR2_Init(WFT2_DeviceResultsF &z)
{
	return 0;
}

void WFT2_CUDAF::cuWFT2_feed_fPadded(cufftComplex *f)
{
}
void WFT2_CUDAF::cuWFF2_SetThreashold(cufftComplex *f)
{
}


}	// namespace WFT_FPA
}	// namespace WFT