#include "WFT2_CUDAf.h"
#include "device_launch_parameters.h"
#include "mem_manager.h"

#include <iostream>
#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>

namespace WFT_FPA{
namespace WFT{

__inline__ __device__
float warpReduceSum(float val) {
	for (int offset = warpSize / 2; offset > 0; offset /= 2)
		val += __shfl_down(val, offset);
	return val;
}
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

/*
 PURPOSE:
	Feed the input f into the Padded matrix m_d_fPadded 
 INPUTS:
	d_f: The input fringe pattern
	iWidth, iHeight: size of the d_f
	iPaddedWidth, iPaddedHeight: FFT preferred size after padding
 OUTPUTS:
	d_fPadded: The padded d_f
*/
__global__ void feed_fPadded_kernel(cufftComplex *d_f, cufftComplex *d_fPadded, int iWidth, int iHeight, int iPaddedWidth, int iPaddedHeight)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int idImg = y * iWidth + x;
	int idPadded = y * iPaddedWidth + x;

	if (y < iHeight && x < iWidth)
	{
		d_fPadded[idPadded].x = d_f[idImg].x;
		d_fPadded[idPadded].y = d_f[idImg].y;
	}
	
	else if (y >= iHeight && y < iPaddedHeight && x >= iWidth && x < iPaddedWidth)
	{
		d_fPadded[idPadded].x = 0;
		d_fPadded[idPadded].y = 0;
	}
}

/*
 PURPOSE:
	Calculate the threshold value for the WFF if it's not specified using Parallel Reduction Algorithm
	thr = 6*sqrt(mean2(abs(f).^2)/3);
 INPUTS:
	in:	 type of cufftComplex input array
	size: size(width*height) of the in
 OUTPUS:
	out: 1-element device array
*/
__global__ void compute_WFF_threshold_kernel(cufftComplex *in, float *out, int size)
{
	float sum = float(0);

	for (int i = threadIdx.x + blockIdx.x * blockDim.x;
		 i < size;
		 i += blockDim.x*gridDim.x)
	{
		float abs = cuCabsf(in[i]);
		sum += abs*abs;
	}

	sum=warpReduceSum(sum);

	if (threadIdx.x % warpSize == 0)
		atomicAdd(out, sum);
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
	, m_d_rThr(nullptr)
	, m_iNumStreams(iNumStreams)
	, m_cudaStreams(nullptr)
	, m_d_fPadded(nullptr)
	, m_d_xf(nullptr)
	, m_d_yf(nullptr)
	, im_d_filtered(nullptr)
	, m_planForwardStreams(nullptr)
	, m_planInverseStreams(nullptr)
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
	, m_d_rThr(nullptr)
	, m_iNumStreams(iNumStreams)
	, m_cudaStreams(nullptr)
	, m_d_fPadded(nullptr)
	, m_d_xf(nullptr)
	, m_d_yf(nullptr)
	, im_d_filtered(nullptr)
	, m_planForwardStreams(nullptr)
	, m_planInverseStreams(nullptr)
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

	cufftDestroy(m_planForwardPadded);

	if (WFT_FPA::WFT::WFT_TYPE::WFF == m_type)
	{
		// Destroy stream-specific stuffs
		for (int i = 0; i < m_iNumStreams; i++)
		{
			cudaStreamDestroy(m_cudaStreams[i]);
			cufftDestroy(m_planForwardStreams[i]);
			cufftDestroy(m_planInverseStreams[i]);
		}
		free(m_cudaStreams);			m_cudaStreams = nullptr;
		free(m_planForwardStreams);		m_planForwardStreams = nullptr;
		free(m_planInverseStreams);		m_planInverseStreams = nullptr;

		WFT_FPA::Utils::cudaSafeFree(m_d_rThr);

		// Free the intermediate results 
		WFT_FPA::Utils::cudaSafeFree(im_d_filtered);
	
	}
}

void WFT2_CUDAF::operator()(
	cufftComplex *d_f,
	WFT2_DeviceResultsF &d_z,
	double &time)
{
	if (WFT_FPA::WFT::WFT_TYPE::WFF == m_type)
		cuWFF2(d_f, d_z, time);
	else if (WFT_FPA::WFT::WFT_TYPE::WFR == m_type)
		cuWFR2(d_f, d_z, time);
}


/* Private functions */

void WFT2_CUDAF:: cuWFF2(cufftComplex *d_f, WFT2_DeviceResultsF &d_z, double &time)
{
	/* Set the threshold m_rThr if it's not specified by the client */
	cuWFF2_SetThreashold(d_f);

	/* Feed the f to its padded version */
	cuWFT2_feed_fPadded(d_f);
	
	/* Pre-compute the FFT of m_d_fPadded */
	cufftExecC2C(m_planForwardPadded, m_d_fPadded, m_d_fPadded, CUFFT_FORWARD);
}
void WFT2_CUDAF::cuWFR2(cufftComplex *d_f, WFT2_DeviceResultsF &d_z, double &time)
{
	/* Pad the f to be prefered size of the FFT */
	cuWFT2_feed_fPadded(d_f);

	/* Pre-compute the FFT of m_d_fPadded */
	cufftExecC2C(m_planForwardPadded, m_d_fPadded, m_d_fPadded, CUFFT_FORWARD);
}

int WFT2_CUDAF::cuWFT2_Initialize(WFT2_DeviceResultsF &d_z)
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

		/* Make the CUFFT plans */
		checkCudaErrors(cufftPlan2d(&m_planForwardPadded, m_iPaddedWidth, m_iPaddedHeight, CUFFT_C2C));

		/* Construct the xf & yf */
		dim3 threads(BLOCK_SIZE_16, BLOCK_SIZE_16);
		dim3 blocks((m_iPaddedWidth + BLOCK_SIZE_16 - 1) / BLOCK_SIZE_16, (m_iPaddedHeight + BLOCK_SIZE_16 - 1) / BLOCK_SIZE_16);
		// Generate xf, yf
		gen_xf_yf_Kernel<<<blocks, threads>>>(m_d_xf, m_d_yf, m_iPaddedWidth, m_iPaddedHeight);
		getLastCudaError("gen_xf_yf_Kernel Launch Failed!");
		// Shift xf, yf to match the FFT's results
		fftshift_xf_yf_kernel<<<blocks, threads>>>(m_d_xf, m_d_yf, m_iPaddedWidth, m_iPaddedHeight);
		getLastCudaError("fftshift_xf_yf_kernel Launch Failed!");

		/*----------------------------------Specific Inititialization for WFF2&WFR2--------------------------------*/
		if (WFT_FPA::WFT::WFT_TYPE::WFF == m_type)
		{
			cuWFF2_Init(d_z);
		}
		else if (WFT_TYPE::WFR == m_type)
		{
			if(-1 == cuWFR2_Init(d_z))
				return -1;
		}
	}

	return 0;
}

void WFT2_CUDAF::cuWFF2_Init(WFT2_DeviceResultsF &d_z)
{
	int iImageSize = m_iWidth * m_iHeight;
	int iPaddedSize = m_iPaddedHeight * m_iPaddedWidth;

	// Allocate memory for the final results
	checkCudaErrors(cudaMalloc((void**)&d_z.m_d_filtered, sizeof(cufftComplex)*iImageSize));
	
	// 1. Allocate memory for intermediate results per-stream
	// 2. Create CUDA streams 
	// 3. Make the CUFFT plans for each stream
	checkCudaErrors(cudaMalloc((void**)&im_d_filtered, sizeof(cufftComplex)*m_iNumStreams*iPaddedSize));
	m_cudaStreams = (cudaStream_t*)malloc(m_iNumStreams*sizeof(cudaStream_t));
	m_planForwardStreams = (cufftHandle*)malloc(sizeof(cufftHandle)*m_iNumStreams);
	m_planInverseStreams = (cufftHandle*)malloc(sizeof(cufftHandle)*m_iNumStreams);

	for (int i = 0; i < m_iNumStreams; i++)
	{
		checkCudaErrors(cudaStreamCreate(&(m_cudaStreams[i])));

		checkCudaErrors(cufftPlan2d(&m_planForwardStreams[i], m_iPaddedWidth, m_iPaddedHeight, CUFFT_C2C));
		checkCudaErrors(cufftSetStream(m_planForwardStreams[i], m_cudaStreams[i]));

		checkCudaErrors(cufftPlan2d(&m_planInverseStreams[i], m_iPaddedWidth, m_iPaddedHeight, CUFFT_C2C));
		checkCudaErrors(cufftSetStream(m_planInverseStreams[i], m_cudaStreams[i]));
	}

	if (m_rThr < 0)
	{
		checkCudaErrors(cudaMalloc((void**)&m_d_rThr, sizeof(float)));
	}
}

int WFT2_CUDAF:: cuWFR2_Init(WFT2_DeviceResultsF &d_z)
{
	return 0;
}

void WFT2_CUDAF::cuWFT2_feed_fPadded(cufftComplex *d_f)
{
	dim3 threads(BLOCK_SIZE_16, BLOCK_SIZE_16);
	dim3 blocks((m_iPaddedWidth + BLOCK_SIZE_16 - 1) / BLOCK_SIZE_16, (m_iPaddedHeight + BLOCK_SIZE_16 - 1) / BLOCK_SIZE_16);

	feed_fPadded_kernel<<<blocks, threads>>>(d_f, m_d_fPadded, m_iWidth, m_iHeight, m_iPaddedWidth, m_iPaddedHeight);
	getLastCudaError("feed_fPadded_kernel Launch Failed!");
}
void WFT2_CUDAF::cuWFF2_SetThreashold(cufftComplex *d_f)
{
	// Set the m_rThr if not set
	if (m_rThr < 0)
	{
		int iImgSize = m_iWidth * m_iHeight;

		// Launch the kernel to compute the threshold
		int blocks = std::min((iImgSize + BLOCK_SIZE_256 - 1) / BLOCK_SIZE_256, 2048);
		compute_WFF_threshold_kernel<<<blocks, BLOCK_SIZE_256>>>(d_f, m_d_rThr, iImgSize);
		getLastCudaError("compute_WFF_threshold_kernel Launch Failed!");

		// Passing back to host
		checkCudaErrors(cudaMemcpy(&m_rThr, m_d_rThr, sizeof(float), cudaMemcpyDeviceToHost));
		m_rThr = 6 * sqrt(m_rThr *(1.0f / float(iImgSize)) / 3.0f);
	}
}


}	// namespace WFT_FPA
}	// namespace WFT