#include "WFT2_CUDAf.h"
#include "device_launch_parameters.h"
#include "mem_manager.h"

#include <iostream>
#include <algorithm>
#include <fstream>
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
	width, height: width and height of the xf and yf matrices
 OUTPUTS:
	d_out_xf, d_out_yf: Generated meshgrid
  */
__global__ 
void gen_xf_yf_Kernel(cufftReal *d_out_xf, 
					  cufftReal *d_out_yf, 
					  int iWidth, int iHeight)
{
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	int j = threadIdx.x + blockIdx.x * blockDim.x;

	int id = i*iWidth + j;

	int iHalfy = iHeight / 2;
	int iHalfx = iWidth / 2;

	if (i < iHeight && j < iWidth)
	{
		d_out_xf[id] = j - iHalfx;
		d_out_yf[id] = i - iHalfy;
	}
}
/*
 PURPOSE:
	2. Do the fftshift on xf and yf to be coincide with the CUFFT's results
 NOTE:
	Currently only even size in each dimension is supported 
 INPUTS:
	width, height: width and height of the xf and yf matrices
 OUTPUTS:
	d_out_xf, d_out_yf: In-place fft-shifted xf, yf

*/
__global__ 
void fftshift_xf_yf_kernel(cufftReal *d_out_xf, 
						   cufftReal *d_out_yf,
						   int iWidth, int iHeight)
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

	if (j < iHalfx && i < iHalfy)
	{
		Tempx = d_out_xf[id];
		Tempy = d_out_yf[id];

		// First Quadrant
		d_out_xf[id] = d_out_xf[id + idQ13];
		d_out_yf[id] = d_out_yf[id + idQ13];

		// Third Quadrant
		d_out_xf[id + idQ13] = Tempx;
		d_out_yf[id + idQ13] = Tempy;
	}
	else if (j >= iHalfx && j < iWidth && i < iHalfy)
	{

		Tempx = d_out_xf[id];
		Tempy = d_out_yf[id];

		// Second Quadrant
		d_out_xf[id] = d_out_xf[id + idQ24];
		d_out_yf[id] = d_out_yf[id + idQ24];

		// Fourth Quadrant
		d_out_xf[id + idQ24] = Tempx;
		d_out_yf[id + idQ24] = Tempy;		
	}
}
/*
 PURPOSE:
	Feed the input f into the Padded matrix m_d_fPadded 
 INPUTS:
	d_in_f: The input fringe pattern
	iWidth, iHeight: size of the d_f
	iPaddedWidth, iPaddedHeight: FFT preferred size after padding
 OUTPUTS:
	d_out_fPadded: The padded d_f
*/
__global__ 
void feed_fPadded_kernel(cufftComplex *d_in_f, 
						 cufftComplex *d_out_fPadded, 
						 int iWidth, int iHeight, 
						 int iPaddedWidth, int iPaddedHeight)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int idImg = y * iWidth + x;
	int idPadded = y * iPaddedWidth + x;

	if (y < iPaddedHeight && x < iPaddedWidth)
	{
		if (y < iHeight && x < iWidth)
		{
			d_out_fPadded[idPadded].x = d_in_f[idImg].x;
			d_out_fPadded[idPadded].y = d_in_f[idImg].y;
		}
		else
		{
			d_out_fPadded[idPadded].x = 0;
			d_out_fPadded[idPadded].y = 0;
		}
	}
}
/*
 PURPOSE:
	Point-wise multiplication of two matrices of complex numbers
 INPUT:
	d_in_a, d_in_b: Two matrices to be multiplied
	iSize: size of the matrices
 OUTPUT:
	d_out_c: The result after multiplication
*/
__global__
void complex_pointwise_multiplication_kernel(cufftComplex *d_in_a, 
											 cufftComplex *d_in_b, 
											 int iSize, 
											 cufftComplex *d_out_c)
{
	for (int i = threadIdx.x + blockIdx.x*blockDim.x;
		 i < iSize;
		 i += blockDim.x*gridDim.x)
	{
		d_out_c[i] = 
			WFT_FPA::Utils::ComplexScale(WFT_FPA::Utils::ComplexMul(d_in_a[i], d_in_b[i]), 1.0f / iSize);
	}
}
/*
 PURPOSE:
	Explicitly Compute the FFT of the Gaussian Window
		Fg=exp(-(xf*2*pi/mm-wxt).^2/2*sigmax*sigmax - (yf*2*pi/nn-wyt).^2/2*sigmay*sigmay)*sn2;
 INPUTS:
	d_in_xf, d_in_yf: meshgrid in frequency domain
	iPaddedWidth, iPaddedHeight: Padded Gaussian Window size
	wxt, wyt: frequencies in integer intervals
	wxi, wyi: steps
	wxl, wyl: lower bound of the frequencies
	sigmax, sigmay: sigma's in x&y directions
	sn2: normalization params (norm2 = 1);
 OUTPUTS:
	d_out_Fg: Fg
*/
__global__
void compute_Fg_kernel(cufftReal *d_in_xf, 
					   cufftReal *d_in_yf, 
					   int iPaddedWidth, int iPaddedHeight, 
					   int wxt, int wyt, float wxi, 
					   float wyi, float wxl, float wyl,
					   float sigmax, float sigmay, 
					   float sn2, 
					   cufftComplex *d_out_Fg)
{
	cufftReal rwxt = wxl + cufftReal(wxt) * wxi;
	cufftReal rwyt = wyl + cufftReal(wyt) * wyi;

	for (int i = threadIdx.x + blockIdx.x * blockDim.x;
		 i < iPaddedHeight*iPaddedWidth;
		 i += blockDim.x * gridDim.x)
	{
		cufftReal tempx = d_in_xf[i] * 2 * cufftReal(M_PI) * (1.0f / iPaddedWidth) - rwxt;
		cufftReal tempy = d_in_yf[i] * 2 * cufftReal(M_PI) * (1.0f / iPaddedHeight) - rwyt;
		tempx = -tempx * tempx * 0.5f * sigmax * sigmax;
		tempy = -tempy * tempy * 0.5f * sigmay * sigmay;
		
		d_out_Fg[i].x = exp(tempx + tempy) * sn2;
		d_out_Fg[i].y = 0;
	}
}

/*-------------------------------------------WFF Specific Utility Kernels-------------------------------------------*/
/*
 PURPOSE:
	Calculate the threshold value for the WFF if it's not specified using Parallel Reduction Algorithm
	thr = 6*sqrt(mean2(abs(f).^2)/3);
 INPUTS:
	d_in:	 type of cufftComplex input array
	size: size(width*height) of the in
 OUTPUS:
	d_out: 1-element device array
*/
__global__ 
void compute_WFF_threshold_kernel(cufftComplex *d_in, float *d_out, int size)
{
	float sum = float(0);

	for (int i = threadIdx.x + blockIdx.x * blockDim.x;
		 i < size;
		 i += blockDim.x*gridDim.x)
	{
		float abs = cuCabsf(d_in[i]);
		sum += abs*abs;
	}

	sum=warpReduceSum(sum);

	if (threadIdx.x % warpSize == 0)
		atomicAdd(d_out, sum);
}
/*
 PURPOSE:
	Initialize all WFF related matrices to 0's
 INPUTS:
	iWidth, iHeight: size of the final results
 OUTPUTS:
	d_out_filtered:
*/
__global__ 
void init_WFF_matrices_kernel(cufftComplex *d_out_filtered, 
							  int iWidth, int iHeight)
{
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	int idImg = y * iWidth + x;

	if (y < iHeight && x < iWidth)
	{
		d_out_filtered[idImg].x = 0;
		d_out_filtered[idImg].y = 0;
	}
}
/*
 PURPOSE:
	Threshold the spectrum sf
 INPUTS:
	iWidth, iHeight: image size
	iPaddedWidth, iPaddedHeight: Padded size
	thr: the threshold
 OUTPUTS:
	d_out_sf: sf after threshold
*/
__global__
void threshold_sf_kernel(cufftComplex *d_out_sf, 
						 int iWidth, int iHeight, 
						 int iPaddedWidth, int iPaddedHeight, 
						 float thr)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	int idPadded = y * iPaddedWidth + x;

	if (y < iPaddedHeight && x < iPaddedWidth)
	{
		if (y < iHeight && x < iWidth)
		{
			if (cuCabsf(d_out_sf[idPadded]) < thr)
			{
				d_out_sf[idPadded].x = 0;
				d_out_sf[idPadded].y = 0;
			}
		}
		else
		{
			d_out_sf[idPadded].x = 0;
			d_out_sf[idPadded].y = 0;
		}
	}
}
/*
 PURPOSE:
	Update the partial results im_d_filtered of each stream
 INPUTS:
	iWidth, iHeight: image size
	iPaddedWidth, iPaddedHeight: Padded size
	d_in_im_sf: spectrum of each stream
 OUTPUTS:
	d_out_im_filtered: filtered image after of each stream
*/
__global__
void update_WFF_partial_filtered_kernel(cufftComplex *d_in_im_sf,
										int iWidth, int iHeight, 
										int iPaddedWidth, int iPaddedHeight, 
										cufftComplex *d_out_im_filtered)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	int idPadded = y * iPaddedWidth + x;
	int idImg = y*iWidth + x;

	if (y < iHeight && x < iWidth)	
	{
		d_out_im_filtered[idImg].x += d_in_im_sf[idPadded].x;
		d_out_im_filtered[idImg].y += d_in_im_sf[idPadded].y;
	}
}
/*
 PURPOSE:
	Update the final z.filtered 
 INPUTS:
	d_in_im_filtered: the partial filtered results in each stream
	imgSize: size of the fringe pattern
 OUTPTS:
	d_out_filtered: the final results
*/
__global__
void update_WFF_final_filtered_kernel(cufftComplex *d_in_im_filtered, 
									  int imgSize, 
									  cufftComplex *d_out_filtered)
{
	for (int i = threadIdx.x + blockIdx.x*blockDim.x;
		 i < imgSize;
		 i += blockDim.x * gridDim.x)
	{
		d_out_filtered[i].x += d_in_im_filtered[i].x;
		d_out_filtered[i].y += d_in_im_filtered[i].y;
	}
}
/*
 PURPOSE:
	Scale the final results 
 INPUTS:
	d_out_filtered: the unscaled final results
	imagSize: size of the fringe pattern
	wxi,wyi: step size of the frequencies
 OUTPUT:
	d_out_filtered: scaled final results
*/
__global__
void scale_WFF_final_filtered_kernel(cufftComplex *d_out_filtered, 
									 int imgSize, 
									 float wxi, float wyi)
{
	float factor = 0.25f * (1.0f / float(M_PI*M_PI)) * wxi * wyi;

	for (int i = threadIdx.x + blockIdx.x*blockDim.x;
		 i < imgSize;
		 i += blockDim.x * gridDim.x)
	{
		d_out_filtered[i].x *= factor;
		d_out_filtered[i].y *= factor;
	}
}
/*----------------------------------------/End WFF Specific Utility Kernels------------------------------------------*/

/*-------------------------------------------WFR Specific Utility Kernels------------------------------------------*/
/*
 PURPOSE:
	Preompute the g used to compute the x.*g, y.*g, cxx&cyy using LS
 INPUTS:
	iWinWidth, iWinHeight: Gaussian Window Size
	iPaddedWidth, iPaddedHeight: Padded size of xg, yg
	sigmax, sigmay: sigma's
 OUTPUTS:
	d_out_g
*/
__global__
void precompute_g_kernel(cufftReal *d_out_g, 
						 int iWinWidth, int iWinHeight, 
						 float sigmax, float sigmay)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int iHW = (iWinWidth - 1) / 2;
	int iHH = (iWinHeight - 1) / 2;

	float xx, yy;

	int idWin = y * iWinWidth + x;

	if (y < iWinHeight && x < iWinWidth)
	{
		xx = float(x - iHW);
		yy = float(y - iHH);

		d_out_g[idWin] = exp(-xx*xx*0.5f*(1.0f / (sigmax*sigmax)) - yy*yy*0.5f*(1.0f / (sigmay*sigmay)));
	}
}
/*
 PURPOSE:
	Compute the (sum(sum(g.*g)))
 INPUTS:
	iWinSize: size of the gaussian window
	d_in_g: g
 OUTPUS:
	d_out_norm2g
*/
__global__
void precompute_norm2g_kernel(cufftReal *d_in_g, int iWinSize, float *d_out_norm2g)
{
	float sum = float(0);

	for (int i = threadIdx.x + blockIdx.x * blockDim.x;
		 i < iWinSize;
		 i += blockDim.x *gridDim.x)
	{
		float tempSqr = d_in_g[i] * d_in_g[i];
		sum += tempSqr;
	}

	sum = warpReduceSum(sum);

	if (threadIdx.x % warpSize == 0)
		atomicAdd(d_out_norm2g, sum);
}
/*
 PURPOSE: 
	compute nomalized g
 INPUTS:
	d_in_norm2g: normalization factor
	iWinszie: Gaussian windows size
 OUTPUTS:
	d_out_g: normalized g
 */
__global__
void precompute_normalized_g_kernel(float *d_in_norm2g, int iWinSize, cufftReal *d_out_g)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x;
		 i < iWinSize;
		 i += blockDim.x * gridDim.x)
	{
		d_out_g[i] = d_out_g[i] * (1.0f / sqrt(d_in_norm2g[0]));
	}
}
/*
 PURPOSE:
	Precompute xg & yg
 INPUTS:
	iWinWidth, iWinHeight: size of the Gaussian Window
	iPaddedWidth, iPaddedHeight: padded size
	d_in_g: the Gaussian Window
 OUTPUS:
	d_out_xg, d_out_yg: the constructed xg&yg
*/
__global__
void precompute_xg_yg_kernel(cufftReal *d_in_g,
							 int iWinWidth, int iWinHeight, 
							 int iPaddedWidth, int iPaddedHeight, 
							 cufftComplex *d_out_xg, 
							 cufftComplex *d_out_yg)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	int idPadded = y * iPaddedWidth + x;
	int idWin = y * iWinWidth + x;

	int iHW = (iWinWidth - 1) / 2;
	int iHH = (iWinHeight - 1) / 2;

	if (y < iPaddedHeight && x < iPaddedWidth)
	{
		if (y < iWinHeight && x < iWinWidth)
		{
			float xx = float(x - iHW);
			float yy = float(y - iHH);

			d_out_xg[idPadded].x = xx * d_in_g[idWin];
			d_out_yg[idPadded].x = yy * d_in_g[idWin];
		}
		else
		{
			d_out_xg[idPadded].x = 0;
			d_out_yg[idPadded].x = 0;
		}
		d_out_xg[idPadded].y = 0;
		d_out_yg[idPadded].y = 0;
	}
}
/*
 PURPOSE:
	Precompute the sum(x.*x.*g) or sum(y.*y.*g)
 INPUTS:
	d_in_xg, d_y_yg: the calculated x.*g, y.*g
	iWinWidth, iWinHeight: Width & Height of the Gaussian Window
	iPaddedWidth, iPaddedHeight: Padded size
 OUTPUS:
	d_out_sumxxg, d_out_sumyyg: sum of x.*x.*g & y.*y.*g	
*/
__global__
void precompute_sum_xxg_yyg_kernel(cufftComplex *d_in_xg, 
								   cufftComplex *d_in_yg, 
								   int iWinWidth, int iWinHeight, 
								   int iPaddedWidth, int iPaddedHeight,
								   float *d_out_sumxxg, 
								   float *d_out_sumyyg)
{
	float sumxxg = float(0);
	float sumyyg = float(0);

	for (int i = threadIdx.x + blockIdx.x * blockDim.x;
		 i < iWinHeight * iWinWidth;
		 i += gridDim.x * blockDim.x)
	{
		int x = i % iWinWidth;
		int y = i / iWinWidth;

		float xx = float(x - iWinWidth);
		float yy = float(y - iWinHeight);
		
		float tempxg = d_in_xg[y*iPaddedWidth + x].x * xx;
		float tempyg = d_in_yg[y*iPaddedWidth + x].x * yy;

		sumxxg += tempxg;
		sumyyg += tempyg;
	}

	sumxxg = warpReduceSum(sumxxg);
	sumyyg = warpReduceSum(sumyyg);

	if (threadIdx.x % warpSize == 0)
	{
		atomicAdd(d_out_sumxxg, sumxxg);
		atomicAdd(d_out_sumyyg, sumyyg);
	}	
}
/*
 PURPOSE: 
	Initialize the final results of WFR to zero's
 INPUTS:
	imgSize: the image sizes
 OUTPUTS:
	d_out_wx, d_out_wy, d_out_phase, d_out_phase_comp, d_out_r, d_out_b, d_out_cxx, d_out_cyy: to be initialized 
*/
__global__
void initialize_WFR_final_results_kernel(int iImgSize,
										 cufftReal* d_out_wx, 
										 cufftReal* d_out_wy,
										 cufftReal* d_out_phase,
										 cufftReal* d_out_phase_comp,
										 cufftReal* d_out_r,
										 cufftReal* d_out_b, 
										 cufftReal* d_out_cxx,
										 cufftReal* d_out_cyy)
{
	for (int i = threadIdx.x + blockIdx.x*blockDim.x;
		 i < iImgSize;
		 i += gridDim.x * blockDim.x)
	{
		d_out_wx[i] = 0;
		d_out_wy[i] = 0;
		d_out_phase[i] = 0;
		d_out_phase_comp[i] = 0;
		d_out_cxx[i] = 0;
		d_out_cyy[i] = 0;
		d_out_b[i] = 0;
		d_out_r[i] = 0;
	}
}
/*
 PURPOSE:
	Initialize per-stream intermediate results
 INPUTS:
	iPaddedSize: padded size 
 OUTPUS:
	d_out_im_wx, d_out_im_wy, d_out_im_p, d_out_im_r: to be initialized
*/
__global__
void initialize_WFR_im_results_kernel(int iImgSize,
									  cufftReal* d_out_im_wx, 
									  cufftReal* d_out_im_wy, 
									  cufftReal* d_out_im_p, 
									  cufftReal* d_out_im_r)
{
	for (int i = threadIdx.x + blockIdx.x*blockDim.x;
		 i < iImgSize;
		 i += gridDim.x * blockDim.x)
	{
		d_out_im_wx[i] = 0;
		d_out_im_wy[i] = 0;
		d_out_im_p[i] = 0;
		d_out_im_r[i] = 0;
	}
}
/*
 PURPOSE:
	Update the r, wx, wy and p of each iteration
 INPUTS:
	d_in_sf: the computed sf
	wxl, wyl: lower-bound of the frequencies
	wxt, wyt: current frequency
	wxi, wyi: step size of the frequencies
	iPaddedSize: padded size
 OUTPUTS:
	d_out_r, d_out_wx, d_out_wy, d_out_p: updated
*/
__global__
void update_r_wx_wy_p_kernel(cufftComplex *d_in_sf,
							 int wxt, float wxl, float wxi, 
							 int wyt, float wyl, float wyi, 
							 int iPaddedWidth, int iPaddedHeight, 
							 int iImgWidth, int iImgHeight,
							 cufftReal* d_out_r, 
							 cufftReal* d_out_wx, 
							 cufftReal* d_out_wy, 
							 cufftReal* d_out_p)
{
	float rwxt = wxl + float(wxt) * wxi;
	float rwyt = wyl + float(wyt) * wyi;

	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	int idImg = y * iImgWidth + x;
	int idPadded = y * iPaddedWidth + x;

	if (y < iImgHeight && x < iImgWidth)
	{
		float abs_sf = cuCabsf(d_in_sf[idPadded]);
		
		if (abs_sf > d_out_r[idImg])
		{
			d_out_r[idImg] = abs_sf;
			d_out_wx[idImg] = rwxt;
			d_out_wy[idImg] = rwyt;
			d_out_p[idImg] = atan2f(d_in_sf[idPadded].y, d_in_sf[idPadded].x);
		}
	}
}
/*
 PURPOSE: 
	Update the final r, wx, wy and p 
 INPUTS:
	d_in_r, d_in_wx, d_in_wy, d_in_p: partial results of each stream
	iImgSize: image size
 OUTPUTS:
	d_out_r, d_out_wx, d_out_wy, d_out_p: final results
*/
__global__
void update_final_r_wx_wy_p_kernel(cufftReal* d_in_r, 
								   cufftReal* d_in_wx, 
								   cufftReal* d_in_wy, 
								   cufftReal* d_in_p,
								   int iImgSize,
								   cufftReal* d_out_r, 
								   cufftReal* d_out_wx, 
								   cufftReal* d_out_wy, 
								   cufftReal* d_out_p)
{
	for (int i = threadIdx.x + blockDim.x * blockIdx.x;
		 i < iImgSize;
		 i += gridDim.x * blockDim.x)
	{
		if(d_in_r[i] > d_out_r[i])
		{
			d_out_r[i] = d_in_r[i];
			d_out_wx[i] = d_in_wx[i];
			d_out_wy[i] = d_in_wy[i];
			d_out_p[i] = d_in_p[i];
		}
	}
}
/*
 PURPOSE:
	Feed the wx&wy into padded cxx&cyy
 INPUTS:
	d_in_wx, d_in_wy
	iWidth, iHeight: image size
	iPaddedWidth, iPaddedHeight: padded size
 OUTPUTS:
	d_out_cxx, d_out_cyy: padded cxx&cyy
*/
__global__
void feed_cxx_cyy_kernel(cufftReal* d_in_wx,
						 cufftReal* d_in_wy, 
						 int iWidth, int iHeight, 
						 int iPaddedWidth, int iPaddedHeight,
						 cufftComplex *d_out_cxx, 
						 cufftComplex *d_out_cyy)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	int idPadded = y * iPaddedWidth + x;
	int idImg = y* iWidth + x;

	if (y < iPaddedHeight && x < iPaddedWidth)
	{
		if (y < iHeight && x < iWidth)
		{
			d_out_cxx[idPadded].x = d_in_wx[idImg];
			d_out_cyy[idPadded].x = d_in_wy[idImg];
		}
		else
		{
			d_out_cxx[idPadded].x = 0;
			d_out_cyy[idPadded].x = 0;
		}
		d_out_cxx[idPadded].y = 0;
		d_out_cyy[idPadded].y = 0;
	}
}
/*
 PURPOSE:
	2D Point-wise multiplication of two matrices of complex numbers
 INPUT:
	d_in_a1, d_in_b1, d_in_a2, d_in_b2: Two sets of matrices to be multiplied
	iSize: size of the matrices
 OUTPUT:
	d_out_c1, d_out_c2: The results after multiplication
*/
__global__
void complex_pointwise_multiplication_2d_kernel(cufftComplex *d_in_a1, 
												cufftComplex *d_in_b1, 
												cufftComplex *d_in_a2, 
												cufftComplex *d_in_b2,
												int iSize, 
												cufftComplex *d_out_c1, 
												cufftComplex *d_out_c2)
{
	for (int i = threadIdx.x + blockIdx.x*blockDim.x;
		 i < iSize;
		 i += blockDim.x*gridDim.x)
	{
		d_out_c1[i] = 
			WFT_FPA::Utils::ComplexScale(WFT_FPA::Utils::ComplexMul(d_in_a1[i], d_in_b1[i]), 1.0f / iSize);
		d_out_c2[i] = 
			WFT_FPA::Utils::ComplexScale(WFT_FPA::Utils::ComplexMul(d_in_a2[i], d_in_b2[i]), 1.0f / iSize);
	}
}
/*
 PURPOSE: 
	Update the results after compensation
 INPUTS:
	d_in_wx, d_in_wy, d_in_r, d_in_p: calculated results used to update the compensated phase
 OUTPUTS:
	d_out_cxx, d_out_cyy, d_out_phase_comp, d_out_b: results 
*/
__global__
void update_final_cxx_cyy_phaseComp_b_kernel(cufftComplex* d_in_cxx,
											 cufftComplex* d_in_cyy, 
											 cufftReal* d_in_r, 
											 cufftReal* d_in_p,
											 int iWidth, int iHeight,
											 int iPaddedWidth, int iPaddedHeight, 
											 float sumxxg, float sumyyg, 
											 float sigmax, float sigmay, 
											 int sx, int sy,
											 cufftReal* d_out_cxx, 
											 cufftReal* d_out_cyy, 
											 cufftReal* d_out_phase_comp, 
											 cufftReal* d_out_b)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	int idImg = y * iWidth + x;
	int idPadded = (y + sy) * iPaddedWidth + x + sx;

	if (x < iWidth && y < iHeight)
	{
		cufftReal cxx = d_in_cxx[idPadded].x;
		cufftReal cyy = d_in_cyy[idPadded].x;

		// curvature estimation
		cufftReal tempCxx = -cxx * (1.0f / sumxxg);
		cufftReal tempCyy = -cyy * (1.0f / sumyyg);

		d_out_cxx[idImg] = tempCxx;
		d_out_cyy[idImg] = tempCyy;
		
		// phase compensation
		cufftReal tempPhaseComp =  d_in_p[idImg] - 0.5f * atanf(sigmax*sigmax*tempCxx) - 0.5f * atanf(sigmay*sigmay*tempCyy);
		d_out_phase_comp[idImg] = atan2f(sin(tempPhaseComp), cos(tempPhaseComp));

		//scale amplitude
		d_out_b[idImg] = d_in_r[idImg] 
			* powf((1 + sigmax*sigmax*sigmax*sigmax*cxx*cxx)*0.25f*(1.0f / M_PI)*(1.0f / (sigmax*sigmax)), 0.25f)
			* powf((1 + sigmay*sigmay*sigmay*sigmay*cyy*cyy)*0.25f*(1.0f / M_PI)*(1.0f / (sigmay*sigmay)), 0.25f);
	}
}

/*----------------------------------------/End WFR Specific Utility Kernels------------------------------------------*/


/*------------------------------------------------/End CUDA Kernels--------------------------------------------------*/




/*--------------------------------------------------WFT2 Implementations-----------------------------------------------*/
WFT2_CUDAF::WFT2_CUDAF(int iWidth, int iHeight,
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
	, im_d_Fg(nullptr)
	, im_d_filtered(nullptr)
	, im_d_r(nullptr)
	, im_d_p(nullptr)
	, im_d_wx(nullptr)
	, im_d_wy(nullptr)
	, im_d_cxxPadded(nullptr)
	, im_d_cyyPadded(nullptr)
	, im_d_xgPadded(nullptr)
	, im_d_ygPadded(nullptr)
	, m_planStreams(nullptr)
{
	// Check the input image size
	//if (iWidth % 2 != 0 || iHeight % 2 != 0)
	//{
	//	std::cout << "GPU implementation of WFT curretly only suppports even image size!" << std::endl;
	//	throw -2;

	//}
	
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

WFT2_CUDAF::WFT2_CUDAF(int iWidth, int iHeight,
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
	, im_d_Fg(nullptr)
	, im_d_filtered(nullptr)
	, im_d_r(nullptr)
	, im_d_p(nullptr)
	, im_d_wx(nullptr)
	, im_d_wy(nullptr)
	, im_d_cxxPadded(nullptr)
	, im_d_cyyPadded(nullptr)
	, im_d_xgPadded(nullptr)
	, im_d_ygPadded(nullptr)
	, m_planStreams(nullptr)
{
	// Check the input image size
	//if (iWidth % 2 != 0 || iHeight % 2 != 0)
	//{
	//	std::cout << "GPU implementation of WFT curretly only suppports even image size!" << std::endl;
	//	throw -2;

	//}
	
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

	cufftDestroy(m_planPadded);

	if (WFT_FPA::WFT::WFT_TYPE::WFF == m_type)
	{
		// Destroy stream-specific stuffs
		for (int i = 0; i < m_iNumStreams; i++)
		{
			cudaStreamDestroy(m_cudaStreams[i]);
			cufftDestroy(m_planStreams[i]);
			WFT_FPA::Utils::cudaSafeFree(im_d_filtered[i]);
			WFT_FPA::Utils::cudaSafeFree(im_d_Fg[i]);
			WFT_FPA::Utils::cudaSafeFree(im_d_Sf[i]);
		}
		free(m_cudaStreams);			m_cudaStreams = nullptr;
		free(m_planStreams);			m_planStreams = nullptr;
		free(im_d_filtered);			im_d_filtered = nullptr;
		free(im_d_Fg);					im_d_Fg = nullptr;
		free(im_d_Sf);					im_d_Sf = nullptr;

		WFT_FPA::Utils::cudaSafeFree(m_d_rThr);
	}

	if (WFT_FPA::WFT::WFT_TYPE::WFR == m_type)
	{
		for (int i = 0; i < m_iNumStreams; i ++)
		{
			cudaStreamDestroy(m_cudaStreams[i]);
			cufftDestroy(m_planStreams[i]);
			WFT_FPA::Utils::cudaSafeFree(im_d_wx[i]);
			WFT_FPA::Utils::cudaSafeFree(im_d_wy[i]);
			WFT_FPA::Utils::cudaSafeFree(im_d_p[i]);
			WFT_FPA::Utils::cudaSafeFree(im_d_r[i]);
		}

		free(m_cudaStreams);	m_cudaStreams = nullptr;
		free(m_planStreams);	m_planStreams = nullptr;
		free(im_d_wx);	im_d_wx = nullptr;
		free(im_d_wy);	im_d_wy = nullptr;
		free(im_d_p);	im_d_p = nullptr;
		free(im_d_r);	im_d_r = nullptr;

		WFT_FPA::Utils::cudaSafeFree(im_d_cxxPadded);
		WFT_FPA::Utils::cudaSafeFree(im_d_cyyPadded);
		WFT_FPA::Utils::cudaSafeFree(im_d_xgPadded);
		WFT_FPA::Utils::cudaSafeFree(im_d_ygPadded);
		WFT_FPA::Utils::cudaSafeFree(m_d_rg_norm2);
		WFT_FPA::Utils::cudaSafeFree(m_d_rxxg_norm2);
		WFT_FPA::Utils::cudaSafeFree(m_d_ryyg_norm2);
	}
}

void WFT2_CUDAF::operator()(cufftComplex *d_f,
							WFT2_DeviceResultsF &d_z,
							double &time)
{
	if (WFT_FPA::WFT::WFT_TYPE::WFF == m_type)
		cuWFF2(d_f, d_z, time);
	else if (WFT_FPA::WFT::WFT_TYPE::WFR == m_type)
		cuWFR2(d_f, d_z, time);
}


/* Private functions */

void WFT2_CUDAF::cuWFF2(cufftComplex *d_f, WFT2_DeviceResultsF &d_z, double &time)
{
	/* CUDA blocks & threads scheduling */
	dim3 threads(BLOCK_SIZE_16, BLOCK_SIZE_16);
	dim3 blocksPadded((m_iPaddedWidth + BLOCK_SIZE_16 - 1) / BLOCK_SIZE_16, (m_iPaddedHeight + BLOCK_SIZE_16 - 1) / BLOCK_SIZE_16);
	dim3 blocksImg((m_iWidth + BLOCK_SIZE_16 - 1) / BLOCK_SIZE_16, (m_iHeight + BLOCK_SIZE_16 - 1) / BLOCK_SIZE_16);
	int blocks1D = std::min((m_iPaddedWidth*m_iPaddedHeight + BLOCK_SIZE_256 - 1) / BLOCK_SIZE_256, 2048);

	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	/* Set the threshold m_rThr if it's not specified by the client */

	cudaEventRecord(start);
	cuWFF2_SetThreashold(d_f);

	/* Feed the f to its padded version */
	cuWFT2_feed_fPadded(d_f);
	
	/* Pre-compute the FFT of m_d_fPadded */
	checkCudaErrors(cufftExecC2C(m_planPadded, m_d_fPadded, m_d_fPadded, CUFFT_FORWARD));

	/* Clear the results if they already contain last results */	
	init_WFF_matrices_kernel<<<blocksImg, threads>>>(d_z.m_d_filtered, m_iWidth, m_iHeight);
	getLastCudaError("init_WFF_matrices_kernel Launch Failed!");

	/* Insert this part inbetween to realize kind of CPU&GPU concurrent execution.
	   map the wl: wi : wh interval to integers from  0 to size = (wyh - wyl)/wyi + 1 in order to divide the 
	   copmutations across threads, since threads indices are more conviniently controlled by integers 	    */
	int iwx = int((m_rWxh - m_rWxl)*(1 / m_rWxi)) + 1;
	int iwy = int((m_rWyh - m_rWyl)*(1 / m_rWyi)) + 1;

	for (int i = 0; i < m_iNumStreams; i++)
	{
		init_WFF_matrices_kernel<<<blocksPadded, threads, 0, m_cudaStreams[i]>>>(im_d_filtered[i], m_iPaddedWidth, m_iPaddedHeight);
		getLastCudaError("init_WFF_matrices_kernel Launch Failed!");
	}

	/*std::vector<std::thread> td(m_iNumStreams);

	for (int i = 0; i < m_iNumStreams; i++)
	{
		td[i] = (std::thread(init_WFF_matrices, im_d_filtered[i], m_iPaddedWidth, m_iPaddedHeight));
	}
	std::for_each(td.begin(), td.end(), std::mem_fn(&std::thread::join));
	getLastCudaError("init_WFF_matrices_kernel Launch Failed!");*/

	/* Start the Real WFF iterations */
	

	int iNumResidue = iwx % m_iNumStreams;
	
	for (int y = 0; y < iwy; y++)
	{
		// Now we have equal number of kernels executed in each stream
		for (int x = iNumResidue; x < iwx; x += m_iNumStreams)
		{
			for (int i = 0; i < m_iNumStreams; i++)
			{
				// Construct Fg
				compute_Fg_kernel<<<blocks1D, BLOCK_SIZE_256, 0, m_cudaStreams[i]>>>(
					m_d_xf, m_d_yf, m_iPaddedWidth, m_iPaddedHeight,
					x + i, y, m_rWxi, m_rWyi, m_rWxl, m_rWyl,
					m_rSigmaX, m_rSigmaY, m_rGaussianNorm2, im_d_Fg[i]);
				getLastCudaError("compute_Fg_kernel Launch Failed!");
				
				// Compute sf=ifft2(Ff.*Fg)
				complex_pointwise_multiplication_kernel<<<blocks1D, BLOCK_SIZE_256, 0, m_cudaStreams[i]>>>(
					m_d_fPadded, im_d_Fg[i], m_iPaddedHeight*m_iPaddedWidth, im_d_Sf[i]);
				getLastCudaError("complex_pointwise_multiplication_kernel Launch Failed!");
				checkCudaErrors(cufftExecC2C(m_planStreams[i], im_d_Sf[i], im_d_Sf[i], CUFFT_INVERSE));

				// Threshold the sf: sf=sf.*(abs(sf)>=thr); 
				threshold_sf_kernel<<<blocksPadded, threads, 0, m_cudaStreams[i]>>>(im_d_Sf[i], m_iWidth, m_iHeight, m_iPaddedWidth, m_iPaddedHeight, m_rThr);
				getLastCudaError("threshold_sf_kernel Launch Failed!");

				// implement of IWFT: conv2(sf,w);
				checkCudaErrors(cufftExecC2C(m_planStreams[i], im_d_Sf[i], im_d_Sf[i], CUFFT_FORWARD));
				complex_pointwise_multiplication_kernel<<<blocks1D, BLOCK_SIZE_256, 0, m_cudaStreams[i]>>>(
					im_d_Sf[i], im_d_Fg[i], m_iPaddedHeight*m_iPaddedWidth, im_d_Sf[i]);
				getLastCudaError("complex_pointwise_multiplication_kernel Launch Failed!");
				checkCudaErrors(cufftExecC2C(m_planStreams[i], im_d_Sf[i], im_d_Sf[i], CUFFT_INVERSE));

				// Update partial results im_d_filtered
				update_WFF_partial_filtered_kernel<<<blocksImg, threads, 0, m_cudaStreams[i]>>>(im_d_Sf[i], m_iWidth, m_iHeight, m_iPaddedWidth, m_iPaddedHeight, im_d_filtered[i]);
				getLastCudaError("update_WFF_partial_filtered_kernel Launch Failed!");
			}
		}
		// Deal with the residues
		for (int x = 0; x < iNumResidue; x++)
		{
			// Construct Fg
			compute_Fg_kernel<<<blocks1D, BLOCK_SIZE_256, 0, m_cudaStreams[x] >>>(
				m_d_xf, m_d_yf, m_iPaddedWidth, m_iPaddedHeight,
				x, y, m_rWxi, m_rWyi, m_rWxl, m_rWyl,
				m_rSigmaX, m_rSigmaY, m_rGaussianNorm2, im_d_Fg[x]);
			getLastCudaError("compute_Fg_kernel Launch Failed!");
			
			// Compute sf=ifft2(Ff.*Fg)
			complex_pointwise_multiplication_kernel<<<blocks1D, BLOCK_SIZE_256, 0, m_cudaStreams[x] >>>(
				m_d_fPadded, im_d_Fg[x], m_iPaddedHeight*m_iPaddedWidth, im_d_Sf[x]);
			getLastCudaError("complex_pointwise_multiplication_kernel Launch Failed!");
			checkCudaErrors(cufftExecC2C(m_planStreams[x], im_d_Sf[x], im_d_Sf[x], CUFFT_INVERSE));

			// Threshold the sf: sf=sf.*(abs(sf)>=thr); 
			threshold_sf_kernel<<<blocksPadded, threads, 0, m_cudaStreams[x]>>>(im_d_Sf[x], m_iWidth, m_iHeight, m_iPaddedWidth, m_iPaddedHeight, m_rThr);
			getLastCudaError("threshold_sf_kernel Launch Failed!");

			// implement of IWFT: conv2(sf,w);
			checkCudaErrors(cufftExecC2C(m_planStreams[x], im_d_Sf[x], im_d_Sf[x], CUFFT_FORWARD));
			complex_pointwise_multiplication_kernel<<<blocks1D, BLOCK_SIZE_256, 0, m_cudaStreams[x]>>>(
				im_d_Sf[x], im_d_Fg[x], m_iPaddedHeight*m_iPaddedWidth, im_d_Sf[x]);
			getLastCudaError("complex_pointwise_multiplication_kernel Launch Failed!");
			checkCudaErrors(cufftExecC2C(m_planStreams[x], im_d_Sf[x], im_d_Sf[x], CUFFT_INVERSE));

			// Update partial results im_d_filtered
			update_WFF_partial_filtered_kernel<<<blocksImg, threads, 0, m_cudaStreams[x]>>>(im_d_Sf[x], m_iWidth, m_iHeight, m_iPaddedWidth, m_iPaddedHeight, im_d_filtered[x]);
			getLastCudaError("update_WFF_partial_filtered_kernel Launch Failed!");
		}
	}
	for (int i = 0; i < m_iNumStreams; i++)
	{
		cudaStreamSynchronize(m_cudaStreams[i]);
	}

	for (int i = 0; i < m_iNumStreams; i++)
	{
		update_WFF_final_filtered_kernel<<<blocks1D, BLOCK_SIZE_256>>>(im_d_filtered[i], m_iWidth*m_iHeight, d_z.m_d_filtered);
		getLastCudaError("update_WFF_final_filtered_kernel Launch Failed!");
	}
	scale_WFF_final_filtered_kernel<<<blocks1D, BLOCK_SIZE_256>>>(d_z.m_d_filtered, m_iWidth*m_iHeight, m_rWxi, m_rWyi);
	getLastCudaError("scale_WFF_final_filtered_kernel Launch Failed!");

	cudaEventRecord(end);
	cudaEventSynchronize(end);

	// Calculate the running time
	float t = 0;
	cudaEventElapsedTime(&t, start, end);
	time = double(t);
}
void WFT2_CUDAF::cuWFR2(cufftComplex *d_f, WFT2_DeviceResultsF &d_z, double &time)
{
	/* Various Sizes */
	int iPaddedSize = m_iPaddedHeight * m_iPaddedWidth;
	int iWinSize = m_iWinHeight * m_iWinWidth;
	int iImgSize = m_iWidth * m_iHeight;

	/* CUDA blocks & threads scheduling */
	dim3 threads(BLOCK_SIZE_16, BLOCK_SIZE_16);
	dim3 blocksPadded((m_iPaddedWidth + BLOCK_SIZE_16 - 1) / BLOCK_SIZE_16, (m_iPaddedHeight + BLOCK_SIZE_16 - 1) / BLOCK_SIZE_16);
	dim3 blocksImg((m_iWidth + BLOCK_SIZE_16 - 1) / BLOCK_SIZE_16, (m_iHeight + BLOCK_SIZE_16 - 1) / BLOCK_SIZE_16);
	int blocks1D_pad = std::min((iPaddedSize+ BLOCK_SIZE_256 - 1) / BLOCK_SIZE_256, 2048);
	int blocks1D_img = std::min((iImgSize + BLOCK_SIZE_256 - 1) / BLOCK_SIZE_256, 2048);

	/* Pad the f to be prefered size of the FFT */
	cuWFT2_feed_fPadded(d_f);

	/* Pre-compute the FFT of m_d_fPadded */
	cufftExecC2C(m_planPadded, m_d_fPadded, m_d_fPadded, CUFFT_FORWARD);

	/* Clear the results if they already contain last results */	
	initialize_WFR_final_results_kernel<<<blocks1D_img, BLOCK_SIZE_256>>>(
		iImgSize, 
		d_z.m_d_wx, d_z.m_d_wy, d_z.m_d_phase, d_z.m_d_phase_comp, d_z.m_d_r, d_z.m_d_b, d_z.m_d_cxx, d_z.m_d_cyy);
	getLastCudaError("initialize_WFR_final_results_kernel Launch Failed!");

	/* Insert this part inbetween to realize kind of CPU&GPU concurrent execution.
	   map the wl: wi : wh interval to integers from  0 to size = (wyh - wyl)/wyi + 1 in order to divide the 
	   copmutations across threads, since threads indices are more conviniently controlled by integers 	    */
	int iwx = int((m_rWxh - m_rWxl)*(1 / m_rWxi)) + 1;
	int iwy = int((m_rWyh - m_rWyl)*(1 / m_rWyi)) + 1;

	for (int i = 0; i < m_iNumStreams; i++)
	{
		initialize_WFR_im_results_kernel<<<blocks1D_img, BLOCK_SIZE_256, 0, m_cudaStreams[i]>>>(
			iImgSize,
			im_d_wx[i], im_d_wy[i], im_d_p[i], im_d_r[i]);
		getLastCudaError("initialize_WFR_im_results_kernel Launch Failed!");
	}

	/* Start the Real WFF iterations */
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	int iNumResidue = iwx % m_iNumStreams;
	cudaEventRecord(start);
	for (int y = 0; y < iwy; y++)
	{
		// Now we have equal number of kernels executed in each stream
		for (int x = iNumResidue; x < iwx; x += m_iNumStreams)
		{
			for (int i = 0; i < m_iNumStreams; i++)
			{
				// Construct Fg
				compute_Fg_kernel<<<blocks1D_pad, BLOCK_SIZE_256, 0, m_cudaStreams[i]>>>(
					m_d_xf, m_d_yf, m_iPaddedWidth, m_iPaddedHeight,
					x + i, y, m_rWxi, m_rWyi, m_rWxl, m_rWyl,
					m_rSigmaX, m_rSigmaY, m_rGaussianNorm2, im_d_Fg[i]);
				getLastCudaError("compute_Fg_kernel Launch Failed!");
				
				// Compute sf=ifft2(Ff.*Fg)
				complex_pointwise_multiplication_kernel<<<blocks1D_pad, BLOCK_SIZE_256, 0, m_cudaStreams[i]>>>(
					m_d_fPadded, im_d_Fg[i], iPaddedSize, im_d_Sf[i]);
				getLastCudaError("complex_pointwise_multiplication_kernel Launch Failed!");
				checkCudaErrors(cufftExecC2C(m_planStreams[i], im_d_Sf[i], im_d_Sf[i], CUFFT_INVERSE));

				// Update r, wx, wy and phase
				update_r_wx_wy_p_kernel<<<blocksImg, threads, 0, m_cudaStreams[i]>>>(
					im_d_Sf[i], x + i, m_rWxl, m_rWxi, y, m_rWyl, m_rWyi, 
					m_iPaddedWidth, m_iPaddedHeight, m_iWidth, m_iHeight, 
					im_d_r[i], im_d_wx[i], im_d_wy[i], im_d_p[i]);
				getLastCudaError("update_r_wx_wy_p_kernel Launch Failed!");
			}
		}

		for (int x = 0; x < iNumResidue; x++)
		{
			// Construct Fg
			compute_Fg_kernel<<<blocks1D_pad, BLOCK_SIZE_256, 0, m_cudaStreams[x]>>>(
				m_d_xf, m_d_yf, m_iPaddedWidth, m_iPaddedHeight,
				x, y, m_rWxi, m_rWyi, m_rWxl, m_rWyl,
				m_rSigmaX, m_rSigmaY, m_rGaussianNorm2, im_d_Fg[x]);
			getLastCudaError("compute_Fg_kernel Launch Failed!");
			
			// Compute sf=ifft2(Ff.*Fg)
			complex_pointwise_multiplication_kernel<<<blocks1D_pad, BLOCK_SIZE_256, 0, m_cudaStreams[x]>>>(
				m_d_fPadded, im_d_Fg[x], iPaddedSize, im_d_Sf[x]);
			getLastCudaError("complex_pointwise_multiplication_kernel Launch Failed!");
			checkCudaErrors(cufftExecC2C(m_planStreams[x], im_d_Sf[x], im_d_Sf[x], CUFFT_INVERSE));

			// Update r, wx, wy and phase
			update_r_wx_wy_p_kernel<<<blocksImg, threads, 0, m_cudaStreams[x]>>>(
				im_d_Sf[x], x, m_rWxl, m_rWxi, y, m_rWyl, m_rWyi, 
				m_iPaddedWidth, m_iPaddedHeight, m_iWidth, m_iHeight, 
				im_d_r[x], im_d_wx[x], im_d_wy[x], im_d_p[x]);
			getLastCudaError("update_r_wx_wy_p_kernel Launch Failed!");
		}
	}
	// Synchronize streams
	for (int i = 0; i < m_iNumStreams; i++)
	{
		cudaStreamSynchronize(m_cudaStreams[i]);
	}

	for (int i = 0; i < m_iNumStreams; i++)
	{
		update_final_r_wx_wy_p_kernel<<<blocks1D_img, BLOCK_SIZE_256>>>(im_d_r[i], im_d_wx[i], im_d_wy[i], im_d_p[i], iImgSize,
			d_z.m_d_r, d_z.m_d_wx, d_z.m_d_wy, d_z.m_d_phase);
		getLastCudaError("update_final_r_wx_wy_p_kernel Launch Failed!");
	}

	cudaEventRecord(end);
	cudaEventSynchronize(end);

	// Calculate the running time
	float t = 0;
	cudaEventElapsedTime(&t, start, end);
	time = double(t);

	/* Do the Least squre fitting to get cx and cy */
	/* Feed the wx & wy into their padded versions*/
	feed_cxx_cyy_kernel<<<blocksPadded, threads>>>(d_z.m_d_wx, d_z.m_d_wy, m_iWidth, m_iHeight, m_iPaddedWidth, m_iPaddedHeight,
		im_d_cxxPadded, im_d_cyyPadded);
	getLastCudaError("feed_cxx_cyy_kernel Launch Failed!");

	// z.cxx=-conv2(z.wx,x.*g,'same')/sum(sum(x.*x.*g));
    // z.cyy=-conv2(z.wy,y.*g,'same')/sum(sum(y.*y.*g)); 
	// Forward FFT
	checkCudaErrors(cufftExecC2C(m_planPadded, im_d_cxxPadded, im_d_cxxPadded, CUFFT_FORWARD));
	checkCudaErrors(cufftExecC2C(m_planPadded, im_d_cyyPadded, im_d_cyyPadded, CUFFT_FORWARD));
	// Pointwise multiplication
	complex_pointwise_multiplication_2d_kernel<<<blocks1D_pad, BLOCK_SIZE_256>>>(im_d_xgPadded, im_d_cxxPadded, im_d_ygPadded, im_d_cyyPadded, iPaddedSize,
		im_d_cxxPadded, im_d_cyyPadded);
	getLastCudaError("complex_pointwise_multiplication_2d_kernel Launch Failed!");
	// Inverse FFT
	checkCudaErrors(cufftExecC2C(m_planPadded, im_d_cxxPadded, im_d_cxxPadded, CUFFT_INVERSE));
	checkCudaErrors(cufftExecC2C(m_planPadded, im_d_cyyPadded, im_d_cyyPadded, CUFFT_INVERSE));

	// Update the compensated results
	update_final_cxx_cyy_phaseComp_b_kernel<<<blocksImg, threads>>>(
		im_d_cxxPadded, im_d_cyyPadded, d_z.m_d_r, d_z.m_d_phase,
		m_iWidth, m_iHeight, m_iPaddedWidth, m_iPaddedHeight, m_rxxg_norm2, m_ryyg_norm2, m_rSigmaX, m_rSigmaY, m_iSx, m_iSy,
		d_z.m_d_cxx, d_z.m_d_cyy, d_z.m_d_phase_comp, d_z.m_d_b);
	getLastCudaError("update_final_cxx_cyy_phaseComp_b_kernel Launch Failed!");
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
		checkCudaErrors(cufftPlan2d(&m_planPadded, m_iPaddedHeight, m_iPaddedWidth, CUFFT_C2C));
		checkCudaErrors(cufftSetStream(m_planPadded, 0));

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
			cuWFR2_Init(d_z);
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
	im_d_Fg = (cufftComplex**)malloc(m_iNumStreams * sizeof(cufftComplex*));
	im_d_filtered = (cufftComplex**)malloc(m_iNumStreams * sizeof(cufftComplex*));
	im_d_Sf = (cufftComplex**)malloc(m_iNumStreams * sizeof(cufftComplex*));
	
	m_cudaStreams = (cudaStream_t*)malloc(m_iNumStreams*sizeof(cudaStream_t));
	m_planStreams = (cufftHandle*)malloc(sizeof(cufftHandle)*m_iNumStreams);

	for (int i = 0; i < m_iNumStreams; i++)
	{
		checkCudaErrors(cudaStreamCreate(&(m_cudaStreams[i])));

		checkCudaErrors(cudaMalloc((void**)&im_d_Fg[i], sizeof(cufftComplex)*iPaddedSize));
		checkCudaErrors(cudaMalloc((void**)&im_d_filtered[i], sizeof(cufftComplex)*iPaddedSize));
		checkCudaErrors(cudaMalloc((void**)&im_d_Sf[i], sizeof(cufftComplex)*iPaddedSize));

		checkCudaErrors(cufftPlan2d(&m_planStreams[i], m_iPaddedHeight, m_iPaddedWidth, CUFFT_C2C));
		checkCudaErrors(cufftSetStream(m_planStreams[i], m_cudaStreams[i]));
	}

	if (m_rThr < 0)
	{
		checkCudaErrors(cudaMalloc((void**)&m_d_rThr, sizeof(float)));
	}
}
void WFT2_CUDAF::cuWFR2_Init(WFT2_DeviceResultsF &d_z)
{
	int iPaddedSize = m_iPaddedHeight * m_iPaddedWidth;
	int iWinSize = m_iWinHeight * m_iWinWidth;
	int iImgSize = m_iWidth * m_iHeight;

	// Allocate memory for the final results
	checkCudaErrors(cudaMalloc((void**)&d_z.m_d_wx, sizeof(cufftReal)*iImgSize));
	checkCudaErrors(cudaMalloc((void**)&d_z.m_d_wy, sizeof(cufftReal)*iImgSize));
	checkCudaErrors(cudaMalloc((void**)&d_z.m_d_r, sizeof(cufftReal)*iImgSize));
	checkCudaErrors(cudaMalloc((void**)&d_z.m_d_phase, sizeof(cufftReal)*iImgSize));
	checkCudaErrors(cudaMalloc((void**)&d_z.m_d_cxx, sizeof(cufftReal)*iImgSize));
	checkCudaErrors(cudaMalloc((void**)&d_z.m_d_cyy, sizeof(cufftReal)*iImgSize));
	checkCudaErrors(cudaMalloc((void**)&d_z.m_d_phase_comp, sizeof(cufftReal)*iImgSize));
	checkCudaErrors(cudaMalloc((void**)&d_z.m_d_b, sizeof(cufftReal)*iImgSize));

	// 1. Allocate memory for intermediate results per-stream
	// 2. Create CUDA streams 
	// 3. Make the CUFFT plans for each stream
	im_d_wx = (cufftReal**)malloc(m_iNumStreams*sizeof(cufftReal*));
	im_d_wy = (cufftReal**)malloc(m_iNumStreams*sizeof(cufftReal*));
	im_d_p = (cufftReal**)malloc(m_iNumStreams*sizeof(cufftReal*));
	im_d_r = (cufftReal**)malloc(m_iNumStreams*sizeof(cufftReal*));
	im_d_Fg = (cufftComplex**)malloc(m_iNumStreams * sizeof(cufftComplex*));
	im_d_Sf = (cufftComplex**)malloc(m_iNumStreams * sizeof(cufftComplex*));

	m_cudaStreams = (cudaStream_t*)malloc(m_iNumStreams*sizeof(cudaStream_t));
	m_planStreams = (cufftHandle*)malloc(sizeof(cufftHandle)*m_iNumStreams);

	for (int i = 0; i < m_iNumStreams; i++)
	{
		checkCudaErrors(cudaStreamCreate(&(m_cudaStreams[i])));

		// Allocate memory for the intermediate arrays
		checkCudaErrors(cudaMalloc((void**)&im_d_Fg[i], sizeof(cufftComplex)*iPaddedSize));
		checkCudaErrors(cudaMalloc((void**)&im_d_Sf[i], sizeof(cufftComplex)*iPaddedSize));
		checkCudaErrors(cudaMalloc((void**)&im_d_wx[i], sizeof(cufftReal)*iImgSize));
		checkCudaErrors(cudaMalloc((void**)&im_d_wy[i], sizeof(cufftReal)*iImgSize));
		checkCudaErrors(cudaMalloc((void**)&im_d_p[i], sizeof(cufftReal)*iImgSize));
		checkCudaErrors(cudaMalloc((void**)&im_d_r[i], sizeof(cufftReal)*iImgSize));		

		checkCudaErrors(cufftPlan2d(&m_planStreams[i], m_iPaddedHeight, m_iPaddedWidth, CUFFT_C2C));
		checkCudaErrors(cufftSetStream(m_planStreams[i], m_cudaStreams[i]));
	}
	// Allocate the memory for corresponding arrays
	checkCudaErrors(cudaMalloc((void**)&im_d_g, sizeof(cufftReal)*iWinSize));
	checkCudaErrors(cudaMalloc((void**)&im_d_cyyPadded, sizeof(cufftComplex)*iPaddedSize));
	checkCudaErrors(cudaMalloc((void**)&im_d_cxxPadded, sizeof(cufftComplex)*iPaddedSize));
	checkCudaErrors(cudaMalloc((void**)&im_d_xgPadded, sizeof(cufftComplex)*iPaddedSize));
	checkCudaErrors(cudaMalloc((void**)&im_d_ygPadded, sizeof(cufftComplex)*iPaddedSize));
	checkCudaErrors(cudaMalloc((void**)&m_d_rg_norm2, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&m_d_rxxg_norm2, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&m_d_ryyg_norm2, sizeof(float)));

	// Pre-compute g, x.*g, y.*g
	dim3 blocks_g((m_iWinWidth + BLOCK_SIZE_16 - 1) / BLOCK_SIZE_16, (m_iWinHeight + BLOCK_SIZE_16 - 1) / BLOCK_SIZE_16);
	dim3 threads(BLOCK_SIZE_16, BLOCK_SIZE_16);
	precompute_g_kernel<<<blocks_g, threads>>>(im_d_g, m_iWinWidth, m_iWinHeight, m_rSigmaX, m_rSigmaY);
	getLastCudaError("precompute_g_kernel Launch Failed!");

	int blocks_g1D = std::min((iWinSize + BLOCK_SIZE_256 - 1) / BLOCK_SIZE_256, 2048);
	precompute_norm2g_kernel<<<blocks_g1D, BLOCK_SIZE_256>>>(im_d_g, iWinSize, m_d_rg_norm2);
	getLastCudaError("precompute_norm2g_kernel Launch Failed!");

	precompute_normalized_g_kernel<<<blocks_g1D, BLOCK_SIZE_256>>>(m_d_rg_norm2, iWinSize, im_d_g);
	getLastCudaError("precompute_normalized_g_kernel Launch Failed!");

	dim3 blocks_xyg((m_iPaddedWidth + BLOCK_SIZE_16 - 1) / BLOCK_SIZE_16, (m_iPaddedHeight + BLOCK_SIZE_16 - 1) / BLOCK_SIZE_16);
	precompute_xg_yg_kernel<<<blocks_xyg, threads>>>(im_d_g, m_iWinWidth, m_iWinHeight, m_iPaddedWidth, m_iPaddedHeight, im_d_xgPadded, im_d_ygPadded);
	getLastCudaError("precompute_xg_yg_kernel Launch Failed!");

	precompute_sum_xxg_yyg_kernel<<<blocks_g1D, BLOCK_SIZE_256>>>(im_d_xgPadded, im_d_ygPadded, m_iWinWidth, m_iWinHeight, m_iPaddedWidth, m_iPaddedHeight, m_d_rxxg_norm2, m_d_ryyg_norm2);
	getLastCudaError("precompute_sum_xxg_yyg_kernel Launch Failed!");

	checkCudaErrors(cudaMemcpy(&m_rxxg_norm2, m_d_rxxg_norm2, sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&m_ryyg_norm2, m_d_ryyg_norm2, sizeof(float), cudaMemcpyDeviceToHost));

	// Free the im_d_g since it's no need furthermore
	checkCudaErrors(cudaFree(im_d_g));	im_d_g = nullptr;

	// Compute the FFT of x.*g & y.*g
	checkCudaErrors(cufftExecC2C(m_planPadded, im_d_xgPadded, im_d_xgPadded, CUFFT_FORWARD));
	checkCudaErrors(cufftExecC2C(m_planPadded, im_d_ygPadded, im_d_ygPadded, CUFFT_FORWARD));
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

/*-----------------------------------------/End WFT2 Implementations-------------------------------------------*/

}	// namespace WFT_FPA
}	// namespace WFT