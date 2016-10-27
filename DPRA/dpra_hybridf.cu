#include "dpra_hybridf.cuh"

#include "helper_cuda.h"
#include "device_launch_parameters.h"

namespace DPRA{

/*---------------------------------------CUDA Kernels----------------------------------*/
/*
 PURPOSE:
	Load the image into its padded version (0's at boundary)
 INPUTS:
	d_in_img: unpadded image
	iImgWidth, iImgHeihgt: image size
	iPaddedWidth, iPaddedHeight: padded size
 OUTPUTS:
	d_out_img_Padded: padded image
*/
__global__
void load_img_padding_kernel(uchar *d_out_img_Padded,
							 const uchar *d_in_img,
							 int iImgWidth,
							 int iImgHeight,
						 	 int iPaddedWidth,
							 int iPaddedHeight)
{
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	const int x = threadIdx.x + blockIdx.x * blockDim.x;

	int idImg = (y - 1)*iImgWidth + x - 1;
	int idPadded = y*iPaddedWidth + x;

	if (y < iPaddedHeight && x < iPaddedWidth)
	{
		if (y > 0 && y < iPaddedHeight - 1 && x>0 && x < iPaddedWidth - 1)
		{
			d_out_img_Padded[idPadded] = d_in_img[idImg];
		}
		else
		{
			d_out_img_Padded[idPadded] = 0;
		}
	}
}	
/* 
 PURPOSE:
	Pre-compute the cos(phi) and sin(phi) with padding of 0 at boundary
 INPUTS:
	d_in_Phi: the ref phi
	iWidth, iHeight: iWidth = iImgWidth +2, iHeight = iImgWidth +2
*/
__global__
void compute_cosPhi_sinPhi_kernel(float *d_out_cosPhi,
								  float *d_out_sinPhi,
								  float *d_in_Phi,
								  const int iWidth,
								  const int iHeight,
								  const int iPaddedWidth,
								  const int iPaddedHeight)
{
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	const int x = threadIdx.x + blockIdx.x * blockDim.x;

	int idPadded = y*iPaddedWidth + x;
	int idImg = (y - 1)*iWidth + x - 1;

	if (y < iPaddedHeight && x < iPaddedWidth)
	{
		if (y > 0 && y < iPaddedHeight - 1 && x > 0 && x < iPaddedWidth - 1)
		{
			float tempPhi = d_in_Phi[idImg];
			d_out_cosPhi[idPadded] = cos(tempPhi);
			d_out_sinPhi[idPadded] = sin(tempPhi);
		}
		else
		{
			d_out_cosPhi[idPadded] = 0;
			d_out_sinPhi[idPadded] = 0;
		}
	}
}

/*
 PURPOSE:
	Generate all matrix A and b for each pixel on GPU
 INPUTS:
	d_in_imgPadded: padded image
	d_in_cosPhi: padded cosPhi
	d_in_sinPhi: padded sinPhi
	iImgWidth, iImgHeight: image size
	iPaddedWidth, iPaddedHeight: padded size
 OUTPUTS:
	d_out_A: matrix A
	d_out_b: vector b
*/
__global__
void generate_A_b_kernel(float *d_out_A,
					     float *d_out_b,
						 const uchar *d_in_imgPadded,
						 const float *d_in_cosphi,
						 const float *d_in_sinphi,
						 const int iImgWidth,
						 const int iImgHeight,
						 const int iPaddedWidth,
						 const int iPaddedHeight)
{
	const int y = threadIdx.y + (BLOCK_SIZE_16 - 2) * blockIdx.y;
	const int x = threadIdx.x + (BLOCK_SIZE_16 - 2) * blockIdx.x;

	float sum_cos = 0, sum_sin = 0, sum_sincos = 0, sum_sin2 = 0, sum_cos2 = 0;
	float sum_ft = 0, sum_ft_cos = 0, sum_ft_sin = 0;

	// Global Memory offset: every block actually begin with 2 overlapped pixels
	__shared__ float cos_phi_sh[BLOCK_SIZE_16][BLOCK_SIZE_16];
	__shared__ float sin_phi_sh[BLOCK_SIZE_16][BLOCK_SIZE_16];
	__shared__ uchar img_sh[BLOCK_SIZE_16][BLOCK_SIZE_16];

	// Load the global mem to shared mem
	if (y < iPaddedHeight && x < iPaddedWidth)
	{
		cos_phi_sh[threadIdx.y][threadIdx.x] = d_in_cosphi[y*iPaddedWidth + x];
		sin_phi_sh[threadIdx.y][threadIdx.x] = d_in_sinphi[y*iPaddedWidth + x];
		img_sh[threadIdx.y][threadIdx.x] = d_in_imgPadded[y*iPaddedWidth + x];
	}
	__syncthreads();	

	// Compute the results within the boundary
	if (y >= 1 && y < iPaddedHeight - 1 && x >= 1 && x < iPaddedWidth - 1 &&
		threadIdx.x != 0 && threadIdx.x != BLOCK_SIZE_16 - 1 &&
		threadIdx.y != 0 && threadIdx.y != BLOCK_SIZE_16 - 1)
	{
		int idA = ((y - 1)*iImgWidth + x - 1) * 9;
		int idb = ((y - 1)*iImgWidth + x - 1) * 3;

		sum_cos = 0, sum_sin = 0, sum_sincos = 0, sum_sin2 = 0, sum_cos2 = 0;
		sum_ft = 0, sum_ft_cos = 0, sum_ft_sin = 0;

		for (int i = threadIdx.y - 1; i <= threadIdx.y + 1; i++)
		{
			for (int j = threadIdx.x - 1; j <= threadIdx.x + 1; j++)
			{
				float cos_phi = cos_phi_sh[i][j];
				float sin_phi = sin_phi_sh[i][j];
				float ft = static_cast<float>(img_sh[i][j]);

				// Elements of A
				sum_cos += cos_phi;
				sum_sin += sin_phi;
				sum_sincos += cos_phi * sin_phi;
				sum_sin2 += sin_phi*sin_phi;
				sum_cos2 += cos_phi*cos_phi;

				// Elements of b
				sum_ft += ft;
				sum_ft_cos += ft * cos_phi;
				sum_ft_sin += ft * sin_phi;
			}
		}
		d_out_A[idA + 0] = 9;			d_out_A[idA + 1] = 0;			d_out_A[idA + 2] = 0;
		d_out_A[idA + 3] = sum_cos;		d_out_A[idA + 4] = sum_cos2;	d_out_A[idA + 5] = 0;
		d_out_A[idA + 6] = sum_sin;		d_out_A[idA + 7] = sum_sincos;	d_out_A[idA + 8] = sum_sin2;

		d_out_b[idb + 0] = sum_ft;
		d_out_b[idb + 1] = sum_ft_cos;
		d_out_b[idb + 2] = sum_ft_sin;
	}
}

/*
 PURPOSE:
	Get the current & deltaPhi on device
*/
__global__
void get_deltaPhi_currPhi_kernel(float *d_out_deltaPhi,
								 float *d_out_currPhi,
								 float *d_in_refPhi,
								 cufftComplex *d_in_filtered,
								 const int iSize)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x;
		 i < iSize;
		 i += blockDim.x * gridDim.x)
	{
		float temp = atan2f(d_in_filtered[i].y, d_in_filtered[i].x);
		float tempRefPhi = d_in_refPhi[i];

		d_out_deltaPhi[i] = temp;
		d_out_currPhi[i] = atan2f(sinf(temp + tempRefPhi), cos(temp + tempRefPhi));
	}
}

/*--------------------------------------End CUDA Kernels--------------------------------*/


void load_img_padding(uchar *d_out_img_Padded,
					  const uchar *d_in_img,
					  int iImgWidth,
					  int iImgHeight,
					  int iPaddedWidth,
					  int iPaddedHeight,
					  const dim3 &blocks,
					  const dim3 &threads)
{
	load_img_padding_kernel<<<blocks, threads>>>(d_out_img_Padded,
												 d_in_img,
												 iImgWidth,
												 iImgHeight,
												 iPaddedWidth,
												 iPaddedHeight);
	getLastCudaError("load_img_padding_kernel launch failed!");
}

void compute_cosPhi_sinPhi(float *d_out_cosPhi,
						   float *d_out_sinPhi,
						   float *d_in_Phi,
						   const int iWidth,
						   const int iHeight,
						   const int iPaddedWidth,
						   const int iPaddedHeight,
						   const dim3 &blocks,
						   const dim3 &threads)
{
	compute_cosPhi_sinPhi_kernel<<<blocks, threads>>>(d_out_cosPhi,
													  d_out_sinPhi,
													  d_in_Phi,
													  iWidth,
													  iHeight,
													  iPaddedWidth,
													  iPaddedHeight);
	getLastCudaError("compute_cosPhi_sinPhi_kernel launch failed!");
}

void get_A_b(float *d_out_A,
			 float *d_out_b,
			 const uchar *d_in_imgPadded,
			 const float *d_in_cosphi,
			 const float *d_in_sinphi,
			 const int iImgWidth,
			 const int iImgHeight,
			 const int iPaddedWidth,
			 const int iPaddedHeight, 
			 const dim3 &blocks,
			 const dim3 &threads)
{
	generate_A_b_kernel<<<blocks, threads>>>(d_out_A, 
											 d_out_b,
											 d_in_imgPadded,
											 d_in_cosphi, 
											 d_in_sinphi, 
											 iImgWidth,
											 iImgHeight,
											 iPaddedWidth, 
											 iPaddedHeight);
	getLastCudaError("generate_A_b_kernel launch failed!");
}

void get_deltaPhi_currPhi(float *d_out_deltaPhi,
						  float *d_out_currPhi,
						  float *d_in_refPhi,
						  cufftComplex *d_in_filtered,
						  const int iSize)
{
	get_deltaPhi_currPhi_kernel<<<8*32, 256>>>(d_out_deltaPhi,
											   d_out_currPhi,
											   d_in_refPhi,
											   d_in_filtered,
											   iSize);
	getLastCudaError("get_deltaPhi_currPhi_kernel launch failed!");
}

}	// namespace DPRA