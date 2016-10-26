#include "dpra_hybridf.h"
#include "mem_manager.h"
#include "cuda.h"

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
void load_img_padded_kernel(uchar *d_out_img_Padded,
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
void compute_cosPhi_sinPhi(float *d_out_cosPhi,
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
void generate_csrValA_b_kernel(float *d_out_A,
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
		d_out_A[idA + 0] = 9;			d_out_A[idA + 1] = sum_cos;		d_out_A[idA + 2] = sum_sin;
		d_out_A[idA + 3] = sum_cos;		d_out_A[idA + 4] = sum_cos2;	d_out_A[idA + 5] = sum_sincos;
		d_out_A[idA + 6] = sum_sin;		d_out_A[idA + 7] = sum_sincos;	d_out_A[idA + 8] = sum_sin2;

		d_out_b[idb + 0] = sum_ft;
		d_out_b[idb + 1] = sum_ft_cos;
		d_out_b[idb + 2] = sum_ft_sin;
	}
}
/*--------------------------------------End CUDA Kernels--------------------------------*/


DPRA_HYBRIDF::DPRA_HYBRIDF(const float *v_Phi0, 
						   const int iWidth, const int iHeight, 
						   const int irefUpdateRate,
						   const int iNumThreads)
	: m_iWidth(iWidth)
	, m_iHeight(iHeight)
	, m_iPaddedHeight(iHeight + 2)
	, m_iPaddedWidth(iWidth + 2)
	, m_rr(irefUpdateRate)
	, m_iNumThreads(iNumThreads)
	, m_PhiRef(iWidth*iHeight, 0)
	, m_PhiCurr(iWidth*iHeight, 0)
	, m_d_PhiRef(nullptr)
	, m_h_A(nullptr)
	, m_h_b(nullptr)
	, m_d_A(nullptr)
	, m_d_b(nullptr)
	, m_d_cosPhi(nullptr)
	, m_d_sinPhi(nullptr)
	, m_d_img(nullptr)
	, m_d_img_Padded(nullptr)
	, m_WFT(iWidth, iHeight, WFT_FPA::WFT::WFT_TYPE::WFF, 
			20, -0.2f, 0.2f, 0.05f, 20, -0.2f, 0.2f, 0.05f, 10,
			m_d_z,1)
	, m_d_deltaPhiWFT(nullptr)
	, m_h_deltaPhiWFT(nullptr)
	, m_threads2D(BLOCK_SIZE_16, BLOCK_SIZE_16)
	, m_blocks_2Dshrunk((m_iPaddedWidth + BLOCK_SIZE_16 - 1) / BLOCK_SIZE_16, (m_iPaddedHeight + BLOCK_SIZE_16 - 1) / BLOCK_SIZE_16)
	, m_blocks_2D((m_iPaddedWidth + BLOCK_SIZE_16 - 1) / BLOCK_SIZE_16, (m_iPaddedHeight + BLOCK_SIZE_16 - 1) / BLOCK_SIZE_16)
{
	int iSize = iWidth * iHeight;
	int iPaddedSize = m_iPaddedWidth * m_iPaddedHeight;

	// Allocate host pinned memory
	WFT_FPA::Utils::cucreateptr(m_h_A, iSize * 9);
	WFT_FPA::Utils::cucreateptr(m_h_b, iSize * 3);
	WFT_FPA::Utils::cucreateptr(m_h_img, iSize);
	WFT_FPA::Utils::cucreateptr(m_h_deltaPhiWFT, iSize);

	// Allocate corresponding device memory
	checkCudaErrors(cudaMalloc((void**)&m_d_A, sizeof(float)*iSize * 9));
	checkCudaErrors(cudaMalloc((void**)&m_d_b, sizeof(float)*iSize * 3));
	checkCudaErrors(cudaMalloc((void**)&m_d_deltaPhiWFT, sizeof(cufftComplex)*iSize));

	// Allocate device memory for computing m_d_A & m_d_b for every pixel
	checkCudaErrors(cudaMalloc((void**)&m_d_cosPhi, sizeof(float)*iPaddedSize));
	checkCudaErrors(cudaMalloc((void**)&m_d_sinPhi, sizeof(float)*iPaddedSize));
	checkCudaErrors(cudaMalloc((void**)&m_d_img, sizeof(uchar)*iSize));
	checkCudaErrors(cudaMalloc((void**)&m_d_img_Padded, sizeof(uchar)*iPaddedSize));

	// Copy the initial v_Phi0 to local device array
	checkCudaErrors(cudaMalloc((void**)&m_d_PhiRef, sizeof(float)*iSize));
	checkCudaErrors(cudaMemcpy(m_d_PhiRef, v_Phi0, sizeof(float)*iSize, cudaMemcpyHostToDevice));

	// Copy the initial v_Phi0 to local host array
	memcpy(m_PhiRef.data(), v_Phi0, sizeof(float)*iSize);

	// Create CUDA event used for timing and synchronizing
	checkCudaErrors(cudaEventCreate(&m_d_event_start));
	checkCudaErrors(cudaEventCreate(&m_d_event_1));
	checkCudaErrors(cudaEventCreate(&m_d_event_2));

}

DPRA_HYBRIDF::~DPRA_HYBRIDF()
{
	checkCudaErrors(cudaEventDestroy(m_d_event_start));
	checkCudaErrors(cudaEventDestroy(m_d_event_1));
	checkCudaErrors(cudaEventDestroy(m_d_event_2));

	WFT_FPA::Utils::cudaSafeFree(m_d_PhiRef);
	WFT_FPA::Utils::cudaSafeFree(m_d_img_Padded);
	WFT_FPA::Utils::cudaSafeFree(m_d_img);
	WFT_FPA::Utils::cudaSafeFree(m_d_sinPhi);
	WFT_FPA::Utils::cudaSafeFree(m_d_cosPhi);
	WFT_FPA::Utils::cudaSafeFree(m_d_deltaPhiWFT);
	WFT_FPA::Utils::cudaSafeFree(m_d_b);
	WFT_FPA::Utils::cudaSafeFree(m_d_A);

	WFT_FPA::Utils::cudestroyptr(m_h_deltaPhiWFT);
	WFT_FPA::Utils::cudestroyptr(m_h_img);
	WFT_FPA::Utils::cudestroyptr(m_h_b);
	WFT_FPA::Utils::cudestroyptr(m_h_A);
}

void DPRA_HYBRIDF::dpra_per_frame(const cv::Mat &img, 
								  std::vector<float> &dPhi,
								  double &time)
{
	int iSize = m_iWidth * m_iHeight;
	int iPaddedSize = m_iPaddedWidth * m_iPaddedHeight;

	/* -------------------- Hybrid CPU and GPU DPRA algorithm ----------------------- */
	
	/* 1. Load the image f into device padded memory */

	cudaEventRecord(m_d_event_start);
	memcpy(m_h_img, img.data, sizeof(uchar)*iSize);
	checkCudaErrors(cudaMemcpyAsync(m_d_img, m_h_img, sizeof(uchar)*iSize, cudaMemcpyHostToDevice));
	
	load_img_padded_kernel<<<m_blocks_2D, m_threads2D>>>(m_d_img_Padded, m_d_img, m_iWidth, m_iHeight, m_iPaddedWidth, m_iPaddedHeight);
	getLastCudaError("load_img_padded_kernel launch failed!");
	cudaEventRecord(m_d_event_1);
	cudaEventSynchronize(m_d_event_1);

	float f_1_time = 0;
	cudaEventElapsedTime(&f_1_time, m_d_event_start, m_d_event_1);
	std::cout << "Step 1 running time is: " << f_1_time << "ms" << std::endl;


	/* 2. construct matrix A and vector b on GPU */

	/* 3. copy A and b from device to host */

	/* 4. Solve Ax = b on host */

	/* 5. Copy the resulted phiWFT array to GPU to get the delta phi */

	/* 6. Run WFF on the device phiWFT */

	/* 7. Get the delta phi and current phi on device */

	/* 8. Update refPhi on device */

	/* 9. Repeat 1 to 7 for the next frames */
}

}	// namespace DPRA