#include "dpra_cudaf.h"

#include <fstream>

namespace DPRA{

/*---------------------------------------CUDA Kernels----------------------------------*/

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
						   const int iPaddedWidth,
						   const int iPaddedHeight)
{
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	const int x = threadIdx.x + blockIdx.x * blockDim.x;

	int idPadded = y*iPaddedWidth + x;

	if (y < iPaddedHeight && x < iPaddedWidth)
	{
		if (y > 0 && y < iPaddedHeight - 1 && x > 0 && x < iPaddedWidth - 1)
		{
			float tempPhi = d_in_Phi[idPadded];
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

__global__
void load_phi0_to_phiref_kernel(float *d_out_phiref,
								const float *d_in_phi0,
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
			d_out_phiref[idPadded] = d_in_phi0[idImg];
		}
		else
		{
			d_out_phiref[idPadded] = 0;
		}
	}
}

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

__global__
void generate_csrRowPtrA_csrColIndA_kernel(int *d_out_csrRowPtrA,
										   int *d_out_csrColIndA,
										   const int iSize)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x;
		 i < iSize;
		 i += blockDim.x * gridDim.x)
	{
		int a1 = i * 9;

		d_out_csrRowPtrA[i * 3 + 0] = a1;
		d_out_csrRowPtrA[i * 3 + 1] = a1 + 3;
		d_out_csrRowPtrA[i * 3 + 2] = a1 + 6;

		a1 = i * 3;
		int a2 = a1 + 1;
		int a3 = a1 + 2;

		d_out_csrColIndA[i * 9 + 0] = a1;
		d_out_csrColIndA[i * 9 + 1] = a2;
		d_out_csrColIndA[i * 9 + 2] = a3;
		d_out_csrColIndA[i * 9 + 3] = a1;
		d_out_csrColIndA[i * 9 + 4] = a2;
		d_out_csrColIndA[i * 9 + 5] = a3;
		d_out_csrColIndA[i * 9 + 6] = a1;
		d_out_csrColIndA[i * 9 + 7] = a2;
		d_out_csrColIndA[i * 9 + 8] = a3;
	}

	if (blockIdx.x == 0 && threadIdx.x == 0)
	{
		d_out_csrRowPtrA[3 * iSize] = 9 * iSize;
	}
}

__global__
void generate_csrValA_b_kernel(float *d_out_csrValA,
							   float *d_out_b,
							   const uchar *d_in_img,
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
		img_sh[threadIdx.y][threadIdx.x] = d_in_img[y*iPaddedWidth + x];
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
		d_out_csrValA[idA + 0] = 9;			d_out_csrValA[idA + 1] = sum_cos;		d_out_csrValA[idA + 2] = sum_sin;
		d_out_csrValA[idA + 3] = sum_cos;	d_out_csrValA[idA + 4] = sum_cos2;		d_out_csrValA[idA + 5] = sum_sincos;
		d_out_csrValA[idA + 6] = sum_sin;	d_out_csrValA[idA + 7] = sum_sincos;	d_out_csrValA[idA + 8] = sum_sin2;

		d_out_b[idb + 0] = sum_ft;
		d_out_b[idb + 1] = sum_ft_cos;
		d_out_b[idb + 2] = sum_ft_sin;
	}
}

/*--------------------------------------End CUDA Kernels--------------------------------*/

DPRA_CUDAF::DPRA_CUDAF(const float *d_Phi0,
					   const int iWidth, const int iHeight,
					   const int irefUpdateRate)
	: m_iImgWidth(iWidth)
	, m_iImgHeight(iHeight)
	, m_iPaddedHeight(iHeight + 2)
	, m_iPaddedWidth(iWidth + 2)
	, m_rr(irefUpdateRate)
	, m_d_PhiRef(nullptr)
	, m_d_PhiCurr(nullptr)
	, m_d_csrValA(nullptr)
	, m_d_csrRowPtrA(nullptr)
	, m_d_csrColIndA(nullptr)
	, m_d_b(nullptr)
	, m_d_cosPhi(nullptr)
	, m_d_sinPhi(nullptr)
	, m_d_WFT(iWidth, iHeight, WFT_FPA::WFT::WFT_TYPE::WFF, m_d_z, 1)
	, m_d_deltaPhi_WFT(nullptr)
{
	int iImgSize = m_iImgWidth * m_iImgHeight;
	int iPaddedSize = m_iPaddedHeight * m_iPaddedWidth;

	dim3 threads(BLOCK_SIZE_16, BLOCK_SIZE_16);
	dim3 blocks((m_iPaddedWidth + BLOCK_SIZE_16 - 1) / BLOCK_SIZE_16, (m_iPaddedHeight + BLOCK_SIZE_16 - 1) / BLOCK_SIZE_16);

	// Copy the d_Phi0 to local device array
	checkCudaErrors(cudaMalloc((void**)&m_d_PhiRef, sizeof(float)*iPaddedSize));
	load_phi0_to_phiref_kernel<<<blocks, threads>>>(m_d_PhiRef, d_Phi0, m_iImgWidth, m_iImgHeight, m_iPaddedWidth, m_iPaddedHeight);
	getLastCudaError("load_phi0_to_phiref_kernel launch failed!");

	// Allocate memory
	checkCudaErrors(cudaMalloc((void**)&m_d_PhiCurr, sizeof(float)*iPaddedSize));
	checkCudaErrors(cudaMalloc((void**)&m_d_cosPhi, sizeof(float)*iPaddedSize));
	checkCudaErrors(cudaMalloc((void**)&m_d_sinPhi, sizeof(float)*iPaddedSize));
	checkCudaErrors(cudaMalloc((void**)&m_d_img_Padded, sizeof(uchar)*iPaddedSize));

	checkCudaErrors(cudaMalloc((void**)&m_d_csrValA, sizeof(float) * 9 * iImgSize));
	checkCudaErrors(cudaMalloc((void**)&m_d_csrRowPtrA, sizeof(int) * (3 * iImgSize + 1)));
	checkCudaErrors(cudaMalloc((void**)&m_d_csrColIndA, sizeof(int) * 9 * iImgSize));
	checkCudaErrors(cudaMalloc((void**)&m_d_b, sizeof(float) * 3 * iImgSize));

	// Allocate WFF phase memory
	checkCudaErrors(cudaMalloc((void**)&m_d_deltaPhi_WFT, sizeof(cufftComplex) * iImgSize));

	// Create cuSolver required handles
	checkCudaErrors(cusolverSpCreate(&m_cuSolverHandle));
	checkCudaErrors(cusparseCreateMatDescr(&m_desrA));
	checkCudaErrors(cusparseSetMatType(m_desrA, CUSPARSE_MATRIX_TYPE_GENERAL));
	checkCudaErrors(cusparseSetMatIndexBase(m_desrA, CUSPARSE_INDEX_BASE_ZERO));

	// Pre-compute the csrRowPtrA & csrColIndA 
	generate_csrRowPtrA_csrColIndA_kernel<<<8*32, 256>>>(m_d_csrRowPtrA, m_d_csrColIndA, iImgSize);
	getLastCudaError("generate_csrRowPtrA_csrColIndA_kernel launch failed!");
}

DPRA_CUDAF::~DPRA_CUDAF()
{
	checkCudaErrors(cusolverSpDestroy(m_cuSolverHandle));
	checkCudaErrors(cusparseDestroyMatDescr(m_desrA));

	WFT_FPA::Utils::cudaSafeFree(m_d_PhiRef);
	WFT_FPA::Utils::cudaSafeFree(m_d_PhiCurr);
	WFT_FPA::Utils::cudaSafeFree(m_d_csrValA);
	WFT_FPA::Utils::cudaSafeFree(m_d_csrRowPtrA);
	WFT_FPA::Utils::cudaSafeFree(m_d_csrColIndA);
	WFT_FPA::Utils::cudaSafeFree(m_d_b);
	WFT_FPA::Utils::cudaSafeFree(m_d_deltaPhi_WFT);
	WFT_FPA::Utils::cudaSafeFree(m_d_cosPhi);
	WFT_FPA::Utils::cudaSafeFree(m_d_sinPhi);
	WFT_FPA::Utils::cudaSafeFree(m_d_img_Padded);
}

void DPRA_CUDAF::operator() (const std::vector<cv::cuda::HostMem> &f,
							 std::vector<std::vector<float>> &dPhi_Sum,
							 double &time)
{

}

void DPRA_CUDAF::operator() (const std::vector<std::string> &fileNames,
							 std::vector<std::vector<float>> &dPhi_Sum,
							 double &time)
{

}

void DPRA_CUDAF::dpra_per_frame(const uchar *d_imag,
								float *&d_dPhi,
								double &time)
{
	dim3 threads(BLOCK_SIZE_16, BLOCK_SIZE_16);
	dim3 blocks((int)ceil((float)m_iPaddedWidth / (BLOCK_SIZE_16 - 2)), (int)ceil((float)m_iPaddedHeight / (BLOCK_SIZE_16 - 2)));
	dim3 blocks1((m_iPaddedWidth + BLOCK_SIZE_16 - 1) / BLOCK_SIZE_16, (m_iPaddedHeight + BLOCK_SIZE_16 - 1) / BLOCK_SIZE_16);

	load_img_padded_kernel<<<blocks1, threads>>>(m_d_img_Padded, d_imag, m_iImgWidth, m_iImgHeight, m_iPaddedWidth, m_iPaddedHeight);
	getLastCudaError("load_img_padded_kernel launch failed!");
	

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	/* Per-frame algorithm starts here */
	// 1. Pre-compute cos(phi) and sin(phi) for each pixel
	compute_cosPhi_sinPhi<<<blocks1, threads>>>(m_d_cosPhi, m_d_sinPhi, m_d_PhiRef, m_iPaddedWidth, m_iPaddedHeight);
	getLastCudaError("compute_cosPhi_sinPhi launch failed!");

	// 2. Construct the csrValA and the b to solve Ax = b
	generate_csrValA_b_kernel<<<blocks, threads>>>(m_d_csrValA, m_d_b, m_d_img_Padded, m_d_cosPhi, m_d_sinPhi, m_iImgWidth, m_iImgHeight, m_iPaddedWidth, m_iPaddedHeight);
	getLastCudaError("generate_csrValA_b_kernel launch failed!");

	// 3. Solve Ax = b (here should be tackled in another manner)
	int iSingularity = -1;
	float tol = 1e-7f;
	cusolverSpScsrlsvqr(m_cuSolverHandle,
						3*m_iImgHeight*m_iImgWidth,
						9*m_iImgWidth*m_iImgHeight,
						m_desrA,
						m_d_csrValA,
						m_d_csrRowPtrA,
						m_d_csrColIndA,
						m_d_b,
						tol,
						0,
						m_d_b,
						&iSingularity);
	 if (0 <= iSingularity)
    {
        printf("WARNING: the matrix is singular at row %d under tol (%E)\n", iSingularity, tol);
    }

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float ftime;
	cudaEventElapsedTime(&ftime, start, stop);
	std::cout << "csvValA_b_kernel running time is: " << ftime << "ms" << std::endl;
}

void DPRA_CUDAF::update_ref_phi()
{
	checkCudaErrors(cudaMemcpy(m_d_PhiRef, m_d_PhiCurr, sizeof(float)*m_iPaddedWidth*m_iPaddedHeight, cudaMemcpyDeviceToDevice));
}

}	// namespace DPRA