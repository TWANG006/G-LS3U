#include "dpra_cudaf.h"
#include "dpra_general.cuh"
#include <fstream>

// TODO
namespace DPRA{

/*---------------------------------------CUDA Kernels----------------------------------*/
__global__ 
void Gaussian_Elimination_3x3_kernel(const float *in_A, 
									 float *out_b,
									 int iSize)
{
	float A[3][4];	// The augmented matrix

	for (int i = threadIdx.x + blockDim.x * blockIdx.x;
		 i < iSize;
		 i += blockDim.x * gridDim.x)
	{
		int idA = i * 9; // Index in Mat A
		int idb = i * 3; // Index in Vec b

		// Load values from A&b to the augmented matrix A per thread
		A[0][0] = 9;		A[0][1] = in_A[idA + 3];	A[0][2] = in_A[idA + 6];	A[0][3] = out_b[idb + 0];
		A[1][0] = A[0][1];	A[1][1] = in_A[idA + 4];	A[1][2] = in_A[idA + 7];	A[1][3] = out_b[idb + 1];
		A[2][0] = A[0][2];	A[2][1] = A[1][2];			A[2][2] = in_A[idA + 8];	A[2][3] = out_b[idb + 2];

		// Gaussian Elimination with partial pivoting algorithm
		for (int k = 0; k < 3; k++)
		{
			// 1. Find the i-th pivot of the following A[k][i] elements
			int i_max = -1;
			float i_pivot = 0.0f;
			
			for (int i = k; i < 3; i++)
			{
				if (fabsf(i_pivot) - fabsf(A[i][k]) <= 1e-6)
				{
					i_pivot = A[i][k];
					i_max = i;
				}

			}

			// 2. swap rows
			for (int j = 0; j < 4; j++)
			{
				float temp = A[i_max][j];
				A[i_max][j] = A[k][j];
				A[k][j] = temp;
			}

			// 3. Triangulate the matrix
			for (int i = k + 1; i < 3; i++)
			{
				float mult = A[i][k] / A[k][k];
				
				for (int j = 0; j < 4; j++)
				{
					A[i][j] = A[i][j] - A[k][j] * mult;
				}
			}
		}

		// 4. Find the solution using backward substitution method
		A[2][3] = A[2][3] / A[2][2];
		A[1][3] = (A[1][3] - A[2][3] * A[1][2]) / A[1][1];
		A[0][3] = (A[0][3] - A[2][3] * A[0][2] - A[1][3] * A[0][1]) / A[0][0];

		// 5. Wirte the results back to out_b
		out_b[idb + 0] = A[0][3];
		out_b[idb + 1] = A[1][3];
		out_b[idb + 2] = A[2][3];
	}
}

__global__
void Update_Delta_Phi_Kernel(const float *in_b,
							 const int iSize,
							 cufftComplex *out_deltaPhiWFT)
{
	for (int i = threadIdx.x + blockDim.x * blockIdx.x;
		 i < iSize;
		 i += gridDim.x * blockDim.x)
	{
		int idb = i * 3;
		float fDeltaPhi = atan2f(-in_b[idb + 2], in_b[idb + 1]);

		out_deltaPhiWFT[i].x = cosf(fDeltaPhi);
		out_deltaPhiWFT[i].y = sinf(fDeltaPhi);
	}
}

/*--------------------------------------End CUDA Kernels--------------------------------*/

DPRA_CUDAF::DPRA_CUDAF(const float *v_Phi0,
					   const int iWidth, const int iHeight,
					   const int irefUpdateRate)
	: m_iImgWidth(iWidth)
	, m_iImgHeight(iHeight)
	, m_iPaddedHeight(iHeight + 2)
	, m_iPaddedWidth(iWidth + 2)
	, m_rr(irefUpdateRate)
	, m_h_deltaPhi(nullptr)
	, m_d_PhiRef(nullptr)
	, m_d_PhiCurr(nullptr)
	, m_d_deltaPhiRef(nullptr)
	, m_d_deltaPhi(nullptr)
	, m_d_A(nullptr)
	, m_d_b(nullptr)
	, m_d_cosPhi(nullptr)
	, m_d_sinPhi(nullptr)
	, m_h_img(nullptr)
	, m_d_img(nullptr)
	, m_d_img_Padded(nullptr)
	, m_d_WFT(iWidth, iHeight, WFT_FPA::WFT::WFT_TYPE::WFF,
			  10, -0.3f, 0.3f, 0.1f, 10, -0.3f, 0.3f, 0.1f, 15,
			  m_d_z, 1)
	, m_d_deltaPhi_WFT(nullptr)
	, m_threads2D(BLOCK_SIZE_16, BLOCK_SIZE_16)
	, m_blocks_2Dshrunk((int)ceil((float)m_iPaddedWidth / (BLOCK_SIZE_16 - 2)), (int)ceil((float)m_iPaddedHeight / (BLOCK_SIZE_16 - 2)))
	, m_blocks_2D((m_iPaddedWidth + BLOCK_SIZE_16 - 1) / BLOCK_SIZE_16, (m_iPaddedHeight + BLOCK_SIZE_16 - 1) / BLOCK_SIZE_16)

{
	int iImgSize = m_iImgWidth * m_iImgHeight;
	int iPaddedSize = m_iPaddedHeight * m_iPaddedWidth;

	// Allocate host pinned memory
	WFT_FPA::Utils::cucreateptr(m_h_img, iImgSize);
	WFT_FPA::Utils::cucreateptr(m_h_deltaPhi, iImgSize);
	
	// Copy the d_Phi0 to local device array
	checkCudaErrors(cudaMalloc((void**)&m_d_PhiRef, sizeof(float)*iImgSize));

	// Allocate memory
	checkCudaErrors(cudaMalloc((void**)&m_d_cosPhi, sizeof(float)*iPaddedSize));
	checkCudaErrors(cudaMalloc((void**)&m_d_sinPhi, sizeof(float)*iPaddedSize));
	checkCudaErrors(cudaMalloc((void**)&m_d_img_Padded, sizeof(uchar)*iPaddedSize));

	checkCudaErrors(cudaMalloc((void**)&m_d_A, sizeof(float) * 9 * iImgSize));
	checkCudaErrors(cudaMalloc((void**)&m_d_b, sizeof(float) * 3 * iImgSize));
	checkCudaErrors(cudaMalloc((void**)&m_d_img, sizeof(uchar)*iImgSize));	
	checkCudaErrors(cudaMalloc((void**)&m_d_PhiCurr, sizeof(float)*iImgSize));
	checkCudaErrors(cudaMalloc((void**)&m_d_deltaPhi_WFT, sizeof(cufftComplex) * iImgSize));
	checkCudaErrors(cudaMalloc((void**)&m_d_deltaPhi, sizeof(float)*iImgSize));

	// Copy the initial v_Phi0 to local device array
	checkCudaErrors(cudaMalloc((void**)&m_d_PhiRef, sizeof(float)*iImgSize));
	checkCudaErrors(cudaMemcpy(m_d_PhiRef, v_Phi0, sizeof(float)*iImgSize, cudaMemcpyHostToDevice));

	// Initialize the reference delta phi to 0's
	checkCudaErrors(cudaMalloc((void**)&m_d_deltaPhiRef, sizeof(float)*iImgSize));
	WFT_FPA::Utils::cuInitialize<float>(m_d_deltaPhiRef, 0, iImgSize);

	// Create CUDA event used for timing and synchronizing
	checkCudaErrors(cudaEventCreate(&m_d_event_start));
	checkCudaErrors(cudaEventCreate(&m_d_event_1));
	checkCudaErrors(cudaEventCreate(&m_d_event_2));
	checkCudaErrors(cudaEventCreate(&m_d_event_3));
	checkCudaErrors(cudaEventCreate(&m_d_event_4));
	checkCudaErrors(cudaEventCreate(&m_d_event_5));
	checkCudaErrors(cudaEventCreate(&m_d_event_6));
	checkCudaErrors(cudaEventCreate(&m_d_event_7));
}

DPRA_CUDAF::~DPRA_CUDAF()
{
	checkCudaErrors(cudaEventDestroy(m_d_event_start));
	checkCudaErrors(cudaEventDestroy(m_d_event_1));
	checkCudaErrors(cudaEventDestroy(m_d_event_2));
	checkCudaErrors(cudaEventDestroy(m_d_event_3));
	checkCudaErrors(cudaEventDestroy(m_d_event_4));
	checkCudaErrors(cudaEventDestroy(m_d_event_5));
	checkCudaErrors(cudaEventDestroy(m_d_event_6));
	checkCudaErrors(cudaEventDestroy(m_d_event_7));

	WFT_FPA::Utils::cudaSafeFree(m_d_PhiRef);
	WFT_FPA::Utils::cudaSafeFree(m_d_PhiCurr);
	WFT_FPA::Utils::cudaSafeFree(m_d_deltaPhiRef);
	WFT_FPA::Utils::cudaSafeFree(m_d_A);
	WFT_FPA::Utils::cudaSafeFree(m_d_b);
	WFT_FPA::Utils::cudaSafeFree(m_d_deltaPhi);
	WFT_FPA::Utils::cudaSafeFree(m_d_deltaPhi_WFT);
	WFT_FPA::Utils::cudaSafeFree(m_d_cosPhi);
	WFT_FPA::Utils::cudaSafeFree(m_d_sinPhi);
	WFT_FPA::Utils::cudaSafeFree(m_d_img);
	WFT_FPA::Utils::cudaSafeFree(m_d_img_Padded);

	WFT_FPA::Utils::cudestroyptr(m_h_img);
	WFT_FPA::Utils::cudestroyptr(m_h_deltaPhi);
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

void DPRA_CUDAF::dpra_per_frame(const cv::Mat &img,
								std::vector<float> &dPhi,
								double &time)
{
	int iSize = m_iImgWidth * m_iImgHeight;
	int iPaddedSize = m_iPaddedWidth * m_iPaddedHeight;

	/* I/O */
	memcpy(m_h_img, img.data, sizeof(uchar)*iSize);

	/* Per-frame algorithm starts here */

	cudaEventRecord(m_d_event_start);

	/* 1. Load the image f into device padded memory */
	checkCudaErrors(cudaMemcpyAsync(m_d_img, m_h_img, sizeof(uchar)*iSize, cudaMemcpyHostToDevice));
	load_img_padding(m_d_img_Padded, m_d_img, m_iImgWidth, m_iImgHeight, m_iPaddedWidth, m_iPaddedHeight, m_blocks_2D, m_threads2D);
	cudaEventRecord(m_d_event_1);
	/* 2. construct matrix A and vector b on GPU */
	compute_cosPhi_sinPhi(m_d_cosPhi, m_d_sinPhi, m_d_PhiRef, m_iImgWidth, m_iImgHeight, m_iPaddedWidth, m_iPaddedHeight, m_blocks_2D, m_threads2D);
	get_A_b(m_d_A, m_d_b, m_d_img_Padded, m_d_cosPhi, m_d_sinPhi, m_iImgWidth, m_iImgHeight, m_iPaddedWidth, m_iPaddedHeight, m_blocks_2Dshrunk, m_threads2D);
	cudaEventRecord(m_d_event_2);
		

	/* 3. Solve Ax = b and construct the m_h_deltaPhiWFT for, each pixel a thread */
	Gaussian_Elimination_3x3_kernel<<<256, 256>>>(m_d_A, m_d_b, iSize);
	getLastCudaError("Gaussian_Elimination_3x3_kernel launch failed!");
	cudaEventRecord(m_d_event_3);

	Update_Delta_Phi_Kernel<<<256, 256>>>(m_d_b, iSize, m_d_deltaPhi_WFT);
	getLastCudaError("Update_Delta_Phi_Kernel launch failed!");
	cudaEventRecord(m_d_event_4);
	

	/* 4. Run the CUDA based WFF */
	double d_wft_time = 0;
	m_d_WFT(m_d_deltaPhi_WFT, m_d_z, d_wft_time);
	
	cudaEventRecord(m_d_event_5);
	/* 5. Get the delta phi and current phi */
	get_deltaPhi_currPhi(m_d_deltaPhi, m_d_PhiCurr, m_d_deltaPhiRef, m_d_PhiRef, m_d_z.m_d_filtered, iSize);
	
	cudaEventRecord(m_d_event_6);
	/* 6. Copy the delta Phi to host */
	checkCudaErrors(cudaMemcpyAsync(m_h_deltaPhi, m_d_deltaPhi, sizeof(float)*iSize, cudaMemcpyDeviceToHost));
	cudaEventRecord(m_d_event_7);
	
	cudaEventSynchronize(m_d_event_7);

	/* END Per-frame algorithm starts here */

	/* I/O */
	memcpy(dPhi.data(), m_h_deltaPhi, sizeof(float)*iSize);

	float f_1_time = 0;
	cudaEventElapsedTime(&f_1_time, m_d_event_start, m_d_event_1);
	float f_2_time = 0;
	cudaEventElapsedTime(&f_2_time, m_d_event_1, m_d_event_2);
	float f_3_time = 0;
	cudaEventElapsedTime(&f_3_time, m_d_event_2, m_d_event_3);
	float f_4_time = 0;
	cudaEventElapsedTime(&f_4_time, m_d_event_3, m_d_event_4);
	float f_5_time = 0;
	cudaEventElapsedTime(&f_5_time, m_d_event_5, m_d_event_6);
	float f_6_time = 0;
	cudaEventElapsedTime(&f_6_time, m_d_event_6, m_d_event_7);

	std::cout << "Step 0 running time is: " << f_1_time << "ms" << std::endl;
	std::cout << "Step 1 running time is: " << f_2_time << "ms" << std::endl;
	std::cout << "Step 2 running time is: " << f_3_time << "ms" << std::endl;
	std::cout << "Step 3 running time is: " << f_4_time << "ms" << std::endl;
	std::cout << "Step 4 running time is: " << d_wft_time << "ms" << std::endl;
	std::cout << "Step 5 running time is: " << f_5_time << "ms" << std::endl;
	std::cout << "Step 6 running time is: " << f_6_time << "ms" << std::endl;


	time = double(f_1_time + f_2_time + f_3_time + f_4_time + f_5_time + f_6_time) + d_wft_time;
}

void DPRA_CUDAF::update_ref_phi()
{
	checkCudaErrors(cudaMemcpyAsync(m_d_PhiRef, m_d_PhiCurr, sizeof(float)*m_iImgWidth*m_iImgHeight, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpyAsync(m_d_deltaPhiRef, m_d_deltaPhi, sizeof(float)*m_iImgWidth*m_iImgHeight, cudaMemcpyDeviceToDevice));
}

}	// namespace DPRA