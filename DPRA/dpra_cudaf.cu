#include "dpra_cudaf.h"
#include "dpra_general.cuh"
#include <fstream>

// TODO
namespace DPRA{

/*---------------------------------------CUDA Kernels----------------------------------*/



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
			  20, -0.2f, 0.2f, 0.1f, 20, -0.2f, 0.2f, 0.1f, 15,
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
}

DPRA_CUDAF::~DPRA_CUDAF()
{
	checkCudaErrors(cudaEventDestroy(m_d_event_start));
	checkCudaErrors(cudaEventDestroy(m_d_event_1));
	checkCudaErrors(cudaEventDestroy(m_d_event_2));
	checkCudaErrors(cudaEventDestroy(m_d_event_3));
	checkCudaErrors(cudaEventDestroy(m_d_event_4));

	WFT_FPA::Utils::cudaSafeFree(m_d_PhiRef);
	WFT_FPA::Utils::cudaSafeFree(m_d_PhiCurr);
	WFT_FPA::Utils::cudaSafeFree(m_d_deltaPhiRef);
	WFT_FPA::Utils::cudaSafeFree(m_d_A);
	WFT_FPA::Utils::cudaSafeFree(m_d_b);
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
	/* 2. construct matrix A and vector b on GPU */
	compute_cosPhi_sinPhi(m_d_cosPhi, m_d_sinPhi, m_d_PhiRef, m_iImgWidth, m_iImgHeight, m_iPaddedWidth, m_iPaddedHeight, m_blocks_2D, m_threads2D);
	get_A_b(m_d_A, m_d_b, m_d_img_Padded, m_d_cosPhi, m_d_sinPhi, m_iImgWidth, m_iImgHeight, m_iPaddedWidth, m_iPaddedHeight, m_blocks_2Dshrunk, m_threads2D);

	cudaEventRecord(m_d_event_1);

	/* 3. Solve Ax = b and construct the m_h_deltaPhiWFT for, each pixel a thread */

	cudaEventRecord(m_d_event_2);

	/* 4. Run the CUDA based WFF */
	double d_wft_time = 0;
	m_d_WFT(m_d_deltaPhi_WFT, m_d_z, d_wft_time);

	cudaEventRecord(m_d_event_3);

	/* 5. Get the delta phi and current phi */
	get_deltaPhi_currPhi(m_d_deltaPhi, m_d_PhiCurr, m_d_deltaPhiRef, m_d_PhiRef, m_d_z.m_d_filtered, iSize);
	/* 6. Copy the delta Phi to host */
	checkCudaErrors(cudaMemcpyAsync(m_h_deltaPhi, m_d_deltaPhi, sizeof(float)*iSize, cudaMemcpyDeviceToHost));
	
	cudaEventRecord(m_d_event_4);
	cudaEventSynchronize(m_d_event_4);

	/* END Per-frame algorithm starts here */

	/* I/O */
	memcpy(dPhi.data(), m_h_deltaPhi, sizeof(float)*iSize);

	float f_12_time = 0;
	cudaEventElapsedTime(&f_12_time, m_d_event_start, m_d_event_1);

	float f_3_time = 0;
	cudaEventElapsedTime(&f_3_time, m_d_event_1, m_d_event_2);

	float f_56_time = 0;
	cudaEventElapsedTime(&f_56_time, m_d_event_2, m_d_event_3);

	time = double(f_12_time + f_56_time) + d_wft_time;
}

void DPRA_CUDAF::update_ref_phi()
{
	checkCudaErrors(cudaMemcpyAsync(m_d_PhiRef, m_d_PhiCurr, sizeof(float)*m_iPaddedWidth*m_iPaddedHeight, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpyAsync(m_d_deltaPhiRef, m_d_deltaPhi, sizeof(float)*m_iImgWidth*m_iImgHeight, cudaMemcpyDeviceToDevice));
}

}	// namespace DPRA