#include "dpra_hybrid.h"
#include <omp.h>
#include <mkl.h>
#include "mem_manager.h"
#include "dpra_general.cuh"

namespace DPRA
{
DPRA_HYBRID::DPRA_HYBRID(const double *v_Phi0, 
						 const int iWidth, const int iHeight, 
						 const int irefUpdateRate,
						 const int iNumThreads)
	: m_iWidth(iWidth)
	, m_iHeight(iHeight)
	, m_iPaddedHeight(iHeight + 2)
	, m_iPaddedWidth(iWidth + 2)
	, m_rr(irefUpdateRate)
	, m_iNumThreads(iNumThreads)
	//, m_PhiRef(iWidth*iHeight, 0)
	//, m_PhiCurr(iWidth*iHeight, 0)
	, m_h_deltaPhi(nullptr)
	, m_d_PhiRef(nullptr)
	, m_d_dPhiRef(nullptr)
	, m_d_deltaPhi(nullptr)
	, m_d_PhiCurr(nullptr)
	, m_h_A(nullptr)
	, m_h_b(nullptr)
	, m_d_A(nullptr)
	, m_d_b(nullptr)
	, m_d_cosPhi(nullptr)
	, m_d_sinPhi(nullptr)
	, m_d_img(nullptr)
	, m_d_img_Padded(nullptr)
	, m_WFT(iWidth, iHeight, WFT_FPA::WFT::WFT_TYPE::WFF, 
			20, -0.15f, 0.15f, 0.05f, 20, -0.15f, 0.15f, 0.05f, 15,
			m_d_z,1)
	, m_d_deltaPhiWFT(nullptr)
	, m_h_deltaPhiWFT(nullptr)
	, m_threads2D(BLOCK_SIZE_16, BLOCK_SIZE_16)
	, m_blocks_2Dshrunk((int)ceil((double)m_iPaddedWidth / (BLOCK_SIZE_16 - 2)), (int)ceil((double)m_iPaddedHeight / (BLOCK_SIZE_16 - 2)))
	, m_blocks_2D((m_iPaddedWidth + BLOCK_SIZE_16 - 1) / BLOCK_SIZE_16, (m_iPaddedHeight + BLOCK_SIZE_16 - 1) / BLOCK_SIZE_16)
{
	int iSize = iWidth * iHeight;
	int iPaddedSize = m_iPaddedWidth * m_iPaddedHeight;

	omp_set_num_threads(m_iNumThreads);

	// Allocate host pinned memory
	WFT_FPA::Utils::cucreateptr(m_h_A, iSize * 9);
	WFT_FPA::Utils::cucreateptr(m_h_b, iSize * 3);
	WFT_FPA::Utils::cucreateptr(m_h_img, iSize);
	WFT_FPA::Utils::cucreateptr(m_h_deltaPhiWFT, iSize);
	WFT_FPA::Utils::cucreateptr(m_h_deltaPhi, iSize);
	

	// Allocate corresponding device memory
	checkCudaErrors(cudaMalloc((void**)&m_d_A, sizeof(double)*iSize * 9));
	checkCudaErrors(cudaMalloc((void**)&m_d_b, sizeof(double)*iSize * 3));
	checkCudaErrors(cudaMalloc((void**)&m_d_img, sizeof(uchar)*iSize));
	checkCudaErrors(cudaMalloc((void**)&m_d_deltaPhiWFT, sizeof(cufftDoubleComplex)*iSize));
	checkCudaErrors(cudaMalloc((void**)&m_d_deltaPhi, sizeof(double)*iSize));

	// Allocate device memory for computing m_d_A & m_d_b for every pixel
	checkCudaErrors(cudaMalloc((void**)&m_d_cosPhi, sizeof(double)*iPaddedSize));
	checkCudaErrors(cudaMalloc((void**)&m_d_sinPhi, sizeof(double)*iPaddedSize));
	checkCudaErrors(cudaMalloc((void**)&m_d_img_Padded, sizeof(uchar)*iPaddedSize));
	checkCudaErrors(cudaMalloc((void**)&m_d_PhiCurr, sizeof(double)*iSize));

	// Copy the initial v_Phi0 to local device array
	checkCudaErrors(cudaMalloc((void**)&m_d_PhiRef, sizeof(double)*iSize));
	checkCudaErrors(cudaMemcpy(m_d_PhiRef, v_Phi0, sizeof(double)*iSize, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&m_d_dPhiRef, sizeof(double)*iSize));
	WFT_FPA::Utils::cuInitialize<double>(m_d_dPhiRef, 0, iSize);


	// Copy the initial v_Phi0 to local host array
	//memcpy(m_PhiRef.data(), v_Phi0, sizeof(float)*iSize);

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

DPRA_HYBRID::~DPRA_HYBRID()
{
	checkCudaErrors(cudaEventDestroy(m_d_event_start));
	checkCudaErrors(cudaEventDestroy(m_d_event_1));
	checkCudaErrors(cudaEventDestroy(m_d_event_2));
	checkCudaErrors(cudaEventDestroy(m_d_event_3));
	checkCudaErrors(cudaEventDestroy(m_d_event_4));
	checkCudaErrors(cudaEventDestroy(m_d_event_5));
	checkCudaErrors(cudaEventDestroy(m_d_event_6));
	checkCudaErrors(cudaEventDestroy(m_d_event_7));
	
	WFT_FPA::Utils::cudaSafeFree(m_d_img_Padded);
	WFT_FPA::Utils::cudaSafeFree(m_d_img);
	WFT_FPA::Utils::cudaSafeFree(m_d_sinPhi);
	WFT_FPA::Utils::cudaSafeFree(m_d_cosPhi);
	WFT_FPA::Utils::cudaSafeFree(m_d_deltaPhiWFT);
	WFT_FPA::Utils::cudaSafeFree(m_d_b);
	WFT_FPA::Utils::cudaSafeFree(m_d_A);
	WFT_FPA::Utils::cudaSafeFree(m_d_PhiRef);
	WFT_FPA::Utils::cudaSafeFree(m_d_PhiCurr);
	WFT_FPA::Utils::cudaSafeFree(m_d_deltaPhi);

	WFT_FPA::Utils::cudestroyptr(m_h_deltaPhiWFT);
	WFT_FPA::Utils::cudestroyptr(m_h_img);
	WFT_FPA::Utils::cudestroyptr(m_h_b);
	WFT_FPA::Utils::cudestroyptr(m_h_A);
	WFT_FPA::Utils::cudestroyptr(m_h_deltaPhi);
}

void DPRA_HYBRID::operator() (const std::vector<cv::Mat> &f, 
							   std::vector<std::vector<double>> &dPhi_Sum,
							   double &time)
{
	/* 1. create per-frame parmeters */
	std::vector<double> deltaPhi(m_iWidth*m_iHeight, 0);

	time = 0;
	double dTime_per_Frame = 0;;

	/* 2. Load frames from f and compute the results */
	for (int i = 0; i < f.size(); i++)
	{
		/* 2.1 Compute dpra per-frame */
		dpra_per_frame(f[i], deltaPhi, dTime_per_Frame);
		
		/* 2.2 Accumulate the time cost */
		time += dTime_per_Frame;

		/* 2.3 Enqueue the results */
		dPhi_Sum.push_back(deltaPhi);

		/* 2.4 Update the reference image evnery iteration */
		if(i % m_rr ==0)
			update_ref_phi();
	}
}

void DPRA_HYBRID::operator() (const std::vector<std::string> &fileNames,
							   std::vector<std::vector<double>> &dPhi_Sum,
							   double &time)
{
	/* 1. create per-frame parmeters */
	std::vector<double> deltaPhi(m_iWidth*m_iHeight, 0);
	cv::Mat f;

	time = 0;
	double dTime_per_Frame = 0;;

	/* 2. Load frames from f and compute the results */
	for (int i = 0; i < fileNames.size(); i++)
	{
		/* 2.0 Load image from files specified by fileNames */
		f = cv::imread(fileNames[i]);
		cv::cvtColor(f, f, CV_BGR2GRAY);

		/* 2.1 Compute dpra per-frame */
		dpra_per_frame(f, deltaPhi, dTime_per_Frame);
		
		/* 2.2 Accumulate the time cost */
		time += dTime_per_Frame;

		/* 2.3 Enqueue the results */
		dPhi_Sum.push_back(deltaPhi);

		/* 2.4 Update the reference image evnery iteration */
		if(i % m_rr ==0)
			update_ref_phi();
	}
}

void DPRA_HYBRID::dpra_per_frame(const cv::Mat &img, 
								  std::vector<double> &dPhi,
								  double &time)
{
	int iSize = m_iWidth * m_iHeight;
	int iPaddedSize = m_iPaddedWidth * m_iPaddedHeight;

	/* I/O */
	memcpy(m_h_img, img.data, sizeof(uchar)*iSize);

	/* -------------------- Hybrid CPU and GPU DPRA algorithm ----------------------- */
	/* 1. Load the image f into device padded memory */
	cudaEventRecord(m_d_event_start);

	checkCudaErrors(cudaMemcpyAsync(m_d_img, m_h_img, sizeof(uchar)*iSize, cudaMemcpyHostToDevice));
	
	load_img_padding(m_d_img_Padded, m_d_img, m_iWidth, m_iHeight, m_iPaddedWidth, m_iPaddedHeight, m_blocks_2D, m_threads2D);

	cudaEventRecord(m_d_event_1);

	/* 2. construct matrix A and vector b on GPU */

	compute_cosPhi_sinPhi(m_d_cosPhi, m_d_sinPhi, m_d_PhiRef, m_iWidth, m_iHeight, m_iPaddedWidth, m_iPaddedHeight, m_blocks_2D, m_threads2D);

	get_A_b(m_d_A, m_d_b, m_d_img_Padded, m_d_cosPhi, m_d_sinPhi, m_iWidth, m_iHeight, m_iPaddedWidth, m_iPaddedHeight, m_blocks_2Dshrunk, m_threads2D);

	/* 3. copy A and b from device to host */
	checkCudaErrors(cudaMemcpyAsync(m_h_A, m_d_A, sizeof(double)*iSize * 9, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpyAsync(m_h_b, m_d_b, sizeof(double)*iSize * 3, cudaMemcpyDeviceToHost));

	cudaEventRecord(m_d_event_2);
	cudaEventSynchronize(m_d_event_2);

	/* 4. Solve Ax = b on host */
	double d_4_start = omp_get_wtime();

	#pragma omp parallel num_threads(m_iNumThreads)
	{
		int nthreads = omp_get_num_threads();
		int idThread = omp_get_thread_num();

		for (int i = idThread; i < iSize; i += nthreads)
		{
			int idA = i * 9;
			int idb = i * 3;

			MKL_INT ipiv[3];

			int infor = LAPACKE_dsysv(LAPACK_COL_MAJOR, 'U', 3, 1, m_h_A + idA, 3, ipiv, m_h_b + idb, 3);
			if (infor > 0)
			{
				printf("The leading minor of order %i is not positive ", infor);
				printf("definite;\nThe solution could not be computed for %d.\n", i);
			}

			// Update delta phi
			double fdeltaPhi = double(atan2(-m_h_b[idb + 2], m_h_b[idb + 1]));
			m_h_deltaPhiWFT[i].x = cos(fdeltaPhi);
			m_h_deltaPhiWFT[i].y = sin(fdeltaPhi);
		}
	}

	double d_4_end = omp_get_wtime();
	double d_4_time = 1000 * (d_4_end - d_4_start);
	
	
	/* 5. Copy the resulted phiWFT array to GPU to get the delta phi */	
	cudaEventRecord(m_d_event_3);

	checkCudaErrors(cudaMemcpyAsync(m_d_deltaPhiWFT, m_h_deltaPhiWFT, sizeof(cufftDoubleComplex)*iSize, cudaMemcpyHostToDevice));

	cudaEventRecord(m_d_event_4);

	/* 6. Run WFF on the device phiWFT */
	double d_6_time = 0;
	m_WFT(m_d_deltaPhiWFT, m_d_z, d_6_time);

	/* 7. Get the delta phi and current phi on device */
	cudaEventRecord(m_d_event_5);
	
	get_deltaPhi_currPhi(m_d_deltaPhi, m_d_PhiCurr, m_d_dPhiRef, m_d_PhiRef, m_d_z.m_d_filtered, iSize);

	cudaEventRecord(m_d_event_6);

	/* 8. Copy the delta Phi to host */
	checkCudaErrors(cudaMemcpyAsync(m_h_deltaPhi, m_d_deltaPhi, sizeof(double)*iSize, cudaMemcpyDeviceToHost));

	cudaEventRecord(m_d_event_7);
	cudaEventSynchronize(m_d_event_7);

	/* I/O */
	memcpy(dPhi.data(), m_h_deltaPhi, sizeof(double)*iSize);
	
	float f_1_time = 0;
	cudaEventElapsedTime(&f_1_time, m_d_event_start, m_d_event_1);

	float f_23_time = 0;
	cudaEventElapsedTime(&f_23_time, m_d_event_1, m_d_event_2);

	float f_5_time = 0;
	cudaEventElapsedTime(&f_5_time, m_d_event_3, m_d_event_4);

	float f_7_time = 0;
	cudaEventElapsedTime(&f_7_time, m_d_event_5, m_d_event_6);

	float f_8_time = 0;
	cudaEventElapsedTime(&f_8_time, m_d_event_6, m_d_event_7);

	std::cout << "Step 1 running time is: " << f_1_time << "ms" << std::endl;
	std::cout << "Step 2&3 running time is: " << f_23_time << "ms" << std::endl;
	std::cout << "Step 4 running time is: " << d_4_time << "ms" << std::endl;
	std::cout << "Step 5 running time is: " << f_5_time << "ms" << std::endl;
	std::cout << "Step 6 running time is: " << d_6_time << "ms" << std::endl;
	std::cout << "Step 7 running time is: " << f_7_time << "ms" << std::endl;
	std::cout << "Step 8 running time is: " << f_8_time << "ms" << std::endl;

	time = double(f_1_time + f_23_time + d_4_time + f_5_time + f_5_time + d_6_time + f_7_time + f_8_time);
}

void DPRA_HYBRID::update_ref_phi()
{
	checkCudaErrors(cudaMemcpyAsync(m_d_PhiRef, m_d_PhiCurr, sizeof(double)*m_iWidth*m_iHeight, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpyAsync(m_d_dPhiRef, m_d_deltaPhi, sizeof(double)*m_iWidth*m_iHeight, cudaMemcpyDeviceToDevice));
}

}	// namespace DPRA