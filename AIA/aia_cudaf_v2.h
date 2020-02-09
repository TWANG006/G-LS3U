#ifndef AIA_CUDAF_YC2_H
#define AIA_CUDAF_YC2_H

#include "WFT-FPA.h"
#include <vector>
#include "opencv2\opencv.hpp"

#include "cuda_runtime.h"
#include "cusparse.h"
#include "cusolverSp.h"
#include "helper_cuda.h"
#include "mem_manager.h"

namespace AIA {
	class WFT_FPA_DLL_EXPORTS AIA_CUDAF_YC2
	{
	public:
		AIA_CUDAF_YC2() = delete;
		AIA_CUDAF_YC2(const AIA_CUDAF_YC2&) = delete;
		AIA_CUDAF_YC2 &operator=(const AIA_CUDAF_YC2&) = delete;

		AIA_CUDAF_YC2(const std::vector<cv::Mat>& v_f);
		AIA_CUDAF_YC2(const int iM,
			const int icols,
			const int irows);
		~AIA_CUDAF_YC2();

		void operator() (// Outputs
			std::vector<float>& v_phi,
			std::vector<float>& v_deltas,
			double &runningtime,
			int &iters,
			float &err,
			// Inputs
			const std::vector<cv::Mat>& v_f,
			int iMaxIterations = 20,
			float dMaxErr = 1e-4,
			int iNumThreads = 1);

	private:
		void computePhi_YC2();
		void computeDelta_YC2();
		float computeMaxError(const float *v_delta,
			const float *v_deltaOld,
			int m);

	private:
		uchar *m_d_img;			// Images in devcie mem space

		// CUDA sparse solver
		cusolverSpHandle_t m_cuSolverHandle;
		cusparseMatDescr_t m_desrA;

		// CSR format for cuSolver
		float *m_d_csrValA1;
		int *m_d_csrColIndA1;
		int *m_d_csrRowPtrA1;
		float *m_d_b1;
		float *m_d_phi;
		float *m_d_A2temp;		// Host-alloc
		float *m_d_A2;
		float *m_d_b2;

		float *m_d_delta;
		float *m_h_delta;		// Host-alloc
		float *m_h_old_delta;
		float *m_h_A2temp;		// Host-alloc
		float *m_h_A2;
		float *m_h_b2;			// Host-alloc

		int m_cols;
		int m_rows;
		int m_N;	// Number of pixels	
		int m_M;	// Number of frames
	};
}	//	namespace AIA
#endif // !AIA_CUDAF_H
#pragma once
