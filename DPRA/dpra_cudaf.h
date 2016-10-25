#ifndef DPRA_CUDAF_H
#define DPRA_CUDAF_H

#include "WFT-FPA.h"
#include "WFT2_CUDAf.h"

#include <string>
#include <vector>

#include "cuda_runtime.h"
#include "cuSparse.h"
#include "cusolverSp.h"
#include "helper_cuda.h"

#include "opencv2\opencv.hpp"

namespace DPRA{

class WFT_FPA_DLL_EXPORTS DPRA_CUDAF
{
public:
	DPRA_CUDAF() = delete;
	DPRA_CUDAF(const DPRA_CUDAF&) = delete;
	DPRA_CUDAF& operator=(const DPRA_CUDAF&) = delete;

	DPRA_CUDAF(const float *d_Phi0,
			   const int iWidth, const int iHeight,
			   const int irefUpdateRate =1);
	~DPRA_CUDAF();

public:
	void operator() (const std::vector<cv::cuda::HostMem> &f,
					 std::vector<std::vector<float>> &dPhi_Sum,
					 double &time);
	void operator() (const std::vector<std::string> &fileNames,
					 std::vector<std::vector<float>> &dPhi_Sum,
					 double &time);

	void dpra_per_frame(const cv::cuda::HostMem &f,
						std::vector<float> &dPhi,
						double &time);

	void update_ref_phi();

private:
	int m_iWidth;
	int m_iHeight;
	int m_rr;
	
	float *m_d_PhiRef;
	float *m_d_PhiCurr;

	cusparseMatDescr_t m_desrA;
	cusolverSpHandle_t m_cuSolverHandle;

	float *m_d_csrValA;
	float *m_d_csrRowPtrA;
	float *m_d_csrColIndA;
	float *m_d_b;

	WFT_FPA::WFT::WFT2_DeviceResultsF m_d_z;
	WFT_FPA::WFT::WFT2_CUDAF m_d_WFT;
	cufftComplex *m_d_deltaPhi_WFT;
};

}	// namespace DPRA

#endif // !DPRA_CUDAF_H
