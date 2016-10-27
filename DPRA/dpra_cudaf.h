#ifndef DPRA_CUDAF_H
#define DPRA_CUDAF_H

#include "WFT-FPA.h"
#include "WFT2_CUDAf.h"
#include "opencv2\opencv.hpp"

#include <string>
#include <vector>

#include "cuda_runtime.h"
#include "cusparse.h"
#include "cusolverSp.h"
#include "helper_cuda.h"
#include "mem_manager.h"


// TODO
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

	void dpra_per_frame(const uchar *d_imag,
						float *&d_dPhi,
						double &time);

	void update_ref_phi();

public:
	int m_iImgWidth;
	int m_iImgHeight;
	int m_iPaddedWidth;
	int m_iPaddedHeight;

	int m_rr;
	
	float *m_d_PhiRef;		// m_iWidth + 2, m_iHeight + 2
	float *m_d_PhiCurr;		// m_iWidth +2, m_iHeight +2

	cusparseMatDescr_t m_desrA;
	cusolverSpHandle_t m_cuSolverHandle;

	float *m_d_cosPhi;		// padded
	float *m_d_sinPhi;		// padded
	uchar *m_d_img_Padded;	// padded

	float *m_d_csrValA;
	int *m_d_csrRowPtrA;
	int *m_d_csrColIndA;
	float *m_d_b;

	WFT_FPA::WFT::WFT2_DeviceResultsF m_d_z;
	WFT_FPA::WFT::WFT2_CUDAF m_d_WFT;
	cufftComplex *m_d_deltaPhi_WFT;
};

}	// namespace DPRA

#endif // !DPRA_CUDAF_H
