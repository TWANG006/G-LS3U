#include "dpra_cudaf.h"
#include "mem_manager.h"

namespace DPRA{

DPRA_CUDAF::DPRA_CUDAF(const float *d_Phi0,
					   const int iWidth, const int iHeight,
					   const int irefUpdateRate)
	: m_iWidth(m_iWidth)
	, m_iHeight(m_iHeight)
	, m_rr(irefUpdateRate)
	, m_d_PhiRef(nullptr)
	, m_d_PhiCurr(nullptr)
	, m_d_csrValA(nullptr)
	, m_d_csrRowPtrA(nullptr)
	, m_d_csrColIndA(nullptr)
	, m_d_b(nullptr)
	, m_d_WFT(iWidth, iHeight, WFT_FPA::WFT::WFT_TYPE::WFF, m_d_z, 1)
	, m_d_deltaPhi_WFT(nullptr)
{
	int iSize = m_iWidth * m_iHeight;

	// Copy the d_Phi0 to local device array
	checkCudaErrors(cudaMalloc((void**)&m_d_PhiRef, sizeof(float)*iSize));
	checkCudaErrors(cudaMemcpy(m_d_PhiRef, d_Phi0, sizeof(float)*iSize, cudaMemcpyDeviceToDevice));

	// Allocate memory
	checkCudaErrors(cudaMalloc((void**)&m_d_PhiCurr, sizeof(float)*iSize));
	checkCudaErrors(cudaMalloc((void**)&m_d_csrValA, sizeof(float) * 9 * iSize));
	checkCudaErrors(cudaMalloc((void**)&m_d_csrRowPtrA, sizeof(float) * (3 * iSize + 1)));
	checkCudaErrors(cudaMalloc((void**)&m_d_csrColIndA, sizeof(float) * 9 * iSize));
	checkCudaErrors(cudaMalloc((void**)&m_d_b, sizeof(float) * 3 * iSize));

	// Allocate WFF phase memory
	checkCudaErrors(cudaMalloc((void**)&m_d_deltaPhi_WFT, sizeof(cufftComplex) * iSize));
}

DPRA_CUDAF::~DPRA_CUDAF()
{
	WFT_FPA::Utils::cudaSafeFree(m_d_PhiRef);
	WFT_FPA::Utils::cudaSafeFree(m_d_PhiCurr);
	WFT_FPA::Utils::cudaSafeFree(m_d_csrValA);
	WFT_FPA::Utils::cudaSafeFree(m_d_csrRowPtrA);
	WFT_FPA::Utils::cudaSafeFree(m_d_csrColIndA);
	WFT_FPA::Utils::cudaSafeFree(m_d_b);
	WFT_FPA::Utils::cudaSafeFree(m_d_deltaPhi_WFT);
}

}	// namespace DPRA