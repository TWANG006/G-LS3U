#ifndef DPRA_CUDAF_H
#define DPRA_CUDAF_H

#include "WFT-FPA.h"
#include "WFT2_CUDAf.h"
#include "cuda_runtime.h"
#include "cuSparse.h"
#include "cusolverSp.h"
#include "helper_cuda.h"

namespace DPRA{

class WFT_FPA_DLL_EXPORTS DPRA_CUDAF
{
public:
	DPRA_CUDAF() = delete;
	DPRA_CUDAF(const DPRA_CUDAF&) = delete;
	DPRA_CUDAF& operator=(const DPRA_CUDAF&) = delete;

	DPRA_CUDAF();

	~DPRA_CUDAF();

};

}	// namespace DPRA

#endif // !DPRA_CUDAF_H
