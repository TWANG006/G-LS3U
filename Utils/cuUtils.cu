#include "Utils.h"

namespace WFT_FPA{
namespace Utils{

template<typename T>
__global__ void initKernel(T * devPtr, 
						   const T val, 
						   const size_t nwords)
{
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(; tidx < nwords; tidx += stride)
        devPtr[tidx] = val;
}


template<typename T>
void cuInitialize(T* devPtr, 
				  const T val, 
				  const size_t nwords)
{
	initKernel<T><<<256, 64>>>(devPtr, val, nwords);
}

template WFT_FPA_DLL_EXPORTS void cuInitialize<float>(float *devPtr, const float val, const size_t nwords);
template WFT_FPA_DLL_EXPORTS void cuInitialize<double>(double *devPtr, const double val, const size_t nwords);
template WFT_FPA_DLL_EXPORTS void cuInitialize<int>(int *devPtr, const int val, const size_t nwords);
template WFT_FPA_DLL_EXPORTS void cuInitialize<unsigned int>(unsigned int *devPtr, const unsigned int val, const size_t nwords);
template WFT_FPA_DLL_EXPORTS void cuInitialize<uchar1>(uchar1 *devPtr, const uchar1 val, const size_t nwords);
template WFT_FPA_DLL_EXPORTS void cuInitialize<uchar2>(uchar2 *devPtr, const uchar2 val, const size_t nwords);
template WFT_FPA_DLL_EXPORTS void cuInitialize<uchar3>(uchar3 *devPtr, const uchar3 val, const size_t nwords);
template WFT_FPA_DLL_EXPORTS void cuInitialize<uchar4>(uchar4 *devPtr, const uchar4 val, const size_t nwords);

}	//	namespace Utils
}	//	namespace WFT-FPA