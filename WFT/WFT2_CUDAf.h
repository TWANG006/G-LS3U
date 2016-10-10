#ifndef WFT2_CUDA_H
#define WFT2_CUDA_H

#include "WFT.h"


namespace WFT_FPA{
namespace WFT{

class WFT_FPA_DLL_EXPORTS WFT2_CUDAf 
{
public:

	// Default parameters are used based on the WFT_TYPE
	WFT2_CUDAf(
		int iWidth, int iHeight,
		WFT_TYPE type,
		WFT2_DeviceResultsF& z,
		int iNumberThreads = 1);

	// Parameters are set by the input parameters
	WFT2_CUDAf(
		int iWidth, int iHeight,
		WFT_TYPE type,
		float rSigmaX, float rWxl, float rWxh, float rWxi,
		float rSigmaY, float rWyl, float rWyh, float rWyi,
		float rThr,
		WFT2_DeviceResultsF &z,
		int iNumberThreads = 1);

	~WFT2_CUDAf();

	// Make this class a callable object (functor)
	void operator() (
		cufftComplex *f, 
		WFT2_DeviceResultsF &z,
		double &time);

private:
	/* Initilaize the WFT2 algorithm 
		1. Calculate the padding size.
		2. Allocate&initialize arrays with the padded size
		3. Make cufft plans								   */
	int WFT2_cuInitialize(WFT2_DeviceResultsF &z);
	void WFF2_cuInit(WFT2_DeviceResultsF &z);
	int  WFR2_cuInit(WFT2_DeviceResultsF &z);
};

}	// namespace WFT_FPA
}	// namespace WFT

#endif // !WFT2_CUDA_H
