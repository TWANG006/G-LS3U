#include "WFT2_CUDAf.h"

namespace WFT_FPA{
namespace WFT{

WFT2_CUDAf::WFT2_CUDAf(
	int iWidth, int iHeight,
	WFT_TYPE type,
	WFT2_DeviceResultsF& z,
	int iNumberThreads)
{}

WFT2_CUDAf::WFT2_CUDAf(
		int iWidth, int iHeight,
		WFT_TYPE type,
		float rSigmaX, float rWxl, float rWxh, float rWxi,
		float rSigmaY, float rWyl, float rWyh, float rWyi,
		float rThr,
		WFT2_DeviceResultsF &z,
		int iNumberThreads)
{

}

WFT2_CUDAf::~WFT2_CUDAf()
{

}

void WFT2_CUDAf::operator()(
	cufftComplex *f,
	WFT2_DeviceResultsF &z,
	double &time)
{

}

}	// namespace WFT_FPA
}	// namespace WFT