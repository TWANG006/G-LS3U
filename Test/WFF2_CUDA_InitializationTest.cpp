//#include "gtest\gtest.h"
//
//#include "WFT2_CUDA.h"
//#include "WFT2_CUDAf.h"
//
//TEST(WFF2_CUDA_Initialization_Single, WFF2_CUDA_Initialization)
//{
//	int iWidth = 132;
//	int iHeight = 132;
//
//	WFT_FPA::WFT::WFT2_DeviceResultsF z;
//	WFT_FPA::WFT::WFT2_CUDAF cuwft(
//		iWidth, iHeight,
//		WFT_FPA::WFT::WFT_TYPE::WFF,
//		z);
//
//	cufftReal *hxf, *hyf;
//	hxf = (cufftReal*)malloc(sizeof(cufftReal) * cuwft.m_iPaddedWidth * cuwft.m_iPaddedHeight);
//	hyf = (cufftReal*)malloc(sizeof(cufftReal) * cuwft.m_iPaddedWidth * cuwft.m_iPaddedHeight);
//
//	checkCudaErrors(cudaMemcpy(hxf, cuwft.m_d_xf, sizeof(cufftReal) * cuwft.m_iPaddedWidth * cuwft.m_iPaddedHeight, cudaMemcpyDeviceToHost));
//	checkCudaErrors(cudaMemcpy(hyf, cuwft.m_d_yf, sizeof(cufftReal) * cuwft.m_iPaddedWidth * cuwft.m_iPaddedHeight, cudaMemcpyDeviceToHost));
//
//	std::ofstream out("xfyf_FromInit.csv", std::ios::out | std::ios::trunc);
//
//	for(int i=0; i<cuwft.m_iPaddedHeight; i++)
//	{
//		for(int j=0; j<cuwft.m_iPaddedWidth; j++)
//		{
//			out << "[" << hxf[i * cuwft.m_iPaddedWidth + j] << "-" << hyf[i * cuwft.m_iPaddedWidth + j] << "]" << ",";
//		}
//		out<<"\n";
//	}
//
//	out.close();
//
//	free(hxf);
//	free(hyf);
//}
//
//
//TEST(WFF2_CUDA_Initialization_Double, WFF2_CUDA_Initialization)
//{
//	int iWidth = 132;
//	int iHeight = 132;
//
//	WFT_FPA::WFT::WFT2_DeviceResults z;
//	WFT_FPA::WFT::WFT2_CUDA cuwft(
//		iWidth, iHeight,
//		WFT_FPA::WFT::WFT_TYPE::WFF,
//		z);
//
//	cufftDoubleReal *hxf, *hyf;
//	hxf = (cufftDoubleReal*)malloc(sizeof(cufftDoubleReal) * cuwft.m_iPaddedWidth * cuwft.m_iPaddedHeight);
//	hyf = (cufftDoubleReal*)malloc(sizeof(cufftDoubleReal) * cuwft.m_iPaddedWidth * cuwft.m_iPaddedHeight);
//
//	checkCudaErrors(cudaMemcpy(hxf, cuwft.m_d_xf, sizeof(cufftDoubleReal) * cuwft.m_iPaddedWidth * cuwft.m_iPaddedHeight, cudaMemcpyDeviceToHost));
//	checkCudaErrors(cudaMemcpy(hyf, cuwft.m_d_yf, sizeof(cufftDoubleReal) * cuwft.m_iPaddedWidth * cuwft.m_iPaddedHeight, cudaMemcpyDeviceToHost));
//
//	std::ofstream out("xfyf_FromInit_double.csv", std::ios::out | std::ios::trunc);
//
//	for(int i=0; i<cuwft.m_iPaddedHeight; i++)
//	{
//		for(int j=0; j<cuwft.m_iPaddedWidth; j++)
//		{
//			out << "[" << hxf[i * cuwft.m_iPaddedWidth + j] << "-" << hyf[i * cuwft.m_iPaddedWidth + j] << "]" << ",";
//		}
//		out<<"\n";
//	}
//
//	out.close();
//
//	free(hxf);
//	free(hyf);
//}