#include "gtest\gtest.h"
#include "Utils.h"
#include "WFT2_CUDAf.h"

TEST(WFF2_CUDA_Computation_Single, WFF2_CUDA_Computation)
{

	/* Load the Fringe Pattern */
	cufftComplex *f = nullptr;
	std::ifstream in("1024.fp");
	int rows, cols;
	if (!WFT_FPA::Utils::cufftComplexMatRead2D(in, f, rows, cols))
		std::cout << "load error" << std::endl;
	std::cout << rows << ", " << cols << std::endl;
	in.close();

	cufftComplex *df;
	cudaMalloc((void**)&df, sizeof(cufftComplex)*cols*rows);
	cudaMemcpy(df, f, sizeof(cufftComplex)*cols*rows, cudaMemcpyHostToDevice);


	WFT_FPA::WFT::WFT2_DeviceResultsF z;
	WFT_FPA::WFT::WFT2_CUDAF cuwft(
		cols, rows,
		WFT_FPA::WFT::WFT_TYPE::WFF,
		z);

	double time = 0;

	cuwft(df,z,time);

	std::cout << "Thres is : " << cuwft.m_rThr << std::endl;
	std::cout << "Time is: " << time << std::endl;

	cufftComplex *h_fPadded = (cufftComplex*)malloc(sizeof(cufftComplex) * cuwft.m_iPaddedWidth * cuwft.m_iPaddedHeight);

	checkCudaErrors(cudaMemcpy(h_fPadded, cuwft.m_d_fPadded, sizeof(cufftComplex) * cuwft.m_iPaddedWidth * cuwft.m_iPaddedHeight, cudaMemcpyDeviceToHost));

	std::ofstream out("device_Padded_f.csv", std::ios::out | std::ios::trunc);

	for(int i=0; i<cuwft.m_iPaddedHeight; i++)
	{
		for(int j=0; j<cuwft.m_iPaddedWidth; j++)
		{
			out << h_fPadded[i * cuwft.m_iPaddedWidth + j].x << "+" << h_fPadded[i * cuwft.m_iPaddedWidth + j].y << "i" << ",";
		}
		out<<"\n";
	}

	free(f);
	free(h_fPadded);
	cudaFree(df);
}