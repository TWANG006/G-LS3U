#include "gtest\gtest.h"

#include "WFT2_CUDAf.h"

#include <iostream>
#include <fstream>

using namespace std;

TEST(WFR2_Computation, WFR2_Computation)
{
	/* Load the Fringe Pattern */
	cufftComplex *f = nullptr;
	std::ifstream in("256.fp");
	int rows, cols;
	if (!WFT_FPA::Utils::cufftComplexMatRead2D(in, f, rows, cols))
		std::cout << "load error" << std::endl;
	std::cout << rows << ", " << cols << std::endl;
	in.close();

	cufftComplex *df;
	cudaMalloc((void**)&df, sizeof(cufftComplex)*cols*rows);
	cudaMemcpy(df, f, sizeof(cufftComplex)*cols*rows, cudaMemcpyHostToDevice);

	WFT_FPA::WFT::WFT2_DeviceResultsF z;

	WFT_FPA::WFT::WFT2_CUDAF wfr(
		cols, rows, 
		WFT_FPA::WFT::WFT_TYPE::WFR, 
		10.0f, -1.0f, 1.0f, 0.025f, 10.0f, -1.0f, 1.0f, 0.025f, -1, 
		z, 1);

	double time = 0;

	wfr(df, z, time);
	
	float *z_r = (float*)malloc(sizeof(float)*cols*rows);

	cudaMemcpy(z_r, z.m_d_phase_comp, sizeof(float)*cols*rows, cudaMemcpyDeviceToHost);

	ofstream out("z_r_phase_comp.csv", std::ios::out | std::ios::trunc);

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			out << z_r[i * cols + j]<< ",";
		}
		out << "\n";
	}
	out.close();


	cout << "WFR Running Time is: " << time << endl;

	free(f);
	cudaFree(df);
	free(z_r);
}