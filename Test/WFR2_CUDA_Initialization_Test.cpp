#include "gtest\gtest.h"

#include "WFT2_CUDAf.h"

#include <iostream>
#include <fstream>

using namespace std;

TEST(WFR2_g_xg_yg, WFR2_Initialization)
{
	WFT_FPA::WFT::WFT2_DeviceResultsF z;

	WFT_FPA::WFT::WFT2_CUDAF wfr(
		132, 132, 
		WFT_FPA::WFT::WFT_TYPE::WFR,
		z);

	/*cufftReal *h_g = (cufftReal*)malloc(sizeof(float) * 61 * 61);

	cudaMemcpy(h_g, wfr.im_d_g, sizeof(float) * 61 * 61, cudaMemcpyDeviceToHost);
	
	
	ofstream out("g.csv", std::ios::out | std::ios::trunc);

	for (int i = 0; i < 61; i++)
	{
		for (int j = 0; j < 61; j++)
		{
			out << h_g[i * 61 + j]<< ",";
		}
		out << "\n";
	}
	out.close();

	free(h_g);*/

	cout <<wfr.m_rxxg_norm2<<", "<<wfr.m_ryyg_norm2<<", " << endl;
}