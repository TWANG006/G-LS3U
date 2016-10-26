#include "gtest\gtest.h"

#include "aia_cpuf.h"
#include "dpra_cudaf.h"

TEST(DPRA_CUDAF_csrValA, DPRA_CUDA_Single)
{
		/* AIA to get the initial phi */
	std::vector<cv::Mat> f;

	cv::Mat img = cv::imread("1.bmp");
	cv::cvtColor(img,
				 img,
				 CV_BGR2GRAY);
	f.push_back(img);

	img = cv::imread("2.bmp");
	cv::cvtColor(img,
				 img,
				 CV_BGR2GRAY);
	f.push_back(img);

	img = cv::imread("3.bmp");
	cv::cvtColor(img,
				 img,
				 CV_BGR2GRAY);
	f.push_back(img);
	

	// computation
	std::vector<float> phi;
	std::vector<float> delta{0.5377f,
    1.8339f,
   -2.2588f};
	double time = 0;
	float err = 0;
	int iter = 0;

	AIA::AIA_CPU_DnF aia;
	aia(phi, delta, time, iter, err, f, 20, 1e-4, 12);

	std::cout << "AIA Running Time: " << time << std::endl;
	std::cout << "AIA Error is: " << err << std::endl;
	std::cout << "AIA Iteration is: " << iter << std::endl;

	std::cout << "Delta is: " << delta[0] << "," << delta[1] << "," << delta[2] << std::endl;

	float *dPhi0, *dphi_curr = nullptr;
	uchar *dImg;
	cudaMalloc((void**)&dPhi0, sizeof(float)*f[0].cols*f[0].rows);
	cudaMalloc((void**)&dImg, sizeof(uchar)*f[0].cols*f[0].rows);
	cudaMemcpy(dPhi0, phi.data(), sizeof(float)*f[0].cols*f[0].rows, cudaMemcpyHostToDevice);
	cudaMemcpy(dImg, f[0].data, sizeof(uchar)*f[0].cols*f[0].rows, cudaMemcpyHostToDevice);



	DPRA::DPRA_CUDAF dpra_cuda_f(dPhi0, f[0].cols, f[0].rows, 1);

	double ddtime = 0;

	dpra_cuda_f.dpra_per_frame(dImg, dphi_curr, ddtime);

	// Check padded refPhi
	float *h_padded_refPhie = (float*)malloc(sizeof(float)*dpra_cuda_f.m_iPaddedWidth*dpra_cuda_f.m_iPaddedHeight);

	checkCudaErrors(cudaMemcpy(h_padded_refPhie, dpra_cuda_f.m_d_PhiRef, sizeof(float)*dpra_cuda_f.m_iPaddedWidth*dpra_cuda_f.m_iPaddedHeight, cudaMemcpyDeviceToHost));

	std::ofstream out("h_padded_refPhi.csv", std::ios::out | std::ios::trunc);

	for (int i = 0; i < dpra_cuda_f.m_iPaddedHeight; i++)
	{
		for (int j = 0; j < dpra_cuda_f.m_iPaddedWidth; j++)
		{
			
			out << h_padded_refPhie[i * dpra_cuda_f.m_iPaddedWidth + j] << ",";
			
		}
		out<<"\n";
	}

	out.close();

	// Check padded cosPhi
	float *h_padded_cosPhi = (float*)malloc(sizeof(float)*dpra_cuda_f.m_iPaddedWidth*dpra_cuda_f.m_iPaddedHeight);

	checkCudaErrors(cudaMemcpy(h_padded_cosPhi, dpra_cuda_f.m_d_cosPhi, sizeof(float)*dpra_cuda_f.m_iPaddedWidth*dpra_cuda_f.m_iPaddedHeight, cudaMemcpyDeviceToHost));

	out.open("h_padded_cosPhi.csv", std::ios::out | std::ios::trunc);

	for (int i = 0; i < dpra_cuda_f.m_iPaddedHeight; i++)
	{
		for (int j = 0; j < dpra_cuda_f.m_iPaddedWidth; j++)
		{
			
			out << h_padded_cosPhi[i * dpra_cuda_f.m_iPaddedWidth + j] << ",";
			
		}
		out<<"\n";
	}

	out.close();

	// Check padded sinPhi
	float *h_padded_sinPhi = (float*)malloc(sizeof(float)*dpra_cuda_f.m_iPaddedWidth*dpra_cuda_f.m_iPaddedHeight);

	checkCudaErrors(cudaMemcpy(h_padded_sinPhi, dpra_cuda_f.m_d_sinPhi, sizeof(float)*dpra_cuda_f.m_iPaddedWidth*dpra_cuda_f.m_iPaddedHeight, cudaMemcpyDeviceToHost));

	out.open("h_padded_sinPhi.csv", std::ios::out | std::ios::trunc);

	for (int i = 0; i < dpra_cuda_f.m_iPaddedHeight; i++)
	{
		for (int j = 0; j < dpra_cuda_f.m_iPaddedWidth; j++)
		{
			
			out << h_padded_sinPhi[i * dpra_cuda_f.m_iPaddedWidth + j] << ",";
			
		}
		out<<"\n";
	}

	out.close();

	// Check padded img_padded
	uchar *h_img_padded = (uchar*)malloc(sizeof(uchar)*dpra_cuda_f.m_iPaddedWidth*dpra_cuda_f.m_iPaddedHeight);

	checkCudaErrors(cudaMemcpy(h_img_padded, dpra_cuda_f.m_d_img_Padded, sizeof(uchar)*dpra_cuda_f.m_iPaddedWidth*dpra_cuda_f.m_iPaddedHeight, cudaMemcpyDeviceToHost));

	out.open("h_img_padded.csv", std::ios::out | std::ios::trunc);

	for (int i = 0; i < dpra_cuda_f.m_iPaddedHeight; i++)
	{
		for (int j = 0; j < dpra_cuda_f.m_iPaddedWidth; j++)
		{
			
			out << (int)h_img_padded[i * dpra_cuda_f.m_iPaddedWidth + j] << ",";
			
		}
		out<<"\n";
	}

	out.close();



	float *h_csrValA = (float*)malloc(sizeof(float)*dpra_cuda_f.m_iImgWidth*dpra_cuda_f.m_iImgHeight * 9);
	float *h_b = (float*)malloc(sizeof(float)*dpra_cuda_f.m_iImgWidth*dpra_cuda_f.m_iImgHeight * 3);

	checkCudaErrors(cudaMemcpy(h_csrValA, dpra_cuda_f.m_d_csrValA, sizeof(float)*dpra_cuda_f.m_iImgWidth*dpra_cuda_f.m_iImgHeight * 9, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_b, dpra_cuda_f.m_d_b, sizeof(float)*dpra_cuda_f.m_iImgWidth*dpra_cuda_f.m_iImgHeight * 3, cudaMemcpyDeviceToHost));

	out.open("csrA_b.csv", std::ios::out | std::ios::trunc);

	for (int i = 0; i < dpra_cuda_f.m_iImgHeight*dpra_cuda_f.m_iImgWidth; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				out << h_csrValA[i * 9 + j * 3 + k]<<",";
			}
			out << h_b[i * 3 + j] << "\n";
		}
		out<<"\n";
	}

	out.close();

	free(h_img_padded);
	free(h_csrValA);
	free(h_b);
	free(h_padded_cosPhi);
	free(h_padded_sinPhi);
	free(h_padded_refPhie);
	cudaFree(dImg);
	cudaFree(dPhi0);
}