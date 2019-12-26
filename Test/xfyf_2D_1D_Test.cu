#include "gtest\gtest.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cufft.h"
#include "helper_cuda.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <fstream>
#include <opencv2\opencv.hpp>
#include "aia_cpuf.h"
#include "dpra_cudaf.h"
#include "cuda_testt.h"

__global__ void Gen_xf_yf_Kernel2D(
	cufftReal *xf, cufftReal *yf, 
	int iWidth, int iHeight)
{
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	int id = i*iWidth + j;
	int iH = iHeight / 2;
	int iW = iWidth / 2;

	if (i < iHeight && j < iWidth)
	{
		xf[id] = j - iW;
		yf[id] = i - iH;
	}
}
__global__ void Gen_xf_yf_Kernel1D(
	cufftReal *xf, cufftReal *yf, 
	int iWidth, int iHeight)
{
	int iH = iHeight / 2;
	int iW = iWidth / 2;

	for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < iWidth * iHeight; i += blockDim.x*gridDim.x)
	{
		int x = i % iWidth;
		int y = i / iWidth;
		
		if (y < iHeight && x < iWidth)
		{
			xf[i] = x - iW;
			yf[i] = y - iH;
		}
	}
}
__global__ void gen_xf_yf_Kernel(cufftReal *xf, cufftReal *yf, int iWidth, int iHeight)
{
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	int j = threadIdx.x + blockIdx.x * blockDim.x;

	int id = i*iWidth + j;

	int iHalfy = iHeight / 2;
	int iHalfx = iWidth / 2;

	if (i < iHeight && j < iWidth)
	{
		xf[id] = j - iHalfx;
		yf[id] = i - iHalfy;
	}
	
}
__global__ void shift_xf_yf_Kernel(cufftReal *xf, cufftReal *yf,  int iWidth, int iHeight)
{
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	int j = threadIdx.x + blockIdx.x * blockDim.x;

	int id = i*iWidth + j;

	int iHalfx = iWidth / 2;
	int iHalfy = iHeight / 2;
	int iSlice = iWidth * iHeight;

	int idQ13 = iSlice / 2 + iHalfx;
	int idQ24 = iSlice / 2 - iHalfx;

	cufftReal Tempx, Tempy;

	if (j < iHalfx)
	{
		if(i < iHalfy)
		{
			Tempx = xf[id];
			Tempy = yf[id];

			// First Quadrant
			xf[id] = xf[id + idQ13];
			yf[id] = yf[id + idQ13];

			// Third Quadrant
			xf[id + idQ13] = Tempx;
			yf[id + idQ13] = Tempy;
		}
	}
	else
	{
		if (i < iHalfy)
		{
			Tempx = xf[id];
			Tempy = yf[id];

			// Second Quadrant
			xf[id] = xf[id + idQ24];
			yf[id] = yf[id + idQ24];

			// Fourth Quadrant
			xf[id + idQ24] = Tempx;
			yf[id + idQ24] = Tempy;
		}
	}
}
__global__
void compute_Fg_kernel(cufftReal *d_in_xf, cufftReal *d_in_yf, int iPaddedWidth, int iPaddedHeight, 
					   int wxt, int wyt, float wxi, float wyi, float wxl, float wyl,
					   float sigmax, float sigmay, float sn2, cufftComplex *d_out_Fg)
{
	cufftReal rwxt = wxl + cufftReal(wxt) * wxi;
	cufftReal rwyt = wyl + cufftReal(wyt) * wyi;

	for (int i = threadIdx.x + blockIdx.x * blockDim.x;
		 i < iPaddedHeight*iPaddedWidth;
		 i += blockDim.x * gridDim.x)
	{
		cufftReal tempx = d_in_xf[i] * 2 * cufftReal(M_PI) * (1.0f / iPaddedWidth) - rwxt;
		cufftReal tempy = d_in_yf[i] * 2 * cufftReal(M_PI) * (1.0f / iPaddedHeight) - rwyt;
		tempx = -tempx * tempx * 0.5f * sigmax * sigmax;
		tempy = -tempy * tempy * 0.5f * sigmay * sigmay;
		
		d_out_Fg[i].x = exp(tempx + tempy)*sn2;
		d_out_Fg[i].y = 0;
	}
}

TEST(XF_YF_2D_1D, KernelTest)
{
	cudaEvent_t start, end2D, end1D;
	cudaEventCreate(&start);
	cudaEventCreate(&end1D);
	cudaEventCreate(&end2D);

	dim3 threads(16, 16);
	dim3 blocks((1120 + 16 - 1) / 16, (1120 + 16 - 1) / 16);

	cufftReal *xf, *xf1, *yf, *yf1;
	checkCudaErrors(cudaMalloc((void**)&xf, sizeof(cufftReal) * 1120 * 1120));
	checkCudaErrors(cudaMalloc((void**)&yf, sizeof(cufftReal) * 1120 * 1120));
	checkCudaErrors(cudaMalloc((void**)&xf1, sizeof(cufftReal) * 1120 * 1120));
	checkCudaErrors(cudaMalloc((void**)&yf1, sizeof(cufftReal) * 1120 * 1120));

	cudaEventRecord(start);
	Gen_xf_yf_Kernel1D <<<32 * 8, 256 >> >(xf1, yf1, 1120, 1120);
	getLastCudaError("1D Kernel Failed.");
	cudaEventRecord(end1D);
	Gen_xf_yf_Kernel2D<<<blocks, threads>>>(xf, yf, 1120, 1120);
	getLastCudaError("2D Kernel Failed.");
	cudaEventRecord(end2D);

	cudaDeviceSynchronize();


	cufftReal *hxf, *hxf1, *hyf, *hyf1;
	hxf = (cufftReal*)malloc(sizeof(cufftReal) * 1120 * 1120);
	hyf = (cufftReal*)malloc(sizeof(cufftReal) * 1120 * 1120);
	hxf1 = (cufftReal*)malloc(sizeof(cufftReal) * 1120 * 1120);
	hyf1 = (cufftReal*)malloc(sizeof(cufftReal) * 1120 * 1120);

	checkCudaErrors(cudaMemcpy(hxf, xf, sizeof(cufftReal) * 1120 * 1120, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(hyf, yf, sizeof(cufftReal) * 1120 * 1120, cudaMemcpyDeviceToHost));
	
	checkCudaErrors(cudaMemcpy(hxf1, xf1, sizeof(cufftReal) * 1120 * 1120, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(hyf1, yf1, sizeof(cufftReal) * 1120 * 1120, cudaMemcpyDeviceToHost));

	for(int i=0; i<1120; i++)
		for(int j=0; j<1120; j++)
		{
			ASSERT_EQ(hxf[i * 1120 + j], hxf1[i * 1120 + j]);
			ASSERT_EQ(hyf[i * 1120 + j], hyf1[i * 1120 + j]);
		}

	float t1, t2;
	cudaEventElapsedTime(&t1, start, end1D);
	cudaEventElapsedTime(&t2, end1D, end2D);

	std::cout << "1D Kernel Execution Time: " << t1 << std::endl;
	std::cout << "2D Kernel Execution Time: " << t2 << std::endl;

	cudaEventDestroy(start);
	cudaEventDestroy(end1D);
	cudaEventDestroy(end2D);
	cudaFree(xf);
	cudaFree(yf);
	cudaFree(xf1);
	cudaFree(yf1);
	free(hxf);
	free(hyf);
	free(hxf1);
	free(hyf1);
}
TEST(XF_YF_2D_with_FFTSHIFT, KernelTest)
{
	int iWidth = 64;
	int iHeight = 50;

	cudaEvent_t start, end2D;
	cudaEventCreate(&start);
	cudaEventCreate(&end2D);

	dim3 threads(16, 16);
	dim3 blocks((iWidth + 16 - 1) / 16, (iHeight + 16 - 1) / 16);
	dim3 blocks1((iWidth + 16 - 1) / 16, (iHeight + 16 - 1) / 16);

	cufftReal *xf, *yf;
	checkCudaErrors(cudaMalloc((void**)&xf, sizeof(cufftReal) * iWidth * iHeight));
	checkCudaErrors(cudaMalloc((void**)&yf, sizeof(cufftReal) * iWidth * iHeight));

	cudaEventRecord(start);
	gen_xf_yf_Kernel<<<blocks, threads>>>(xf, yf, iWidth, iHeight);
	getLastCudaError("2D Kernel Failed.");
	shift_xf_yf_Kernel<<<blocks1, threads>>>(xf, yf, iWidth, iHeight);
	getLastCudaError("2D Shift Failed.");
	cudaEventRecord(end2D);

	cudaDeviceSynchronize();


	cufftReal *hxf, *hyf;
	hxf = (cufftReal*)malloc(sizeof(cufftReal) * iWidth * iHeight);
	hyf = (cufftReal*)malloc(sizeof(cufftReal) * iWidth * iHeight);

	checkCudaErrors(cudaMemcpy(hxf, xf, sizeof(cufftReal) * iWidth * iHeight, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(hyf, yf, sizeof(cufftReal) * iWidth * iHeight, cudaMemcpyDeviceToHost));

	std::ofstream out("xfyf.csv", std::ios::out | std::ios::trunc);

	for(int i=0; i<iHeight; i++)
	{
		for(int j=0; j<iWidth; j++)
		{
			out << "[" << hxf[i * iWidth + j] << "-" << hyf[i * iWidth + j] << "]" << ",";
		}
		out<<"\n";
	}

	out.close();

	float t1;
	cudaEventElapsedTime(&t1, start, end2D);

	std::cout << "1D Kernel Execution Time: " << t1 << std::endl;

	cudaEventDestroy(start);
	cudaEventDestroy(end2D);
	cudaFree(xf);
	cudaFree(yf);

	free(hxf);
	free(hyf);

}
////TEST(Fg_Computation, KernelTest)
////{
////	int iWidth = 192;
////	int iHeight = 192;
////
////	dim3 threads(16, 16);
////	dim3 blocks((iWidth + 16 - 1) / 16, (iHeight + 16 - 1) / 16);
////	dim3 blocks1((iWidth + 16 - 1) / 16, (iHeight + 16 - 1) / 16);
////
////	cufftReal *xf, *yf;
////	cufftComplex *fg;
////	checkCudaErrors(cudaMalloc((void**)&xf, sizeof(cufftReal) * iWidth * iHeight));
////	checkCudaErrors(cudaMalloc((void**)&yf, sizeof(cufftReal) * iWidth * iHeight));
////	checkCudaErrors(cudaMalloc((void**)&fg, sizeof(cufftComplex) * iWidth * iHeight));
////
////	gen_xf_yf_Kernel<<<blocks, threads>>>(xf, yf, iWidth, iHeight);
////	getLastCudaError("gen_xf_yf_Kernel Kernel Failed.");
////
////	shift_xf_yf_Kernel<<<blocks1, threads>>>(xf, yf, iWidth, iHeight);
////	getLastCudaError("2D Shift Failed.");
////
////	compute_Fg_kernel<<<32*8, 256>>>(xf, yf, iWidth, iHeight, 0, 0, 0.1f, 0.1f, -2.3f, -2.3f, 10, 10, 35.4491f, fg);
////	getLastCudaError("compute_Fg_kernel Kernel Failed.");
////
////	cufftComplex *hfg;
////	cufftReal *hfx, *hfy;
////	hfg = (cufftComplex*)malloc(sizeof(cufftComplex) * iWidth * iHeight);
////	hfx = (cufftReal*)malloc(sizeof(cufftReal) * iWidth * iHeight);
////	hfy = (cufftReal*)malloc(sizeof(cufftReal) * iWidth * iHeight);
////
////	checkCudaErrors(cudaMemcpy(hfg, fg, sizeof(cufftComplex) * iWidth * iHeight, cudaMemcpyDeviceToHost));
////	checkCudaErrors(cudaMemcpy(hfx, xf, sizeof(cufftReal) * iWidth * iHeight, cudaMemcpyDeviceToHost));
////	checkCudaErrors(cudaMemcpy(hfy, yf, sizeof(cufftReal) * iWidth * iHeight, cudaMemcpyDeviceToHost));
////
////	std::ofstream out("CUDAFg.csv", std::ios::out | std::ios::trunc);
////
////	for(int i=0; i<iHeight; i++)
////	{
////		for(int j=0; j<iWidth; j++)
////		{
////			out << hfg[i * iWidth + j].x << "+" << hfg[i * iWidth + j].y << "i" << ",";
////		}
////		out<<"\n";
////	}
////
////	out.close();
////
////	out.open("xfyf.csv", std::ios::out | std::ios::trunc);
////
////	for(int i=0; i<iHeight; i++)
////	{
////		for(int j=0; j<iWidth; j++)
////		{
////			out << "[" << hfx[i * iWidth + j] << "-" << hfy[i * iWidth + j] << "]" << ",";
////		}
////		out<<"\n";
////	}
////
////	out.close();
////
////	cudaFree(xf);
////	cudaFree(yf);
////	cudaFree(fg);
////	free(hfg);
////	free(hfx);
////	free(hfy);
////}
//
//#define BLOCK_SIZE_16 16
//
//__global__
//void generate_csrValA_b_kernel(float *d_out_csrValA,
//							   float *d_out_b,
//							   const uchar *d_in_img,
//							   const float *d_in_phi,
//							   const int iWidth,
//							   const int iHeight)
//{
//	const int y = threadIdx.y + (BLOCK_SIZE_16 - 2) * blockIdx.y;
//	const int x = threadIdx.x + (BLOCK_SIZE_16 - 2) * blockIdx.x;
//
//	int idA = (y*iWidth + x) * 9;
//
//	float sum_cos = 0, sum_sin = 0, sum_sincos = 0, sum_sin2 = 0, sum_cos2 = 0;
//	float sum_ft = 0, sum_ft_cos = 0, sum_ft_sin = 0;
//
//	// Global Memory offset: every block actually begin with 2 overlapped pixels
//	__shared__ float phi_sh[BLOCK_SIZE_16][BLOCK_SIZE_16];
//	__shared__ uchar img_sh[BLOCK_SIZE_16][BLOCK_SIZE_16];
//
//	// Load the global mem to shared mem
//	if (y < iHeight && x < iWidth)
//	{
//		phi_sh[threadIdx.y][threadIdx.x] = d_in_phi[y*iWidth + x];
//		img_sh[threadIdx.y][threadIdx.x] = d_in_img[y*iWidth + x];
//	}
//	__syncthreads();	
//
//	if (y < iHeight && x < iWidth)
//	{
//		// Compute the results within the boundary
//		if (y >= 1 && y < iHeight - 1 && x >= 1 && x < iWidth - 1 &&
//			threadIdx.x != 0 && threadIdx.x != BLOCK_SIZE_16 - 1 &&
//			threadIdx.y != 0 && threadIdx.y != BLOCK_SIZE_16 - 1)
//		{
//			sum_cos = 0, sum_sin = 0, sum_sincos = 0, sum_sin2 = 0, sum_cos2 = 0;
//			sum_ft = 0, sum_ft_cos = 0, sum_ft_sin = 0;
//
//			for (int i = threadIdx.y - 1; i <= threadIdx.y + 1; i++)
//			{
//				for (int j = threadIdx.x - 1; j <= threadIdx.x + 1; j++)
//				{
//					float cos_phi = cos(phi_sh[i][j]);
//					float sin_phi = sin(phi_sh[i][j]);
//					float ft = static_cast<float>(img_sh[i][j]);
//
//					// Elements of A
//					sum_cos += cos_phi;
//					sum_sin += sin_phi;
//					sum_sincos += cos_phi * sin_phi;
//					sum_sin2 += sin_phi*sin_phi;
//					sum_cos2 += cos_phi*cos_phi;
//
//					// Elements of b
//					sum_ft += ft;
//					sum_ft_cos += ft * cos_phi;
//					sum_ft_sin += ft * sin_phi;
//				}
//			}
//			d_out_csrValA[idA + 0] = 9;			d_out_csrValA[idA + 1] = sum_cos;		d_out_csrValA[idA + 2] = sum_sin;
//			d_out_csrValA[idA + 3] = sum_cos;	d_out_csrValA[idA + 4] = sum_cos2;		d_out_csrValA[idA + 5] = sum_sincos;
//			d_out_csrValA[idA + 6] = sum_sin;	d_out_csrValA[idA + 7] = sum_sincos;	d_out_csrValA[idA + 8] = sum_sin2;
//		}
//		// Deal with boundary
//		if ((y == 0 && blockIdx.y == 0) ||
//		    (x == 0 && blockIdx.x == 0) ||
//			(y == iHeight - 1 && blockIdx.y == gridDim.y - 1) ||
//			(x == iWidth - 1 && blockIdx.x == gridDim.x - 1))
//		{
//			sum_cos = 0, sum_sin = 0, sum_sincos = 0, sum_sin2 = 0, sum_cos2 = 0;
//			sum_ft = 0, sum_ft_cos = 0, sum_ft_sin = 0;
//
//			int yl = -1, yh = 1, xl = -1, xh = 1;
//
//			if (y == 0)				
//			{
//				yl = 0;	yh = 1;
//			}
//			if (y == iHeight - 1)	
//			{
//				yl = -1;	yh = 0;
//			}
//			if (x == 0)		
//			{
//				xl = 0;	xh = 1;
//			}
//			if (x == iWidth - 1)
//			{
//				xl = -1;	xh = 0;
//			}
//
//			for (int i = yl; i <= yh; i++)
//			{
//				for (int j = xl; j <= xh; j++)
//				{
//					float cos_phi = cos(phi_sh[threadIdx.y + i][threadIdx.x + j]);
//					float sin_phi = sin(phi_sh[threadIdx.y + i][threadIdx.x + j]);
//					float ft = static_cast<float>(img_sh[threadIdx.y + i][threadIdx.x + j]);
//
//					// Elements of A
//					sum_cos += cos_phi;
//					sum_sin += sin_phi;
//					sum_sincos += cos_phi * sin_phi;
//					sum_sin2 += sin_phi*sin_phi;
//					sum_cos2 += cos_phi*cos_phi;
//
//					// Elements of b
//					sum_ft += ft;
//					sum_ft_cos += ft * cos_phi;
//					sum_ft_sin += ft * sin_phi;
//				}
//			}
//			d_out_csrValA[idA + 0] = 9;			d_out_csrValA[idA + 1] = sum_cos;		d_out_csrValA[idA + 2] = sum_sin;
//			d_out_csrValA[idA + 3] = sum_cos;	d_out_csrValA[idA + 4] = sum_cos2;		d_out_csrValA[idA + 5] = sum_sincos;
//			d_out_csrValA[idA + 6] = sum_sin;	d_out_csrValA[idA + 7] = sum_sincos;	d_out_csrValA[idA + 8] = sum_sin2;
//		}
//	}
//}
//
//void launch()
//{
//		/* AIA to get the initial phi */
//	std::vector<cv::Mat> f;
//
//	cv::Mat img = cv::imread("1.bmp");
//	cv::cvtColor(img,
//				 img,
//				 CV_BGR2GRAY);
//	f.push_back(img);
//
//	img = cv::imread("2.bmp");
//	cv::cvtColor(img,
//				 img,
//				 CV_BGR2GRAY);
//	f.push_back(img);
//
//	img = cv::imread("3.bmp");
//	cv::cvtColor(img,
//				 img,
//				 CV_BGR2GRAY);
//	f.push_back(img);
//	
//
//	// computation
//	std::vector<float> phi;
//	std::vector<float> delta{-1.4577f,
//   -0.8285f,
//    1.3368f};
//	double time = 0;
//	float err = 0;
//	int iter = 0;
//
//	AIA::AIA_CPU_DnF aia;
//	aia(phi, delta, time, iter, err, f, 20, 1e-4, 12);
//
//	std::cout << "AIA Running Time: " << time << std::endl;
//	std::cout << "AIA Error is: " << err << std::endl;
//	std::cout << "AIA Iteration is: " << iter << std::endl;
//
//	std::cout << "Delta is: " << delta[0] << "," << delta[1] << "," << delta[2] << std::endl;
//
//	int iSize = f[0].cols * f[0].rows;
//
//	float *m_d_csrValA, *m_d_b;
//
//
//	float *dPhi0, *dphi_curr = nullptr;
//	uchar *dImg;
//	cudaMalloc((void**)&dPhi0, sizeof(float)*f[0].cols*f[0].rows);
//	cudaMalloc((void**)&dImg, sizeof(uchar)*f[0].cols*f[0].rows);
//	cudaMemcpy(dPhi0, phi.data(), sizeof(float)*f[0].cols*f[0].rows, cudaMemcpyHostToDevice);
//	cudaMemcpy(dImg, f[0].data, sizeof(uchar)*f[0].cols*f[0].rows, cudaMemcpyHostToDevice);
//	checkCudaErrors(cudaMalloc((void**)&m_d_csrValA, sizeof(float) * 9 * iSize));
//	checkCudaErrors(cudaMalloc((void**)&m_d_b, sizeof(float) * 3 * iSize));
//
//
//	dim3 threads(BLOCK_SIZE_16, BLOCK_SIZE_16);
//	dim3 blocks((int)ceil((float)f[0].cols / (BLOCK_SIZE_16 - 2)), (int)ceil((float) f[0].rows / (BLOCK_SIZE_16 - 2)));
//	
//	cudaEvent_t start, stop;
//	cudaEventCreate(&start);
//	cudaEventCreate(&stop);
//
//	cudaEventRecord(start);
//	generate_csrValA_b_kernel<<<blocks, threads>>>(m_d_csrValA, m_d_b, dImg, dPhi0, f[0].cols, f[0].rows);
//	getLastCudaError("generate_csrValA_b_kernel launch failed!");
//	cudaEventRecord(stop);
//	cudaEventSynchronize(stop);
//
//	float ftime;
//	cudaEventElapsedTime(&ftime, start, stop);
//	std::cout << "csvValA_b_kernel running time is: " << ftime << "ms" << std::endl;
//
//	float *h_csrValA = (float*)malloc(sizeof(float)* f[0].cols* f[0].rows * 9);
//
//	checkCudaErrors(cudaMemcpy(h_csrValA, m_d_csrValA, sizeof(float)*f[0].cols* f[0].rows * 9, cudaMemcpyDeviceToHost));
//
//	std::ofstream out("csrA.csv", std::ios::out | std::ios::trunc);
//
//	for (int i = 0; i < f[0].cols* f[0].rows; i++)
//	{
//		for (int j = 0; j < 3; j++)
//		{
//			for (int k = 0; k < 3; k++)
//			{
//				out << h_csrValA[i * 9 + j * 3 + k]<<",";
//			}
//			out<<"\n";
//		}
//		out<<"\n";
//	}
//
//	out.close();
//
//	free(h_csrValA);
//
//	cudaFree(dImg);
//	cudaFree(dPhi0);
//}