//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include "cufft.h"
//#include "helper_cuda.h"
//#include "gtest\gtest.h"
//#include <fstream>
//
//__global__ void Gen_xf_yf_Kernel2D(
//	cufftReal *xf, cufftReal *yf, 
//	int iWidth, int iHeight)
//{
//	int i = threadIdx.y + blockIdx.y * blockDim.y;
//	int j = threadIdx.x + blockIdx.x * blockDim.x;
//	int id = i*iWidth + j;
//	int iH = iHeight / 2;
//	int iW = iWidth / 2;
//
//	if (i < iHeight && j < iWidth)
//	{
//		xf[id] = j - iW;
//		yf[id] = i - iH;
//	}
//}
//__global__ void Gen_xf_yf_Kernel1D(
//	cufftReal *xf, cufftReal *yf, 
//	int iWidth, int iHeight)
//{
//	int iH = iHeight / 2;
//	int iW = iWidth / 2;
//
//	for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < iWidth * iHeight; i += blockDim.x*gridDim.x)
//	{
//		int x = i % iWidth;
//		int y = i / iWidth;
//		
//		if (y < iHeight && x < iWidth)
//		{
//			xf[i] = x - iW;
//			yf[i] = y - iH;
//		}
//	}
//}
//__global__ void gen_xf_yf_Kernel(cufftReal *xf, cufftReal *yf, int iWidth, int iHeight)
//{
//	int i = threadIdx.y + blockIdx.y * blockDim.y;
//	int j = threadIdx.x + blockIdx.x * blockDim.x;
//
//	int id = i*iWidth + j;
//
//	int iHalfy = iHeight / 2;
//	int iHalfx = iWidth / 2;
//
//	if (i < iHeight && j < iWidth)
//	{
//		xf[id] = j - iHalfx;
//		yf[id] = i - iHalfy;
//	}
//	
//}
//__global__ void shift_xf_yf_Kernel(cufftReal *xf, cufftReal *yf,  int iWidth, int iHeight)
//{
//	int i = threadIdx.y + blockIdx.y * blockDim.y;
//	int j = threadIdx.x + blockIdx.x * blockDim.x;
//
//	int id = i*iWidth + j;
//
//	int iHalfx = iWidth / 2;
//	int iHalfy = iHeight / 2;
//	int iSlice = iWidth * iHeight;
//
//	int idQ13 = iSlice / 2 + iHalfx;
//	int idQ24 = iSlice / 2 - iHalfx;
//
//	cufftReal Tempx, Tempy;
//
//	if (j < iHalfx)
//	{
//		if(i < iHalfy)
//		{
//			Tempx = xf[id];
//			Tempy = yf[id];
//
//			// First Quadrant
//			xf[id] = xf[id + idQ13];
//			yf[id] = yf[id + idQ13];
//
//			// Third Quadrant
//			xf[id + idQ13] = Tempx;
//			yf[id + idQ13] = Tempy;
//		}
//	}
//	else
//	{
//		if (i < iHalfy)
//		{
//			Tempx = xf[id];
//			Tempy = yf[id];
//
//			// Second Quadrant
//			xf[id] = xf[id + idQ24];
//			yf[id] = yf[id + idQ24];
//
//			// Fourth Quadrant
//			xf[id + idQ24] = Tempx;
//			yf[id + idQ24] = Tempy;
//		}
//	}
//}
//TEST(XF_YF_2D_1D, KernelTest)
//{
//	cudaEvent_t start, end2D, end1D;
//	cudaEventCreate(&start);
//	cudaEventCreate(&end1D);
//	cudaEventCreate(&end2D);
//
//	dim3 threads(16, 16);
//	dim3 blocks((1120 + 16 - 1) / 16, (1120 + 16 - 1) / 16);
//
//	cufftReal *xf, *xf1, *yf, *yf1;
//	checkCudaErrors(cudaMalloc((void**)&xf, sizeof(cufftReal) * 1120 * 1120));
//	checkCudaErrors(cudaMalloc((void**)&yf, sizeof(cufftReal) * 1120 * 1120));
//	checkCudaErrors(cudaMalloc((void**)&xf1, sizeof(cufftReal) * 1120 * 1120));
//	checkCudaErrors(cudaMalloc((void**)&yf1, sizeof(cufftReal) * 1120 * 1120));
//
//	cudaEventRecord(start);
//	Gen_xf_yf_Kernel1D << <32 * 8, 256 >> >(xf1, yf1, 1120, 1120);
//	getLastCudaError("1D Kernel Failed.");
//	cudaEventRecord(end1D);
//	Gen_xf_yf_Kernel2D<<<blocks, threads>>>(xf, yf, 1120, 1120);
//	getLastCudaError("2D Kernel Failed.");
//	cudaEventRecord(end2D);
//
//	cudaDeviceSynchronize();
//
//
//	cufftReal *hxf, *hxf1, *hyf, *hyf1;
//	hxf = (cufftReal*)malloc(sizeof(cufftReal) * 1120 * 1120);
//	hyf = (cufftReal*)malloc(sizeof(cufftReal) * 1120 * 1120);
//	hxf1 = (cufftReal*)malloc(sizeof(cufftReal) * 1120 * 1120);
//	hyf1 = (cufftReal*)malloc(sizeof(cufftReal) * 1120 * 1120);
//
//	checkCudaErrors(cudaMemcpy(hxf, xf, sizeof(cufftReal) * 1120 * 1120, cudaMemcpyDeviceToHost));
//	checkCudaErrors(cudaMemcpy(hyf, yf, sizeof(cufftReal) * 1120 * 1120, cudaMemcpyDeviceToHost));
//	
//	checkCudaErrors(cudaMemcpy(hxf1, xf1, sizeof(cufftReal) * 1120 * 1120, cudaMemcpyDeviceToHost));
//	checkCudaErrors(cudaMemcpy(hyf1, yf1, sizeof(cufftReal) * 1120 * 1120, cudaMemcpyDeviceToHost));
//
//	for(int i=0; i<1120; i++)
//		for(int j=0; j<1120; j++)
//		{
//			ASSERT_EQ(hxf[i * 1120 + j], hxf1[i * 1120 + j]);
//			ASSERT_EQ(hyf[i * 1120 + j], hyf1[i * 1120 + j]);
//		}
//
//	float t1, t2;
//	cudaEventElapsedTime(&t1, start, end1D);
//	cudaEventElapsedTime(&t2, end1D, end2D);
//
//	std::cout << "1D Kernel Execution Time: " << t1 << std::endl;
//	std::cout << "2D Kernel Execution Time: " << t2 << std::endl;
//
//	cudaEventDestroy(start);
//	cudaEventDestroy(end1D);
//	cudaEventDestroy(end2D);
//	cudaFree(xf);
//	cudaFree(yf);
//	cudaFree(xf1);
//	cudaFree(yf1);
//	free(hxf);
//	free(hyf);
//	free(hxf1);
//	free(hyf1);
//}
//TEST(XF_YF_2D_with_FFTSHIFT, KernelTest)
//{
//	int iWidth = 64;
//	int iHeight = 50;
//
//	cudaEvent_t start, end2D;
//	cudaEventCreate(&start);
//	cudaEventCreate(&end2D);
//
//	dim3 threads(16, 16);
//	dim3 blocks((iWidth + 16 - 1) / 16, (iHeight + 16 - 1) / 16);
//	dim3 blocks1((iWidth + 16 - 1) / 16, (iHeight + 16 - 1) / 16);
//
//	cufftReal *xf, *yf;
//	checkCudaErrors(cudaMalloc((void**)&xf, sizeof(cufftReal) * iWidth * iHeight));
//	checkCudaErrors(cudaMalloc((void**)&yf, sizeof(cufftReal) * iWidth * iHeight));
//
//	cudaEventRecord(start);
//	gen_xf_yf_Kernel<<<blocks, threads>>>(xf, yf, iWidth, iHeight);
//	getLastCudaError("2D Kernel Failed.");
//	shift_xf_yf_Kernel<<<blocks1, threads>>>(xf, yf, iWidth, iHeight);
//	getLastCudaError("2D Shift Failed.");
//	cudaEventRecord(end2D);
//
//	cudaDeviceSynchronize();
//
//
//	cufftReal *hxf, *hyf;
//	hxf = (cufftReal*)malloc(sizeof(cufftReal) * iWidth * iHeight);
//	hyf = (cufftReal*)malloc(sizeof(cufftReal) * iWidth * iHeight);
//
//	checkCudaErrors(cudaMemcpy(hxf, xf, sizeof(cufftReal) * iWidth * iHeight, cudaMemcpyDeviceToHost));
//	checkCudaErrors(cudaMemcpy(hyf, yf, sizeof(cufftReal) * iWidth * iHeight, cudaMemcpyDeviceToHost));
//
//	std::ofstream out("xfyf.csv", std::ios::out | std::ios::trunc);
//
//	for(int i=0; i<iHeight; i++)
//	{
//		for(int j=0; j<iWidth; j++)
//		{
//			out << "[" << hxf[i * iWidth + j] << "-" << hyf[i * iWidth + j] << "]" << ",";
//		}
//		out<<"\n";
//	}
//
//	out.close();
//
//	float t1;
//	cudaEventElapsedTime(&t1, start, end2D);
//
//	std::cout << "1D Kernel Execution Time: " << t1 << std::endl;
//
//	cudaEventDestroy(start);
//	cudaEventDestroy(end2D);
//	cudaFree(xf);
//	cudaFree(yf);
//
//	free(hxf);
//	free(hyf);
//
//}