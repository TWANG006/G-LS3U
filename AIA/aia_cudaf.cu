#include "aia_cudaf.h"
#include <time.h>
#include <functional>
#include <random>
#include <omp.h>
#include <mkl.h>

#include "device_launch_parameters.h"

namespace AIA{

__inline__ __device__
float warpReduceSum(float val) {
	for (int offset = warpSize / 2; offset > 0; offset /= 2)
		val += __shfl_down(val, offset);
	return val;
}

/*---------------------------------------CUDA Kernels----------------------------------*/
__global__
void generate_csrValA1_rhs1_kernel(float* d_out_csrValA1,
								   float* d_out_csr_rhs1,
								   uchar* d_in_img,
								   float* d_in_delta,
								   const int M,
								   const int N)
{
	for (int j = threadIdx.x + blockDim.x *blockIdx.x;
		 j < N;
		 j += blockDim.x * gridDim.x)
	{
		float a3 = 0, a4 = 0, a6 = 0, a7 = 0, a8 = 0;
		float b0 = 0, b1 = 0, b2 = 0;

		for (int i = 0; i < M; i++)
		{
			float delta = d_in_delta[i];
			float cos_delta = cos(delta);
			float sin_delta = sin(delta);
			float Iij = static_cast<float>(d_in_img[i*N + j]);

			a3 += cos_delta;
			a4 += cos_delta*cos_delta;
			a6 += sin_delta;
			a7 += sin_delta*cos_delta;
			a8 += sin_delta*sin_delta;
			b0 += Iij;
			b1 += Iij * cos_delta;
			b2 += Iij * sin_delta;
		}

		d_out_csrValA1[j * 9 + 0] = M;
		d_out_csrValA1[j * 9 + 1] = a3;
		d_out_csrValA1[j * 9 + 2] = a6;
		d_out_csrValA1[j * 9 + 3] = a3;
		d_out_csrValA1[j * 9 + 4] = a4;
		d_out_csrValA1[j * 9 + 5] = a7;
		d_out_csrValA1[j * 9 + 6] = a6;
		d_out_csrValA1[j * 9 + 7] = a7;
		d_out_csrValA1[j * 9 + 8] = a8;

		d_out_csr_rhs1[j * 3 + 0] = b0;
		d_out_csr_rhs1[j * 3 + 1] = b1;
		d_out_csr_rhs1[j * 3 + 2] = b2;
	}
}

__global__
void generate_csrColIndA1_csrRowPtrA1_kernel(int* d_out_csrColIndA1,
											 int* d_out_csrRowPtrA1,
											 const int N)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x;
		 i < N;
		 i += blockDim.x * gridDim.x)
	{
		int a1 = i * 9;
		
		d_out_csrRowPtrA1[i * 3 + 0] = a1;
		d_out_csrRowPtrA1[i * 3 + 1] = a1 + 3;
		d_out_csrRowPtrA1[i * 3 + 2] = a1 + 6;

		a1 = i * 3;
		int a2 = a1 + 1;
		int a3 = a1 + 2;

		d_out_csrColIndA1[i * 9 + 0] = a1;
		d_out_csrColIndA1[i * 9 + 1] = a2;
		d_out_csrColIndA1[i * 9 + 2] = a3;
		d_out_csrColIndA1[i * 9 + 3] = a1;
		d_out_csrColIndA1[i * 9 + 4] = a2;
		d_out_csrColIndA1[i * 9 + 5] = a3;
		d_out_csrColIndA1[i * 9 + 6] = a1;
		d_out_csrColIndA1[i * 9 + 7] = a2;
		d_out_csrColIndA1[i * 9 + 8] = a3;
	}

	// Last ele of csrRowIndA is nnz + csrRowIndA(0)
	if (blockIdx.x == 0 && threadIdx.x ==0)
		d_out_csrRowPtrA1[3 * N] = 9 * N;
}

__global__
void get_phi_kernel(float *d_out_phi, float* d_in_x, int N)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x;
		 i < N;
		 i += gridDim.x * blockDim.x)
	{
		d_out_phi[i] = atan2f(-d_in_x[i * 3 + 2], d_in_x[i * 3 + 1]);
	}
}

__global__
void get_final_phi_kernel(float* d_out_phi, float* d_in_delta, int N)
{
	float delta0 = d_in_delta[0];

	for (int i = threadIdx.x + blockIdx.x * blockDim.x;
		 i < N;
		 i += gridDim.x * blockDim.x)
	{
		float temp = d_out_phi[i] + delta0;
		d_out_phi[i] = atan2f(sin(temp), cos(temp));
	}
}

__global__
void generate_A2_kernel(float *d_out_A2temp, int N, float* d_in_phi)
{
	if (blockIdx.x == 0 && threadIdx.x < 5)
	{
		d_out_A2temp[threadIdx.x] = 0;
	}

	float sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0, sum5 = 0;

	for (int j = threadIdx.x + blockIdx.x *blockDim.x;
		 j < N;
		 j += gridDim.x * blockDim.x)
	{
		float cos_phi = cos(d_in_phi[j]);
		float sin_phi = sin(d_in_phi[j]);

		sum1 += cos_phi;
		sum2 += cos_phi*cos_phi;
		sum3 += sin_phi;
		sum4 += cos_phi*sin_phi;
		sum5 += sin_phi*sin_phi;
	}	

	sum1 = warpReduceSum(sum1);
	sum2 = warpReduceSum(sum2);
	sum3 = warpReduceSum(sum3);
	sum4 = warpReduceSum(sum4);
	sum5 = warpReduceSum(sum5);

	if (threadIdx.x % warpSize == 0)
	{
		atomicAdd(&d_out_A2temp[0], sum1);
		atomicAdd(&d_out_A2temp[1], sum2);
		atomicAdd(&d_out_A2temp[2], sum3);
		atomicAdd(&d_out_A2temp[3], sum4);
		atomicAdd(&d_out_A2temp[4], sum5);
	}
}
__global__
void generate_b2_kernel(float *d_out_b2, int i, int N, float* d_in_phi, uchar* d_in_img)
{
	float b1 = 0, b2 = 0, b3 = 0;

	if (blockIdx.x == 0 && threadIdx.x < 3)
	{
		d_out_b2[i * 3 + threadIdx.x] = 0;
	}

	for (int j = threadIdx.x + blockIdx.x *blockDim.x;
		 j < N;
		 j += gridDim.x * blockDim.x)
	{
		float Iij = static_cast<float>(d_in_img[i*N + j]);
		float cos_phi = cos(d_in_phi[j]);
		float sin_phi = sin(d_in_phi[j]);

		b1 += Iij;
		b2 += Iij * cos_phi;
		b3 += Iij * sin_phi;
	}	

	b1 = warpReduceSum(b1);
	b2 = warpReduceSum(b2);
	b3 = warpReduceSum(b3);

	if (threadIdx.x % warpSize == 0)
	{
		atomicAdd(&d_out_b2[i * 3 + 0], b1);
		atomicAdd(&d_out_b2[i * 3 + 1], b2);
		atomicAdd(&d_out_b2[i * 3 + 2], b3);
	}
}

/*--------------------------------------End CUDA Kernels--------------------------------*/

AIA_CUDAF::AIA_CUDAF(const std::vector<cv::Mat>& v_f)
	: m_d_img(nullptr)
	, m_d_csrColIndA1(nullptr)
	, m_d_csrValA1(nullptr)
	, m_d_csrRowPtrA1(nullptr)
	, m_d_b1(nullptr)
	, m_d_phi(nullptr)
	, m_d_delta(nullptr)
	, m_h_delta(nullptr)
	, m_h_A2(9, 0)
	, m_h_b2(nullptr)
{
	// Get params
	m_M = v_f.size();
	m_N = v_f[0].cols*v_f[0].rows;
	m_cols = v_f[0].cols;
	m_rows = v_f[0].rows;

	// Allocate required pinned host memory
	m_h_old_delta = (float*)malloc(sizeof(float)*m_M);
	WFT_FPA::Utils::hcreateptr(m_h_delta, sizeof(float)*m_M);
	WFT_FPA::Utils::hcreateptr(m_h_A2temp, sizeof(float) * 5);
	WFT_FPA::Utils::hcreateptr(m_h_b2, sizeof(float)*m_M * 3);

	// Allocate device memory
	checkCudaErrors(cudaMalloc((void**)&m_d_csrValA1, sizeof(float) * 9 * m_N));
	checkCudaErrors(cudaMalloc((void**)&m_d_csrRowPtrA1, sizeof(int)*(3 * m_N + 1)));
	checkCudaErrors(cudaMalloc((void**)&m_d_csrColIndA1, sizeof(int) * 9 * m_N));
	checkCudaErrors(cudaMalloc((void**)&m_d_b1, sizeof(float) * 3 * m_N));
	checkCudaErrors(cudaMalloc((void**)&m_d_phi, sizeof(float) * m_cols * m_rows));
	checkCudaErrors(cudaMalloc((void**)&m_d_delta, sizeof(float) * m_M));
	checkCudaErrors(cudaMalloc((void**)&m_d_img, sizeof(uchar)*m_M*m_N));
	checkCudaErrors(cudaMalloc((void**)&m_d_b2, sizeof(float)*m_M * 3));
	checkCudaErrors(cudaMalloc((void**)&m_d_A2temp, sizeof(float) * 5));

	// Initialize the csrRowPtrA & csrColIndA here because they remain at the same patterns
	generate_csrColIndA1_csrRowPtrA1_kernel<<<8*32, 256>>>(m_d_csrColIndA1, m_d_csrRowPtrA1, m_N);
	getLastCudaError("generate_csrColIndA1_csrRowPtrA1_kernel launch failed!");

	// Create cuSolver required handles
	checkCudaErrors(cusolverSpCreate(&m_cuSolverHandle));
	checkCudaErrors(cusparseCreateMatDescr(&m_desrA));
	checkCudaErrors(cusparseSetMatType(m_desrA, CUSPARSE_MATRIX_TYPE_GENERAL));
	checkCudaErrors(cusparseSetMatIndexBase(m_desrA, CUSPARSE_INDEX_BASE_ZERO));
}

AIA_CUDAF::AIA_CUDAF(const int iM,
					 const int icols,
					 const int irows)
	: m_M(iM)
	, m_cols(icols)
	, m_rows(irows)
	, m_h_A2(9, 0)
{
	m_N = icols * irows;

	// Allocate required pinned host memory
	m_h_old_delta = (float*)malloc(sizeof(float)*m_M);
	WFT_FPA::Utils::hcreateptr(m_h_delta, sizeof(float)*m_M);
	WFT_FPA::Utils::hcreateptr(m_h_A2temp, sizeof(float) * 5);
	WFT_FPA::Utils::hcreateptr(m_h_b2, sizeof(float)*m_M * 3);

	// Allocate device memory
	checkCudaErrors(cudaMalloc((void**)&m_d_csrValA1, sizeof(float) * 9 * m_N));
	checkCudaErrors(cudaMalloc((void**)&m_d_csrRowPtrA1, sizeof(int)*(3 * m_N + 1)));
	checkCudaErrors(cudaMalloc((void**)&m_d_csrColIndA1, sizeof(int) * 9 * m_N));
	checkCudaErrors(cudaMalloc((void**)&m_d_b1, sizeof(float) * 3 * m_N));
	checkCudaErrors(cudaMalloc((void**)&m_d_phi, sizeof(float) * m_cols * m_rows));
	checkCudaErrors(cudaMalloc((void**)&m_d_delta, sizeof(float) * m_M));
	checkCudaErrors(cudaMalloc((void**)&m_d_img, sizeof(uchar)*m_M*m_N));
	checkCudaErrors(cudaMalloc((void**)&m_d_b2, sizeof(float)*m_M * 3));
	checkCudaErrors(cudaMalloc((void**)&m_d_A2temp, sizeof(float) * 5));

	// Initialize the csrRowPtrA & csrColIndA here because they remain at the same patterns
	generate_csrColIndA1_csrRowPtrA1_kernel<<<8*32, 256>>>(m_d_csrColIndA1, m_d_csrRowPtrA1, m_N);
	getLastCudaError("generate_csrColIndA1_csrRowPtrA1_kernel launch failed!");

	// Create cuSolver required handles
	checkCudaErrors(cusolverSpCreate(&m_cuSolverHandle));
	checkCudaErrors(cusparseCreateMatDescr(&m_desrA));
	checkCudaErrors(cusparseSetMatType(m_desrA, CUSPARSE_MATRIX_TYPE_GENERAL));
	checkCudaErrors(cusparseSetMatIndexBase(m_desrA, CUSPARSE_INDEX_BASE_ZERO));
}

AIA_CUDAF::~AIA_CUDAF()
{
	checkCudaErrors(cusolverSpDestroy(m_cuSolverHandle));
	checkCudaErrors(cusparseDestroyMatDescr(m_desrA));

	WFT_FPA::Utils::hdestroyptr(m_h_A2temp);
	WFT_FPA::Utils::hdestroyptr(m_h_b2);
	WFT_FPA::Utils::hdestroyptr(m_h_delta);

	free(m_h_old_delta);	m_h_old_delta = nullptr;
	WFT_FPA::Utils::cudaSafeFree(m_d_csrValA1);
	WFT_FPA::Utils::cudaSafeFree(m_d_csrColIndA1);
	WFT_FPA::Utils::cudaSafeFree(m_d_csrRowPtrA1);
	WFT_FPA::Utils::cudaSafeFree(m_d_b1);
	WFT_FPA::Utils::cudaSafeFree(m_d_phi);
	WFT_FPA::Utils::cudaSafeFree(m_d_delta);
	WFT_FPA::Utils::cudaSafeFree(m_d_A2temp);
	WFT_FPA::Utils::cudaSafeFree(m_d_b2);
}

void AIA_CUDAF::operator() (//Outputs
			 				std::vector<float>& v_phi,
							std::vector<float>& v_deltas,
							double &runningtime,
							int &iters,
							float &err,
							// Inputs
							const std::vector<cv::Mat>& v_f,
							int iMaxIterations,
							float dMaxErr,
							int iNumThreads)
{
	omp_set_num_threads(iNumThreads);

	/* If the number of frames != the number of initial deltas, randomly generate 
	   deltas */	
	if (m_M != v_deltas.size())
	{
		// Mimic the real randomness by varying the seed
		std::default_random_engine generator;
		generator.seed(time(nullptr));

		// Random numbers picked from standard normal distribution g(0,1)
		std::normal_distribution<float> distribution(0.0f, 1.0f);

		for (int i = 0; i < m_M; i++)
		{
			m_h_delta[i] = distribution(generator);
		}
	}
	else
	{
		for (int i = 0; i < m_M; i++)
		{
			m_h_delta[i] = v_deltas[i];
		}
	}

	// Copy the images to device
	for (int i = 0; i < m_M; i++)
	{
		checkCudaErrors(cudaMemcpy(&m_d_img[i*m_N], v_f[i].data, sizeof(uchar)*m_N, cudaMemcpyHostToDevice));
	}
	
	err = dMaxErr * 2.0f;
	iters = 0;

	
	double start = omp_get_wtime();

	/* Begin the real algorithm */
	while (err > dMaxErr && iters < iMaxIterations)
	{
		for(int i=0; i<m_M; i++)
		{
			m_h_old_delta[i] = m_h_delta[i];
		}

		// Step 1: pixel-by-pixel iterations
		
		computePhi();

		// Step 2: frame-by-frame iterations
		computeDelta();

		// Step 3: update & check convergence criterion
		iters++;
		err = computeMaxError(m_h_delta, m_h_old_delta, m_M);
	}

	double end = omp_get_wtime();
	runningtime = 1000.0f*(end - start);
	

	/* One more round to calculate phi once delta is good enough */
	computePhi();

	/* Get the final phi and  deltas */
	get_final_phi_kernel<<<8*32, 256>>>(m_d_phi, m_d_delta, m_N);
	getLastCudaError("get_final_phi_kernel launch failed!");
	
	v_phi.resize(m_N);
	
	cudaMemcpy(v_phi.data(), m_d_phi, sizeof(float)*m_N, cudaMemcpyDeviceToHost);

	v_deltas.resize(m_M);

	for (int i = m_M - 1; i >= 0; i--)
	{
		m_h_delta[i] -= m_h_delta[0];
		m_h_delta[i] = atan2(sin(m_h_delta[i]), cos(m_h_delta[i]));
		v_deltas[i] = m_h_delta[i];
	}
}

void AIA_CUDAF::computePhi()
{
	// Load the new deltas
	checkCudaErrors(cudaMemcpy(m_d_delta, m_h_delta, sizeof(float)*m_M, cudaMemcpyHostToDevice));

	/*cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);*/

	// Generate csrValA & RHS
	generate_csrValA1_rhs1_kernel<<<8*32, 256>>>(m_d_csrValA1, m_d_b1, m_d_img, m_d_delta, m_M, m_N);
	getLastCudaError("generate_csrValA1_rhs1_kernel launch failed!");

	//std::vector<float> h_b(m_N*3);
	//cudaMemcpy(h_b.data(), m_d_b1, sizeof(float) * 3 * m_N, cudaMemcpyDeviceToHost);

	float tol = 1e-4f;
	int singularity = -1;
	checkCudaErrors(cusolverSpScsrlsvchol(m_cuSolverHandle, 3*m_N, 9*m_N, m_desrA, 
									 	  m_d_csrValA1, m_d_csrRowPtrA1, m_d_csrColIndA1, m_d_b1,
										  tol, 0, m_d_b1, &singularity));
	if (0 <= singularity)
    {
        printf("WARNING: the matrix is singular at row %d under tol (%E)\n", singularity, tol);
    }

	// Update phi
	get_phi_kernel<<<8*32, 256>>>(m_d_phi, m_d_b1, m_N);
	getLastCudaError("get_phi_kernel launch failed!");

	/*cudaEventRecord(end);
	cudaEventSynchronize(end);
	float t = 0;
	cudaEventElapsedTime(&t, start, end);*/
	/*std::vector<float>h_phi(m_N);
	cudaMemcpy(h_phi.data(), m_d_phi, sizeof(float)*m_N, cudaMemcpyDeviceToHost);*/
}

void AIA_CUDAF::computeDelta()
{
	// Generate A2
	generate_A2_kernel<<<8*32, 256>>>(m_d_A2temp, m_N, m_d_phi);
	getLastCudaError("generate_A2_kernel launch failed!");

	checkCudaErrors(cudaMemcpy(m_h_A2temp, m_d_A2temp, sizeof(float) * 5, cudaMemcpyDeviceToHost));

	// Generate b2
	for (int i = 0; i < m_M; i++)
	{
		generate_b2_kernel<<<8*32, 256>>>(m_d_b2, i, m_N, m_d_phi, m_d_img);
		getLastCudaError("generate_b2_kernel launch failed!");
	}

	checkCudaErrors(cudaMemcpy(m_h_b2, m_d_b2, sizeof(float) * 3 * m_M, cudaMemcpyDeviceToHost));

	//cudaDeviceSynchronize();

	m_h_A2[0] = float(m_N);		m_h_A2[1] = 0;				m_h_A2[2] = 0;
	m_h_A2[3] = m_h_A2temp[0];	m_h_A2[4] = m_h_A2temp[1];	m_h_A2[5] = 0;
	m_h_A2[6] = m_h_A2temp[2];	m_h_A2[7] = m_h_A2temp[3];	m_h_A2[8] = m_h_A2temp[4];


	/*std::cout << "A2: " << std::endl;

	for(int i=0; i<3;i++)
	{
		for(int j=0;j<3;j++)
		{
			std::cout<<m_h_A2[i*3+j]<<",";
		}
		std::cout << std::endl;
	}*/

	/* Solve the Ax = b */
	int info = LAPACKE_sposv(LAPACK_COL_MAJOR, 'U', 3, m_M, m_h_A2.data(), 3, m_h_b2, 3);
	 /* Check for the positive definiteness */
	if (info > 0) {
		printf("The leading minor of order %i is not positive ", info);
		printf("definite;\nThe solution could not be computed.\n");
		exit(1);
	}

	for (int i = 0; i < m_M; i++)
	{
		m_h_delta[i] = atan2(-m_h_b2[i * 3 + 2], m_h_b2[i * 3 + 1]);
	}
}

float AIA_CUDAF::computeMaxError(const float *v_delta,
								 const float *v_deltaOld,
								 int m)
{
	std::vector<float> abs;

	for (int i = 0; i < m; i++)
	{
		abs.push_back(std::abs(v_delta[i] - v_deltaOld[i]));
	}
	std::sort(abs.begin(), abs.end(), std::greater<float>());

	return abs[0];
}

}	//	namespace AIA

//// Load the new deltas
//checkCudaErrors(cudaMemcpy(m_d_delta, m_h_delta, sizeof(float)*m_M, cudaMemcpyHostToDevice));
//
//// Generate csrValA & RHS
//generate_csrValA1_rhs1_kernel << <8 * 32, 256 >> >(m_d_csrValA1, m_d_b1, m_d_img, m_d_delta, m_M, m_N);
//getLastCudaError("generate_csrValA1_rhs1_kernel launch failed!");
//
//float *csrValA = (float*)malloc(sizeof(float) * 9 * m_N);
//int *csrRowPtrA = (int*)malloc(sizeof(int) * (3 * m_N + 1));
//int *csrColIndA = (int*)malloc(sizeof(int) * 9 * m_N);
//
//checkCudaErrors(cudaMemcpy(csrValA, m_d_csrValA1, sizeof(float) * 9 * m_N, cudaMemcpyDeviceToHost));
//checkCudaErrors(cudaMemcpy(csrColIndA, m_d_csrColIndA1, sizeof(int) * 9 * m_N, cudaMemcpyDeviceToHost));
//checkCudaErrors(cudaMemcpy(csrRowPtrA, m_d_csrRowPtrA1, sizeof(int) *(3 * m_N + 1), cudaMemcpyDeviceToHost));
//
//
//
///*for(int i=0; i<m_N; i++)
//{
//for(int j=0; j<9; j++)
//{
//std::cout<<csrColIndA[i * 9 + j]<<",";
//}
//for(int j=0; j<3; j++)
//{
//std::cout<<csrRowPtrA[i * 3 + j]<<",";
//}
//std::cout << "\n";
//}*/
//
//for (int i = 0; i < 3; i++)
//{
//	for (int j = 0; j < 3; j++)
//	{
//		std::cout << csrValA[i * 3 + j] << ",";
//	}
//	std::cout << std::endl;
//}
//
//int issym = 0;
//checkCudaErrors(cusolverSpXcsrissymHost(m_cuSolverHandle, 3 * m_N, 9 * m_N, m_desrA, csrRowPtrA, csrRowPtrA + 1, csrColIndA, &issym));
//
//if (!issym)
//{
//	printf("Error: A has no symmetric pattern, please use LU or QR \n");
//	exit(EXIT_FAILURE);
//}
//
//free(csrValA);
//free(csrRowPtrA);
//free(csrColIndA);