#include "aia_cpuf.h"

#include <time.h>
#include <random>
#include <functional>

#include <omp.h>
#include <mkl.h>

namespace AIA{

AIA_CPU_DnF::~AIA_CPU_DnF()
{}

void AIA_CPU_DnF::operator()(// Outputs
							 std::vector<float>& v_phi,
							 std::vector<float>& v_deltas,
							 double &runningtime,
							 int &iIters,
							 float &dErr,
							 // Inputs
							 const std::vector<cv::Mat>& v_f,
							 int iMaxIterations,
							 float dMaxErr,
							 int iNumThreads)
{
	omp_set_num_threads(iNumThreads);

	/* If the number of frames != the number of initial deltas, randomly generate 
	   deltas */
	if (v_f.size() != v_deltas.size())
	{
		// Mimic the real randomness by varying the seed
		std::default_random_engine generator;
		generator.seed(time(nullptr));

		// Random numbers picked from standard normal distribution g(0,1)
		std::normal_distribution<float> distribution(0.0f, 1.0f);

		v_deltas.push_back(0);
		for (int i = 0; i < v_f.size() - 1; i++)
		{
			v_deltas.push_back(distribution(generator));
		}
	}
	
	// Assign values for m_N & m_M
	m_M = v_f.size();
	m_N = v_f[0].cols * v_f[0].rows;
	m_rows = v_f[0].rows;
	m_cols = v_f[0].cols;	

	// Allocate space for m_v_b & m_v_phi
	v_phi.resize(m_N);
	std::vector<float> v_b_phi(m_N * 3, 0);
	std::vector<float> v_b_delta(m_M * 3, 0);
	std::vector<float> v_A(9, 0);

	dErr = dMaxErr * 2.0f;
	iIters = 0;
	std::vector<float> v_deltaOld = v_deltas;

	double start = omp_get_wtime();

	/* Begin the real algorithm */
	while (dErr > dMaxErr && iIters < iMaxIterations)
	{
		v_deltaOld = v_deltas;

		// Step 1: pixel-by-pixel iterations
		computePhi(v_A, v_b_phi, v_phi, v_deltas, v_f);

		// Step 2: frame-by-frame iterations
		computeDelta(v_A, v_b_delta, v_deltas, v_phi, v_f);

		// Step 3: update & check convergence criterion
		iIters++;
		dErr = computeMaxError(v_deltas, v_deltaOld);
	}

	double end = omp_get_wtime();
	runningtime = 1000.0*(end - start);

	/* One more round to calculate phi once delta is good enough */
	computePhi(v_A, v_b_phi, v_phi, v_deltas, v_f);

	#pragma omp parallel for
	for (int i = 0; i < v_phi.size(); i++)
	{
		v_phi[i] += v_deltas[0];
		v_phi[i] = atan2(sin(v_phi[i]), cos(v_phi[i]));
	}

	for (int i = v_deltas.size() - 1; i >= 0; i--)
	{
		v_deltas[i] -= v_deltas[0];
		v_deltas[i] = atan2(sin(v_deltas[i]), cos(v_deltas[i]));
	}
}

void AIA_CPU_DnF::computePhi(std::vector<float>& v_A, 
						 	 std::vector<float>& v_b_phi,
							 std::vector<float>& v_phi,
							 const std::vector<float>& v_deltas,
							 const std::vector<cv::Mat>& v_f)
{
	/* pixel-by-pixel iterations */

	/* Construct m_v_A only once and use accross multiple RHS vectors m_v_b */
	// Save the Cholesky factorized version of m_v_A 
	float dA3 = 0, dA4 = 0, dA6 = 0, dA7 = 0, dA8 = 0;

	for (int i = 0; i < m_M; i++)
	{
		float cos_delta = cos(v_deltas[i]);
		float sin_delta = sin(v_deltas[i]);

		dA3 += cos_delta;	dA4 += cos_delta*cos_delta;
		dA6 += sin_delta;	dA7 += cos_delta*sin_delta; dA8 += sin_delta*sin_delta;
	}

	v_A[0] = float(m_M);	v_A[1] = 0;		v_A[2] = 0;
	v_A[3] = dA3;			v_A[4] = dA4;	v_A[5] = 0;
	v_A[6] = dA6;			v_A[7] = dA7;	v_A[8] = dA8;

	/* Construct the RHS's m_v_b[N-by-3] */
	#pragma omp parallel for
	for (int j = 0; j < m_N; j++)
	{
		int y = j / m_cols;
		int x = j % m_cols;

		float b0 = 0, b1 = 0, b2 = 0;

		for (int i = 0; i < m_M; i++)
		{
			float dI = static_cast<float>(v_f[i].at<uchar>(y, x));

			b0 += dI;
			b1 += dI * cos(v_deltas[i]);
			b2 += dI * sin(v_deltas[i]);
		}
		v_b_phi[j * 3 + 0] = b0;
		v_b_phi[j * 3 + 1] = b1;
		v_b_phi[j * 3 + 2] = b2;
	}

	/* Solve the Ax = b */
	int info = LAPACKE_sposv(LAPACK_COL_MAJOR, 'U', 3, m_N, v_A.data(), 3, v_b_phi.data(), 3);
	 /* Check for the positive definiteness */
	if (info > 0) {
		printf("The leading minor of order %i is not positive ", info);
		printf("definite;\nThe solution could not be computed.\n");
		exit(1);
	}

	#pragma omp parallel for
	for (int j = 0; j < m_N; j++)
	{
		v_phi[j] = atan2(-v_b_phi[j * 3 + 2], v_b_phi[j * 3 + 1]);
	}
}

void AIA_CPU_DnF::computeDelta(std::vector<float>& v_A,
							   std::vector<float>& v_b_delta,
							   std::vector<float>& v_deltas,
							   const std::vector<float>& v_phi,
							   const std::vector<cv::Mat>& v_f)
{
	/* Frame-by-frame iterations */
	/* Construct m_v_A only once and use accross multiple RHS vectors m_v_b */
	float dA3 = 0, dA4 = 0, dA6 = 0, dA7 = 0, dA8 = 0;

	#pragma omp parallel for default(shared) reduction(+: dA3, dA4, dA6, dA7, dA8)
	for (int j = 0; j < m_N; j++)
	{
		float cos_phi = cos(v_phi[j]);
		float sin_phi = sin(v_phi[j]);

		dA3 += cos_phi;
		dA4 += cos_phi*cos_phi;
		dA6 += sin_phi;
		dA7 += cos_phi*sin_phi;
		dA8 += sin_phi*sin_phi;
	}

	v_A[0] = float(m_N);	v_A[1] = 0;		v_A[2] = 0;
	v_A[3] = dA3;			v_A[4] = dA4;	v_A[5] = 0;
	v_A[6] = dA6;			v_A[7] = dA7;	v_A[8] = dA8;

	/* Construct the RHS's m_v_b[M-by-3] */
	for (int i = 0; i < m_M; i++)
	{
		float b0 = 0, b1 = 0, b2 = 0;

		#pragma omp parallel for default(shared) reduction(+: b0, b1, b2)
		for (int j = 0; j < m_N; j++)
		{
			int y = j / m_cols;
			int x = j % m_cols;

			float dI = static_cast<float>(v_f[i].at<uchar>(y, x));
			float cos_phi = cos(v_phi[j]);
			float sin_phi = sin(v_phi[j]);

			b0 += dI;
			b1 += dI * cos_phi;
			b2 += dI * sin_phi;
		}

		v_b_delta[i * 3 + 0] = b0;
		v_b_delta[i * 3 + 1] = b1;
		v_b_delta[i * 3 + 2] = b2;
	}

	/* Solve the Ax = b */

	/* Solve the Ax = b */
	int info = LAPACKE_sposv(LAPACK_COL_MAJOR, 'U', 3, m_M, v_A.data(), 3, v_b_delta.data(), 3);
	 /* Check for the positive definiteness */
	if (info > 0) {
		printf("The leading minor of order %i is not positive ", info);
		printf("definite;\nThe solution could not be computed.\n");
		exit(1);
	}

	for (int i = 0; i < m_M; i++)
	{
		v_deltas[i] = atan2(-v_b_delta[i * 3 + 2], v_b_delta[i * 3 + 1]);
	}
}

float AIA_CPU_DnF::computeMaxError(const std::vector<float> &v_delta, 
								   const std::vector<float>& v_deltaOld)
{
	std::vector<float> v_abs;

	for (int i = 0; i < v_delta.size(); i++)
	{
		v_abs.push_back(abs(v_delta[i] - v_deltaOld[i]));
	}

	std::sort(v_abs.begin(), v_abs.end(), std::greater<float>());

	return v_abs[0];
}

}	// namepsace AIA