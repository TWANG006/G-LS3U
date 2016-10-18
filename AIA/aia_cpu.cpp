#include "aia_cpu.h"

#include <time.h>
#include <random>

#include <omp.h>
#include <mkl.h>


namespace AIA{

AIA_CPU_Dn::AIA_CPU_Dn()
	: m_v_A(9, 0)
{}

AIA_CPU_Dn::~AIA_CPU_Dn()
{}

void AIA_CPU_Dn::operator()(
	// Outputs
	std::vector<double>& v_phi,
	double &runningtime,
	// Inputs
	const std::vector<cv::Mat>& v_f,
	const std::vector<double> v_deltas,
	int iMaxIterations,
	double dMaxErr,
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
		std::normal_distribution<double> distribution(0.0, 1.0);

		for (int i = 0; i < v_f.size(); i++)
		{
			m_v_delta.push_back(distribution(generator));
		}
	}
	else
	{
		m_v_delta = v_deltas;
	}
	
	// Assign values for m_N & m_M
	m_M = v_f.size();
	m_N = v_f[0].cols * v_f[0].rows;
	m_rows = v_f[0].rows;
	m_cols = v_f[0].cols;

	// Allocate space for m_v_b & m_v_phi
	m_v_phi.resize(m_N);
	m_v_b_phi.resize(m_N * 3);
	m_v_b_delta.resize(m_M * 3);

	double dErr = dMaxErr * 2.0;
	int iIters = 0;
	std::vector<double> v_deltaOld = m_v_delta;

	double start = omp_get_wtime();

	/* Begin the real algorithm */
	while (dErr > dMaxErr && iIters < iMaxIterations)
	{
		v_deltaOld = m_v_delta;

		// Step 1: pixel-by-pixel iterations
		computePhi(v_f);

		// Step 2: frame-by-frame iterations
		computeDelta(v_f);

		// Step 3: update & check convergence criterion
		iIters++;
		dErr = computeMaxError(m_v_delta, v_deltaOld);
	}

	double end = omp_get_wtime();
	runningtime = end - start;

	/* One more round to calculate phi once delta is good enough */
	computePhi(v_f);

	#pragma omp parallel for
	for (int i = 0; i < m_v_phi.size(); i++)
	{
		m_v_phi[i] += m_v_delta[0];
		m_v_phi[i] = atan2(sin(m_v_phi[i]), cos(m_v_phi[i]));
	}

	for (int i = 0; i < m_v_delta.size(); i++)
	{
		m_v_delta[i] -= m_v_delta[0];
		m_v_delta[i] = atan2(sin(m_v_delta[i]), cos(m_v_delta[i]));
	}

	// Output the results
	v_phi = m_v_phi;
}

void AIA_CPU_Dn::computePhi(const std::vector<cv::Mat>& v_f)
{
	/* pixel-by-pixel iterations */

	/* Construct m_v_A only once and use accross multiple RHS vectors m_v_b */
	// Save the Cholesky factorized version of m_v_A 
	double dA3 = 0, dA4 = 0, dA6 = 0, dA7 = 0, dA8 = 0;

	for (int i = 0; i < m_M; i++)
	{
		double cos_delta = cos(m_v_delta[i]);
		double sin_delta = sin(m_v_delta[i]);

		dA3 += cos_delta;	dA4 += cos_delta*cos_delta;
		dA6 += sin_delta;	dA7 += cos_delta*sin_delta; dA8 += sin_delta*sin_delta;
	}

	m_v_A[0] = m_M;	m_v_A[1] = 0;	m_v_A[2] = 0;
	m_v_A[3] = dA3;	m_v_A[4] = dA4;	m_v_A[5] = 0;
	m_v_A[6] = dA6;	m_v_A[7] = dA7;	m_v_A[8] = dA8;

	/* Construct the RHS's m_v_b[N-by-3] */
	#pragma omp parallel for
	for (int j = 0; j < m_N; j++)
	{
		int y = j % m_cols;
		int x = j / m_cols;

		for (int i = 0; i < m_M; i++)
		{
			double dI = static_cast<double>(v_f[i].at<uchar>(y, x));

			m_v_b_phi[j * 3 + 0] += dI;
			m_v_b_phi[j * 3 + 1] += dI * cos(m_v_delta[i]);
			m_v_b_phi[j * 3 + 2] += dI * sin(m_v_delta[i]);
		}
	}

	/* Solve the Ax = b */
	int info = LAPACKE_dposv(LAPACK_COL_MAJOR, 'U', 3, m_N, m_v_A.data(), 3, m_v_b_phi.data(), 3);
	 /* Check for the positive definiteness */
	if (info > 0) {
		printf("The leading minor of order %i is not positive ", info);
		printf("definite;\nThe solution could not be computed.\n");
		exit(1);
	}

	#pragma omp parallel for
	for (int j = 0; j < m_N; j++)
	{
		m_v_phi[j] = atan2(-m_v_b_phi[j * 3 + 2], m_v_b_phi[j * 3 + 1]);
	}
}

void AIA_CPU_Dn::computeDelta(const std::vector<cv::Mat>& v_f)
{
	/* Frame-by-frame iterations */
	/* Construct m_v_A only once and use accross multiple RHS vectors m_v_b */
	double dA3 = 0, dA4 = 0, dA6 = 0, dA7 = 0, dA8 = 0;

	#pragma omp parallel for default(shared) reduction(+: dA3, dA4, dA6, dA7, dA8)
	for (int i = 0; i < m_N; i++)
	{
		double cos_phi = cos(m_v_phi[i]);
		double sin_phi = sin(m_v_phi[i]);

		dA3 += cos_phi;
		dA4 += cos_phi*cos_phi;
		dA6 += sin_phi;
		dA7 += cos_phi*sin_phi;
		dA8 += sin_phi*sin_phi;
	}

	m_v_A[0] = m_M;	m_v_A[1] = 0;	m_v_A[2] = 0;
	m_v_A[3] = dA3;	m_v_A[4] = dA4;	m_v_A[5] = 0;
	m_v_A[6] = dA6;	m_v_A[7] = dA7;	m_v_A[8] = dA8;

	/* Construct the RHS's m_v_b[M-by-3] */
	for (int i = 0; i < m_M; i++)
	{
		double b0 = 0, b1 = 0, b2 = 0;

		#pragma omp parallel for default(shared) reduction(+: b0, b1, b2)
		for (int j = 0; j < m_N; j++)
		{
			int y = j % m_cols;
			int x = j / m_cols;

			double dI = static_cast<double>(v_f[i].at<uchar>(y, x));
			double cos_phi = cos(m_v_phi[j]);
			double sin_phi = sin(m_v_phi[j]);

			b0 += dI;
			b1 += dI * cos_phi;
			b2 += dI * sin_phi;
		}

		m_v_b_delta[i * 3 + 0] = b0;
		m_v_b_delta[i * 3 + 1] = b1;
		m_v_b_delta[i * 3 + 2] = b2;
	}

	/* Solve the Ax = b */
	int info = LAPACKE_dposv(LAPACK_COL_MAJOR, 'U', 3, m_M, m_v_A.data(), 3, m_v_b_delta.data(), 3);
	 /* Check for the positive definiteness */
	if (info > 0) {
		printf("The leading minor of order %i is not positive ", info);
		printf("definite;\nThe solution could not be computed.\n");
		exit(1);
	}

	for (int i = 0; i < m_M; i++)
	{
		m_v_delta[i] = atan2(-m_v_b_delta[i * 3 + 2], m_v_b_delta[i * 3 + 1]);
	}
}

double AIA_CPU_Dn::computeMaxError(const std::vector<double> &v_delta, const std::vector<double>& v_deltaOld)
{
	double dMaxErr = -1.0;
	for (int i = 0; i < v_delta.size(); i++)
	{
		double dErr = abs(v_delta[i] - v_deltaOld[i]);

		if (dErr - dMaxErr > 1e-7)
		{
			dMaxErr = dErr;
		}
	}

	return dMaxErr;
}

}	// namespace AIA