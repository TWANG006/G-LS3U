#include "dpra_cpu.h"

#include <omp.h>
#include <mkl.h>
#include "mem_manager.h"

namespace WFT_FPA{
namespace DPRA{

DPRA_CPU::DPRA_CPU(const double *v_Phi0,
				   const int iWidth, const int iHeight,
				   const int irefUpdateRate,
				   const int iNumThreads)
	: m_iWidth(iWidth)
	, m_iHeight(iHeight)
	, m_rr(irefUpdateRate)
	, m_iNumThreads(iNumThreads)
	, m_PhiRef(iWidth*iHeight, 0)
	, m_PhiCurr(iWidth*iHeight, 0)
	, m_A(iNumThreads * 9, 0)
	, m_b(iNumThreads * 3, 0)
	, m_WFT(WFT::WFT2_cpu(iWidth, iHeight, WFT::WFT_TYPE::WFF, 
						  20, -0.2, 0.2, 0.1, 20, -0.2, 0.2, 0.1, 15, 
						  m_z))
	/*, m_deltaPhi(iWidth*iHeight, 0)*/
	//, m_deltaPhiRef(iWidth*iHeight, 0)
{
	int iSize = iWidth*iHeight;

	// Allocate memory
	/*WFT_FPA::Utils::hcreateptr(m_A, 9 * iNumThreads);
	WFT_FPA::Utils::hcreateptr(m_b, 3 * iNumThreads);
	WFT_FPA::Utils::hcreateptr(m_PhiRef, iSize);
	WFT_FPA::Utils::hcreateptr(m_deltaPhi, iSize);
	WFT_FPA::Utils::hcreateptr(m_deltaPhiRef, iSize);*/

	m_dPhiWFT = (fftw_complex*)fftw_malloc(sizeof(fftw_complex*)*iSize);

	// Load the initial phase map
	for (int i = 0; i < iSize; i++)
	{
		m_PhiRef[i] = v_Phi0[i];
	}

	// Construct the WFT

}

DPRA_CPU::~DPRA_CPU()
{
	/*if (m_A)	WFT_FPA::Utils::hdestroyptr(m_A);
	if (m_b)	WFT_FPA::Utils::hdestroyptr(m_b);
	if (m_PhiRef) WFT_FPA::Utils::hdestroyptr(m_PhiRef);*/
	fftw_free(m_dPhiWFT);	m_dPhiWFT = nullptr;
}


void DPRA_CPU::operator()(const std::vector<cv::Mat> &f, 
						  std::vector<std::vector<double>> &dPhi_Sum,
						  double &time)
{
	if (f.empty())
	{
		std::cout << "No input fringe patterns! Error!" << std::endl;
		return;
	}
	
	std::vector<double> dPhi_per_frame(m_iWidth*m_iHeight, 0);

	time = 0;

	/* Iterate though all frames */
	for (int i = 0; i < f.size(); i++)
	{
		double time_1frame = 0;
		/* Compute the DPRA for 1 frame */
		dpra_per_frame(f[i], dPhi_per_frame, time_1frame);
		time += time_1frame;

		/* Save the 1 frame results */
		dPhi_Sum.push_back(dPhi_per_frame);

		/* Update reference */
		if (i % m_rr == 0)
		{
			m_PhiRef = m_PhiCurr;
		}
	}
}

void DPRA_CPU::dpra_per_frame(const cv::Mat &f, 
							  std::vector<double> &dPhi,
							  double &time)
{
	// If the input images have the same size as initialized
	if (f.cols != m_iWidth || f.rows != m_iHeight)
		{
			std::cout << "Size of the input FP is not equal to the size in the initialization " << std::endl;
			return;
		}

	int iSize = m_iWidth * m_iHeight;
	
	// If demodulatedPhi has not been allocated, allocate it
	if (dPhi.empty())
		dPhi.resize(iSize);

	// Set the default number of threads
	omp_set_num_threads(m_iNumThreads);

	/* Main algorithm for DPRA using windows size 3-by-3 */
	// 1. Least-squre fitting
	double start = omp_get_wtime();
	#pragma omp parallel num_threads(m_iNumThreads)
	{
		int nthreads = omp_get_num_threads();
		int idThread = omp_get_thread_num();

		// Intermediate IDs of A and b to solve Ax = b
		int id_A = idThread * 9;
		int id_B = idThread * 3;

		for (int i = idThread; i < m_iHeight; i += nthreads)
		{
			for (int j = 0; j < m_iWidth; j++)
			{
				int idImg = i*m_iWidth + j;

				// Use local sum's to avoid false sharing
				double sum_cos = 0, sum_sin = 0, sum_sincos = 0, sum_sin2 = 0, sum_cos2 = 0;
				double sum_ft = 0, sum_ft_cos = 0, sum_ft_sin = 0;

				// Get the neighbors
				for (int m = std::max(i - 1, 0); m <= std::min(i + 1, m_iHeight - 1); m++)
				{
					for (int n = std::max(j - 1, 0); n <= std::min(j + 1, m_iWidth - 1); n++)
					{
						int idNeighbor = m*m_iWidth + n;

						double cos_phi = cos(m_PhiRef[idNeighbor]);
						double sin_phi = sin(m_PhiRef[idNeighbor]);
						double ft = static_cast<double>(f.at<uchar>(m, n));

						// Elements of A
						sum_cos += cos_phi;
						sum_sin += sin_phi;
						sum_sincos += cos_phi * sin_phi;
						sum_sin2 += sin_phi * sin_phi;
						sum_cos2 += cos_phi * cos_phi;

						// Elements of B
						sum_ft += ft;
						sum_ft_cos += ft * cos_phi;
						sum_ft_sin += ft * sin_phi;
					}
				}

				// Construct A & B
				m_A[id_A + 0] = 9;			m_A[id_A + 1] = 0;			m_A[id_A + 2] = 0;
				m_A[id_A + 3] = sum_cos;	m_A[id_A + 4] = sum_cos2;	m_A[id_A + 5] = 0;
				m_A[id_A + 6] = sum_sin;	m_A[id_A + 7] = sum_sincos;	m_A[id_A + 8] = sum_sin2;

				m_b[id_B + 0] = sum_ft;
				m_b[id_B + 1] = sum_ft_cos;
				m_b[id_B + 2] = sum_ft_sin;

				// Solve Ax = b & Check for the positive definiteness
				int info = LAPACKE_dposv(LAPACK_COL_MAJOR, 'U', 3, 1, m_A.data() + id_A, 3, m_b.data() + id_B, 3);
				if (info > 0)
				{
					printf("The leading minor of order %i is not positive ", info);
					printf("definite;\nThe solution could not be computed.\n");
					return;
				}
				
				// Update delta phi
				double m_deltaPhi = atan2(-m_b[id_B + 2], m_b[id_B + 1]);
				m_dPhiWFT[idImg][0] = cos(m_deltaPhi);
				m_dPhiWFT[idImg][1] = sin(m_deltaPhi);
			}
		}
	}
	double end = omp_get_wtime();
	time = end - start;		// DPRA 1-frame time

	// 2. WFF denoising
	double dWFT_time = 0;
	m_WFT(m_dPhiWFT, m_z, dWFT_time);

	time += dWFT_time;		// DPRA + WFF

	// Update phi using the calculated delta phi
	for (int i = 0; i < iSize; i++)
	{
		dPhi[i] = atan2(m_dPhiWFT[i][1], m_dPhiWFT[i][0]);
		m_PhiCurr[i] = atan2(sin(dPhi[i] + m_PhiRef[i]), cos(dPhi[i] + m_PhiRef[i]));
	}
}

}	// namespace DPRA
}	// namespace WFT_FPA