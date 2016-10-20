#ifndef AIA_CPU_H
#define AIA_CPU_H

#include "WFT-FPA.h"
#include <vector>
#include "opencv2\opencv.hpp"

namespace AIA{

class WFT_FPA_DLL_EXPORTS AIA_CPU_Dn
{
public:
	AIA_CPU_Dn(const AIA_CPU_Dn&) = delete;
	AIA_CPU_Dn &operator=(const AIA_CPU_Dn&) = delete;

	AIA_CPU_Dn() = default;
	~AIA_CPU_Dn();

	/*
	  PURPOSE:
		Make this class a callable object (functor)
	  INPUTS:
		f: frames of images
		v_deltas: the phase shifts
		iMaxIterations: max iterations of the algorithm
		dErr: max error threshold
	  OUTPUTS:
		phi: calculated phi's
		time: calculation time
	*/
	void operator() (// Outputs
			 		 std::vector<double>& phi,
					 std::vector<double>& delta,
					 double &runningtime,
					 int &iters,
					 double &err,
					 // Inputs
					 const std::vector<cv::Mat>& f,
					 int iMaxIterations = 20,
					 double dMaxErr = 1e-4,
					 int iNumThreads = 1);

private:
	void computePhi(std::vector<double>& v_A, 
					std::vector<double>& v_b_phi,
					std::vector<double>& v_phi,
					const std::vector<double>& v_deltas,
					const std::vector<cv::Mat>& v_f);

	void computeDelta(std::vector<double>& v_A,
					  std::vector<double>& v_b_delta,
					  std::vector<double>& v_deltas,
					  const std::vector<double>& v_phi,
					  const std::vector<cv::Mat>& v_f);

	double computeMaxError(const std::vector<double> &v_delta, 
						   const std::vector<double> &v_deltaOld);

private:
	int m_cols;
	int m_rows;
	int m_N;	// Number of pixels	
	int m_M;	// Number of frames
};

}	// namespace AIA

#endif // !AIA_CPU_H
