#ifndef AIA_CPUF_H
#define AIA_CPUF_H

#include "WFT-FPA.h"
#include <vector>
#include "opencv2\opencv.hpp"

namespace AIA{

/*
 PURPOSE:
	Implementation of the AIA using dense matrix solver, i.e,
	construct A once and solve for multiple b's on the RHS
*/
class WFT_FPA_DLL_EXPORTS AIA_CPU_DnF
{
public:
	AIA_CPU_DnF(const AIA_CPU_DnF&) = delete;
	AIA_CPU_DnF &operator=(const AIA_CPU_DnF&) = delete;

	AIA_CPU_DnF() = default;
	~AIA_CPU_DnF();

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
			 		 std::vector<float>& v_phi,
					 std::vector<float>& v_deltas,
					 double &runningtime,
					 int &iters,
					 float &err,
					 // Inputs
					 const std::vector<cv::Mat>& v_f,
					 int iMaxIterations = 20,
					 float dMaxErr = 1e-4,
					 int iNumThreads = 1);

private:
	void computePhi(std::vector<float>& v_A, 
					std::vector<float>& v_b_phi,
					std::vector<float>& v_phi,
					const std::vector<float>& v_deltas,
					const std::vector<cv::Mat>& v_f);

	void computeDelta(std::vector<float>& v_A, 
					  std::vector<float>& v_b_delta,
					  std::vector<float>& v_deltas,
					  const std::vector<float>& v_phi,
					  const std::vector<cv::Mat>& v_f);

	float computeMaxError(const std::vector<float> &v_delta, 
						  const std::vector<float> &v_deltaOld);

private:
	int m_cols;
	int m_rows;
	int m_N;	// Number of pixels	
	int m_M;	// Number of frames
};

}	// namespace AIA

#endif // !AIA_CPUF_H
