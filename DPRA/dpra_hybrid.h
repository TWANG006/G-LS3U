#ifndef DPRA_HYBRID_H
#define DPRA_HYBRID_H

#include "dpra_hybrid.cuh"

#include "WFT-FPA.h"
#include "Utils.h"
#include "WFT2_CUDA.h"

#include <memory>
#include <vector>
#include "opencv2\opencv.hpp"

namespace DPRA{

class WFT_FPA_DLL_EXPORTS DPRA_HYBRID
{
public:
	DPRA_HYBRID() = delete;
	DPRA_HYBRID(const DPRA_HYBRID&) = delete;
	DPRA_HYBRID &operator=(const DPRA_HYBRID&) = delete;

public:
	DPRA_HYBRID(const double *v_Phi0, 
				const int iWidth, const int iHeight, 
				const int irefUpdateRate = 1,
				const int iNumThreads = 1);
	
	~DPRA_HYBRID();

	void operator() (const std::vector<cv::Mat> &f, 
					 std::vector<std::vector<double>> &dPhi_Sum,
					 double &time);
	void operator() (const std::vector<std::string> &fileNames,
					 std::vector<std::vector<double>> &dPhi_Sum,
					 double &time);

	void dpra_per_frame(const cv::Mat &f, 
						std::vector<double> &dPhi,
						double &time);

	void update_ref_phi();

private:
	// Context parameters
	int m_iWidth;
	int m_iHeight;
	int m_iPaddedWidth;
	int m_iPaddedHeight;

	int m_rr;
	int m_iNumThreads;

	/* DPRA parameters */
	//std::vector<double> m_PhiRef;	// Reference phase
	//std::vector<double> m_PhiCurr;	// Current phase
	double *m_h_deltaPhi;

	double *m_d_PhiRef;
	double *m_d_deltaPhi;
	double *m_d_PhiCurr;
	
	double *m_h_A;			// Matrix A
	double *m_h_b;			// Vector b

	double *m_d_A;
	double *m_d_b;

	double *m_d_cosPhi;		// Padded
	double *m_d_sinPhi;		// Padded
	uchar *m_h_img;
	uchar *m_d_img;			
	uchar *m_d_img_Padded;	// Padded

	/* Parameters used for WFF */
	WFT_FPA::WFT::WFT2_DeviceResults m_d_z;
	WFT_FPA::WFT::WFT2_CUDA m_WFT;	
	cufftDoubleComplex *m_h_deltaPhiWFT;
	cufftDoubleComplex *m_d_deltaPhiWFT;

	cudaEvent_t m_d_event_start;
	cudaEvent_t m_d_event_1;
	cudaEvent_t m_d_event_2;
	cudaEvent_t m_d_event_3;
	cudaEvent_t m_d_event_4;
	cudaEvent_t m_d_event_5;
	cudaEvent_t m_d_event_6;
	cudaEvent_t m_d_event_7;

	dim3 m_threads2D;
	dim3 m_blocks_2Dshrunk;
	dim3 m_blocks_2D;
};

}	// namespace DPRA

#endif // !DPRA_HYBRID_H
