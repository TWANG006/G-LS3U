#ifndef DPRA_HYBRIDF_H
#define DPRA_HYBRIDF_H

#include "WFT-FPA.h"
#include "Utils.h"
#include "WFT2_CUDAf.h"

#include <memory>
#include <vector>
#include "opencv2\opencv.hpp"

namespace DPRA{

class WFT_FPA_DLL_EXPORTS DPRA_HYBRIDF
{
public:
	DPRA_HYBRIDF() = delete;
	DPRA_HYBRIDF(const DPRA_HYBRIDF&) = delete;
	DPRA_HYBRIDF &operator=(const DPRA_HYBRIDF&) = delete;

public:
	DPRA_HYBRIDF(const float *v_Phi0, 
				 const int iWidth, const int iHeight, 
				 const int irefUpdateRate = 1,
				 const int iNumThreads = 1);
	
	~DPRA_HYBRIDF();

	void operator() (const std::vector<cv::Mat> &f, 
					 std::vector<std::vector<float>> &dPhi_Sum,
					 double &time);
	void operator() (const std::vector<std::string> &fileNames,
					 std::vector<std::vector<float>> &dPhi_Sum,
					 double &time);

	void dpra_per_frame(const cv::Mat &f, 
						std::vector<float> &dPhi,
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
	//std::vector<float> m_PhiRef;	// Reference phase
	//std::vector<float> m_PhiCurr;	// Current phase
	float *m_h_deltaPhi;

	float *m_d_PhiRef;
	float *m_d_dPhiRef;
	float *m_d_deltaPhi;
	float *m_d_PhiCurr;
	
	float *m_h_A;			// Matrix A
	float *m_h_b;			// Vector b

	float *m_d_A;
	float *m_d_b;

	float *m_d_cosPhi;		// Padded
	float *m_d_sinPhi;		// Padded
	uchar *m_h_img;
	uchar *m_d_img;			
	uchar *m_d_img_Padded;	// Padded

	/* Parameters used for WFF */
	WFT_FPA::WFT::WFT2_DeviceResultsF m_d_z;
	WFT_FPA::WFT::WFT2_CUDAF m_WFT;	
	cufftComplex *m_h_deltaPhiWFT;
	cufftComplex *m_d_deltaPhiWFT;

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



#endif // !DPRA_HYBRIDF_H
