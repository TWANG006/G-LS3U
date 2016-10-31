#ifndef DPRA_CPUF_H
#define DPRA_CPUF_H

#include "WFT-FPA.h"
#include "Utils.h"
#include "WFT2_CPUf.h"

#include <memory>
#include <vector>
#include "opencv2\opencv.hpp"

namespace DPRA{

class WFT_FPA_DLL_EXPORTS DPRA_CPUF
{
public:
	DPRA_CPUF() = delete;
	DPRA_CPUF(const DPRA_CPUF&) = delete;
	DPRA_CPUF &operator=(const DPRA_CPUF&) = delete;

/*
 Purpose:
	Constructor
 Inputs:
	v_Phi0: array of iWidth*iHeight initial phase
	iWidth, iHeight: image size
	ireUpdateRate: how many frames would reference phase be updated
	iNumThreads: number of CPU-threads used					*/
	DPRA_CPUF(const float *v_Phi0, 
			  const int iWidth, const int iHeight, 
			  const int irefUpdateRate = 1,
			  const int iNumThreads = 1);
	
	~DPRA_CPUF();

public:
/*
 Purpose:
	Compute the DPRA of f.size() frames of fringe patterns
 Inputs:
	f: some frames of fringe patterns in CV_U8C1 format
 Outputs:
	dPhi_Sum: delta phase change for every frame
	iTime: running time										*/
	void operator() (const std::vector<cv::Mat> &f, 
					 std::vector<std::vector<float>> &dPhi_Sum,
					 double &time);
	void operator() (const std::vector<std::string> &fileNames,
					 std::vector<std::vector<float>> &dPhi_Sum,
					 double &time);

/*
 Purpose:
	Compute the DPRA of 1 frame
 Inputs:
	f: 1 frame of fringe pattern
 Outputs:
	dPhi: delta phase change for 1 frame					*/
	void dpra_per_frame(const cv::Mat &f, 
						std::vector<float> &dPhi,
						double &time);

	void update_ref_phi();

private:
	// Context parameters
	int m_iWidth;
	int m_iHeight;
	int m_rr;
	int m_iNumThreads;

	/* DPRA parameters */
	std::vector<float> m_PhiRef;		// Reference phase
	std::vector<float> m_PhiCurr;		// Current phase
	std::vector<float> m_A;			// Matrix A
	std::vector<float> m_b;			// Vector b

	/* Parameters used for WFF */
	WFT_FPA::WFT::WFT2_HostResultsF m_z;
	WFT_FPA::WFT::WFT2_cpuF m_WFT;	
	fftwf_complex *m_dPhiWFT;
};

}	//	namespace DPRA

#endif // !DPRA_CPUF_H
