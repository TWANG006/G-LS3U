#ifndef DPRA_CPU_H
#define DPRA_CPU_H

#include "WFT-FPA.h"
#include "Utils.h"
#include "WFT2_CPU.h"

#include <memory>
#include <string>
#include <vector>
#include "opencv2\opencv.hpp"

namespace DPRA{

class WFT_FPA_DLL_EXPORTS DPRA_CPU
{
public:
	DPRA_CPU() = delete;
	DPRA_CPU(const DPRA_CPU&) = delete;
	DPRA_CPU &operator=(const DPRA_CPU&) = delete;

/*
 Purpose:
	Constructor
 Inputs:
	v_Phi0: array of iWidth*iHeight initial phase
	iWidth, iHeight: image size
	ireUpdateRate: how many frames would reference phase be updated
	iNumThreads: number of CPU-threads used					*/
	DPRA_CPU(const double *v_Phi0, 
			 const int iWidth, const int iHeight, 
			 const int irefUpdateRate = 1,
			 const int iNumThreads = 1);
	
	~DPRA_CPU();

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
					 std::vector<std::vector<double>> &dPhi_Sum,
					 double &time);
	void operator() (const std::vector<std::string> &fileNames,
					 std::vector<std::vector<double>> &dPhi_Sum,
					 double &time);

/*
 Purpose:
	Compute the DPRA of 1 frame
 Inputs:
	f: 1 frame of fringe pattern
 Outputs:
	dPhi: delta phase change for 1 frame					*/
	void dpra_per_frame(const cv::Mat &f, 
						std::vector<double> &dPhi,
						double &time);

	void update_ref_phi();

public:
	// Context parameters
	int m_iWidth;
	int m_iHeight;
	int m_rr;
	int m_iNumThreads;

	/* DPRA parameters */
	std::vector<double> m_PhiRef;		// Reference phase
	std::vector<double> m_PhiCurr;		// Current phase
	std::vector<double> m_A;			// Matrix A
	std::vector<double> m_b;			// Vector b

	/* Parameters used for WFF */
	WFT_FPA::WFT::WFT2_HostResults m_z;
	WFT_FPA::WFT::WFT2_cpu m_WFT;	
	fftw_complex *m_dPhiWFT;
};

}	//	namespace DPRA

#endif // !DPRA_CPU_H
