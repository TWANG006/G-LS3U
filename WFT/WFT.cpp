#include "WFT.h"
//#include <iostream>

namespace WFT_FPA{
namespace WFT{

/* WFT Specific Utility functions */
/* FUNCTION: int getFirstGreater(int x) */
int getFirstGreater(int x)
{
	int first = 0, last = int(OPT_FFT_SIZE.size()) -1;
	while (first <= last)
	{
		int mid = (first + last) / 2;

		if(OPT_FFT_SIZE[mid] >= x)
			last = mid - 1;
		else
			first = mid + 1;
	}
	return last + 1 == OPT_FFT_SIZE.size() ? -1 : last + 1;
}

/* Double precision Data structures for WFT Results */
WFT2_HostResults::WFT2_HostResults()
	: m_filtered(nullptr)
	, m_wx(nullptr)
	, m_wy(nullptr)
	, m_phase(nullptr)
	, m_phase_comp(nullptr)
	, m_b(nullptr)
	, m_r(nullptr)
	, m_cxx(nullptr)
	, m_cyy(nullptr)
{}

WFT2_HostResults::~WFT2_HostResults()
{
	fftw_free(m_filtered);	m_filtered = nullptr;

	free(m_wx);				m_wx = nullptr;
	free(m_phase);			m_phase = nullptr;
	free(m_phase_comp);		m_phase_comp = nullptr;
	free(m_b);				m_b = nullptr;
	free(m_r);				m_r = nullptr;
	free(m_cxx);			m_cxx = nullptr;
	free(m_cyy);			m_cyy = nullptr;
}

/* Single precision Data structures for WFT Results */
WFT2_HostResultsF::WFT2_HostResultsF()
	: m_filtered(nullptr)
	, m_wx(nullptr)
	, m_wy(nullptr)
	, m_phase(nullptr)
	, m_phase_comp(nullptr)
	, m_b(nullptr)
	, m_r(nullptr)
	, m_cxx(nullptr)
	, m_cyy(nullptr)
{}

WFT2_HostResultsF::~WFT2_HostResultsF()
{
	fftwf_free(m_filtered);	m_filtered = nullptr;

	free(m_wx);				m_wx = nullptr;
	free(m_phase);			m_phase = nullptr;
	free(m_phase_comp);		m_phase_comp = nullptr;
	free(m_b);				m_b = nullptr;
	free(m_r);				m_r = nullptr;
	free(m_cxx);			m_cxx = nullptr;
	free(m_cyy);			m_cyy = nullptr;
}


/* Double precision Data structures for WFT Results */
WFT2_DeviceResults::WFT2_DeviceResults()
	: m_d_filtered(nullptr)
	, m_d_wx(nullptr)
	, m_d_wy(nullptr)
	, m_d_phase(nullptr)
	, m_d_phase_comp(nullptr)
	, m_d_b(nullptr)
	, m_d_r(nullptr)
	, m_d_cxx(nullptr)
	, m_d_cyy(nullptr)
{}

WFT2_DeviceResults::~WFT2_DeviceResults()
{
	cudaFree(m_d_filtered);			m_d_filtered = nullptr;

	cudaFree(m_d_wx);				m_d_wx = nullptr;
	cudaFree(m_d_phase);			m_d_phase = nullptr;
	cudaFree(m_d_phase_comp);		m_d_phase_comp = nullptr;
	cudaFree(m_d_b);				m_d_b = nullptr;
	cudaFree(m_d_r);				m_d_r = nullptr;
	cudaFree(m_d_cxx);				m_d_cxx = nullptr;
	cudaFree(m_d_cyy);				m_d_cyy = nullptr;
}


/* Single precision Data structures for WFT Results */
WFT2_DeviceResultsF::WFT2_DeviceResultsF()
	: m_d_filtered(nullptr)
	, m_d_wx(nullptr)
	, m_d_wy(nullptr)
	, m_d_phase(nullptr)
	, m_d_phase_comp(nullptr)
	, m_d_b(nullptr)
	, m_d_r(nullptr)
	, m_d_cxx(nullptr)
	, m_d_cyy(nullptr)
{}

WFT2_DeviceResultsF::~WFT2_DeviceResultsF()
{
	cudaFree(m_d_filtered);			m_d_filtered = nullptr;

	cudaFree(m_d_wx);				m_d_wx = nullptr;
	cudaFree(m_d_phase);			m_d_phase = nullptr;
	cudaFree(m_d_phase_comp);		m_d_phase_comp = nullptr;
	cudaFree(m_d_b);				m_d_b = nullptr;
	cudaFree(m_d_r);				m_d_r = nullptr;
	cudaFree(m_d_cxx);				m_d_cxx = nullptr;
	cudaFree(m_d_cyy);				m_d_cyy = nullptr;
}

}	// namespace WFT
}	// namespace WFT_FPA