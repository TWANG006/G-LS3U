#ifndef WFT_FPA_H
#define WFT_FPA_H

const int BLOCK_SIZE_256 = 256;
const int BLOCK_SIZE_128 = 128;
const int BLOCK_SIZE_64 = 64;
const int BLOCK_SIZE_16 = 16;

//!- Macro for library dll export utility
#if defined (_WIN32) 
#		if defined (WFT_FPA_DLL_EXPORTS_MODE)
#			define WFT_FPA_DLL_EXPORTS __declspec(dllexport)
#		else
#			define WFT_FPA_DLL_EXPORTS __declspec(dllimport)
#		endif
#else
#    define WFT_FPA_DLL_EXPORTS
#endif

//!- WFT_FPA engine version
#define VERSION "1.0"

//!- Debugging macro
#ifdef WFT_FPA_DEBUG_MSG
#define DEBUG_MSG(x) do {std::cout<<"[WFT_FPA_DEBUG]: "<<x<<std::endl;} while(0)
#else
#define DEBUG_MSG(x) do {} while(0);
#endif // TW_DEBUG_MSG



#include <string>
#include <vector>
#include <cfloat>
#include <cuda_runtime.h>
#include <cufft.h>

/* The fftw3 lib should either be enabled by install MKL or use the
 3rd fftw lib downloaded online */
#include <fftw3.h>


namespace WFT_FPA
{
//#ifdef WFT_FPA_DOUBLE
//	using float = double;
//	using fftw3Plan = fftw_plan;
//	using fftwf_complex = fftw_complex;
//	using cudaComplex = cufftDoubleComplex;
//#else
//	using float = float;
//	using fftw3Plan = fftwf_plan;
//	using fftwf_complex = fftwf_complex;
//	using cudaComplex = cufftComplex;
//#endif // WFT_FPA_DOUBLE
//
//	using int_t = int;
//	using uint_t = unsigned int;
//	using uint8 = unsigned char;

	enum class PARALLEL_COMPUTING_TYPE
	{
		Sequential,
		Multicore,
		CUDA
	};

namespace WFT
{
	enum class WFT_TYPE
	{
		WFF,	// Windowed Fourier Filter
		WFR		// Windowed Fourier Ridges
	};
}	// namespace WFT
}	// namespace WFT_FPA


#endif // WFT_FPA_H