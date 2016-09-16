#ifndef WFT_FPA_H
#define WFT_FPA_H


#define WFT_FPA_PI 3.14159265358979323846
#define WFT_FPA_TWOPI 6.28318530717958647692

#define BLOCK_SIZE_256 256
#define BLOCK_SIZE_128 128
#define BLOCK_SIZE_64 64

//!- Macro for library dll export utility
#if defined (_WIN32)
#    if defined (WFT_FPA_DLL_EXPORTS_MODE)
#        define WFT_FPA_DLL_EXPORTS __declspec(dllexport)
#    else
#        define WFT_FPA_DLL_EXPORTS __declspec(dllimport)
#    endif
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

/* The fftw3 lib should either be enabled by install MKL or use the
 3rd fftw lib downloaded online */
#include <fftw3.h>


namespace WFT_FPA
{
#ifdef WFT_FPA_DOUBLE
	using real_t = double;
	using fftw3Plan = fftw_plan;
	using fftw3Complex = fftw_complex;
#else
	using real_t = float;
	using fftw3Plan = fftwf_plan;
	using fftw_complex = fftwf_complex;
#endif // WFT_FPA_DOUBLE
	using int_t = int;
	using uint_t = unsigned int;
	using uint8 = unsigned char;

	enum class WFT_FPA_TYPE
	{
		WFF,
		WFR
	};

	enum class PARALLEL_COMPUTING_TYPE
	{
		Sequential,
		Multicore,
		CUDA
	};
} //WFT_FPA


#endif // WFT_FPA_H