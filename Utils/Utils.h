#ifndef WFT_UTILS_H
#define WFT_UTILS_H

#include "WFT-FPA.h"
#include <fftw3.h>

namespace WFT_FPA{

/* Compute the complex multiplication (x + yi)(u + vi) = (xu - yv) + (xv + yu)i */
template<typename T>
void fftwComplexMul(T& out, const T& in1, const T& in2);

/* Compute the complex scale (x + yi)*s */
template<typename T, typename R>
void fftwComplexScale(T& out, const R s);

template<typename T, typename R>
R fftwComplexAbs(const T& in);

/* Formatted output of FFTW3 complex numbers */
template<typename T>
void fftwComplexPrint(const T& in);

WFT_FPA_DLL_EXPORTS int add(int a, int b);
	
}	//namespace WFT-FPA
#endif // !WFT_UTILS_H
