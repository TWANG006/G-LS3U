#include "Utils.h"
#include <math.h>
#include <iostream>

namespace WFT_FPA
{

template<typename T>
void fftwComplexMul(T& out, const T& in1, const T& in2)
{
	out[0] = in1[0] * in2[0] - in1[1] * in2[1];
	out[1] = in1[0] * in2[1] + in1[1] * in2[0];
}

template<typename T, typename R>
void fftwComplexScale(T& out, const R s)
{
	out[0] = out[0] * s;
	out[1] = out[1] * s;
}

template<typename T, typename R>
R fftwComplexAbs(const T& in)
{
	return sqrt(in[0]*in[0] + in[1]*in[1]);
}

template<typename T>
void fftwComplexPrint(const T& in)
{
	std::cout<<in[0]<<"+"<<"("<<in[1]<<"i)";
}

int add(int a, int b)
{
	return a+b;
}


/* Template type assignment */
template WFT_FPA_DLL_EXPORTS void fftwComplexMul<fftwf_complex>(fftwf_complex& out, const fftwf_complex& in1, const fftwf_complex& int2);
template WFT_FPA_DLL_EXPORTS void fftwComplexMul<fftw_complex>(fftw_complex& out, const fftw_complex& in1, const fftw_complex& int2);
template WFT_FPA_DLL_EXPORTS void fftwComplexScale<fftwf_complex,float>(fftwf_complex& out, const float s);
template WFT_FPA_DLL_EXPORTS void fftwComplexScale<fftw_complex, double>(fftw_complex& out, const double s);
template WFT_FPA_DLL_EXPORTS float fftwComplexAbs<fftwf_complex, float>(const fftwf_complex& in);
template WFT_FPA_DLL_EXPORTS double fftwComplexAbs<fftwf_complex, double>(const fftwf_complex& in);
template WFT_FPA_DLL_EXPORTS void fftwComplexPrint<fftwf_complex>(const fftwf_complex & in);
template WFT_FPA_DLL_EXPORTS void fftwComplexPrint<fftw_complex>(const fftw_complex & in);

} // WFT_FPA