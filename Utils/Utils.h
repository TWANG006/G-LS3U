#ifndef WFT_UTILS_H
#define WFT_UTILS_H

#include "WFT-FPA.h"
#include <fftw3.h>
#include "cuda_runtime.h"
#include "cufft.h"
#include "helper_cuda.h"
#include <fstream>

namespace WFT_FPA{
namespace Utils{

/*---------------------------------------------HOST Methods-------------------------------------------*/

WFT_FPA_DLL_EXPORTS void DisplayMemoryUsed(size_t size);
	
/* Compute the complex multiplication (x + yi)(u + vi) = (xu - yv) + (xv + yu)i */
WFT_FPA_DLL_EXPORTS void fftwComplexMul(fftwf_complex& out, const fftwf_complex& in1, const fftwf_complex& in2);
WFT_FPA_DLL_EXPORTS void fftwComplexMul(fftw_complex& out, const fftw_complex& in1, const fftw_complex& in2);
/* Compute the complex scale (x + yi)*s */
WFT_FPA_DLL_EXPORTS void fftwComplexScale(fftw_complex& out, const double s);
WFT_FPA_DLL_EXPORTS void fftwComplexScale(fftwf_complex& out, const float s);
/* Compute the absolute values abs(z) = sqrt(x^2+y^2) */
WFT_FPA_DLL_EXPORTS float fftwComplexAbs(const fftwf_complex& in);
WFT_FPA_DLL_EXPORTS double fftwComplexAbs(const fftw_complex& in);
/* Compute the phase angle (in radians) of the complex angle(z) */
WFT_FPA_DLL_EXPORTS float fftwComplexAngle(const fftwf_complex& in);
WFT_FPA_DLL_EXPORTS double fftwComplexAngle(const fftw_complex& in);
/* Formatted output of FFTW3 complex numbers */
WFT_FPA_DLL_EXPORTS void fftwComplexPrint(const fftwf_complex& in);
WFT_FPA_DLL_EXPORTS void fftwComplexPrint(const fftw_complex& in);
/* FFTW3 2D Matrix I/O : matrix is stored in Row-Major */
WFT_FPA_DLL_EXPORTS bool fftwComplexMatRead2D(std::istream& in, fftwf_complex *&f, int& rows, int& cols);
WFT_FPA_DLL_EXPORTS bool fftwComplexMatRead2D(std::istream& in, fftw_complex *&f, int& rows, int& cols);
WFT_FPA_DLL_EXPORTS void fftwComplexMatWrite2D(std::ostream& out, fftwf_complex *f, const int rows, const int cols);
WFT_FPA_DLL_EXPORTS void fftwComplexMatWrite2D(std::ostream& out, fftw_complex *f, const int rows, const int cols);

/* CUFFT 2D Matrix I/O: matrix is stored in Row-Major */
WFT_FPA_DLL_EXPORTS bool cufftComplexMatRead2D(std::istream& in, cufftComplex *&f, int& rows, int& cols);
WFT_FPA_DLL_EXPORTS bool cufftComplexMatRead2D(std::istream& in, cufftDoubleComplex *&f, int& rows, int& cols);
WFT_FPA_DLL_EXPORTS void cufftComplexMatWrite2D(std::ostream& out, cufftComplex *f, const int rows, const int cols);
WFT_FPA_DLL_EXPORTS void cufftComplexMatWrite2D(std::ostream& out, cufftDoubleComplex *f, const int rows, const int cols);
/* Formatted output of CUFFT complex numbers */
WFT_FPA_DLL_EXPORTS void cufftComplexPrint(const cufftComplex& in);
WFT_FPA_DLL_EXPORTS void cufftComplexPrint(const cufftDoubleComplex& in);

/*-------------------------------------------Device Methods-----------------------------------------------*/
/* cufft Complex number scaling */
static	__device__ __host__
inline cufftComplex ComplexScale(cufftComplex a, float s)
{
	cufftComplex c;
	c.x = s*a.x;
	c.y = s*a.y;
	return c;
}
static	__device__ __host__
inline cufftDoubleComplex ComplexScale(cufftDoubleComplex a, float s)
{
	cufftDoubleComplex c;
	c.x = s*a.x;
	c.y = s*a.y;
	return c;
}

/* cufft Complex number multiplication */
static __device__ __host__
inline cufftComplex ComplexMul(cufftComplex a, cufftComplex b)
{
	cufftComplex c;
	c.x = a.x * b.x - a.y * b.y;
	c.y = a.x * b.y + a.y * b.x;
	return c;
}
static __device__ __host__
inline cufftDoubleComplex ComplexMul(cufftDoubleComplex a, cufftDoubleComplex b)
{
	cufftDoubleComplex c;
	c.x = a.x * b.x - a.y * b.y;
	c.y = a.x * b.y + a.y * b.x;
	return c;
}

template<typename T>
void cuInitialize(T* devPtr, const T val, const size_t nwords);

}	//	namespace Utils
}	//	namespace WFT-FPA
#endif // !WFT_UTILS_H
