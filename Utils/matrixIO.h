#ifndef MATRIXIO_H
#define MATRIXIO_H

#include <cstdio>
#include <cstdlib>
#include <cctype>
#include "cuda_runtime.h"

/* 2D matrix I/O in Row-major using binary file format */
inline long ELT(const int iNumCols, const int y, const int x)
{
	return (long(y)*long(iNumCols) + long(x));
}

inline int ReadMatrixSizeFromStream(FILE * file, int * rows, int * cols)
{
	if (fread(rows, sizeof(int), 1, file) < 1)
		return 1;
	if (fread(cols, sizeof(int), 1, file) < 1)
		return 1;

	return 0;
}
inline int WriteMatrixHeaderToStream(FILE * file, int rows, int cols)
{
	if (fwrite(&rows, sizeof(int), 1, file) < 1)
		return 1;
	if (fwrite(&cols, sizeof(int), 1, file) < 1)
		return 1;

	return 0;
}

namespace WFT_FPA{
namespace Utils{

//Read matrix into cudaHostAlloc memory
template <class T>
int cuReadMatrixFromDisk(const char * filename, int * rows, int * cols, T ** matrix);
template <class T>
int ReadMatrixFromDisk(const char * filename, int * rows, int * cols, T ** matrix);
template <class T>
int ReadMatrixFromStream(FILE * file, int rows, int cols, T * matrix);


template <class T>
int WriteMatrixToDisk(const char * filename, int rows, int cols, T * matrix);
template <class T>
int WriteMatrixToStream(FILE * file, int rows, int cols, T * matrix);

// prints the matrix to standard output in Matlab format
template <class T>
void PrintMatrixInMatlabFormat(int rows, int cols, T * U);

#include "matrixIO.inl"

}	// namespace Utils
}	// namespace WFT_FPA

#endif // !MATRIXIO_H
