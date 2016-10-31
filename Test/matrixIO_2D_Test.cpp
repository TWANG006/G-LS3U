#include "gtest\gtest.h"

#include "matrixIO.h"
#include <vector>
#include <iostream>

using namespace std;


TEST(matrixIO2D, matrixIO)
{
	vector<float> A{
		1, 2, 3, 4,
		5, 6, 7, 8
	};

	int rows = 2, cols = 4;
	char *filename = "A.bin";
	WFT_FPA::Utils::WriteMatrixToDisk("A.bin", 2, 4, A.data());

	WFT_FPA::Utils::PrintMatrixInMatlabFormat(2, 4, A.data());

	float *B = nullptr;

	WFT_FPA::Utils::ReadMatrixFromDisk(filename, &rows, &cols, &B);

	WFT_FPA::Utils::PrintMatrixInMatlabFormat(2, 4, B);

	free(B);
}