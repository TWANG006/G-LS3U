#include <iostream>
#include <vector>

#include "Utils.h"
#include "WFT.h"
#include "WFT2_CPUf.h"
#include "gtest\gtest.h"
#include "opencv2\opencv.hpp"
#include "aia_cudaf.h"
#include "dpra_cpu.h"
#include "aia_cpuf.h"
#include "dpra_cudaf.h"
#include "cuda_testt.h"

using namespace std;


int main(int argc, char** argv)
{
	/* Need to be revised*/

	::testing::InitGoogleTest(&argc, argv);
	RUN_ALL_TESTS();

	return 0;
}