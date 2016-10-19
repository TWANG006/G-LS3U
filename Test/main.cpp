#include <iostream>
#include <vector>

#include "Utils.h"
#include "WFT.h"
#include "WFT2_CPU.h"
#include "gtest\gtest.h"
#include "opencv2\opencv.hpp"

using namespace std;


int main(int argc, char** argv)
{
	::testing::InitGoogleTest(&argc, argv);
	RUN_ALL_TESTS();

	return 0;
}