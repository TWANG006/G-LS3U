#include "gtest\gtest.h"
#include "Utils.h"

TEST(Utils_ComplexOperation, Utils_Test)
{
	fftwf_complex in1, in2, out;
	in1[0] = in2[1] = 1;	// 1 + 2i
	in1[1] = in2[0] = 2;	// 2 + i

	WFT_FPA::fftwComplexMul<fftwf_complex>(out, in1, in2);	
	EXPECT_EQ(0, out[0]);
	EXPECT_EQ(5, out[1]);

	float temp1 = WFT_FPA::fftwComplexAbs<fftwf_complex, float>(out);
	EXPECT_EQ(5, temp1);

	WFT_FPA::fftwComplexScale<fftwf_complex, float>(out,2);
	EXPECT_EQ(0, out[0]);
	EXPECT_EQ(10, out[1]);

	float temp2 = WFT_FPA::fftwComplexAbs<fftwf_complex, float>(out);
	EXPECT_EQ(10, temp2);
}