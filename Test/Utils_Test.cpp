#include "gtest\gtest.h"
#include "Utils.h"


TEST(Utils_ComplexOperation, Utils_Test)
{
	fftw_complex in1, in2, out;
	in1[0] = in2[1] = 2;	// 1 + 2i
	in2[0] = in1[1] = 1;	// 2 + 1i

	WFT_FPA::fftwComplexPrint(in1);
	WFT_FPA::fftwComplexPrint(in2);

	std::cout<<std::endl;

	WFT_FPA::fftwComplexMul(out, in1, in2);	
	EXPECT_EQ(0, out[0]);
	EXPECT_EQ(5, out[1]);

	double temp1 = WFT_FPA::fftwComplexAbs(out);
	EXPECT_EQ(5, temp1);

	WFT_FPA::fftwComplexScale(out,2);
	EXPECT_EQ(0, out[0]);
	EXPECT_EQ(10, out[1]);

	double temp2 = WFT_FPA::fftwComplexAbs(out);
	EXPECT_EQ(10, temp2);
}

//TEST(Utils_fftwComplexIO, Utils_Test)
//{
//	// single-precision test
//	std::ifstream in("f.txt");
//	fftwf_complex *f = nullptr;
//	int row, col;
//
//	if(in.is_open())
//	{
//		if(WFT_FPA::fftwComplexMatRead2D(in, f, row, col))
//		{
//			for (int i = 0; i < row*col; i++)
//			{
//				WFT_FPA::fftwComplexPrint(f[i]);
//				std::cout<<"\n";
//			}		
//		}			
//	}
//	in.close();
//
//	std::ofstream out("fout.txt", std::ios::out | std::ios::trunc);
//	WFT_FPA::fftwComplexMatWrite2D(out, f, row, col);
//
//	fftwf_free(f);
//	out.close();
//
//
//	// double-precision test
//	fftw_complex *ff = nullptr;
//	in.open("fout.txt");
//	if(in.is_open())
//	{
//		if(WFT_FPA::fftwComplexMatRead2D(in, ff, row, col))
//		{
//			for (int i = 0; i < row*col; i++)
//			{
//				WFT_FPA::fftwComplexPrint(ff[i]);
//				std::cout<<"\n";
//			}	
//		}		
//	}
//	in.close();
//
//	fftw_free(ff);
//}

