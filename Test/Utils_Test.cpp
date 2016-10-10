#include "gtest\gtest.h"
#include "Utils.h"


TEST(Utils_cufftIO, Utils_Test)
{
		// single-precision test
	std::ifstream in("f.txt");
	cufftComplex *f = nullptr;
	int row, col;

	if(in.is_open())
	{
		if(WFT_FPA::Utils::cufftComplexMatRead2D(in, f, row, col))
		{
			for (int i = 0; i < row*col; i++)
			{
				WFT_FPA::Utils::cufftComplexPrint(f[i]);
				std::cout<<"\n";
			}		
		}			
	}
	in.close();

	std::ofstream out("fout.txt", std::ios::out | std::ios::trunc);
	WFT_FPA::Utils::cufftComplexMatWrite2D(out, f, row, col);

	free(f);
	out.close();


	// double-precision test
	cufftDoubleComplex *ff = nullptr;
	in.open("fout.txt");
	if(in.is_open())
	{
		if(WFT_FPA::Utils::cufftComplexMatRead2D(in, ff, row, col))
		{
			for (int i = 0; i < row*col; i++)
			{
				WFT_FPA::Utils::cufftComplexPrint(ff[i]);
				std::cout<<"\n";
			}	
		}		
	}
	in.close();

	free(ff);
}

//TEST(Utils_ComplexOperation, Utils_Test)
//{
//	fftw_complex in1, in2, out;
//	in1[0] = in2[1] = 2;	// 1 + 2i
//	in2[0] = in1[1] = 1;	// 2 + 1i
//
//	WFT_FPA::Utils::fftwComplexPrint(in1);
//	WFT_FPA::Utils::fftwComplexPrint(in2);
//
//	std::cout<<std::endl;
//
//	WFT_FPA::Utils::fftwComplexMul(out, in1, in2);	
//	EXPECT_EQ(0, out[0]);
//	EXPECT_EQ(5, out[1]);
//
//	double temp1 = WFT_FPA::Utils::fftwComplexAbs(out);
//	EXPECT_EQ(5, temp1);
//
//	WFT_FPA::Utils::fftwComplexScale(out,2);
//	EXPECT_EQ(0, out[0]);
//	EXPECT_EQ(10, out[1]);
//
//	double temp2 = WFT_FPA::Utils::fftwComplexAbs(out);
//	EXPECT_EQ(10, temp2);
//
//	double angle = WFT_FPA::Utils::fftwComplexAngle(out);
//
//	std::cout << "Angle of ";
//	WFT_FPA::Utils::fftwComplexPrint(out);
//	std::cout << "is: " << angle << std::endl;
//}

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



