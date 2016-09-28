#include <iostream>
#include <vector>

#include "Utils.h"
#include "WFT.h"
#include "WFT2_CPU.h"
#include "gtest\gtest.h"

using namespace std;

class Test
{
public:



	void operator()(int a, int b, int c, int d)
	{
		cout<<a+b+c+d<<endl;
	}

private:
	int x;
};

int main(int argc, char** argv)
{
	::testing::InitGoogleTest(&argc, argv);
	RUN_ALL_TESTS();

	return 0;
}