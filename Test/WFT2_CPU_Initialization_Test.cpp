#include "gtest\gtest.h"

#include "WFT2_CPU.h"

TEST(WFT2_CPU_Init, WFT2_CPU)
{
	WFT_FPA::WFT::WFT2_HostResults z;
	WFT_FPA::WFT::WFT2_cpu(128, 128, WFT_FPA::WFT::WFT_TYPE::WFF,z);


}