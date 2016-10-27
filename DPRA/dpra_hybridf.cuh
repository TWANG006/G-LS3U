#ifndef DPRA_HYBRID_CUH
#define DPRA_HYBRID_CUH

#include "WFT-FPA.h"
#include "opencv2\opencv.hpp"

namespace DPRA{

WFT_FPA_DLL_EXPORTS void load_img_padding(uchar *d_out_img_Padded,
										  const uchar *d_in_img,
										  int iImgWidth,
										  int iImgHeight,
										  int iPaddedWidth,
										  int iPaddedHeight,
										  const dim3 &blocks,
										  const dim3 &threads);

WFT_FPA_DLL_EXPORTS void compute_cosPhi_sinPhi(float *d_out_cosPhi,
											   float *d_out_sinPhi,
											   float *d_in_Phi,
											   const int iWidth,
											   const int iHeight,
											   const int iPaddedWidth,
											   const int iPaddedHeight,
											   const dim3 &blocks,
											   const dim3 &threads);

WFT_FPA_DLL_EXPORTS void get_A_b(float *d_out_A,
								 float *d_out_b,
								 const uchar *d_in_imgPadded,
								 const float *d_in_cosphi,
								 const float *d_in_sinphi,
								 const int iImgWidth,
								 const int iImgHeight,
								 const int iPaddedWidth,
								 const int iPaddedHeight, 
								 const dim3 &blocks,
								 const dim3 &threads);

WFT_FPA_DLL_EXPORTS void get_deltaPhi_currPhi(float *d_out_deltaPhi,
											  float *d_out_currPhi,
											  float *d_in_refPhi,
											  cufftComplex *d_in_filtered,
											  const int iSize);
}
#endif // !DPRA_HYBRID_CUH
