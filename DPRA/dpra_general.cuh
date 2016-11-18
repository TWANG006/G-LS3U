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

// Double-precision methods

WFT_FPA_DLL_EXPORTS void get_deltaPhi_currPhi(float *d_out_deltaPhi,
											  float *d_out_currPhi,
											  float *d_in_dphiRef,
											  float *d_in_refPhi,
											  cufftComplex *d_in_filtered,
											  const int iSize);

WFT_FPA_DLL_EXPORTS void compute_cosPhi_sinPhi(double *d_out_cosPhi,
											   double *d_out_sinPhi,
											   double *d_in_Phi,
											   const int iWidth,
											   const int iHeight,
											   const int iPaddedWidth,
											   const int iPaddedHeight,
											   const dim3 &blocks,
											   const dim3 &threads);

WFT_FPA_DLL_EXPORTS void get_A_b(double *d_out_A,
								 double *d_out_b,
								 const uchar *d_in_imgPadded,
								 const double *d_in_cosphi,
								 const double *d_in_sinphi,
								 const int iImgWidth,
								 const int iImgHeight,
								 const int iPaddedWidth,
								 const int iPaddedHeight, 
								 const dim3 &blocks,
								 const dim3 &threads);

WFT_FPA_DLL_EXPORTS void get_deltaPhi_currPhi(double *d_out_deltaPhi,
											  double *d_out_currPhi,
											  double *d_in_dphiRef,
											  double *d_in_refPhi,
											  cufftDoubleComplex *d_in_filtered,
											  const int iSize);

WFT_FPA_DLL_EXPORTS void update_dphiRef(float *d_out_dphiRef,
									    const float *d_in_dphi,
										const int iSize);

WFT_FPA_DLL_EXPORTS void update_dphiRef(double *d_out_dphiRef,
									    const double *d_in_dphi,
										const int iSize);
}
#endif // !DPRA_HYBRID_CUH
