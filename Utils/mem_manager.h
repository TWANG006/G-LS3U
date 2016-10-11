#ifndef MEM_MANAGER_H
#define MEM_MANAGER_H

#include <cstdio>
#include <cstdlib>
#include "cuda_runtime.h"
#include "helper_cuda.h"

namespace WFT_FPA{
namespace Utils{

//!= Host memory allo/deallo-cation methods.
template<typename T>
void hcreateptr(T*& ptr, 
				size_t size);
template<typename T>
void hcreateptr(T**& ptr, 
				size_t row, 
				size_t col);
template<typename T>
void hcreateptr(T***& ptr, 
				size_t row, 
				size_t col, 
				size_t height);
template<typename T>
void hcreateptr(T****& ptr, 
				size_t a, 
				size_t b, 
				size_t c, 
				size_t d);

template<typename T>
void hdestroyptr(T*& ptr);
template<typename T>
void hdestroyptr(T**& ptr);
template<typename T>
void hdestroyptr(T***& ptr);
template<typename T>
void hdestroyptr(T****& ptr);

//!- Pinned host memory allo/deallo-cation methods
template<typename T>
void cucreateptr(T*& ptr, 
				 size_t size);
template<typename T>
void cucreateptr(T**& ptr, 
				 size_t row, 
				 size_t col);
template<typename T>
void cucreateptr(T***& ptr, 
				 size_t row, 
				 size_t col, 
				 size_t height);
template<typename T>
void cucreateptr(T****& ptr, 
				 size_t a, 
				 size_t b, 
				 size_t c, 
				 size_t d);

template<typename T>
void cudestroyptr(T*&ptr);
template<typename T>
void cudestroyptr(T**&ptr);
template<typename T>
void cudestroyptr(T***&ptr);
template<typename T>
void cudestroyptr(T****&ptr);


template<typename T>
void cudaSafeFree(T*&ptr);

#include "mem_manager.inl"

}	//	namespace Utils
}	//	namespace WFT-FPA

#endif // !MEM_MANAGER_H
