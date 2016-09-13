#ifndef WFT_FPA_H
#define WFT_FPA_H


#define WFT_FPA_PI 3.14159265358979323846
#define WFT_FPA_TWOPI 6.28318530717958647692

#define BLOCK_SIZE_256 256
#define BLOCK_SIZE_128 128
#define BLOCK_SIZE_64 64

//!- Macro for library dll export utility
#if defined (_WIN32)
#    if defined (WFT_FPA_DLL_EXPORTS_MODE)
#        define WFT_FPA_DLL_EXPORTS __declspec(dllexport)
#    else
#        define WFT_FPA_DLL_EXPORTS __declspec(dllimport)
#    endif
#else
#    define WFT_FPA_DLL_EXPORTS
#endif

//!- WFT_FPA engine version
#define VERSION "1.0"

//!- Debugging macro
#ifdef WFT_FPA_DEBUG_MSG
#define DEBUG_MSG(x) do {std::cout<<"[WFT_FPA_DEBUG]: "<<x<<std::endl;} while(0)
#else
#define DEBUG_MSG(x) do {} while(0);
#endif // TW_DEBUG_MSG

#endif // WFT_FPA_H