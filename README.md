# Research code for real-time reference-based dynamic phase retrieval G-LS3U
This repository contains the code for the real-time reference based phase retrieval aglorithm [1]. 

## Checklist
### Utils: 
Utility functions written for both CPU and GPU, including fast 2D matrix I/O, host and device memory manager, etc.

### AIA 
Implementation of the AIA algorithm for fast and accurate phase extraction using random phase shifting [2], which is used to comptue the initial phase distribution. 

    - [x] Multi-core double precition 
    - [x] Multi-core single precition
    - [x] CUDA single precition
    - [ ] CUDA double precition
    
### DPRA
Core algorithm of the real-time reference-based dynamic phase retrieval algorithm [1].

    - [x] Multi-core double precition 
    - [x] Multi-core single precition
    - [x] CUDA single precition
    - [ ] CUDA double precition
    
    **Note**: The followiwng two implementations are just for testing purpose. Their performance is between the multi-core and CUDA 
    implementaitons 
    
    - [x] Hybrid CPU and GPU, double precition
    - [x] Hybrid CPU and GPU, single precition

### WFT
Improved implementaiton of the parallel Windowed Fourier Transform (WFT) algorithm [3].  

  Windowed Fourier Filtering (WFF) algorithm (Used in the proposed G-LS3U algorithm): .
  
    - [x] Multi-core double precition 
    - [x] Multi-core single precition
    - [x] CUDA single precition
    - [x] CUDA double precition
    
  Windowed Fourier Ridges (WFR) algorithm:
  
    - [x] Multi-core double precition 
    - [x] Multi-core single precition
    - [x] CUDA single precition
    - [x] CUDA double precition

### App_DPRA
A demonstration application written for the G-LS3U algorithm. The application can take a sequence of capdtured fringe patterns and exatract the phase distributions amogn frames then build them into a video. ***Note***: use the AIA function provided int he application's UI to calculate an initial phase distribution first before using any other functionalities. 

## Project dependencies
1. [Intel Math Kernel Library (MKL)] (https://software.intel.com/en-us/performance-libraries): using fftw3 to do fast Fourier transform (FFT) and LAPACK routine to solve linear system in parlalel on CPU.
2. [CUDA 8.0] (https://developer.nvidia.com/cuda-80-ga2-download-archive): for parallel computing on NVIDIA GPUs.
3. CUFFT: associated with CUDA, for perform parallel FFT on GPU.
4. [Qt 5.5] (https://www1.qt.io/qt5-5/): for GUI and multi-media used in App_DPRA.
5. [OpenCV 3.1] (https://opencv.org/opencv-3-1.html): for fast and convenient image I/O.

***Note***: On Windows OS using Visual Studio 2013, install these dependencies and your are all set to compile and run the programs. 

## Reference
[1] [Wang, T., Kai, L., & Kemao, Q. (2017). Real-time reference-based dynamic phase retrieval algorithm for optical measurement. Applied Optics, 56(27), 7726-7733.] (https://doi.org/10.1364/AO.56.007726)

[2] [Wang, Z., & Han, B. (2004). Advanced iterative algorithm for phase extraction of randomly phase-shifted interferograms. Optics letters, 29(14), 1671-1673.] (https://doi.org/10.1364/OL.29.001671)

[3] [Gao, W., Huyen, N. T. T., Loi, H. S., & Kemao, Q. (2009). Real-time 2D parallel windowed Fourier transform for fringe pattern analysis using Graphics Processing Unit. Optics express, 17(25), 23147-23152.] (https://doi.org/10.1364/OE.17.023147)

