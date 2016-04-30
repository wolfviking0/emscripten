/**
* emcuda.js
* Licence : https://github.com/wolfviking0/webcl-translator/blob/master/LICENSE
*
* Created by Anthony Liot.
* Copyright (c) 2013 Anthony Liot. All rights reserved.
*
*/

#ifndef __emscripten_cuda_h__
#define __emscripten_cuda_h__

#include <stdio.h>
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

extern struct KernelParams;

extern struct texture {
	int addressMode[3];
	int channelDesc[4];
    int filterMode;
    int normalized;   
};

enum 
{
    CUDA_INT         = 0,
    CUDA_FLOAT       = 1,
    CUDA_POINTER     = 2,
};

extern int cudaRunKernelFunc(const char * kernel_name, const char * kernel_source, const char* option, int blocksPerGrid, int threadsPerBlock, KernelParams params);

extern int cudaRunKernelDimFunc(const char * kernel_name, const char * kernel_source, const char* option, dim3 blocksPerGrid, dim3 threadsPerBlock, KernelParams params);

#ifdef __cplusplus
}
#endif


#endif /* __emscripten_cuda_h__ */

