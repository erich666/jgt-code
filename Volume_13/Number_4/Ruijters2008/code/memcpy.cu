/*--------------------------------------------------------------------------*\
Copyright (c) 2008-2009, Danny Ruijters. All rights reserved.
http://www.dannyruijters.nl/cubicinterpolation/
This file is part of CUDA Cubic B-Spline Interpolation (CI).

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
*  Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
*  Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
*  Neither the name of the copyright holders nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are
those of the authors and should not be interpreted as representing official
policies, either expressed or implied.
\*--------------------------------------------------------------------------*/

#ifndef _MEMCPY_CUDA_H_
#define _MEMCPY_CUDA_H_

#include <stdio.h>
#include <cutil.h>
#include "math_func.cu"


//--------------------------------------------------------------------------
// Declare the typecast CUDA kernels
//--------------------------------------------------------------------------
template<class T> __global__ void Cast(float* destination, const T* source)
{
	extern __device__ void error(void);  //non existing function
	error();  //ensure that we won't compile any un-specialized types
}

template<> __global__ void Cast<uchar>(float* destination, const uchar* source)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	destination[index] = (1.0/255.0) * source[index];
}

template<> __global__ void Cast<schar>(float* destination, const schar* source)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	destination[index] = (1.0/127.0) * source[index];
}

template<> __global__ void Cast<ushort>(float* destination, const ushort* source)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	destination[index] = (1.0/65535.0) * source[index];
}

template<> __global__ void Cast<short>(float* destination, const short* source)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	destination[index] = (1.0/32767.0) * source[index];
}

//--------------------------------------------------------------------------
// Declare the typecast templated function
// This function can be called directly in C++ programs
//--------------------------------------------------------------------------

//! Allocate GPU memory and copy a voxel volume from CPU to GPU memory
//! and cast it to the normalized floating point format
//! @return the pointer to the GPU copy of the voxel volume
//! @param host  pointer to the voxel volume in CPU (host) memory
//! @param width   volume width in number of voxels
//! @param height  volume height in number of voxels
//! @param depth   volume depth in number of voxels
template<class T> extern float* CastVolumeHostToDevice(const T* host, uint width, uint height, uint depth)
{
	T* temp = 0;
	float* device = 0;
	const uint voxelsPerSlice = width * height;
	const size_t nrOfBytesTemp = voxelsPerSlice * sizeof(T);
	const size_t nrOfBytesFloat = voxelsPerSlice * depth * sizeof(float);
	CUDA_SAFE_CALL(cudaMalloc((void**)&device, nrOfBytesFloat));
	CUDA_SAFE_CALL(cudaMalloc((void**)&temp, nrOfBytesTemp));

	dim3 dimBlock(min(PowTwoDivider(voxelsPerSlice), 256));
	dim3 dimGrid(voxelsPerSlice / dimBlock.x);
	size_t offset = 0;
	
	for (uint slice = 0; slice < depth; slice++, offset += voxelsPerSlice)
	{
		CUDA_SAFE_CALL(cudaMemcpy(temp, host + offset, nrOfBytesTemp, cudaMemcpyHostToDevice));
		Cast<T><<<dimGrid, dimBlock>>>(device + offset, temp);
		CUT_CHECK_ERROR("Cast kernel failed");
	}

	CUDA_SAFE_CALL(cudaFree(temp));  //free the temp GPU volume
	return device;
}

//--------------------------------------------------------------------------
// Declare specialized "C" linkage typcast functions
//--------------------------------------------------------------------------
extern "C" float* CastUCharVolumeHostToDevice(const uchar* host, uint width, uint height, uint depth)
{
	return CastVolumeHostToDevice(host, width, height, depth);
}

extern "C" float* CastCharVolumeHostToDevice(const schar* host, uint width, uint height, uint depth)
{
	return CastVolumeHostToDevice(host, width, height, depth);
}

extern "C" float* CastUShortVolumeHostToDevice(const ushort* host, uint width, uint height, uint depth)
{
	return CastVolumeHostToDevice(host, width, height, depth);
}

extern "C" float* CastShortVolumeHostToDevice(const short* host, uint width, uint height, uint depth)
{
	return CastVolumeHostToDevice(host, width, height, depth);
}

//--------------------------------------------------------------------------
// Copy floating point data from and to the GPU
//--------------------------------------------------------------------------

//! Allocate GPU memory and copy a voxel volume from CPU to GPU memory
//! @return the pointer to the GPU copy of the voxel volume
//! @param host  pointer to the voxel volume in CPU (host) memory
//! @param width   volume width in number of voxels
//! @param height  volume height in number of voxels
//! @param depth   volume depth in number of voxels
extern "C"
float* CopyVolumeHostToDevice(const float* host, uint width, uint height, uint depth)
{
	float* device = 0;
	const size_t nrOfBytes = width * height * depth * sizeof(float);
	CUDA_SAFE_CALL(cudaMalloc((void**)&device, nrOfBytes));
	CUDA_SAFE_CALL(cudaMemcpy(device, host, nrOfBytes, cudaMemcpyHostToDevice));
	return device;
}

//! Copy a voxel volume from GPU to CPU memory, and free the GPU memory
//! @param host  pointer to the voxel volume copy in CPU (host) memory
//! @param device  pointer to the voxel volume in GPU (device) memory
//! @param width   volume width in number of voxels
//! @param height  volume height in number of voxels
//! @param depth   volume depth in number of voxels
//! @note The \host CPU memory should be pre-allocated
extern "C"
void CopyVolumeDeviceToHost(float* host, float* device, uint width, uint height, uint depth)
{
	const size_t nrOfBytes = width * height * depth * sizeof(float);
	CUDA_SAFE_CALL(cudaMemcpy(host, device, nrOfBytes, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(device));  //free the GPU volume
}

#endif  //_MEMCPY_CUDA_H_
