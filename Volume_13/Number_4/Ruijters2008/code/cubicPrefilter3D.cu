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

#ifndef _3D_CUBIC_BSPLINE_PREFILTER_H_
#define _3D_CUBIC_BSPLINE_PREFILTER_H_

#include <stdio.h>
#include <cutil.h>
#include "cubicPrefilter_kernel.cu"

#define MAX_DIMENSION 512
#define MEM_INTERLEAVE 32

//--------------------------------------------------------------------------
// Global CUDA procedures
//--------------------------------------------------------------------------
__global__ void SamplesToCoefficients3DX_simple(
	float* volume,		// in-place processing
	uint width,			// width of the volume
	uint height,		// height of the volume
	uint depth)			// depth of the volume
{
	// process lines in x-direction
	const uint y = blockIdx.x * blockDim.x + threadIdx.x;
	const uint z = blockIdx.y * blockDim.y + threadIdx.y;
	const uint startIdx = (z * height + y) * width;
	float* line = volume + startIdx;  //direct access

	ConvertToInterpolationCoefficients(line, width);
}

__global__ void SamplesToCoefficients3DX(
	float* volume,		// in-place processing
	uint width,			// width of the volume
	uint height,		// height of the volume
	uint depth)			// depth of the volume
{
	// process lines in x-direction
	const uint y = blockIdx.x * blockDim.x + threadIdx.x;
	const uint z = blockIdx.y * blockDim.y + threadIdx.y;
	const uint startIdx = (z * height + y) * width;
	float line[MAX_DIMENSION];

	// access the memory in an interleaved manner, to gain some performance
	for (uint offset=0; offset < MEM_INTERLEAVE; offset++)
		for (uint x=offset, i=startIdx+offset; x < width; x+=MEM_INTERLEAVE, i+=MEM_INTERLEAVE)
			line[x] = volume[i];

	ConvertToInterpolationCoefficients(line, width);

	for (uint offset=0; offset < MEM_INTERLEAVE; offset++)
		for (uint x=offset, i=startIdx+offset; x < width; x+=MEM_INTERLEAVE, i+=MEM_INTERLEAVE)
			volume[i] = line[x];
}

__global__ void SamplesToCoefficients3DY(
	float* volume,		// in-place processing
	uint width,			// width of the volume
	uint height,		// height of the volume
	uint depth)			// depth of the volume
{
	// process lines in y-direction
	const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint z = blockIdx.y * blockDim.y + threadIdx.y;
	const uint startIdx = z * height * width + x;
	float line[MAX_DIMENSION];

	// copy the line to fast local memory
	for (uint y = 0, i = startIdx; y < height; y++) {
		line[y] = volume[i];
		i += width;
	}

	ConvertToInterpolationCoefficients(line, height);

	// copy the line back to the volume
	for (uint y = 0, i = startIdx; y < height; y++) {
		volume[i] = line[y];
		i += width;
	}
}

__global__ void SamplesToCoefficients3DZ(
	float* volume,		// in-place processing
	uint width,			// width of the volume
	uint height,		// height of the volume
	uint depth)			// depth of the volume
{
	// process lines in z-direction
	const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
	const uint startIdx = y * width + x;
	const uint slice = height * width;
	float line[MAX_DIMENSION];

	// copy the line to fast local memory
	for (uint z = 0, i = startIdx; z < depth; z++) {
		line[z] = volume[i];
		i += slice;
	}

	ConvertToInterpolationCoefficients(line, height);

	// copy the line back to the volume
	for (uint z = 0, i = startIdx; z < depth; z++) {
		volume[i] = line[z];
		i += slice;
	}
}

#undef MAX_DIMENSION
#undef MEM_INTERLEAVE

//--------------------------------------------------------------------------
// Exported functions
//--------------------------------------------------------------------------

//! Convert the voxel values into cubic b-spline coefficients
//! @param volume  pointer to the voxel volume in GPU (device) memory
//! @param width   volume width in number of voxels
//! @param height  volume height in number of voxels
//! @param depth   volume depth in number of voxels
extern "C"
void CubicBSplinePrefilter3D(float* volume, uint width, uint height, uint depth)
{
	// Try to determine the optimal block dimensions
	uint dimX = min(min(PowTwoDivider(width), PowTwoDivider(height)), 64);
	uint dimY = min(min(PowTwoDivider(depth), PowTwoDivider(height)), 512/dimX);
	dim3 dimBlock(dimX, dimY);

	// Replace the voxel values by the b-spline coefficients
	dim3 dimGridX(height / dimBlock.x, depth / dimBlock.y);
	SamplesToCoefficients3DX<<<dimGridX, dimBlock>>>(volume, width, height, depth);
	CUT_CHECK_ERROR("SamplesToCoefficients3DX kernel failed");

	dim3 dimGridY(width / dimBlock.x, depth / dimBlock.y);
	SamplesToCoefficients3DY<<<dimGridY, dimBlock>>>(volume, width, height, depth);
	CUT_CHECK_ERROR("SamplesToCoefficients3DY kernel failed");

	dim3 dimGridZ(width / dimBlock.x, height / dimBlock.y);
	SamplesToCoefficients3DZ<<<dimGridZ, dimBlock>>>(volume, width, height, depth);
	CUT_CHECK_ERROR("SamplesToCoefficients3DZ kernel failed");
}

//! Convert the voxel values into cubic b-spline coefficients
//! @param volume  pointer to the voxel volume in GPU (device) memory
//! @param width   volume width in number of voxels
//! @param height  volume height in number of voxels
//! @param depth   volume depth in number of voxels
//! @note Prints stopwatch feedback
extern "C"
void CubicBSplinePrefilter3DTimer(float* volume, uint width, uint height, uint depth)
{
	printf("\nCubic B-Spline Prefilter timer:\n");
	uint hTimer;
	CUT_SAFE_CALL(cutCreateTimer(&hTimer));
	CUT_SAFE_CALL(cutResetTimer(hTimer));
    CUT_SAFE_CALL(cutStartTimer(hTimer));

	// Try to determine the optimal block dimensions
	uint dimX = min(min(PowTwoDivider(width), PowTwoDivider(height)), 64);
	uint dimY = min(min(PowTwoDivider(depth), PowTwoDivider(height)), 512/dimX);
	dim3 dimBlock(dimX, dimY);

	// Replace the voxel values by the b-spline coefficients
	dim3 dimGridX(height / dimBlock.x, depth / dimBlock.y);
	SamplesToCoefficients3DX<<<dimGridX, dimBlock>>>(volume, width, height, depth);
	CUT_CHECK_ERROR("SamplesToCoefficients3DX kernel failed");

	CUT_SAFE_CALL(cutStopTimer(hTimer));
    double timerValueX = cutGetTimerValue(hTimer);
    printf("x-direction : %f msec\n", timerValueX);
	CUT_SAFE_CALL(cutResetTimer(hTimer));
    CUT_SAFE_CALL(cutStartTimer(hTimer));

	dim3 dimGridY(width / dimBlock.x, depth / dimBlock.y);
	SamplesToCoefficients3DY<<<dimGridY, dimBlock>>>(volume, width, height, depth);
	CUT_CHECK_ERROR("SamplesToCoefficients3DY kernel failed");

	CUT_SAFE_CALL(cutStopTimer(hTimer));
    double timerValueY = cutGetTimerValue(hTimer);
    printf("y-direction : %f msec\n", timerValueY);
	CUT_SAFE_CALL(cutResetTimer(hTimer));
    CUT_SAFE_CALL(cutStartTimer(hTimer));

	dim3 dimGridZ(width / dimBlock.x, height / dimBlock.y);
	SamplesToCoefficients3DZ<<<dimGridZ, dimBlock>>>(volume, width, height, depth);
	CUT_CHECK_ERROR("SamplesToCoefficients3DZ kernel failed");

	CUT_SAFE_CALL(cutStopTimer(hTimer));
    double timerValueZ = cutGetTimerValue(hTimer);
    printf("z-direction : %f msec\n", timerValueZ);
	printf("total : %f msec\n\n", timerValueX+timerValueY+timerValueZ);
}

#endif  //_3D_CUBIC_BSPLINE_PREFILTER_H_
