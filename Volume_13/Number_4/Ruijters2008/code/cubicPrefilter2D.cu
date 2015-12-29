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

#ifndef _2D_CUBIC_BSPLINE_PREFILTER_H_
#define _2D_CUBIC_BSPLINE_PREFILTER_H_

#include <stdio.h>
#include <cutil.h>
#include "cubicPrefilter_kernel.cu"

#define MAX_DIMENSION 512
#define MEM_INTERLEAVE 32

// ***************************************************************************
// *	Global GPU procedures
// ***************************************************************************
__global__ void SamplesToCoefficients2DX_simple(
	float* image,		// in-place processing
	uint width,			// width of the volume
	uint height)		// height of the volume
{
	// process lines in x-direction
	const uint y = blockIdx.x * blockDim.x + threadIdx.x;
	float* line = image + y * width;  //direct access

	ConvertToInterpolationCoefficients(line, width);
}

__global__ void SamplesToCoefficients2DX(
	float* image,		// in-place processing
	uint width,			// width of the volume
	uint height)		// height of the volume
{
	// process lines in x-direction
	const uint y = blockIdx.x * blockDim.x + threadIdx.x;
	const uint startIdx = y * width;
	float line[MAX_DIMENSION];

	// access the memory in an interleaved manner, to gain some performance
	for (uint offset=0; offset < MEM_INTERLEAVE; offset++)
		for (uint x=offset, i=startIdx+offset; x < width; x+=MEM_INTERLEAVE, i+=MEM_INTERLEAVE)
			line[x] = image[i];

	ConvertToInterpolationCoefficients(line, width);

	for (uint offset=0; offset < MEM_INTERLEAVE; offset++)
		for (uint x=offset, i=startIdx+offset; x < width; x+=MEM_INTERLEAVE, i+=MEM_INTERLEAVE)
			image[i] = line[x];
}

__global__ void SamplesToCoefficients2DY(
	float* image,		// in-place processing
	uint width,			// width of the volume
	uint height)		// height of the volume
{
	// process lines in y-direction
	const uint x = blockIdx.x * blockDim.x + threadIdx.x;
	float line[MAX_DIMENSION];

	// copy the line to fast local memory
	for (uint y = 0, i = x; y < height; y++) {
		line[y] = image[i];
		i += width;
	}

	ConvertToInterpolationCoefficients(line, height);

	// copy the line back to the volume
	for (uint y = 0, i = x; y < height; y++) {
		image[i] = line[y];
		i += width;
	}
}

#undef MAX_DIMENSION
#undef MEM_INTERLEAVE

// ***************************************************************************
// *	Exported functions
// ***************************************************************************

//! Convert the pixel values into cubic b-spline coefficients
//! @param image  pointer to the image bitmap in GPU (device) memory
//! @param width   image width in number of pixels
//! @param height  image height in number of pixels
extern "C"
void CubicBSplinePrefilter2D(float* image, uint width, uint height)
{
	dim3 dimBlockX(min(PowTwoDivider(height), 64));
	dim3 dimGridX(height / dimBlockX.x);
	SamplesToCoefficients2DX<<<dimGridX, dimBlockX>>>(image, width, height);
	CUT_CHECK_ERROR("SamplesToCoefficients2DX kernel failed");

	dim3 dimBlockY(min(PowTwoDivider(width), 64));
	dim3 dimGridY(width / dimBlockY.x);
	SamplesToCoefficients2DY<<<dimGridY, dimBlockY>>>(image, width, height);
	CUT_CHECK_ERROR("SamplesToCoefficients2DY kernel failed");
}

//! Convert the pixel values into cubic b-spline coefficients
//! @param image  pointer to the image bitmap in GPU (device) memory
//! @param width   image width in number of pixels
//! @param height  image height in number of pixels
//! @note Prints stopwatch feedback
extern "C"
void CubicBSplinePrefilter2DTimer(float* image, uint width, uint height)
{
	printf("\nCubic B-Spline Prefilter timer:\n");
	unsigned int hTimer;
	CUT_SAFE_CALL(cutCreateTimer(&hTimer));
	CUT_SAFE_CALL(cutResetTimer(hTimer));
    CUT_SAFE_CALL(cutStartTimer(hTimer));

	dim3 dimBlockX(min(PowTwoDivider(height), 64));
	dim3 dimGridX(height / dimBlockX.x);
	SamplesToCoefficients2DX<<<dimGridX, dimBlockX>>>(image, width, height);
	CUT_CHECK_ERROR("SamplesToCoefficients2DX kernel failed");

	CUT_SAFE_CALL(cutStopTimer(hTimer));
    double timerValueX = cutGetTimerValue(hTimer);
    printf("x-direction : %f msec\n", timerValueX);
	CUT_SAFE_CALL(cutResetTimer(hTimer));
    CUT_SAFE_CALL(cutStartTimer(hTimer));

	dim3 dimBlockY(min(PowTwoDivider(width), 64));
	dim3 dimGridY(width / dimBlockY.x);
	SamplesToCoefficients2DY<<<dimGridY, dimBlockY>>>(image, width, height);
	CUT_CHECK_ERROR("SamplesToCoefficients2DY kernel failed");

	CUT_SAFE_CALL(cutStopTimer(hTimer));
    double timerValueY = cutGetTimerValue(hTimer);
    printf("y-direction : %f msec\n", timerValueY);
	printf("total : %f msec\n\n", timerValueX+timerValueY);
}

#endif  //_2D_CUBIC_BSPLINE_PREFILTER_H_
