/*--------------------------------------------------------------------------*\
Copyright 2008-2009 Danny Ruijters.
http://www.dannyruijters.nl/cubicinterpolation/
This file is part of CUDA Cubic B-Spline Interpolation (CI).

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
*  Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
*  Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
*  Neither the name of the above copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
\*--------------------------------------------------------------------------*/

#include <stdio.h>
#include <cutil.h>
#include <memcpy.cu>
#include <cubicPrefilter2D.cu>
#include <cubicFilter2D_kernel.cu>

#define PI ((double)3.14159265358979323846264338327950288419716939937510)
texture<float, 2, cudaReadModeElementType> coeffs;  //2D texture


__global__ void
interpolate_kernel(float* output, uint width, float2 extent, float2 a, float2 shift, bool masking)
{
	uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	uint i = __umul24(y, width) + x;

	float x0 = (float)x;
	float y0 = (float)y;
	float x1 = a.x * x0 - a.y * y0 + shift.x;
	float y1 = a.x * y0 + a.y * x0 + shift.y;

	bool inside =
		-0.5f < x1 && x1 < (extent.x - 0.5f) &&
		-0.5f < y1 && y1 < (extent.y - 0.5f);

	if (masking && !inside)
	{
		output[i] = 0.0f;
	}
	else
	{
		output[i] = interpolate_bicubic_fast(coeffs, x1, y1);
	}
}


extern "C" float* interpolate(
	uint width, uint height, double angle,
	double xShift, double yShift, double xOrigin, double yOrigin, int masking)
{
	// Prepare the geometry
	angle *= PI / 180.0;
	float2 a = make_float2((float)cos(angle), (float)sin(angle));
	double x0 = a.x * (xShift + xOrigin) - a.y * (yShift + yOrigin);
	double y0 = a.y * (xShift + xOrigin) + a.x * (yShift + yOrigin);
	xShift = xOrigin - x0;
	yShift = yOrigin - y0;

	// Allocate the output image
	float* output;
	CUDA_SAFE_CALL(cudaMalloc((void**)&output, width * height * sizeof(float)));

	// Visit all pixels of the output image and assign their value
	dim3 blockSize(min(PowTwoDivider(width), 16), min(PowTwoDivider(height), 16));
	dim3 gridSize(width / blockSize.x, height / blockSize.y);
	float2 shift = make_float2((float)xShift, (float)yShift);
	float2 extent = make_float2((float)width, (float)height);
	interpolate_kernel<<<gridSize, blockSize>>>(output, width, extent, a, shift, masking != 0);
	CUT_CHECK_ERROR("kernel failed");

	return output;
}


extern "C" void initTexture(float* bsplineCoeffs, uint width, uint height)
{
	// Create the B-spline coefficients texture
	cudaChannelFormatDesc channelDescCoeff = cudaCreateChannelDesc<float>();
	cudaArray *coeffArray = 0;
	CUDA_SAFE_CALL(cudaMallocArray(&coeffArray, &channelDescCoeff, width, height));
	CUDA_SAFE_CALL(cudaMemcpyToArray(coeffArray, 0, 0, bsplineCoeffs, width * height * sizeof(float), cudaMemcpyDeviceToDevice));
	CUDA_SAFE_CALL(cudaBindTextureToArray(coeffs, coeffArray, channelDescCoeff));
	coeffs.normalized = false;  // access with normalized texture coordinates
	coeffs.filterMode = cudaFilterModeLinear;
}
