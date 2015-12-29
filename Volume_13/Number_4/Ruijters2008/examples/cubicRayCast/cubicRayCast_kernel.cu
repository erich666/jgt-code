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

#include <stdio.h>
#include <cutil.h>
#include <memcpy.cu>
#include <cubicPrefilter3D.cu>
#include <cubicFilter3D_kernel.cu>

texture<uchar, 3, cudaReadModeNormalizedFloat> tex;  //3D texture
texture<float, 3, cudaReadModeElementType> coeffs;  //3D texture


__device__ float sampleTexture(float3 coord, uint filterMethod)
{
	// read from 3D texture
	switch (filterMethod)
	{
		case 0:  //nearest neighbor
		case 1: return interpolate_trilinear(tex, coord);  //linear
		case 2: return interpolate_tricubic_simple(coeffs, coord);  //simple cubic
		case 3: return interpolate_tricubic_fast(coeffs, coord);  //fast cubic
		case 4: return interpolate_tricubic_fast(tex, coord);  //non-prefiltered, fast cubic
		default: return 0.0f;
	}
}


__global__ void rayCast(float4* output, const float3* rayCoords0, const float3* rayCoords1,
						uint imageWidth, float3 volumeExtent, uint filterMethod)
{
	uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	uint i = __umul24(y, imageWidth) + x;

	float3 start = volumeExtent * rayCoords0[i];
	float3 end = volumeExtent * rayCoords1[i];
	float3 dir = end - start;
	float rayLength = length(dir);

	float3 rayColor = make_float3(0.0f);
	float rayAlpha = 0.0f;

	if (rayLength > 0.1f)
	{
		const float alphaThreshold = 0.95f;  //opacity threshold for early ray termination of saturated rays
		const float density = 0.1f;
		const float brightness = 1.5f;
		float sampleDistance = 0.5f / rayLength;  //distance between the samples
		
		for (float t = 0.0f; t < 1.0f && rayAlpha < alphaThreshold; t += sampleDistance)
		{
			float sample = brightness * sampleTexture(start + t * dir, filterMethod);
			float4 lookup = make_float4(sample, sample, sample, density * sample);

			// Under operator
			rayColor += (1.0f - rayAlpha) * lookup.w * make_float3(lookup.x, lookup.y, lookup.z);
			rayAlpha += (1.0f - rayAlpha) * lookup.w;
		}
	}

	// write output color
	output[i] = make_float4(rayColor, rayAlpha);
}


// render image using CUDA
extern "C" void render(float4* output, float3* rayCoords[2], uint2 imageExtent, uint3 volumeSize, uint filterMethod)
{
	// set texture parameters
	tex.filterMode = (filterMethod == 0) ? cudaFilterModePoint : cudaFilterModeLinear;

	// call CUDA kernel, writing results to PBO
	const dim3 blockSize(min(PowTwoDivider(imageExtent.x), 16), min(PowTwoDivider(imageExtent.y), 8));
	const dim3 gridSize(imageExtent.x / blockSize.x, imageExtent.y / blockSize.y);
	const float3 volumeExtent = make_float3((float)volumeSize.x, (float)volumeSize.y, (float)volumeSize.z);
	rayCast<<<gridSize, blockSize>>>(output, rayCoords[0], rayCoords[1], imageExtent.x, volumeExtent, filterMethod);
	CUT_CHECK_ERROR("kernel failed");
}


// intialize the textures, and calculate the cubic B-spline coefficients
extern "C" void initCuda(const uchar* voxels, uint3 volumeSize)
{
	// calculate the b-spline coefficients
	float* bsplineCoeffs = CastUCharVolumeHostToDevice(voxels, volumeSize.x, volumeSize.y, volumeSize.z);
	CubicBSplinePrefilter3DTimer(bsplineCoeffs, volumeSize.x, volumeSize.y, volumeSize.z);

	// create the b-spline coefficients texture
	cudaChannelFormatDesc channelDescCoeff = cudaCreateChannelDesc<float>();
	cudaArray *coeffArray = 0;
	cudaExtent volumeExtent = make_cudaExtent(volumeSize.x, volumeSize.y, volumeSize.z);
	CUDA_SAFE_CALL(cudaMalloc3DArray(&coeffArray, &channelDescCoeff, volumeExtent));
	// copy data to 3D array
	cudaMemcpy3DParms copyParams = {0};
	copyParams.extent   = volumeExtent;
	copyParams.srcPtr   = make_cudaPitchedPtr((void*)bsplineCoeffs, volumeSize.x*sizeof(float), volumeSize.x, volumeSize.y);
	copyParams.dstArray = coeffArray;
	copyParams.kind     = cudaMemcpyDeviceToDevice;
	CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams));
	// bind array to 3D texture
	CUDA_SAFE_CALL(cudaBindTextureToArray(coeffs, coeffArray, channelDescCoeff));
	coeffs.normalized = false;  //access with absolute texture coordinates
	coeffs.filterMode = cudaFilterModeLinear;
	CUDA_SAFE_CALL(cudaFree(bsplineCoeffs));  //they are now in the coeffs texture, we do not need this anymore

	// Now create a texture with the original sample values for nearest neighbor and linear interpolation
	// Note that if you are going to do cubic interpolation only, you can remove the following code
	// create 3D array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar>();
	cudaArray *volumeArray = 0;
	CUDA_SAFE_CALL(cudaMalloc3DArray(&volumeArray, &channelDesc, volumeExtent));
	// copy data to 3D array
	copyParams.srcPtr   = make_cudaPitchedPtr((void*)voxels, volumeSize.x*sizeof(uchar), volumeSize.x, volumeSize.y);
	copyParams.dstArray = volumeArray;
	copyParams.kind     = cudaMemcpyHostToDevice;
	CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams));
	// bind array to 3D texture
	CUDA_SAFE_CALL(cudaBindTextureToArray(tex, volumeArray, channelDesc));
	tex.normalized = false;  //access with absolute texture coordinates
}
