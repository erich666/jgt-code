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

#ifndef _CUBIC3D_KERNEL_H_
#define _CUBIC3D_KERNEL_H_

#include "bspline_kernel.cu"

//! Trilinearly interpolated texture lookup, using unnormalized coordinates.
//! This function merely serves as a reference for the tricubic versions.
//! @param tex  3D texture
//! @param coord  unnormalized 3D texture coordinate
template<class T, enum cudaTextureReadMode mode>
__device__ float interpolate_trilinear(texture<T, 3, mode> tex, float3 coord)
{
	return tex3D(tex, coord.x, coord.y, coord.z);
}

//! Tricubic interpolated texture lookup, using unnormalized coordinates.
//! Straight forward implementation, using 64 nearest neighbour lookups.
//! @param tex  3D texture
//! @param coord  unnormalized 3D texture coordinate
template<class T, enum cudaTextureReadMode mode>
__device__ float interpolate_tricubic_simple(texture<T, 3, mode> tex, float3 coord)
{
	// transform the coordinate from [0,extent] to [-0.5, extent-0.5]
	const float3 coord_grid = coord - 0.5;
	float3 index = floor(coord_grid);
	const float3 fraction = coord_grid - index;
	index = index + 0.5;  //move from [-0.5, extent-0.5] to [0, extent]

	float result = 0.0;
	for (float z=-1; z < 2.5; z++)  //range [-1, 2]
	{
		float bsplineZ = bspline(z-fraction.z);
		float w = index.z + z;
		for (float y=-1; y < 2.5; y++)
		{
			float bsplineYZ = bspline(y-fraction.y) * bsplineZ;
			float v = index.y + y;
			for (float x=-1; x < 2.5; x++)
			{
				float bsplineXYZ = bspline(x-fraction.x) * bsplineYZ;
				float u = index.x + x;
				result += bsplineXYZ * tex3D(tex, u, v, w);
			}
		}
	}
	return result;
}

//! Tricubic interpolated texture lookup, using unnormalized coordinates.
//! Fast implementation, using 8 trilinear lookups.
//! @param tex  3D texture
//! @param coord  unnormalized 3D texture coordinate
template<class T, enum cudaTextureReadMode mode>
__device__ float interpolate_tricubic_fast(texture<T, 3, mode> tex, float3 coord)
{
	// shift the coordinate from [0,extent] to [-0.5, extent-0.5]
	const float3 coord_grid = coord - 0.5;
	const float3 index = floor(coord_grid);
	const float3 fraction = coord_grid - index;
	float3 w0, w1, w2, w3;
	bspline_weights(fraction, w0, w1, w2, w3);

	const float3 g0 = w0 + w1;
	const float3 g1 = w2 + w3;
	const float3 h0 = (w1 / g0) - 0.5 + index;  //h0 = w1/g0 - 1, move from [-0.5, extent-0.5] to [0, extent]
	const float3 h1 = (w3 / g1) + 1.5 + index;  //h1 = w3/g1 + 1, move from [-0.5, extent-0.5] to [0, extent]

	// fetch the eight linear interpolations
	// weighting and fetching is interleaved for performance and stability reasons
	float tex000 = tex3D(tex, h0.x, h0.y, h0.z);
	float tex100 = tex3D(tex, h1.x, h0.y, h0.z);
	tex000 = lerp(tex100, tex000, g0.x);  //weigh along the x-direction
	float tex010 = tex3D(tex, h0.x, h1.y, h0.z);
	float tex110 = tex3D(tex, h1.x, h1.y, h0.z);
	tex010 = lerp(tex110, tex010, g0.x);  //weigh along the x-direction
	tex000 = lerp(tex010, tex000, g0.y);  //weigh along the y-direction
	float tex001 = tex3D(tex, h0.x, h0.y, h1.z);
	float tex101 = tex3D(tex, h1.x, h0.y, h1.z);
	tex001 = lerp(tex101, tex001, g0.x);  //weigh along the x-direction
	float tex011 = tex3D(tex, h0.x, h1.y, h1.z);
	float tex111 = tex3D(tex, h1.x, h1.y, h1.z);
	tex011 = lerp(tex111, tex011, g0.x);  //weigh along the x-direction
	tex001 = lerp(tex011, tex001, g0.y);  //weigh along the y-direction

	return lerp(tex001, tex000, g0.z);  //weigh along the z-direction
}


#endif // _CUBIC3D_KERNEL_H_
