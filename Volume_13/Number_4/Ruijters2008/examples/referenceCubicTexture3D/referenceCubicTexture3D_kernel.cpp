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

#include <windows.h>
#undef min
#undef max
#include <stdlib.h>
#include <stdio.h>
#include <xutility>
#include <cutil_math_bugfixes.h>
#include <cutil.h>

#include <cubicPrefilter3D.hpp>
#include <cubicFilter3D_kernel.hpp>
#include <cubicFilter3D_kernel_sse.hpp>

//texture<uchar, 3, cudaReadModeNormalizedFloat> tex;  //3D texture
//texture<float, 3, cudaReadModeElementType> coeffs;  //3D texture
float* coeffs = NULL;


struct params
{
	uchar* output;
	uint2 imageExtent;
	uint3 volumeSize;
	float3 volumeExtent;
	float w;
	uint filterMethod;
	uint y_start;
	uint y_end;
};


void render_kernel(uint x, uint y, const params& p)
{
	const float u = x / (float)p.imageExtent.x;
	const float v = y / (float)p.imageExtent.y;
	const float3 coord = p.volumeExtent * make_float3(u, v, p.w);

	// read from 3D texture
	float voxel;
	switch (p.filterMethod)
	{
		case 0: voxel = interpolate_tricubic_simple(coeffs, coord, p.volumeSize); break;  //simple cubic
		case 1: voxel = interpolate_tricubic_SSE(coeffs, coord, p.volumeSize); break;  //SSE accelerated
	}

	// write output color
	const uint i = y * p.imageExtent.x + x;
	p.output[i] = uchar(std::max(0.0f, std::min(voxel, 1.0f)) * 255);
}


DWORD WINAPI render_thread(void* ptr)
{
	params* p = (params*)ptr;

	for (uint y = p->y_start; y < p->y_end; y++)
	for (uint x = 0; x < p->imageExtent.x; x++)
	{
		render_kernel(x, y, *p);
	}

	return 0;
}


// render image
extern "C" void render(uchar* output, uint2 imageExtent, uint3 volumeSize, float w, uint filterMethod, uint nrOfThreads)
{
	float3 volumeExtent = make_float3((float)volumeSize.x, (float)volumeSize.y, (float)volumeSize.z);
	params p = {output, imageExtent, volumeSize, volumeExtent, w, filterMethod, 0, imageExtent.y};

#ifdef _NO_MULTITHREADING
	render_thread(&p);
#else
	HANDLE* hndls = new HANDLE[nrOfThreads];
	params* prms = new params[nrOfThreads];

	for (uint thread = 0; thread < nrOfThreads; thread++)
	{
		prms[thread] = p;
		prms[thread].y_start = thread * imageExtent.y / nrOfThreads;
		prms[thread].y_end = (thread+1) * imageExtent.y / nrOfThreads;
		DWORD threadId = 0;
		hndls[thread] = CreateThread(NULL, 0, render_thread, &prms[thread], 0, &threadId);
	}

	WaitForMultipleObjects(nrOfThreads, hndls, TRUE, INFINITE);
	delete[] hndls;
	delete[] prms;
#endif
}


// calculate the cubic B-spline coefficients
extern "C" void prefilter(const uchar* voxels, uint3 volumeSize)
{
	// calculate the b-spline coefficients
	coeffs = new float[volumeSize.x * volumeSize.y * volumeSize.z];

	uint index = 0;
	for (uint z = 0; z < volumeSize.z; z++)
	for (uint y = 0; y < volumeSize.y; y++)
	for (uint x = 0; x < volumeSize.x; x++)
	{
		coeffs[index] = voxels[index] / 255.0f;
		index++;
	}

	CubicBSplinePrefilter3DTimer(coeffs, volumeSize.x, volumeSize.y, volumeSize.z);
}
