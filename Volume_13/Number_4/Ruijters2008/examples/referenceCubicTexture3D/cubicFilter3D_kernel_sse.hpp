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

#include <intrin.h>

const __m128 mc0 = _mm_set_ps(1.0f/6.0f, -0.5f, -0.5f, 1.0f/6.0f);
const __m128 mc1 = _mm_set_ps(0.0f, 2.0f/3.0f, 2.0f/3.0f, 0.0f);

inline __m128 bsplineSSE(float fraction)
{
	// Creat all 4 weights
	const float one_frac = 1.0f - fraction;

	//w0 = 1.0f/6.0f * one_frac*one_frac*one_frac;
	//w1 = 2.0f/3.0f - 0.5f * fraction*fraction*(2.0f-fraction);
	//w2 = 2.0f/3.0f - 0.5f * one_frac*one_frac*(2.0f-one_frac);
	//w3 = 1.0f/6.0f * fraction*fraction*fraction;

	__m128 m0 = _mm_set_ps(one_frac, fraction, one_frac, fraction);
	__m128 m1 = _mm_set_ps(one_frac, 2.0f-fraction, 2.0f-one_frac, fraction);
	m0 = _mm_mul_ps(m0, m0);
	m0 = _mm_mul_ps(m0, m1);
	m0 = _mm_mul_ps(mc0, m0);
	return _mm_add_ps(mc1, m0);
}


inline float dot_product(__m128 a, __m128 b)
{
#if defined(SSE4)
	__m128 m = _mm_dp_ps(a, b, 0xff);
	return m.m128_f32[0];
#elif defined(SSE3)
	__m128 m = _mm_mul_ps(a, b);
	m = _mm_hadd_ps(m, m);
	m = _mm_hadd_ps(m, m);
	return m.m128_f32[0];
#else
	__m128 m = _mm_mul_ps(a, b);
	return m.m128_f32[0] + m.m128_f32[1] + m.m128_f32[2] + m.m128_f32[3];
#endif
}


inline __m128 convolute_loop(__m128 bspline, __m128 m[4])
{
#if defined(SSE4)
	return _mm_set_ps(
		dot_product(bspline, m[0]),
		dot_product(bspline, m[1]),
		dot_product(bspline, m[2]),
		dot_product(bspline, m[3]));
#else
	_MM_TRANSPOSE4_PS(m[3], m[2], m[1], m[0]);
	return
		_mm_add_ps( _mm_add_ps( _mm_add_ps(
			_mm_mul_ps(m[0], _mm_shuffle_ps(bspline, bspline, _MM_SHUFFLE(3,3,3,3))),
			_mm_mul_ps(m[1], _mm_shuffle_ps(bspline, bspline, _MM_SHUFFLE(2,2,2,2)))),
			_mm_mul_ps(m[2], _mm_shuffle_ps(bspline, bspline, _MM_SHUFFLE(1,1,1,1)))),
			_mm_mul_ps(m[3], _mm_shuffle_ps(bspline, bspline, _MM_SHUFFLE(0,0,0,0))));
#endif
}


float interpolate_tricubic_SSE(float* tex, float3 coord, uint3 volumeExtent)
{
	// transform the coordinate from [0,extent] to [-0.5, extent-0.5]
	const __m128 coord_grid = _mm_sub_ps(_mm_set_ps(coord.x, coord.y, coord.z, 0.5f), _mm_set1_ps(0.5f));  //coord_grid = coord - 0.5f;
	__m128 indexF = _mm_cvtepi32_ps(_mm_cvttps_epi32(coord_grid));  //indexF = floor(coord_grid);
	const __m128 fraction = _mm_sub_ps(coord_grid, indexF);  //fraction = coord_grid - indexF;
	// clamp between 1 and volumeExtent-3
	indexF = _mm_max_ps(indexF, _mm_set1_ps(1.0f));
	indexF = _mm_min_ps(indexF, _mm_cvtepi32_ps(
		_mm_sub_epi32(_mm_set_epi32(volumeExtent.x, volumeExtent.y, volumeExtent.z, 4), _mm_set1_epi32(3))));

	// note that x,y,z are located in registers 3,2,1
	__m128 bspline_x = bsplineSSE(fraction.m128_f32[3]);
	__m128 bspline_y = bsplineSSE(fraction.m128_f32[2]);
	__m128 bspline_z = bsplineSSE(fraction.m128_f32[1]);

	// load the data
	__m128 m0[16];
	__m128i index = _mm_sub_epi32(_mm_cvttps_epi32(indexF), _mm_set1_epi32(1));  //index = indexF - 1
	const float* p0 = tex + (index.m128i_i32[1] * volumeExtent.y + index.m128i_i32[2]) * volumeExtent.x + index.m128i_i32[3];
	const size_t slice = volumeExtent.x * volumeExtent.y;
	for (int z=0, i=0; z<4; z++)
	{
		const float* p1 = p0 + z * slice;
		for (int y=0; y<4; y++, i++)
		{
			m0[i] = _mm_set_ps(p1[0], p1[1], p1[2], p1[3]);
			p1 += volumeExtent.x;
		}
	}

	// convolution
	__m128 m1[4] = {
		convolute_loop(bspline_x, m0),
		convolute_loop(bspline_x, m0+4),
		convolute_loop(bspline_x, m0+8),
		convolute_loop(bspline_x, m0+12)};
	return dot_product(bspline_z, convolute_loop(bspline_y, m1) );
}

