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

// Cubic B-spline function
// The 3rd order Maximal Order and Minimum Support function, that it is maximally differentiable.
inline float bspline(float t)
{
	t = fabs(t);
	const float a = 2.0f - t;

	if (t < 1.0f) return 2.0f/3.0f - 0.5f*t*t*a;
	else if (t < 2.0f) return a*a*a / 6.0f;
	else return 0.0f;
}


inline float tex3D(float* tex, uint x, uint y, uint z, uint3 volumeExtent)
{
	if (x < volumeExtent.x && y < volumeExtent.y && z < volumeExtent.z)
		return tex[(z * volumeExtent.y + y) * volumeExtent.x + x];
	else
		return 0.0f;
}


float interpolate_tricubic_simple(float* tex, float3 coord, uint3 volumeExtent)
{
	// transform the coordinate from [0,extent] to [-0.5, extent-0.5]
	const float3 coord_grid = coord - 0.5;
	float3 indexF = floor(coord_grid);
	const float3 fraction = coord_grid - indexF;
	int3 index = make_int3((int)indexF.x, (int)indexF.y, (int)indexF.z);
	//index = index + 0.5;  //move from [-0.5, extent-0.5] to [0, extent]

	float result = 0.0;
	for (int z=-1; z <= 2; z++)  //range [-1, 2]
	{
		float bsplineZ = bspline(z-fraction.z);
		int w = index.z + z;
		for (int y=-1; y <= 2; y++)
		{
			float bsplineYZ = bspline(y-fraction.y) * bsplineZ;
			int v = index.y + y;
			for (int x=-1; x <= 2; x++)
			{
				float bsplineXYZ = bspline(x-fraction.x) * bsplineYZ;
				int u = index.x + x;
				result += bsplineXYZ * tex3D(tex, u, v, w, volumeExtent);
			}
		}
	}
	return result;
}

