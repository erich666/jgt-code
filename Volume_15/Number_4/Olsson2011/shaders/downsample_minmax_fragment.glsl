#version 330 compatibility
/****************************************************************************/
/* Copyright (c) 2011, Ola Olsson, Ulf Assarsson
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
/****************************************************************************/
#include "globals.glsl"

#if NUM_MSAA_SAMPLES == 0
uniform sampler2D depthTex;
#else // NUM_MSAA_SAMPLES != 0
uniform sampler2DMS depthTex;
#endif // NUM_MSAA_SAMPLES == 0

out vec2 resultMinMax;

vec3 unProject(vec2 fragmentPos, float fragmentDepth)
{
  vec4 pt = inverseProjectionMatrix * vec4(fragmentPos.x * 2.0 - 1.0, fragmentPos.y * 2.0 - 1.0, 2.0 * fragmentDepth - 1.0, 1.0);
  return vec3(pt.x, pt.y, pt.z) / pt.w;
}


vec3 fetchPosition(vec2 p, float d)
{
  vec2 fragmentPos = vec2(p.x * invFbSize.x, p.y * invFbSize.y);
  return unProject(fragmentPos, d);
}


void main()
{
	vec2 minMax = vec2(1.0f, -1.0f);
	ivec2 offset = ivec2(gl_FragCoord.xy) * ivec2(LIGHT_GRID_TILE_DIM_X, LIGHT_GRID_TILE_DIM_Y);
	ivec2 end = min(fbSize, offset + ivec2(LIGHT_GRID_TILE_DIM_X, LIGHT_GRID_TILE_DIM_Y));

	for (int j = offset.y; j < end.y; ++j)
	{
		for (int i = offset.x; i < end.x; ++i)
		{
#if NUM_MSAA_SAMPLES == 0
		  float d = texelFetch(depthTex, ivec2(i,j), 0).x;
			if (d < 1.0)
			{
				minMax.x = min(minMax.x, d);
				minMax.y = max(minMax.y, d);
			}
#else // NUM_MSAA_SAMPLES != 0
			for (int sampleIndex = 0; sampleIndex < NUM_MSAA_SAMPLES; ++sampleIndex)
			{
			  float d = texelFetch(depthTex, ivec2(i,j), sampleIndex).x;
				if (d < 1.0)
				{
					minMax.x = min(minMax.x, d);
					minMax.y = max(minMax.y, d);
				}
			}
#endif //NUM_MSAA_SAMPLES == 0
		}
	}

	// somewhat roundabout way to get to view space depth.
	minMax = vec2(fetchPosition(vec2(0.0, 0.0), minMax.x).z, fetchPosition(vec2(0.0, 0.0), minMax.y).z);
	resultMinMax = minMax;
}
