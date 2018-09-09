/****************************************************************************/
/* Copyright (c) 2011, Markus Billeter, Ola Olsson, Erik Sintorn and Ulf Assarsson
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
#ifndef _chag_utils_Rendering_h_
#define _chag_utils_Rendering_h_

#include <linmath/float4x4.h>
#include <linmath/float3.h>
#include <math.h>

namespace chag
{

/**
 * The resulting matrix is created identically to gluPerspective()
 * and takes identical parameters.
 */
inline float4x4 perspectiveMatrix(float fov, float aspectRatio, float n, float f)
{
  using namespace chag;

  float4x4 m = make_identity<float4x4>();
	m[3][3] = 0.0f;
	float b = -1.0f / (f-n);
	float cotanFOV = 1.0f / tanf(fov*(float)g_pi/360.f);
	m[0][0] = cotanFOV / aspectRatio;
	m[1][1] = cotanFOV;
	m[2][2] = (f+n)*b;
	m[2][3] = -1.0f;
	m[3][2] = 2.0f*n*f*b;
	return m;
}



/**
 */
inline float4x4 lookAt(const float3 &eyePosition, const float3 &lookAt, const float3 &desiredUp)
{
  using namespace chag;

  float3 forward = normalize(lookAt - eyePosition);
  float3 side = normalize(cross(forward, desiredUp));
  float3 up = cross(side, forward);

  float4x4 m = make_identity<float4x4>();
  m[0][0] = side.x;
  m[1][0] = side.y;
  m[2][0] = side.z;

  m[0][1] = up.x;
  m[1][1] = up.y;
  m[2][1] = up.z;

  m[0][2] = -forward.x;
  m[1][2] = -forward.y;
  m[2][2] = -forward.z;

  return m * make_translation(-eyePosition);
}



}; // namespace chag



#endif // _chag_utils_Rendering_h_
