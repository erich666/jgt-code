/****************************************************************************/
/* Copyright (c) 2011, Ola Olsson
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
/**
 * Derived / Ported from hlsl code included with the Intel 2010 talk
 * 'Deferred Rendering for Current and Future Rendering Pipelines'
 * http://visual-computing.intel-research.net/art/publications/deferred_rendering/
 */
// Original copyright notice:

// Copyright 2010 Intel Corporation
// All Rights Reserved
//
// Permission is granted to use, copy, distribute and prepare derivative works of this
// software for any purpose and without fee, provided, that the above copyright notice
// and this statement appear in all copies.  Intel makes no representations about the
// suitability of this software for any purpose.  THIS SOFTWARE IS PROVIDED "AS IS."
// INTEL SPECIFICALLY DISCLAIMS ALL WARRANTIES, EXPRESS OR IMPLIED, AND ALL LIABILITY,
// INCLUDING CONSEQUENTIAL AND OTHER INDIRECT DAMAGES, FOR THE USE OF THIS SOFTWARE,
// INCLUDING LIABILITY FOR INFRINGEMENT OF ANY PROPRIETARY RIGHTS, AND INCLUDING THE
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  Intel does not
// assume any responsibility for any errors which may appear in this software nor any
// responsibility to update it.

#ifndef _ClipRegion_h_
#define _ClipRegion_h_


//--------------------------------------------------------------------------------------
// Bounds computation utilities, similar to PointLightBounds.cpp
void updateClipRegionRoot(float nc,          // Tangent plane x/y normal coordinate (view space)
                          float lc,          // Light x/y coordinate (view space)
                          float lz,          // Light z coordinate (view space)
                          float lightRadius,
                          float cameraScale, // Project scale for coordinate (_11 or _22 for x/y respectively)
                          float &clipMin,
                          float &clipMax)
{
  float nz = (lightRadius - nc * lc) / lz;
  float pz = (lc * lc + lz * lz - lightRadius * lightRadius) / (lz - (nz / nc) * lc);

  if (pz < 0.0f) 
  {
    float c = -nz * cameraScale / nc;
    if (nc < 0.0f) 
    {        
      // Left side boundary
      clipMin = std::max(clipMin, c);
    } 
    else 
    {                          // Right side boundary
      clipMax = std::min(clipMax, c);
    }
  }
}

void updateClipRegion(float lc,          // Light x/y coordinate (view space)
                      float lz,          // Light z coordinate (view space)
                      float lightRadius,
                      float cameraScale, // Project scale for coordinate (_11 or _22 for x/y respectively)
                      float &clipMin,
                      float &clipMax)
{
  float rSq = lightRadius * lightRadius;
  float lcSqPluslzSq = lc * lc + lz * lz;
  float d = rSq * lc * lc - lcSqPluslzSq * (rSq - lz * lz);

  if (d >= 0.0f) 
  {
    float a = lightRadius * lc;
    float b = sqrt(d);
    float nx0 = (a + b) / lcSqPluslzSq;
    float nx1 = (a - b) / lcSqPluslzSq;

    updateClipRegionRoot(nx0, lc, lz, lightRadius, cameraScale, clipMin, clipMax);
    updateClipRegionRoot(nx1, lc, lz, lightRadius, cameraScale, clipMin, clipMax);
  }
}

// Returns bounding box [min.xy, max.xy] in clip [-1, 1] space.
chag::float4 computeClipRegion(const chag::float3 &lightPosView, float lightRadius, float cameraNear, const chag::float4x4 &projection)
{
  using namespace chag;
    // Early out with empty rectangle if the light is too far behind the view frustum
  chag::float4 clipRegion = make_vector(1.0f, 1.0f, -1.0f, -1.0f);
  if (lightPosView.z - lightRadius <= -cameraNear) 
  {
    chag::float2 clipMin = make_vector(-1.0f, -1.0f);
    chag::float2 clipMax = make_vector( 1.0f,  1.0f);

    updateClipRegion(lightPosView.x, lightPosView.z, lightRadius, projection(1,1), clipMin.x, clipMax.x);
    updateClipRegion(lightPosView.y, lightPosView.z, lightRadius, projection(2,2), clipMin.y, clipMax.y);

    clipRegion = make_vector4(clipMin, clipMax);
  }

  return clipRegion;
}

#endif // _ClipRegion_h_
