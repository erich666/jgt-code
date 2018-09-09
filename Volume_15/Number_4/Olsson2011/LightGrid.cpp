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
#include "LightGrid.h"

#include <algorithm>
#include <linmath/float4.h>
#include <utils/Assert.h>
#include <utils/Math.h>

#include "ClipRegion.h"

using chag::max;
using chag::min;
using chag::swap;

#define MY_PROFILE_SCOPE(a)
#define MY_PROFILE_COUNTER(a, c)




static LightGrid::ScreenRect findScreenSpaceBounds(const chag::float4x4 &projection, chag::float3 pt, float rad, int width, int height, float near)
{
  chag::float4 reg = computeClipRegion(pt, rad, near, projection);
  reg = -reg;

  swap(reg.x, reg.z);
  swap(reg.y, reg.w);
  reg *= 0.5f;
  reg += 0.5f;

  static const chag::float4 zeros = { 0.0f, 0.0f, 0.0f, 0.0f };
  static const chag::float4 ones = { 1.0f, 1.0f, 1.0f, 1.0f };
  reg = chag::clamp(reg, zeros, ones);
  
  LightGrid::ScreenRect result;
  result.min.x = int(reg.x * float(width));
  result.min.y = int(reg.y * float(height));
  result.max.x = int(reg.z * float(width));
  result.max.y = int(reg.w * float(height));

  ASSERT(result.max.x <= uint32_t(width));
  ASSERT(result.max.y <= uint32_t(height));

  return result;
}


inline bool testDepthBounds(const chag::float2 &zRange, const Light& light)
{
	// Note that since in view space greater depth means _smaller_ z value (i.e. larger _negative_ Z values), it all gets turned inside out. 
	// Fairly easy to get confused...
	float lightMin = light.position.z + light.range;
	float lightMax = light.position.z - light.range;

  return (zRange.y < lightMin && zRange.x > lightMax);
}


void LightGrid::build(const chag::uint2 tileSize, const chag::uint2 resolution, const Lights &lights, const chag::float4x4 &modelView, const chag::float4x4 &projection, float near, const std::vector<chag::float2> &gridMinMaxZ)
{
  using namespace chag;
  MY_PROFILE_SCOPE("LightGridBuild");

	m_gridMinMaxZ = gridMinMaxZ;
	m_minMaxGridValid = !gridMinMaxZ.empty();

	const float2 *gridMinMaxZPtr = m_minMaxGridValid ? &m_gridMinMaxZ[0] : 0;

  m_tileSize = tileSize;
  MY_PROFILE_COUNTER("GridTileX", m_tileSize.x);
  MY_PROFILE_COUNTER("GridTileY", m_tileSize.y);
  m_gridDim = (resolution + tileSize - 1) / tileSize;
  MY_PROFILE_COUNTER("GridDimX", m_gridDim.x);
  MY_PROFILE_COUNTER("GridDimY", m_gridDim.y);
  m_maxTileLightCount = 0;

  buildRects(resolution, lights, modelView, projection, near);

  memset(m_gridOffsets, 0, sizeof(m_gridOffsets));
  memset(m_gridCounts, 0, sizeof(m_gridCounts));

#define GRID_OFFSETS(_x_,_y_) (m_gridOffsets[_x_ + _y_ * LIGHT_GRID_MAX_DIM_X])
#define GRID_COUNTS(_x_,_y_) (m_gridCounts[_x_ + _y_ * LIGHT_GRID_MAX_DIM_X])

  int totalus = 0;
  {  
  for (size_t i = 0; i < m_screenRects.size(); ++i)
  {
    ScreenRect r = m_screenRects[i];
		Light light = m_viewSpaceLights[i];

    chag::uint2 l = clamp(r.min / tileSize, make_vector<uint32_t>(0,0), m_gridDim + 1);
    chag::uint2 u = clamp((r.max + tileSize - 1) / tileSize, make_vector<uint32_t>(0,0), m_gridDim + 1);

    for (uint32_t y = l.y; y < u.y; ++y)
    {
      for (uint32_t x = l.x; x < u.x; ++x)
      {
				if (!m_minMaxGridValid || testDepthBounds(gridMinMaxZPtr[y * m_gridDim.x + x], light))
        {
          GRID_COUNTS(x, y) += 1;
          ++totalus;
        }
      }
    }
  }
  }
  m_tileLightIndexLists.resize(totalus);
#ifdef _DEBUG
  if (!m_tileLightIndexLists.empty())
  {
    memset(&m_tileLightIndexLists[0], 0, m_tileLightIndexLists.size() * sizeof(m_tileLightIndexLists[0]));
  }
#endif // _DEBUG
  MY_PROFILE_COUNTER("GridDataCount", int(totalus));

  uint32_t offset = 0;
  {  
  for (uint32_t y = 0; y < m_gridDim.y; ++y)
  {
    for (uint32_t x = 0; x < m_gridDim.x; ++x)
    {
      uint32_t count = GRID_COUNTS(x,y);
      // set offset to be just past end, then decrement while filling in
      GRID_OFFSETS(x,y) = offset + count;
      offset += count;

      // for debug/profiling etc.
      m_maxTileLightCount = chag::max(m_maxTileLightCount, count);
    }
  }
  }
	if (m_screenRects.size() && !m_tileLightIndexLists.empty())
  {
    int *data = &m_tileLightIndexLists[0];
    for (size_t i = 0; i < m_screenRects.size(); ++i)
    {
      uint32_t lightId = uint32_t(i);
			
			Light light = m_viewSpaceLights[i];
      ScreenRect r = m_screenRects[i];

      chag::uint2 l = clamp(r.min / tileSize, make_vector<uint32_t>(0,0), m_gridDim + 1);
      chag::uint2 u = clamp((r.max + tileSize - 1) / tileSize, make_vector<uint32_t>(0,0), m_gridDim + 1);

      for (uint32_t y = l.y; y < u.y; ++y)
      {
        for (uint32_t x = l.x; x < u.x; ++x)
        {
				if (!m_minMaxGridValid || testDepthBounds(gridMinMaxZPtr[y * m_gridDim.x + x], light))
          {
            // store reversely into next free slot
            uint32_t offset = GRID_OFFSETS(x, y) - 1;
            data[offset] = lightId;
            GRID_OFFSETS(x,y) = offset;
          }
        }
      }
    }
  }
#undef GRID_COUNTS
#undef GRID_OFFSETS
}



void LightGrid::buildRects(const chag::uint2 resolution, const Lights &lights, const chag::float4x4 &modelView, const chag::float4x4 &projection, float near)
{
  MY_PROFILE_SCOPE("BuildRects");

  m_viewSpaceLights.clear();
  m_screenRects.clear();

  for (uint32_t i = 0; i < lights.size(); ++i)
  {
    const Light &l = lights[i];
    chag::float3 vp = transformPoint(modelView, l.position);
    ScreenRect rect = findScreenSpaceBounds(projection, vp, l.range, resolution.x, resolution.y, near);

    if (rect.min.x < rect.max.x && rect.min.y < rect.max.y)
    {
      m_screenRects.push_back(rect);
      // save light in model space
      m_viewSpaceLights.push_back(make_light(vp, l));
    }
  }
}
