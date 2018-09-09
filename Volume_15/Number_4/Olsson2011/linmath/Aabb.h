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
#ifndef _chag_Aabb_h
#define _chag_Aabb_h

#include "float3.h"
#include "Common.h"


namespace chag
{

/**
 * An aabb defined by the min and max extrema.
 */
class Aabb
{
public:
  float3 min;
  float3 max;

  /**
   */
  const float3 getCentre() const { return (min + max) * 0.5f; }

  /**
   */
  const float3 getHalfSize() const { return (max - min) * 0.5f; }

  /**
   */
  float getVolume() const { float3 d = max - min; return d.x * d.y * d.z; }

  /**
   */
  float getSurfaceArea() const { float3 d = max - min;  return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);  }
};



/**
 */
inline Aabb combine(const Aabb &a, const Aabb &b)
{
  Aabb result = { min(a.min, b.min), max(a.max, b.max) };
  return result;
}

/**
 *
 */
inline Aabb combine(const Aabb &a, const float3 &pt)
{
  Aabb result = { min(a.min, pt), max(a.max, pt) };
  return result;
}
/**
 * creates an aabb that has min = FLT_MAX and max = -FLT_MAX.
 */
Aabb make_inverse_extreme_aabb();


/**
 */
Aabb make_aabb(const float3 &min, const float3 &max);

/**
 */
inline Aabb make_aabb(const float3 &position, const float radius)
{
  Aabb result = { position - radius, position + radius };
  return result;
}

/**
 */
Aabb make_aabb(const float3 *positions, const size_t numPositions);

/**
 */
inline bool overlaps(const Aabb &a, const Aabb &b)
{
  return a.max.x > b.min.x && a.min.x < b.max.x
    &&   a.max.y > b.min.y && a.min.y < b.max.y
    &&   a.max.z > b.min.z && a.min.z < b.max.z;

}


/**
 */
Aabb operator * (const float4x4 &tfm, const Aabb &a);


} // namespace chag

#endif // _chag_Aabb_h
