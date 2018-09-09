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
#include "Aabb.h"
#include "float4x4.h"
#include <float.h>
#include <algorithm>

namespace chag
{


Aabb make_aabb(const float3 &min, const float3 &max)
{
  Aabb result = { min, max };
  return result;
}


Aabb make_inverse_extreme_aabb()
{
  return make_aabb(make_vector(FLT_MAX, FLT_MAX, FLT_MAX), make_vector(-FLT_MAX, -FLT_MAX, -FLT_MAX));
}



Aabb make_aabb(const float3 *positions, const size_t numPositions)
{
  Aabb result = make_inverse_extreme_aabb();

  for (size_t i = 0; i < numPositions; ++i)
  {
    result = combine(result, positions[i]);
  }

  return result;
}



Aabb operator * (const float4x4 &tfm, const Aabb &a)
{
  Aabb result = { tfm.getTranslation(), tfm.getTranslation() };

  for (int i = 1; i < 4; ++i)
  {
    for (int j = 1; j < 4; ++ j)
    {
      float e = tfm(i,j) * a.min[j - 1];
      float f = tfm(i,j) * a.max[j - 1];
      if (e < f)
      {
        result.min[i - 1] += e;
        result.max[i - 1] += f;
      }
      else
      {
        result.min[i - 1] += f;
        result.max[i - 1] += e;
      }
    }
  }
  return result;
}



} // namespace chag
