
/*
 *  Copyright 2009, 2010 Grove City College
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef tangere_NdotVShader_t
#define tangere_NdotVShader_t

#include <Common/rtCore/RGB.t>

#include <Math/FixedPoint.t>
#include <Math/Math.h>

#include <tangere/Context.t>
#include <tangere/Material.t>
#include <tangere/Ray.t>
#include <tangere/Scene.t>
#include <tangere/Shader.t>
#include <tangere/Triangle.t>

namespace tangere
{
  using Math::fp16;
  using Math::fp31;

  template<typename T>
  struct NdotVShader : public Shader<T>
  {
    RGB<T> shade(const RenderContext<T>& rc,
                 const Ray<T>&           ray,
                 const HitRecord<T>&     hit,
                 const Material<T>&      material,
                       int               depth) const
    {
      if (material.emissive != rtCore::RGB<T>::Zero)
        return material.emissive;

      const Vector<T>& dir   = ray.dir();
      const Point<T>   hp    = ray.org() + dir*hit.getMinT();
      const Vector<T>  n     = hit.getTriangle()->normal(rc, ray, hp, hit);
      const T          NdotV = Abs(Dot(n, dir));

      return (NdotV*material.diffuse);
    }
  };

  template<>
  RGB<int> NdotVShader<int>::shade(const RenderContext<int>& rc,
                                   const Ray<int>&           ray,
                                   const HitRecord<int>&     hit,
                                   const Material<int>&      material,
                                         int                 depth) const
  {
    if (material.emissive != rtCore::RGB<int>::Zero)
      return material.emissive;

    // Ray direction components are fixed-point numbers on the range [-1, 1)
    // MinT is a full 32 bit scalar, so we multiply and shift down by
    // 31 bits to get a displacement down the ray to add to the origin
    const Vector<int>& dir = ray.dir();
    const Point<int>   hp  = ray.org() + dir*hit.getMinT();
    const Vector<int>  n   = hit.getTriangle()->normal(rc, ray, hp, hit);

    // 31 bit fixed-point dot product down to 16 bit fixed-point for RGB<int>
    const fp16 NdotV = fp16(fp31(Abs(Dot(n, dir))));

    // Color multiplication, shift down 16 after multiplying
    // because its 16-bit fixed point
    return (RGB<int>(NdotV, NdotV, NdotV)*material.diffuse);
  }

} // namespace tangere

#endif // tangere_NdotVShader_t
