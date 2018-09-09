
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

#ifndef tangere_LambertianShader_t
#define tangere_LambertianShader_t

#include <Common/rtCore/RGB.t>

#include <Math/FixedPoint.t>
#include <Math/Math.h>

#include <tangere/BVH.t>
#include <tangere/Context.t>
#include <tangere/HitRecord.t>
#include <tangere/Light.t>
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
  struct LambertianShader : public Shader<T>
  {
    RGB<T> shade(const RenderContext<T>& rc,
                 const Ray<T>&           ray,
                 const HitRecord<T>&     hit,
                 const Material<T>&      material,
                       int               depth) const
    {
      if (material.emissive != rtCore::RGB<T>::Zero)
        return material.emissive;

      const Scene<T>* scene = rc.getScene();
      const BVH<T>*   bvh   = scene->getBVH();
      const vector<const Light<T>*>& lights = scene->getLights();

      const Vector<T>& dir = ray.dir();
      const Point<T>   hp  = ray.org() + dir*hit.getMinT();
            Vector<T>  n   = hit.getTriangle()->normal(rc, ray, hp, hit);

      const T side = Dot(n, dir);
      if (side > 0)
        n = -n;

      RGB<T> light = material.ka*scene->getAmbient();

      const Light<T>* const* begin = &lights[0];
      const Light<T>* const* end   = begin + lights.size();
      while (begin != end)
      {
        RGB<T>    lcolor;
        Vector<T> ldir;
        const T distance = (*begin++)->getLight(lcolor, ldir, rc, hp);
        const T cosPhi   = Dot(n, ldir);
        if (cosPhi > 0)
        {
          HitRecord<T> shadowHit(distance);
          Ray<T> shadowRay(hp, ldir);
          bvh->intersect(shadowHit, rc, shadowRay);
          if (!shadowHit.getTriangle())
            light += ((lcolor*material.kd)*cosPhi);
        }
      }

      return (light*material.diffuse)*(1.f/lights.size());
    }
  };

  template<>
  RGB<int> LambertianShader<int>::shade(const RenderContext<int>& rc,
                                        const Ray<int>&           ray,
                                        const HitRecord<int>&     hit,
                                        const Material<int>&      material,
                                              int                 depth) const
  {
    if (material.emissive != rtCore::RGB<int>::Zero)
      return material.emissive;

    const Scene<int>* scene = rc.getScene();
    const BVH<int>*   bvh   = scene->getBVH();
    const vector<const Light<int>*>& lights = scene->getLights();

    // Ray direction components are fixed-point numbers on the range [-1, 1)
    // MinT is a full 32 bit scalar, so we multiply and shift down by
    // 31 bits to get a displacement down the ray to add to the origin
    const Vector<int>& dir = ray.dir();
    const Point<int>   hp  = ray.org() + dir*hit.getMinT();
          Vector<int>  n   = hit.getTriangle()->normal(rc, ray, hp, hit);

    // 31 bit fixed-point dot product
    const fp31 side = fp31(Dot(n, dir));
    if (side > fp31::Zero)
      n = -n;

    // Colors and diffuse/ambient are stored in 2^15 capped fixed-point,
    // shift down by 16 when multiplying them
    RGB<int> light = material.ka*scene->getAmbient();

    const Light<int>* const* begin = &lights[0];
    const Light<int>* const* end   = begin + lights.size();
    while (begin != end)
    {
      RGB<int>    lcolor;
      Vector<int> ldir;
      const int distance = (*begin++)->getLight(lcolor, ldir, rc, hp);
      //cosPhi is a 31 bit fixed point computed by fixed point dot product
      const fp31 cosPhi  = fp31(Dot(n, ldir));
      if (cosPhi > fp31::Zero)
      {
        HitRecord<int> shadowHit(distance);
        Ray<int>       shadowRay(hp, ldir);
        bvh->intersect(shadowHit, rc, shadowRay);
        if (!shadowHit.getTriangle())
        {
          // Convert cosPhi to 16 bit fixed point because a color expects
          // to be multiplied by a 16 bit fixed point int.
          light += (lcolor*material.kd)*int(fp16(cosPhi));
        }
      }
    }

    // Color multiplication, shift down 16 after multiplying
    // because its 16-bit fixed point
    return (light*material.diffuse)*FLOAT_TO_FIXED16(1.f/lights.size());
  }

} // namespace tangere

#endif // tangere_LambertianShader_t
