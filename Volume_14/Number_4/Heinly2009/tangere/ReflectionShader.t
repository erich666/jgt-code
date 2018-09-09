
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

#ifndef tangere_ReflectionShader_t
#define tangere_ReflectionShader_t

#include <climits>

#include <Common/rtCore/RGB.t>

#include <Math/FixedPoint.h>
#include <Math/FixedPoint.t>
#include <Math/Math.h>

#include <tangere/BVH.t>
#include <tangere/Context.t>
#include <tangere/Flags.h>
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
  struct ReflectionShader : public Shader<T>
  {
    virtual RGB<T> shade(const RenderContext<T>& rc,
                         const Ray<T>&           ray,
                         const HitRecord<T>&     hit,
                         const Material<T>&      material,
                               int               depth) const
    {
      if (depth >= MAX_DEPTH) 
        return rtCore::RGB<T>::Zero;

      const RGB<T> refl    = material.specular;
      const RGB<T> notRefl = rtCore::RGB<T>::One - refl;

      const Scene<T>* scene = rc.getScene();
      const BVH<T>*   bvh = scene->getBVH();
      const vector<const Light<T>* >& lights = scene->getLights();

      const Vector<T>& dir = ray.dir();
      const Point<T>   hp  = ray.org() + dir*hit.getMinT();
            Vector<T>  n   = hit.getTriangle()->normal(rc, ray, hp, hit);
      T cosTheta = Dot(n, dir);
      if(cosTheta > 0)
      {
        n        = -n;
        cosTheta = -cosTheta;
      }

      const T      k  = 1+cosTheta; // (1 - (-cosTheta)) -> (1 + cosTheta)
      const T      k2 = k*k;
      const T      k4 = k2*k2;
      const T      k5 = k4*k;
      const RGB<T> R  = (refl + notRefl*k5);

      const Vector<T>    reflDir = dir - 2*cosTheta*n;
      const Ray<T>       reflRay(hp, reflDir);
            HitRecord<T> reflHit;
      bvh->intersect(reflHit, rc, reflRay);

      RGB<T> result;
      if(reflHit.getTriangle())
      {
        const Shader<T>*   shader   = reflHit.getShader();
        const uint&        mID      = reflHit.getMaterialID();
        const Material<T>& reflMat = bvh->getMaterials()[mID];

        result = R*shader->shade(rc, reflRay, reflHit, reflMat, depth+1);
      }
      else
      {
        result = R*scene->getBackground(reflRay.dir());
      }

      const Light<T>*const* begin = &lights[0];
      const Light<T>*const* end = begin + lights.size();
      RGB<T> light = rtCore::RGB<T>::Zero;
      while(begin != end)
      {
        RGB<T>    lightColor;
        Vector<T> lightDirection;
        const T dist   = (*begin++)->getLight(lightColor, lightDirection, rc, hp);
        const T cosphi = Dot(n, lightDirection);
        if(cosphi > 0)
        {
          const Ray<T> shadowRay(hp, lightDirection);
          HitRecord<T> shadowHit(dist);
          bvh->intersect(shadowHit, rc, shadowRay);

          if(!shadowHit.getTriangle())
          {
            const Vector<T> H        = (lightDirection - dir).normal();
            const T         cosAlpha = Dot(H, n);
            if(cosAlpha > 0)
            {
              const T phongTerm = Pow(cosAlpha, material.exp);
              light += lightColor*phongTerm;
            }
          }
        }
      }

      return (result + light*refl);
    }
  };

  template<>
  RGB<int> ReflectionShader<int>::shade(const RenderContext<int>& rc,
                                        const Ray<int>&           ray,
                                        const HitRecord<int>&     hit,
                                        const Material<int>&      material,
                                              int                 depth) const
    {      
      if (depth >= MAX_DEPTH) 
        return rtCore::RGB<int>::Zero;

      const RGB<int> refl    = material.specular;
      const RGB<int> notRefl = rtCore::RGB<int>::One - refl;      

      const Scene<int>*                 scene  = rc.getScene();
      const BVH<int>*                   bvh    = scene->getBVH();
      const vector<const Light<int>* >& lights = scene->getLights();

      const Vector<int>& dir = ray.dir();
      const Point<int>   hp  = ray.org() + dir*hit.getMinT();
            Vector<int>  n   = hit.getTriangle()->normal(rc, ray, hp, hit);

      //cosTheta is a 31 bit fixed point computed from a dot product
      fp31 cosTheta = fp31(Dot(n, dir));
      if(cosTheta > fp31::Zero)
      {
        n        = -n;
        cosTheta = -cosTheta;
      }

      // Compute Schlick's approximation with 31-bit fixed point value
      const fp31 k  = fp31::One + cosTheta; // (1 - (-cosTheta)) -> (1 + cosTheta)
      const fp31 k2 = k*k;
      const fp31 k4 = k2*k2;
      const fp31 k5 = k4*k;

      // NOTE(jsh) - shift down by 15 because k5 is 31 bit fixed point and color
      //             is 16 bit fixed point, (31 - 16) = 15
      const RGB<int> R = refl + notRefl*fp16(k5);

#ifdef USE_32_BIT_SHADERS
      Vector<int> reflDir;
      // Shift down by two to avoid overflow in calculation
      reflDir[0] = ((dir[0] >> 2) - MUL31(int(cosTheta), n[0] >> 1)) << 2;
      reflDir[1] = ((dir[1] >> 2) - MUL31(int(cosTheta), n[1] >> 1)) << 2;
      reflDir[2] = ((dir[2] >> 2) - MUL31(int(cosTheta), n[2] >> 1)) << 2;
#else
      // Compute reflection direction vector components
      // Requires promotion to 64 bit to avoid overflow on multiplication
      const int64_t rx64 = int64_t(dir[0]) - 
                           (MUL31_64(int64_t(cosTheta), n[0]) << 1);
      const int64_t ry64 = int64_t(dir[1]) -
                           (MUL31_64(int64_t(cosTheta), n[1]) << 1);
      const int64_t rz64 = int64_t(dir[2]) -
                           (MUL31_64(int64_t(cosTheta), n[2]) << 1);

      Vector<int> reflDir;
      reflDir[0] = int(rx64);
      reflDir[1] = int(ry64);
      reflDir[2] = int(rz64);
#endif

      const Ray<int> reflRay(hp, reflDir);
      HitRecord<int> reflHit;
      bvh->intersect(reflHit, rc, reflRay);

      RGB<int> result;
      if(reflHit.getTriangle())
      {
        const Shader<int>*   shader  = reflHit.getShader();
        const uint&          mID     = reflHit.getMaterialID();
        const Material<int>& reflMat = bvh->getMaterials()[mID];

        result = R*shader->shade(rc, reflRay, reflHit, reflMat, depth+1);
      }
      else
      {
        result = R*scene->getBackground(reflRay.dir());
      }

      const Light<int>*const* begin = &lights[0];
      const Light<int>*const* end = begin + lights.size();
      RGB<int> light = rtCore::RGB<int>::Zero;
      while(begin != end)
      {
        RGB<int> lightColor;
        Vector<int> lightDirection;
        const int dist = (*begin++)->getLight(lightColor, lightDirection, rc, hp);
        // cosphi is a 31-bit fixed point computed from dot product
        const fp31 cosphi = fp31(Dot(n, lightDirection));
        if(cosphi > fp31::Zero)
        {
          const Ray<int> shadowRay(hp, lightDirection);
          HitRecord<int> shadowHit(dist);
          bvh->intersect(shadowHit, rc, shadowRay);

          if(!shadowHit.getTriangle())
          {
            // NOTE(jsh) - shift each vector down by 2 because shifting by 1 is
            //             required to avoid obvious overflow (we can't
            //             represent vectors of length greater than 1) but we
            //             shift an additional bit to avoid an occasional
            //             overflow when computing the length squared in the
            //             normalize call
            const Vector<int> H = ((lightDirection >> 2) - (dir >> 2)).normal();

            // cosAlpha is a 31 bit fixed point number computed from dot product
            const fp31 cosAlpha = fp31(Dot(H, n));
            if(cosAlpha > fp31::Zero)
            {
              // NOTE(jsh) - use powi31 because cosAlpha is 31 bit fixed point
              //             and exp is a normal int
              const fp31 phongTerm = fp31(Math::Pow(cosAlpha, material.exp));

              // NOTE(jsh) - shift down by 15 because phongTerm is 31 fixed
              //             point and when multiplying by a color it
              //             expects a 16 bit fixed point number
              light += lightColor*fp16(phongTerm);
            }
          }
        }
      }

      //16 bit fixed point color multiplication requires shift down by 16
      return (result + light*refl);
    }

} // namespace tangere

#endif // tangere_ReflectionShader_t
