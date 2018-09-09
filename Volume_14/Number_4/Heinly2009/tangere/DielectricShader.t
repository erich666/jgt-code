
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

#ifndef tangere_DielectricShader_t
#define tangere_DielectricShader_t

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
  using Math::Pow;
  using Math::Sqrt;

  template<typename T>
  struct DielectricShader : public Shader<T>
  {

    RGB<T> shade(const RenderContext<T>& rc,
                 const Ray<T>&           ray,
                 const HitRecord<T>&     hit,
                 const Material<T>&      material,
                       int               depth) const
    {
      static const double ETA    = 1.5;
      static const T      eta    = T(ETA);
      static const T      invEta = T(1/ETA);

      if (depth >= MAX_DEPTH)
        return rtCore::RGB<T>::Zero;

      const Scene<T>*                scene  = rc.getScene();
      const vector<const Light<T>*>& lights = scene->getLights();

      const Vector<T>& dir    = ray.dir();
      const Point<T>   hp     = ray.org() + dir*hit.getMinT();
            Vector<T>  n      = hit.getTriangle()->normal(rc, ray, hp, hit);
      const Vector<T>  phongN = n;

      T    NdotV = Dot(n, dir);
      T    etaInverse;
      bool exiting;
      if (NdotV > 0)
      {
        // Exiting surface
        n          = -n;
        etaInverse = eta;
        exiting    = true;
      } 
      else
      {
        // Entering surface
        NdotV      = -NdotV;
        etaInverse = invEta;
        exiting    = false;
      }

      const Vector<T> reflDir = dir + 2*NdotV*n;
      const T         NdotT2  = 1 + (NdotV*NdotV - 1)*(etaInverse*etaInverse);

      RGB<T> result;
      if (NdotT2 < 0)
      {
        // Total internal reflection
        const Ray<T> reflRay(hp, reflDir);
        HitRecord<T> reflHit;
        scene->getBVH()->intersect(reflHit, rc, reflRay);

        if (reflHit.getTriangle())
        {
          const Shader<T>*   shader   = reflHit.getShader();
          const uint&        mID      = reflHit.getMaterialID();
          const Material<T>& reflMat  = scene->getBVH()->getMaterials()[mID];

          result = reflHit.getShader()->shade(rc, reflRay, reflHit, reflMat,
                                              depth+1);
        }
        else
        {
          result = scene->getBackground(reflDir);
        }
      } 
      else
      {
        const T NdotT = Sqrt(NdotT2);
        const T k     = 1 - Min(NdotV, NdotT);
        const T k2    = k*k;
        const T k4    = k2*k2;
        const T k5    = k4*k;
        const T refl  = material.r0*(1-k5)+k5;

        // Reflective ray
        const Ray<T> reflRay(hp, reflDir);
        HitRecord<T> reflHit;
        scene->getBVH()->intersect(reflHit, rc, reflRay);

        if (reflHit.getTriangle())
        {
          const Shader<T>*   shader  = reflHit.getShader();
          const uint&        mID     = reflHit.getMaterialID();
          const Material<T>& reflMat = scene->getBVH()->getMaterials()[mID];

          result = refl*reflHit.getShader()->shade(rc, reflRay, reflHit,
                                                   reflMat, depth+1);
        }
        else
        {
          result = refl*scene->getBackground(reflDir);
        }
        
        // Transparency ray
        const Vector<T> transpDir = etaInverse*dir +
          (etaInverse*NdotV - NdotT)*n;

        const Ray<T> transpRay(hp, transpDir);
        HitRecord<T> transpHit;
        scene->getBVH()->intersect(transpHit, rc, transpRay);

        if (transpHit.getTriangle())
        {
          const Shader<T>*   shader    = transpHit.getShader();
          const uint&        mID       = transpHit.getMaterialID();
          const Material<T>& transpMat = scene->getBVH()->getMaterials()[mID];

          result += (1 - refl)*shader->shade(rc, transpRay, transpHit,
                                             transpMat, depth+1);
        }
        else
        {
          result += (1 - refl)*scene->getBackground(transpDir);
        }
      }

      const Light<T>*const* begin = &lights[0];
      const Light<T>*const* end = begin + lights.size();
      while (begin != end)
      {
        RGB<T>    lightColor;
        Vector<T> lightDirection;
        const T dist   = (*begin++)->getLight(lightColor, lightDirection,
                                              rc, hp);
        const T cosphi = Dot(phongN, lightDirection);
        
        if (!exiting && cosphi > 0)
        {
          if (cosphi > 0)
          {
            HitRecord<T> shadowHit(dist);
            const Ray<T> shadowRay(hp, lightDirection);
            scene->getBVH()->intersect(shadowHit, rc, shadowRay);

            if (!shadowHit.getTriangle())
            {
              Vector<T> H = lightDirection + (exiting ? dir : -dir);
              H.normalize();
              const T cosAlpha = Dot(H, phongN);
              if (cosAlpha > 0)
              {
                const T phongTerm = Pow(cosAlpha, material.exp);
                result += lightColor*phongTerm;
              }
            }
          }
        }
      }

      return result;
    }
  };

  template<>
  RGB<int> DielectricShader<int>::shade(const RenderContext<int>& rc,
                                        const Ray<int>&           ray,
                                        const HitRecord<int>&     hit,
                                        const Material<int>&      material,
                                              int               depth) const
  {
    static const double ETA    = 1.5;
    static const fp16   eta    = fp16(ETA);
    static const fp16   invEta = fp16(1/ETA);

    if (depth >= MAX_DEPTH)
      return rtCore::RGB<int>::Zero;

    const Scene<int>* scene = rc.getScene();
    const vector<const Light<int>*>& lights = scene->getLights();

    const Vector<int>& dir    = ray.dir();
    const Point<int>   hp     = ray.org() + dir*hit.getMinT();
          Vector<int>  n      = hit.getTriangle()->normal(rc, ray, hp, hit);
    const Vector<int>  phongN = n;

    // Initially use 31-bit fixed point for NdotV
    fp31 NdotV31 = fp31(Dot(n, dir));
    fp16 etaInverse;
    bool exiting;
    if (NdotV31 > fp31::Zero)
    {
      // Exiting surface
      n          = -n;
      etaInverse = eta;
      exiting    = true;
    } 
    else
    {
      // Entering surface
      NdotV31    = -NdotV31;
      etaInverse = invEta;
      exiting    = false;
    }

#ifdef USE_32_BIT_SHADERS
    Vector<int> reflDir;
    // Shift down by two to avoid overflow in calculation
    reflDir[0] = ((dir[0] >> 2) + MUL31(int(NdotV31), n[0] >> 1)) << 2;
    reflDir[1] = ((dir[1] >> 2) + MUL31(int(NdotV31), n[1] >> 1)) << 2;
    reflDir[2] = ((dir[2] >> 2) + MUL31(int(NdotV31), n[2] >> 1)) << 2;
#else
    // Compute the reflectance direction in 64 bit so that we don't overflow
    // part of the way through the calculation. When multiplying NdotV with
    // the n, convert the 31 bit fixed point NdotV to an int so that
    // the MUL31_64 macro can use it correctly. The macro then returns a
    // 64 bit value
    const int64_t rx64 = int64_t(dir[0]) + (MUL31_64(int(NdotV31), n[0]) << 1);
    const int64_t ry64 = int64_t(dir[1]) + (MUL31_64(int(NdotV31), n[1]) << 1);
    const int64_t rz64 = int64_t(dir[2]) + (MUL31_64(int(NdotV31), n[2]) << 1);

    Vector<int> reflDir;
    reflDir[0] = int(rx64);
    reflDir[1] = int(ry64);
    reflDir[2] = int(rz64);
#endif

    // Convert NdotV from 31 bit fixed point to 16 bit fixed point for use
    // in the rest of this function
    const fp16 NdotV16 = fp16(NdotV31);
    const fp16 NdotT2  = fp16::One + (NdotV16*NdotV16 - fp16::One) *
      etaInverse*etaInverse;

    RGB<int> result;
    if (NdotT2 < fp16::Zero)
    {
      // Total internal reflection
      const Ray<int> reflRay(hp, reflDir);
      HitRecord<int> reflHit;
      scene->getBVH()->intersect(reflHit, rc, reflRay);

      if (reflHit.getTriangle())
      {
        const Shader<int>*   shader  = reflHit.getShader();
        const uint&          mID     = reflHit.getMaterialID();
        const Material<int>& reflMat = scene->getBVH()->getMaterials()[mID];

        result = reflHit.getShader()->shade(rc, reflRay, reflHit, reflMat,
                                            depth+1);
      }
      else
      {
        result = scene->getBackground(reflDir);
      }
    } 
    else
    {
      const fp16 NdotT = Sqrt(NdotT2);
      const fp16 k     = fp16::One - Min(NdotV16, NdotT);
      const fp16 k2    = k*k;
      const fp16 k4    = k2*k2;
      const fp16 k5    = k4*k;
      // Material.r0 is an int which represents a 31 bit fixed point number.
      // We need to create a fp31 object from this number so that it is
      // interpreted correctly and then we can cast it to a fp16
      const fp16 refl  = fp16(fp31(material.r0))*(fp16::One - k5) + k5;

      // Reflective ray
      const Ray<int> reflRay(hp, reflDir);
      HitRecord<int> reflHit;
      scene->getBVH()->intersect(reflHit, rc, reflRay);

      if (reflHit.getTriangle())
      {
        const Shader<int>*   shader  = reflHit.getShader();
        const uint&          mID     = reflHit.getMaterialID();
        const Material<int>& reflMat = scene->getBVH()->getMaterials()[mID];

        result = int(refl)*reflHit.getShader()->shade(rc, reflRay, reflHit,
                                                      reflMat, depth+1);
      }
      else
      {
        result = int(refl)*scene->getBackground(reflDir);
      }
      
#ifdef USE_32_BIT_SHADERS
      Vector<int> transpDir;
      transpDir[0] = (MUL16(int(etaInverse), dir[0] >> 1) +
                      MUL16(int(etaInverse*NdotV16 - NdotT), n[0] >> 1)) << 1;
      transpDir[1] = (MUL16(int(etaInverse), dir[1] >> 1) +
                      MUL16(int(etaInverse*NdotV16 - NdotT), n[1] >> 1)) << 1;
      transpDir[2] = (MUL16(int(etaInverse), dir[2] >> 1) +
                      MUL16(int(etaInverse*NdotV16 - NdotT), n[2] >> 1)) << 1;
#else
      // Transparency ray
      // We first convert the 16 bit fixed point etaInverse to 64 bits so that
      // it can be used by the multiplication macro. The macro performs a 16 bit
      // fixed point multiply because a 16 bit fixed point number is being
      // multiplied with a 31 bit fixed point number and we want the answer to
      // be in 31 bit fixed point. The same is also true for the next calulation
      // where the 16 bit fixed point (etaInverse*NdotV16 - NdotT) is multiplied
      // with the 31 bit fixed point normal
      int64_t tX = MUL16_64(int64_t(etaInverse), dir[0]) +
                   MUL16_64(int64_t(etaInverse*NdotV16 - NdotT), n[0]);
      int64_t tY = MUL16_64(int64_t(etaInverse), dir[1]) +
                   MUL16_64(int64_t(etaInverse*NdotV16 - NdotT), n[1]);
      int64_t tZ = MUL16_64(int64_t(etaInverse), dir[2]) +
                   MUL16_64(int64_t(etaInverse*NdotV16 - NdotT), n[2]);

      Vector<int> transpDir;
      // We shift down by 1 here to avoid an overflow when normalizing
      transpDir[0] = int(tX) >> 1;
      transpDir[1] = int(tY) >> 1;
      transpDir[2] = int(tZ) >> 1;
      transpDir.normalize();
#endif

      const Ray<int> transpRay(hp, transpDir);
      HitRecord<int> transpHit;
      scene->getBVH()->intersect(transpHit, rc, transpRay);

      if (transpHit.getTriangle())
      {
        const Shader<int>*   shader    = transpHit.getShader();
        const uint&          mID       = transpHit.getMaterialID();
        const Material<int>& transpMat = scene->getBVH()->getMaterials()[mID];

        // Color's expect to be multiplied by an int which represents a 16 bit
        // fixed point number so we must cast (fp16::One - refl) to an int
        result += int(fp16::One - refl)*shader->shade(rc, transpRay, transpHit,
                                                      transpMat, depth+1);
      }
      else
      {
        // Color's expect to be multiplied by an int which represents a 16 bit
        // fixed point number so we must cast (fp16::One - refl) to an int
        result += int(fp16::One - refl)*scene->getBackground(transpDir);
      }
    }

    const Light<int>*const* begin = &lights[0];
    const Light<int>*const* end = begin + lights.size();
    while (begin != end)
    {
      RGB<int>    lightColor;
      Vector<int> lightDirection;
      const int dist = (*begin++)->getLight(lightColor, lightDirection,
                                            rc, hp);
      const int cosphi = Dot(phongN, lightDirection);
      
      if (!exiting && cosphi > 0)
      {
        if (cosphi > 0)
        {
          HitRecord<int> shadowHit(dist);
          const Ray<int> shadowRay(hp, lightDirection);
          scene->getBVH()->intersect(shadowHit, rc, shadowRay);

          if (!shadowHit.getTriangle())
          {
            // We shift both lightDirection and dir down by 2 to avoid an
            // overflow when normalizing vector H
            Vector<int> H = (lightDirection >> 2) + (exiting ? (dir >> 2) :
                                                     -(dir >> 2));
            H.normalize();

            const fp31 cosAlpha = fp31(Dot(H, phongN));
            if (cosAlpha > fp31::Zero)
            {
              const fp31 phongTerm = Math::Pow(cosAlpha, material.exp);
              // We must convert the fp31 phongTerm to a 16 bit fixed point int
              // because that is what a color expects to be multiplied by
              result += lightColor*int(fp16(phongTerm));
            }
          }
        }
      }
    }

    return result;
  }

} // namespace tangere

#endif // tangere_DielectricShader_t
