
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

#ifndef tangere_BBox_t
#define tangere_BBox_t

#include <iostream>
using std::istream;
using std::ostream;

#include <Math/Math.h>

#include <tangere/Flags.h>
#include <tangere/Ray.t>

namespace tangere
{

  ///////////////////////////////////////////////////////////////////////////////
  // Using declarations

  using Math::Point;
  using Math::Vector;

  ///////////////////////////////////////////////////////////////////////////////
  // Class template definition

  template<typename T>
  struct BBox
  {
    inline BBox()
    {
      bounds[0] = Math::Point<T>::Max;
      bounds[1] = Math::Point<T>::Min;
    }

    inline ~BBox()
    {
      //no-op
    }

    inline bool intersect(T& thit, const Ray<T>& ray) const
    {
#if defined(USE_INVDIR)
      const Point<T>& org  = ray.org();
      const T*        inv  = ray.inv();
      const uint*     sign = ray.sign();

      const T tminX = (bounds[  sign[0]][0] - org[0])*inv[0];
      const T tmaxY = (bounds[1-sign[1]][1] - org[1])*inv[1];

      if (tminX > tmaxY)
        return false;

      const T tmaxX = (bounds[1-sign[0]][0] - org[0])*inv[0];
      const T tminY = (bounds[  sign[1]][1] - org[1])*inv[1];

      if (tmaxX < tminY)
        return false;

      T tmin = (tminX > tminY ? tminX : tminY);
      const T tmaxZ = (bounds[1-sign[2]][2] - org[2])*inv[2];

      if (tmin > tmaxZ)
        return false;

      T tmax = (tmaxX < tmaxY ? tmaxX : tmaxY);
      const T tminZ = (bounds[  sign[2]][2] - org[2])*inv[2];
    
      if (tmax < tminZ)
        return false;

      tmin = (tmin > tminZ ? tmin : tminZ);
      tmax = (tmax < tmaxZ ? tmax : tmaxZ);

      if (tmin <= tmax)
      {
        thit = tmin;
        return true;
      }

      return false;
#else
      const Point<T>&  org  = ray.org();
      const Vector<T>& dir  = ray.dir();
      const uint*      sign = ray.sign();

      const T tminX = (bounds[  sign[0]][0] - org[0])/dir[0];
      const T tmaxY = (bounds[1-sign[1]][1] - org[1])/dir[1];

      if (tminX > tmaxY)
        return false;

      const T tmaxX = (bounds[1-sign[0]][0] - org[0])/dir[0];
      const T tminY = (bounds[  sign[1]][1] - org[1])/dir[1];

      if (tmaxX < tminY)
        return false;

      T tmin = (tminX > tminY ? tminX : tminY);
      const T tmaxZ = (bounds[1-sign[2]][2] - org[2])/dir[2];

      if (tmin > tmaxZ)
        return false;

      T tmax = (tmaxX < tmaxY ? tmaxX : tmaxY);
      const T tminZ = (bounds[  sign[2]][2] - org[2])/dir[2];
    
      if (tmax < tminZ)
        return false;

      tmin = (tmin > tminZ ? tmin : tminZ);
      tmax = (tmax < tmaxZ ? tmax : tmaxZ);

      if (tmin <= tmax)
      {
        thit = tmin;
        return true;
      }

      return false;
#endif // defined(USE_INVDIR)
    }

    inline void extend(const BBox<T>& b)
    {
      bounds[0] = Min(b.bounds[0], bounds[0]);
      bounds[1] = Max(b.bounds[1], bounds[1]);
    }

    inline void extend(const Point<T>& p)
    {
      bounds[0] = Min(p, bounds[0]);
      bounds[1] = Max(p, bounds[1]);
    }

    inline void extend(const Vector<T>& v)
    {
      bounds[0] = Min(Point<T>(v), bounds[0]);
      bounds[1] = Max(Point<T>(v), bounds[1]);
    }

    inline void reset()
    {
      bounds[0] = Math::Point<T>::Max;
      bounds[1] = Math::Point<T>::Min;
    }

    inline Point<T> center() const
    {
      return (bounds[0] + bounds[1])/2;
    }

    inline Vector<T> diagonal() const
    {
      return (bounds[1] - bounds[0]);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Stream I/O

    friend istream& operator>>(istream& in, BBox<T>& b)
    {
      in >> b.bounds[0] >> b.bounds[1];
      return in;
    }

    friend ostream& operator<<(ostream& out, const BBox<T>& b)
    {
      out << b.bounds[0] << ' ' << b.bounds[1];
      return out;
    }

    Point<T> bounds[2];
  };

  ///////////////////////////////////////////////////////////////////////////////
  // Template specialization - BBox<int>

  template<>
  inline bool BBox<int>::intersect(int& t, const Ray<int>& ray) const
  {
#if defined(USE_INVDIR)
    const Point<int>& org  = ray.org();
    const int64_t*    inv  = ray.inv();
    const uint*       sign = ray.sign();

    // We shift the (bounds - org) calculation down by dBits because the ray
    // inverse has been shifted up by 'apprximately' that much too. This
    // depends on whether or not the dBits accuracy optimization is being used.
    // If it is not being used, then we shift down by the amount that the
    // inverse had been shifted up. If we are using the optimization, then we
    // shift down by an amount smaller than what we shifted up by.
    const int64_t tminX = (int64_t(bounds[  sign[0]][0] -
                           org[0]) >> dBits) * inv[0];
    const int64_t tmaxY = (int64_t(bounds[1-sign[1]][1] -
                           org[1]) >> dBits) * inv[1];

    if (tminX > tmaxY)
      return false;

    const int64_t tmaxX = (int64_t(bounds[1-sign[0]][0] -
                           org[0]) >> dBits) * inv[0];
    const int64_t tminY = (int64_t(bounds[  sign[1]][1] -
                           org[1]) >> dBits) * inv[1];

    if (tmaxX < tminY)
      return false;

          int64_t tmin  = (tminX > tminY ? tminX : tminY);
    const int64_t tmaxZ = (int64_t(bounds[1-sign[2]][2] -
                           org[2]) >> dBits) * inv[2];

    if (tmin > tmaxZ)
      return false;

          int64_t tmax  = (tmaxX < tmaxY ? tmaxX : tmaxY);
    const int64_t tminZ = (int64_t(bounds[  sign[2]][2] -
                           org[2]) >> dBits)* inv[2];

    if (tmax < tminZ)
      return false;

    tmin = (tmin > tminZ ? tmin : tminZ);
    tmax = (tmax < tmaxZ ? tmax : tmaxZ);

    if (tmin <= tmax)
    {
#if defined(USE_D_BITS)
      // If the dBits accuracy optimization is being used then the t value at
      // this point is (cBits - dBits) too large and we need to shift down
      // by the difference.
      tmin >>= cdDiff;
#endif // defined(USE_D_BITS)

      if (tmin > INT_MAX || tmin < -INT_MAX)
        return false;

      t = int(tmin);
      return true;
    }

    return false;
#else
    const Point<int>&  org  = ray.org();
    const Vector<int>& dir  = ray.dir();
    const uint*        sign = ray.sign();

    int64_t tmin, tmax;
    if (dir[0] == 0)
    {
      tmin = -INT_MAX;
      tmax =  INT_MAX;
    }
    else
    {
      tmin = DIV31_64(bounds[  sign[0]][0] - org[0], dir[0]);
      tmax = DIV31_64(bounds[1-sign[0]][0] - org[0], dir[0]);
    }

    if (dir[1] == 0)
    {
      tmin = -INT_MAX;
      tmax =  INT_MAX;
    }
    else
    {
      const int64_t tminY = DIV31_64(bounds[  sign[1]][1] - org[1], dir[1]);
      const int64_t tmaxY = DIV31_64(bounds[1-sign[1]][1] - org[1], dir[1]);

     	if (tmin > tmaxY || tmax < tminY)
    		return false;

      tmin  = (tmin > tminY ? tmin : tminY);
      tmax  = (tmax < tmaxY ? tmax : tmaxY);
    }

    if (dir[2] == 0)
    {
      tmin = -INT_MAX;
      tmax =  INT_MAX;
    }
    else
    {
      const int64_t tminZ = DIV31_64(bounds[  sign[2]][2] - org[2], dir[2]);
      const int64_t tmaxZ = DIV31_64(bounds[1-sign[2]][2] - org[2], dir[2]);

     	if (tmin > tmaxZ || tmax < tminZ)
    		return false;

      tmin  = (tmin > tminZ ? tmin : tminZ);
      tmax  = (tmax < tmaxZ ? tmax : tmaxZ);
    }

    if (tmin <= tmax)
    {
      if (tmin > INT_MAX || tmin < -INT_MAX)
        return false;

      t = static_cast<int>(tmin);
      return true;
    }

    return false;
#endif // defined(USE_INVDIR)
  }

} // namespace tangere

#endif // tangere_BBox_h
