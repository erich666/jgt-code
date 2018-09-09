
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

#ifndef tangere_Camera_t
#define tangere_Camera_t

#include <Math/Math.h>

#include <tangere/Ray.t>
#include <tangere/SceneScaler.h>

namespace tangere
{

  /////////////////////////////////////////////////////////////////////////////
  // Using declarations

  using Math::Point;
  using Math::Vector;

  /////////////////////////////////////////////////////////////////////////////
  // Class template definition

  template<typename T>
  class Camera
  {
  public: 
    inline Camera(const MathF::Point&  e_,
                  const MathF::Point&  lookat,
                  const MathF::Vector& up,
                        float          fov,
                        float          aspect,
                  const SceneScaler&   /* unused */) :
      e(e_)
    {
      w = lookat - e;
      w.normalize();

      u = Cross(w, up);
      v = Cross(u, w);

      float ulen = tanf(0.5f*fov*DegreesToRadians);
      u.normalize();
      u *= ulen;

      float vlen = ulen/aspect;
      v.normalize();
      v *= vlen;
    }

    inline Camera()
    {
      // no-op
    }

    inline ~Camera()
    {
      // no-op
    }

    inline Ray<T> generate(float x, float y) const
    {
      Vector<T> d = w + u*x + v*y;
      d.normalize();

      return Ray<T>(e, d);
    }

    inline const Point<T>&  getEye()   const { return e; }
    inline const Vector<T>& getRight() const { return u; }
    inline const Vector<T>& getUp()    const { return v; }
    inline const Vector<T>& getIn()    const { return w; }

  private:
    const Point<T>  e;
          Vector<T> u;
          Vector<T> v;
          Vector<T> w;
  };

  //////////////////////////////////////////////////////////////////////////////
  // Template specializations - Camera<int>

  template<>
  inline Camera<int>::Camera(const MathF::Point&  eF,
                             const MathF::Point&  lookat,
                             const MathF::Vector& up,
                                   float          fov,
                                   float          aspect,
                             const SceneScaler&   scaler) :
    e(scaler.scale(eF))
  {
    MathF::Vector wf = lookat - eF;
    wf.normalize();

    MathF::Vector uf = Cross(wf, up);
    MathF::Vector vf = Cross(uf, wf);

    float ulen = tanf(0.5f*fov*DegreesToRadians);
    uf.normalize();
    uf *= ulen;

    float vlen = ulen/aspect;
    vf.normalize();
    vf *= vlen;

    // Convert ONB to fixed point and multiply by 0.5f so that when we compute
    // the ray direction in the generate function we don't overflow.
    u = Vector<int>(0.5f*uf);
    v = Vector<int>(0.5f*vf);
    w = Vector<int>(0.5f*wf);
  }

  template<>
  inline Ray<int> Camera<int>::generate(float x, float y) const
  {
    const int xi = FLOAT_TO_FIXED31(x);
    const int yi = FLOAT_TO_FIXED31(y);

    // This calculation won't overflow because u, v, and w have been
    // divided by 2
    Vector<int> d = w + xi*u + yi*v;
    d.normalize();

    return Ray<int>(e, d);
  }

} // namespace tangere

#endif // tangere_Camera_t
