
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

#ifndef tangere_Triangle_h
#define tangere_Triangle_h

#include <cfloat>

#include <iostream>
using std::ostream;

#include <Common/Utility/MinMax.t>

#include <Math/Math.h>

#include <tangere/Context.t>
#include <tangere/HitRecord.t>
#include <tangere/Ray.t>

namespace tangere
{

  /////////////////////////////////////////////////////////////////////////////
  // Using declarations

  using Math::Abs;
  using Math::Point;
  using Math::Vector;

  using Utility::Max;
  using Utility::Min;

  /////////////////////////////////////////////////////////////////////////////
  // Forward declarations

  template<typename T>
  struct Shader;

  /////////////////////////////////////////////////////////////////////////////
  // Class template definition

  template<typename T>
  class Triangle
  {
  public:
    Triangle(const Point<T>&  p0,
             const Point<T>&  p1,
             const Point<T>&  p2,
             const Vector<T>& v0n,
             const Vector<T>& v1n,
             const Vector<T>& v2n,
                   uint       mID_,
                   Shader<T>* s) :
      mID(mID_), shader(s)
    {
      vn[0] = v0n;
      vn[1] = v1n;
      vn[2] = v2n;

      const Point<T>  points[3] = { p0, p1, p2 };
      const Vector<T> e[2]      = { points[1] - points[0],
                                    points[2] - points[0] };

      /////////////////////////////////////////////////////////////////////////
      // Wald Projection Method Init Code

      Vector<T> norm = Cross(e[0], e[1]);

      // Determine longest component of the normal
      r = (fabsf(norm[0]) > fabsf(norm[1]) ? 0 : 1);
      r = (fabsf(norm[r]) > fabsf(norm[2]) ? r : 2);
      p = (r + 1) % 3;
      q = (r + 2) % 3;

      np = norm[p] / norm[r];
      nq = norm[q] / norm[r];

      // Compute distance to origin, use N'
      d = points[0][r] + points[0][q]*nq + points[0][p]*np;

      // Let b = e1, c = e2, A = v1
      const Vector<T>& b = e[0];
      const Vector<T>& c = e[1];
      const Point<T>&  A = points[0];

      // Beta's precomputed data
      KBp =                   -b[q] / (b[p]*c[q] - b[q]*c[p]);
      KBq =                    b[p] / (b[p]*c[q] - b[q]*c[p]);
      KBd = (b[q]*A[p] - b[p]*A[q]) / (b[p]*c[q] - b[q]*c[p]);

      // Gamma's precomputed data
      KGp =                    c[q] / (b[p]*c[q] - b[q]*c[p]);
      KGq =                   -c[p] / (b[p]*c[q] - b[q]*c[p]);
      KGd = (c[p]*A[q] - c[q]*A[p]) / (b[p]*c[q] - b[q]*c[p]);
    }

    inline Triangle()
    {
      // no-op
    }

    inline ~Triangle()
    {
      // no-op
    }

    inline void intersect(      HitRecord<T>&     hit,
                          const RenderContext<T>& rc,
                          const Ray<T>&           ray) const
    {
      const Vector<T>& dir   = ray.dir();
      const T          denom = dir[r] +  dir[p]*np + dir[q]*nq;
      if (Abs<T>(denom) < ::Constants<T>::Epsilon)
        return;

      const Point<T>&  org   = ray.org();
      const T          numer = d - org[r] - org[p]*np - org[q]*nq;
      const T          thit  = numer/denom;
      if (thit < ::Constants<T>::Epsilon)
        return;

      // Compute hit point projected to the u-v plane
      const T Hp = org[p] + dir[p]*thit;
      const T Hq = org[q] + dir[q]*thit;

      // Compute first barycentric coordinate
      const T beta = KBp*Hp + KBq*Hq + KBd;
      if (beta < 0 || beta > 1)
        return;

      // Compute second barycentric coordinate
      const T gamma = KGp*Hp + KGq*Hq + KGd;
      if (gamma < 0 || beta + gamma > 1)
        return;

      hit.hit(thit, this, Vector<T>(1 - beta - gamma, gamma, beta), mID);
    }

    inline Vector<T> normal(const RenderContext<T>& rc,
                            const Ray<T>&           ray,
                            const Point<T>&         hitPoint,
                            const HitRecord<T>&     hit) const
    {
      const Vector<T> bary = hit.getBary();
      const Vector<T> na   = bary[0]*vn[0];
      const Vector<T> nb   = bary[1]*vn[1];
      const Vector<T> nc   = bary[2]*vn[2];

      // Interpolate normal
      return Vector<T>(na[0] + nb[0] + nc[0],
                       na[1] + nb[1] + nc[1],
                       na[2] + nb[2] + nc[2]).normal();
    }

    inline uint getMID() const
    {
      return mID;
    }

    inline const Shader<T>* getShader() const
    {
      return shader;
    }

    inline void setShader(const Shader<T>* s)
    {
      shader = s;
    }

    friend ostream& operator<<(ostream& out, const Triangle<T>& t)
    {
      out << t.d << ' ' << t.np << ' ' << t.nq << " XXX(cpg) - not yet implemented";
      return out;
    }

  private:
    T d;  // distance along normal to origin
    T np; // p-component of the normal vector, normalized by nr
    T nq; // q-component of the normal vector, normalized by nr

    // Beta barycentric coordinate:  Beta = KBq*Hq + KBp*Hp + KBd
    T KBp; // bp / (bpcq - bqcp)
    T KBq; // -bv / (bpcq - bqcp)
    T KBd; // (bqAp - bpAq) / (bpcq - bqcp)

    // Gamma barycentric coordinate:  Gamma = KGv*Hv + KGu*Hu + KBd
    T KGp; // cq / (bpcq - bqcp)
    T KGq; // -cp / (bpcq - bqcp)
    T KGd; // (cpAq - Apcq) / (bpcq - bqcp)

    uint r : 2;
    uint p : 2;
    uint q : 2;

    Vector<T>  vn[3];
    const Shader<T>* shader;
    uint       mID;
  };

  /////////////////////////////////////////////////////////////////////////////
  // Template specialization - Triangle<int>

  template<>
  class Triangle<int>
  {
  public:
    Triangle(const Point<int>&  p0,
             const Point<int>&  p1,
             const Point<int>&  p2,
             const Vector<int>& v0n,
             const Vector<int>& v1n,
             const Vector<int>& v2n,
                   uint         mID_,
                   Shader<int>* s) :
      mID(mID_), shader(s)
    {
      vn[0] = v0n;
      vn[1] = v1n;
      vn[2] = v2n;

      Point<float> pf[3];
      for (uint i = 0; i < 3; ++i)
      {
        pf[0][i] = float(p0[i]);
        pf[1][i] = float(p1[i]);
        pf[2][i] = float(p2[i]);
      }

      const Vector<float> ef[2] = { pf[1] - pf[0], pf[2] - pf[0] };
      const Vector<float> nf    = Cross(ef[0], ef[1]);
    
      r = (fabsf(nf[0]) > fabsf(nf[1]) ? 0 : 1);
      r = (fabsf(nf[2]) > fabsf(nf[r]) ? 2 : r);

      const uint p = mod3[r + 1];
      const uint q = mod3[r + 2];

      const float npf = nf[p] / nf[r];
      const float nqf = nf[q] / nf[r];

      np = int(npf * double(INT_MAX)); // NOTE(jsh) - a float cannot represent
      nq = int(nqf * double(INT_MAX)); //             INT_MAX precisely

      pp = p0[p];
      pq = p0[q];

      d = p0[r] + MUL31(pp, np) + MUL31(pq, nq);

      float e0pf = ef[0][p] / nf[r];
      float e0qf = ef[0][q] / nf[r];

      float e1pf = ef[1][p] / nf[r];
      float e1qf = ef[1][q] / nf[r];

      const float eMax = Max(fabsf(e0pf), fabsf(e0qf),
                             fabsf(e1pf), fabsf(e1qf));
      maxEdgeComponent = Max(eMax, maxEdgeComponent);

      // Pack these floats into their integer equivalents so that they can
      // be referenced later when applying the edge bias.
      e0p = *reinterpret_cast<int*>(&e0pf);
      e0q = *reinterpret_cast<int*>(&e0qf);
      e1p = *reinterpret_cast<int*>(&e1pf);
      e1q = *reinterpret_cast<int*>(&e1qf);
    }

    inline Triangle()
    {
      // no-op
    }

    inline ~Triangle()
    {
      // no-op
    }

    inline static void computeEdgeBias()
    {
      edgeBias  = -int(ceilf(logf(maxEdgeComponent)/logf(2.f)));
      edgeScale = powf(2.f, float(edgeBias));
    }

    inline void applyEdgeBias()
    {
      // Unpack floats from the integer members, scale them, and then convert
      // them to their final integer representation.
      e0p = FLOAT_TO_FIXED31(*reinterpret_cast<float*>(&e0p)*edgeScale);
      e0q = FLOAT_TO_FIXED31(*reinterpret_cast<float*>(&e0q)*edgeScale);
      e1p = FLOAT_TO_FIXED31(*reinterpret_cast<float*>(&e1p)*edgeScale);
      e1q = FLOAT_TO_FIXED31(*reinterpret_cast<float*>(&e1q)*edgeScale);
    }

    inline void intersect(      HitRecord<int>&     hit,
                          const RenderContext<int>& rc,
                          const Ray<int>&           ray) const
    {
      const Vector<int>& dir       = ray.dir();
      const int64_t      omega[3]  = { dir[0], dir[1], dir[2] };
      const uint         p         = mod3[r+1];
      const uint         q         = mod3[r+2];

      const int64_t denom = omega[r] + MUL31(omega[p], np) + MUL31(omega[q], nq);
      if (denom == 0)
        return;

      const Point<int>&  org       = ray.org();
      const int64_t      origin[3] = { org[0], org[1], org[2] };
      const int64_t      numer     = (((d - origin[r]) << mMinusOne)
                                      - origin[p]*np - origin[q]*nq);

      const int64_t t64 = numer/denom;
      if (t64 > INT_MAX || t64 < -INT_MAX)
        return;

      const int thit = int(t64);

      if (thit < ::Constants<int>::Epsilon)
        return;

      const int64_t kp = origin[p] + MUL31(thit, omega[p]) - pp;
      const int64_t kq = origin[q] + MUL31(thit, omega[q]) - pq;

      // Compute first barycentric coordinate (note that it is shifted
      // up by edgeBias)
      const int64_t u  = int64_t(e0p)*kq - int64_t(e0q)*kp;
      if (u < 0)
        return;

      // Compute second barycentric coordinate (note that it is shifted
      // up by edgeBias)
      const int64_t v = int64_t(e1q)*kp - int64_t(e1p)*kq;
      if (v < 0)
        return;

      if (((u + v) >> edgeBias) > (int64_t(1) << mMinusOne))
        return;
    
      int beta  = int(u >> edgeBias);
      int gamma = int(v >> edgeBias);
      int alpha = INT_MAX - beta - gamma;
      hit.hit(thit, this, Vector<int>(alpha, gamma, beta), mID);
    }

    inline Vector<int> normal(const RenderContext<int>& rc,
                              const Ray<int>& ray,
                              const Point<int>& hitPoint,
                              const HitRecord<int>& hit) const
    {
      const Vector<int> bary = hit.getBary();
      // Shift vertex normals down by 2 to avoid overflow when computing
      // the interpolated normal
      const Vector<int> na   = (bary[0]*vn[0])>>2;
      const Vector<int> nb   = (bary[1]*vn[1])>>2;
      const Vector<int> nc   = (bary[2]*vn[2])>>2;

      // Interpolate normal
      return Vector<int>(na[0] + nb[0] + nc[0],
                         na[1] + nb[1] + nc[1],
                         na[2] + nb[2] + nc[2]).normal();
    }

    static inline int   getEdgeBias()         { return edgeBias;         }
    static inline float getMaxEdgeComponent() { return maxEdgeComponent; }
    static inline float getEdgeScale()        { return edgeScale;        }
    static inline float getMaxEdge()          { return maxEdge;          }
    static inline float getMinEdge()          { return minEdge;          }
  
    static inline void setEdgeBias(int eb)           { edgeBias         = eb; }
    static inline void setMaxEdgeComponent(float ec) { maxEdgeComponent = ec; }
    static inline void setEdgeScale(float es)        { edgeScale        = es; }
    static inline void setMaxEdge(float me)          { maxEdge          = me; }
    static inline void setMinEdge(float me)          { minEdge          = me; }

    inline uint getMID() const
    {
      return mID;
    }

    inline const Shader<int>* getShader() const
    {
      return shader;
    }

    static inline void resetStatics()
    {
      edgeBias         =  0;
      maxEdge          = -FLT_MAX;
      minEdge          =  FLT_MAX;
      maxEdgeComponent = -FLT_MAX;
      edgeScale        =  0;
    }

    inline void setShader(const Shader<int>* s)
    {
      shader = s;
    }

    friend ostream& operator<<(ostream& out, const Triangle<int>& t)
    {
      out << t.d << ' ' << t.np << ' ' << t.nq << ' '
          << t.pp << ' ' << t.e0p << ' ' << t.e1p << ' '
          << t.pq << ' ' << t.e0q << ' ' << t.e1q;
      return out;
    }

  private:
    static       int   edgeBias;
    static       float maxEdgeComponent;
    static       float edgeScale;
    static       float maxEdge;
    static       float minEdge;
    static const uint  mod3[5];

    int d;   // distance to origin along N'

    int e0p; // p-component of edge 0, scaled by Nr (see note); fixed point
    int e0q; // q-component of edge 0, scaled by Nr; fixed point

    int e1p; // p-component of edge 1, scaled by Nr (see note); fixed point
    int e1q; // q-component of edge 1, scaled by Nr (see note); fixed point

    int np;  // p-component of the normal, scaled by nr; fixed point
    int nq;  // q-component of the normal, scaled by nr; fixed point

    uint r  :  2; // index of longest component in the normal
    uint pp : 30; // p-component of the anchor point; not scaled, integer-land
    uint pq;      // q-component of the anchor point; not scaled, integer-land

    Vector<int> vn[3];          // vector of vertex normals
    uint mID;                   // material ID reference
    const Shader<int>* shader;  // pointer to shader
  };

} // namespace tangere

#endif // tangere_Triangle_h
