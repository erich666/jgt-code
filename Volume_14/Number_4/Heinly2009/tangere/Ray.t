
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

#ifndef tangere_Ray_t
#define tangere_Ray_t

#include <climits>

#include <iosfwd>
using std::ostream;

#include <Common/Types.h>

#include <Math/Math.h>

#include <tangere/Constants.h>
#include <tangere/Flags.h>

namespace tangere
{

  using Math::Abs;
  using Math::Point;
  using Math::Vector;

  template<typename T>
  class Ray
  {
  public:
    inline Ray(const Point<T>& o_, const Vector<T>& d_) :
      o(o_), d(d_)
    {
      s[0] = (d[0] < 0);
      s[1] = (d[1] < 0);
      s[2] = (d[2] < 0);

#if defined(USE_INVDIR)
      i[0] = 1/d[0];
      i[1] = 1/d[1];
      i[2] = 1/d[2];
#endif // defined(USE_INVDIR)
    }

    inline ~Ray()
    {
      // no-op
    }

    inline const Point<T>&  org()  const { return o; }
    inline const Vector<T>& dir()  const { return d; }
    inline const uint*      sign() const { return s; }
#if defined(USE_INVDIR)
    inline const T*         inv()  const { return i; }
#endif // defined(USE_INVDIR)

    inline friend ostream& operator<<(ostream& out, const Ray& r)
    {
      out << r.o << ' ' << r.d;
      return out;
    }

  private:
    Point<T>  o;
    Vector<T> d;
    uint      s[3];
#if defined(USE_INVDIR)
    T         i[3];
#endif // defined(USE_INVDIR)
  };

  ///////////////////////////////////////////////////////////////////////////////
  // Template specialization - Ray<int>

  template<>
  class Ray<int>
  {
  public:
    inline Ray(const Point<int>& o_, const Vector<int>& d_) :
      o(o_), d(d_)
    {
      // NOTE(cpg) - Force direction to have non-zero component
      d[0] = (d[0] == 0 ? 1 : d[0]);
      d[1] = (d[1] == 0 ? 1 : d[1]);
      d[2] = (d[2] == 0 ? 1 : d[2]);

      s[0] = (d[0] < 0);
      s[1] = (d[1] < 0);
      s[2] = (d[2] < 0);

#if defined(USE_INVDIR)
      const int64_t numer = int64_t(1) << (31 + cBits);

#if defined(USE_D_BITS)
      // NOTE(jsh) - Force the direction to have a magnitude greater than or
      //             equal to 32 while using it in the inverse direction
      //             calculation. This allows us determine the maximum number
      //             of bits that we would require to represent the inv.
      //             The calculation (-2*s[0] + 1) returns a -1 if the sign
      //             represents a negative number and 1 if it is positive
      /*
      i[0] = (Abs(d[0]) >= 1 << cdDiffShift ?
             (numer / d[0]) : (-2*s[0] + 1) * numer >> cdDiffShift);
      i[1] = (Abs(d[1]) >= 1 << cdDiffShift ?
             (numer / d[1]) : (-2*s[1] + 1) * numer >> cdDiffShift);
      i[2] = (Abs(d[2]) >= 1 << cdDiffShift ?
             (numer / d[2]) : (-2*s[2] + 1) * numer >> cdDiffShift);
      */
      i[0] = (Abs(d[0]) >= 1 << cdDiffShift ?
              (numer / d[0]) : (-(s[0] << 1) + 1) * numer >> cdDiffShift);
      i[1] = (Abs(d[1]) >= 1 << cdDiffShift ?
              (numer / d[1]) : (-(s[1] << 1) + 1) * numer >> cdDiffShift);
      i[2] = (Abs(d[2]) >= 1 << cdDiffShift ?
              (numer / d[2]) : (-(s[2] << 1) + 1) * numer >> cdDiffShift);
#else
      i[0] = numer / d[0];
      i[1] = numer / d[1];
      i[2] = numer / d[2];
#endif // defined(USE_D_BITS)
#endif // defined(USE_INVDIR)
    }

    inline ~Ray()
    {
      // no-op
    }

    inline const Point<int>&  org()  const { return o; }
    inline const Vector<int>& dir()  const { return d; }
    inline const uint*        sign() const { return s; }
#if defined(USE_INVDIR)
    inline const int64_t*     inv()  const { return i; }
#endif // defined(USE_INVDIR)

    inline friend ostream& operator<<(ostream& out, const Ray& r)
    {
      out << r.o << ' ' << r.d;
      return out;
    }

  private:
    Point<int>  o;
    Vector<int> d;
    uint        s[3];
#if defined(USE_INVDIR)
    int64_t     i[3];
#endif // defined(USE_INVDIR)
  };

} // namespace tangere

#endif // tangere_Ray_h
