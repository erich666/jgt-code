
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

#ifndef Math_FixedPoint_t
#define Math_FixedPoint_t

#include <climits>
#include <iostream>
using std::istream;
using std::ostream;

#include <Common/Types.h>

namespace Math
{

  /////////////////////////////////////////////////////////////////////////////
  // Class template definition

  template<int N>
  class FixedPoint
  {
  public:

    ///////////////////////////////////////////////////////////////////////////
    // Static members

    static const FixedPoint Max;
    static const FixedPoint Min;
    static const FixedPoint One;
    static const FixedPoint Zero;

    inline FixedPoint(const FixedPoint& f) :
      val(f.val)
    {
      // no-op
    }

    inline FixedPoint(int i) :
      val(i)
    {
      // no-op
    }

    inline FixedPoint() :
      val(0)
    {
      // no-op
    }

    template<int M>
    inline explicit FixedPoint(const FixedPoint<M>& f)
    {
      if (M > N)
        val = (((f.val) >> ((M - N) - 1)) + 1) >> 1;
      else // if (M < N)
        val = f.val << (N - M);
    }

    inline explicit FixedPoint(float f)
    {
      val = static_cast<int>(static_cast<double>(f) * (one - 1));
    }

    inline explicit FixedPoint(double d)
    {
      val = static_cast<int>(d * (one - 1));
    }

    inline ~FixedPoint()
    {
      // no-op
    }

    inline operator float() const
    {
      return static_cast<float>(static_cast<double>(val) / one);
    }

    inline operator double() const
    {
      return static_cast<double>(val) / one;
    }

    inline operator int() const
    {
      return val;
    }

    inline operator int64_t() const
    {
      return static_cast<int64_t>(val);
    }

    inline operator uint64_t() const
    {
      return static_cast<uint64_t>(val);
    }

    inline FixedPoint& operator=(const FixedPoint& f)
    {
      val = f.val;
      return *this;
    }

    inline FixedPoint operator-() const
    {
      return FixedPoint(-val);
    }

    inline FixedPoint operator+(const FixedPoint& f) const
    {
      return FixedPoint(val + f.val);
    }

    inline FixedPoint operator-(const FixedPoint& f) const
    {
      return FixedPoint(val - f.val);
    }

    inline FixedPoint operator*(const FixedPoint& f) const
    {
      const int64_t val64   = static_cast<int64_t>(val);
      const int64_t fVal64  = static_cast<int64_t>(f.val);
      const int64_t shifted = (((val64 * fVal64) >> (N - 1)) + 1) >> 1;
      return FixedPoint(static_cast<int>(shifted));
    }

    inline FixedPoint operator/(const FixedPoint& f) const
    {
      const int64_t val64  = static_cast<int64_t>(val);
      const int64_t fVal64 = static_cast<int64_t>(f.val);
      return FixedPoint(static_cast<int>((val64 << N) / fVal64));
    }

    inline FixedPoint operator<<(uint shift) const
    {
      return FixedPoint(val << shift);
    }

    inline FixedPoint operator>>(uint shift) const
    {
      return FixedPoint(val >> shift);
    }

    inline FixedPoint& operator+=(const FixedPoint& f)
    {
      val += f.val;
      return *this;
    }

    inline FixedPoint& operator-=(const FixedPoint& f)
    {
      val -= f.val;
      return *this;
    }

    inline FixedPoint& operator*=(const FixedPoint& f)
    {
      const int64_t val64   = static_cast<int64_t>(val);
      const int64_t fVal64  = static_cast<int64_t>(f.val);
      const int64_t shifted = (((val64 * fVal64) >> (N - 1)) + 1) >> 1;
      val = static_cast<int>(shifted);
      return *this;
    }

    inline FixedPoint& operator/=(const FixedPoint& f)
    {
      const int64_t val64  = static_cast<int64_t>(val);
      const int64_t fVal64 = static_cast<int64_t>(f.val);
      val = static_cast<int>((val64 << N) / fVal64);
      return *this;
    }

    inline FixedPoint& operator<<=(uint shift)
    {
      val <<= shift;
      return *this;
    }

    inline FixedPoint& operator>>=(uint shift)
    {
      val >>= shift;
      return *this;
    }

    inline bool operator==(const FixedPoint& f) const
    {
      return (val == f.val);
    }

    inline bool operator!=(const FixedPoint& f) const
    {
      return (val != f.val);
    }

    inline bool operator>(const FixedPoint& f) const
    {
      return (val > f.val);
    }

    inline bool operator<(const FixedPoint& f) const
    {
      return (val < f.val);
    }

    inline bool operator>=(const FixedPoint& f) const
    {
      return (val >= f.val);
    }

    inline bool operator<=(const FixedPoint& f) const
    {
      return (val <= f.val);
    }

    template<int M>
    friend class FixedPoint;

    friend ostream& operator<<(ostream& out, const FixedPoint& f)
    {
      out << "(" << f.val << " = " << double(f) << ")";
      return out;
    }

    friend istream& operator>>(istream& in, const FixedPoint& f)
    {
      in >> f.val;
      return in;
    }

  private:
    static const int one;
    int val;
  };

  /////////////////////////////////////////////////////////////////////////////
  // Helper function + specialization for (N == 31)

  // XXX(cpg) - this helper function is necessary to deal with silliness in g++:
  //            multiply defined symbols when FixedPoint<31>::One is specalized

  template<int N>
  inline int OneHelper()
  {
    return (1 << N);
  }

  template<>
  inline int OneHelper<31>()
  {
    return INT_MAX;
  }

  /////////////////////////////////////////////////////////////////////////////
  // Static member definitions

  template<int N>
  const FixedPoint<N> FixedPoint<N>::Max(INT_MAX);

  template<int N>
  const FixedPoint<N> FixedPoint<N>::Min(1);

  template<int N>
  const FixedPoint<N> FixedPoint<N>::One(OneHelper<N>());
  
  template<int N>
  const FixedPoint<N> FixedPoint<N>::Zero(0);

  template<int N>
  // const int FixedPoint<N>::one(1 << N);
  const int FixedPoint<N>::one = (1 << N);

  /////////////////////////////////////////////////////////////////////////////
  // Type definitions

  typedef FixedPoint<16> fp16;
  typedef FixedPoint<31> fp31;

} // namespace Math 

#endif // Math_FixedPoint_t
