
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

#ifndef Math_Math_h
#define Math_Math_h

#include <Math/FixedPoint.h>
#include <Math/FixedPoint.t>
#include <Math/Point.t>
#include <Math/Vector.t>

#include <Common/Types.h>

////////////////////////////////////////////////////////////////////////////////
// MathF namespace

namespace MathF
{

  //////////////////////////////////////////////////////////////////////////////
  // Type definitions

  typedef Math::Point<float>  Point;
  typedef Math::Vector<float> Vector;

  //////////////////////////////////////////////////////////////////////////////
  // Forward declarations

  /*
  extern const Point& pEpsilon;
  extern const Point& pMax;
  extern const Point& pMin;
  extern const Point& pOne;
  extern const Point& pZero;

  extern const Vector& vEpsilon;
  extern const Vector& vMax;
  extern const Vector& vMin;
  extern const Vector& vOne;
  extern const Vector& vZero;
  */

} // namespace MathF

namespace MathI
{

  //////////////////////////////////////////////////////////////////////////////
  // Type definitions

  typedef Math::Point<int>  Point;
  typedef Math::Vector<int> Vector;

  //////////////////////////////////////////////////////////////////////////////
  // Forward declarations

  /*
  extern const Point& pEpsilon;
  extern const Point& pMax;
  extern const Point& pMin;
  extern const Point& pOne;
  extern const Point& pZero;

  extern const Vector& vEpsilon;
  extern const Vector& vMax;
  extern const Vector& vMin;
  extern const Vector& vOne;
  extern const Vector& vZero;
  */

} // namespace MathI

////////////////////////////////////////////////////////////////////////////////
// Math namespace

namespace Math
{

  //////////////////////////////////////////////////////////////////////////////
  // Helper functions

  template<typename T>
  inline T Abs(const T& x)
  {
    return (x < 0 ? -x : x);
  }

  template<typename T>
  inline T Pow(const T& x, const T& y)
  {
    return T(pow(double(x), double(y)));
  }

  template<typename T>
  inline T Sqrt(const T& x)
  {
    // Convert to double and then back to T
    return T(sqrt(double(x)));
  }

  inline int SqrtI(uint64_t x)
  {
    uint64_t m, y, b;
    m = 0x4000000000000000LL;
    y = 0;

    while (m != 0)
    {
      b = y | m;
      y >>= 1;

      // NOTE(jsh) - the following if statement could be replaced with code that
      //             uses a double word shift, but the shift is more expensive
      //             than the if
      if (x >= b)
      {
        x = x - b;
        y = y | m;
      }

      m >>= 2;
    }

    return static_cast<int>(y);
  }

  template<int N>
  inline FixedPoint<N> Sqrt(const FixedPoint<N>& x)
  {
    // NOTE(jsh) - need to shift a fixed point number up by N first before
    //             performing a sqrt
    return SqrtI(uint64_t(x) << N);
  }

  // NOTE(jsh) - the first parameter is a 31 bit fixed point number and the
  //             second parameter is a normal integer power
  inline fp31 Pow(fp31 x, int n)
  {
    fp31 y = fp31::One;

    while (true)
    {
      if (n & 1)
        y *= x;
      n >>= 1;
      if (n == 0)
        return y;
      x *= x;
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Template specializations

  template<>
  inline float Pow(const float& x, const float& y)
  {
    return powf(x, y);
  }

  template<>
  inline float Sqrt(const float& x)
  {
    return sqrtf(x);
  }

  template<>
  inline int Pow(const int& x, const int& y)
  {
    // XXX(cpg) - now yet implemented
    return 1;
  }

  template<>
  inline int Sqrt(const int& x)
  {
    // NOTE(jsh) - need to shift a fixed point number up by 31 first before
    //             performing a sqrt
    return SqrtI(static_cast<uint64_t>(x) << 31);
  }

} // namespace Math

#endif // Math_Math_h
