
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

#ifndef Math_Point_t
#define Math_Point_t

#include <cmath>

#include <limits>
using std::numeric_limits;

#include <iostream>
using std::istream;
using std::ostream;

#include <typeinfo>

#include <Common/Utility/MinMax.t>
#include <Common/Utility/OutputCC.h>
#include <Common/Constants.h>
#include <Common/Types.h>

namespace Math
{

  ///////////////////////////////////////////////////////////////////////////////
  // Forward declarations

  template<typename T>
  class Point;

  template<typename T>
  class Vector;

  template<typename T>
  Point<T> operator*(const T&, const Point<T>&);

  ///////////////////////////////////////////////////////////////////////////////
  // Class template definition

  template<typename T>
  class Point
  {
  public:

    /////////////////////////////////////////////////////////////////////////////
    // Static members

    static const Point Epsilon;
    static const Point One;
    static const Point Max;
    static const Point Min;
    static const Point Zero;

    /////////////////////////////////////////////////////////////////////////////
    // Constructors

    inline Point(const T& x, const T& y, const T& z)
    {
      e[0] = x;
      e[1] = y;
      e[2] = z;
    }

    inline Point(const Point& p)
    {
      e[0] = p.e[0];
      e[1] = p.e[1];
      e[2] = p.e[2];
    }

    inline Point()
    {
      e[0] = 0;
      e[1] = 0;
      e[2] = 0;
    }

    /////////////////////////////////////////////////////////////////////////////
    // Destructor

    inline ~Point()
    {
      // no-op
    }

    /////////////////////////////////////////////////////////////////////////////
    // Member operators

    inline T& operator[](uint i)
    {
      return e[i];
    }

    inline const T& operator[](uint i) const
    {
      return e[i];
    }

    inline Point operator-() const
    {
      return Point(-e[0], -e[1], -e[2]);
    }

    inline Point operator<<(uint /* unused */) const 
    {
      // NOTE(cpg) - effectively a no-op for anything but Point<int>; see
      //             template specializations below

      FatalError("Point<" << typeid(T).name()
                 << ">::operator<<(uint) - should not be called!" << endl);

      // To quiet warnings...
      return *this;
    }

    inline Point operator>>(uint /* unused */) const 
    {
      // NOTE(cpg) - effectively a no-op for anything but Point<int>; see
      //             template specializations below

      FatalError("Point<" << typeid(T).name()
                 << ">::operator<<(uint) - should not be called!" << endl);

      // To quiet warnings...
      return *this;
    }
    
    inline Point& operator<<=(uint /* unused */)  
    {
      // NOTE(cpg) - effectively a no-op for anything but Point<int>; see
      //             template specializations below

      FatalError("Point<" << typeid(T).name()
                 << ">::operator<<(uint) - should not be called!" << endl);

      // To quiet warnings...
      return *this;
    }

    inline Point& operator>>=(uint /* unused */)  
    {
      // NOTE(cpg) - effectively a no-op for anything but Point<int>; see
      //             template specializations below

      FatalError("Point<" << typeid(T).name()
                 << ">::operator<<(uint) - should not be called!" << endl);

      // To quiet warnings...
      return *this;
    }

    /////////////////////////////////////////////////////////////////////////////
    // Point<T>-Point<T> operators

    inline Point operator+(const Point& p) const
    {
      return Point(e[0] + p.e[0], e[1] + p.e[1], e[2] + p.e[2]);
    }

    inline Vector<T> operator-(const Point& p) const
    {
      return Vector<T>(e[0] - p.e[0], e[1] - p.e[1], e[2] - p.e[2]);
    }

    ///////////////////////////////////////////////////////////////////////////
    // Point<T>-Vector<T> interoperability

    inline Point operator+(const Vector<T>& v) const
    {
      return Point(e[0] + v.e[0], e[1] + v.e[1], e[2] + v.e[2]);
    }

    inline Point operator-(const Vector<T>& v) const
    {
      return Point(e[0] - v.e[0], e[1] - v.e[1], e[2] - v.e[2]);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Point<T>-T interoperability

    inline Point operator/(const T& s) const
    {
      const T inv = 1/s;
      return Point(e[0]*inv, e[1]*inv, e[2]*inv);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Friend classes, functions

    friend class Vector<T>;

    friend Point<T> Math::operator* <> (const T&, const Point<T>&);

    inline friend Point Min(const Point& p0, const Point& p1)
    {
      return Point(Utility::Min(p0.e[0], p1.e[0]),
                   Utility::Min(p0.e[1], p1.e[1]),
                   Utility::Min(p0.e[2], p1.e[2]));
    }

    inline friend Point Max(const Point& p0, const Point& p1)
    {
      return Point(Utility::Max(p0.e[0], p1.e[0]),
                   Utility::Max(p0.e[1], p1.e[1]),
                   Utility::Max(p0.e[2], p1.e[2]));
    }

    ////////////////////////////////////////////////////////////////////////////
    // Stream I/O

    inline friend istream& operator>>(istream& in, Point& p)
    {
      char junk;
      in >> junk;
      in >> p.e[0] >> p.e[1] >> p.e[2];
      in >> junk;

      return in;
    }

    inline friend ostream& operator<<(ostream& out, const Point& p)
    {
      out << '(' << p.e[0] << ' ' << p.e[1] << ' ' << p.e[2] << ')';
      return out;
    }

  private:
    T e[3];
  };

  //////////////////////////////////////////////////////////////////////////////
  // Static members

  template<typename T>
  const Point<T> Point<T>::Epsilon(::Constants<T>::Epsilon,
                                   ::Constants<T>::Epsilon,
                                   ::Constants<T>::Epsilon);

  template<typename T>
  const Point<T> Point<T>::Max(numeric_limits<T>::max(),
                               numeric_limits<T>::max(),
                               numeric_limits<T>::max());

  template<typename T>
  const Point<T> Point<T>::Min(-numeric_limits<T>::max(),
                               -numeric_limits<T>::max(),
                               -numeric_limits<T>::max());

  template<typename T>
  const Point<T> Point<T>::One(1, 1, 1);

  template<typename T>
  const Point<T> Point<T>::Zero(0, 0, 0);

  ///////////////////////////////////////////////////////////////////////////////
  // Friend functions

  template<typename T>
  inline Point<T> operator*(const T& s, const Point<T>& v)
  {
    // NOTE(cpg) - Win32 doesn't recognize this definition as a friend of Vector
#if defined(WIN32)
    return Point<T>(s*v[0], s*v[1], s*v[2]);
#else
    return Point<T>(s*v.e[0], s*v.e[1], s*v.e[2]);
#endif // defined(WIN32)
  }

  //////////////////////////////////////////////////////////////////////////////
  // Template specializations - Point<int>

  template<>
  inline Point<int> Point<int>::operator/(const int& s) const
  {
    return Point<int>(e[0]/s, e[1]/s, e[2]/s);
  }

  template<>
  inline Point<int> Point<int>::operator<<(uint i) const 
  {
    return Point<int>(e[0] << i, e[1] << i, e[2] << i);
  }

  template<>
  inline Point<int> Point<int>::operator>>(uint i) const 
  {
    return Point<int>(e[0] >> i, e[1] >> i, e[2] >> i);
  }

  template<>
  inline Point<int>& Point<int>::operator<<=(const uint i) 
  {
    e[0] <<= i;
    e[1] <<= i;
    e[2] <<= i;

    return *this;
  }

  template<>
  inline Point<int>& Point<int>::operator>>=(const uint i) 
  {
    e[0] >>= i;
    e[1] >>= i;
    e[2] >>= i;

    return *this;
  }

} // namespace Math

#endif // Math_Point_t
