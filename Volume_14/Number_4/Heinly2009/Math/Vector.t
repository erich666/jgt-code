
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

#ifndef Math_Vector_t
#define Math_Vector_t

#include <cmath>

#include <limits>
using std::numeric_limits;

#include <iostream>
using std::istream;
using std::ostream;

#include <typeinfo>

#include <Common/rtCore/RGB.t>
#include <Common/Utility/OutputCC.h>
#include <Common/Constants.h>
#include <Common/Types.h>

#include <Math/FixedPoint.h>

#define NORMALIZE_ADJUST 7

namespace Math
{

  ///////////////////////////////////////////////////////////////////////////////
  // Forward declarations

  template<typename T>
  class Point;

  template<typename T>
  class Vector;

  template<typename T>
  Vector<T> operator*(const T&, const Vector<T>&);

  template<typename T>
  Vector<T> operator/(const T&, const Vector<T>&);

  template<typename T>
  T Dot(const Vector<T>&, const Vector<T>&);

  template<typename T>
  Vector<T> Cross(const Vector<T>&, const Vector<T>&);

  ///////////////////////////////////////////////////////////////////////////////
  // Class template definition

  template<typename T>
  class Vector
  {
  public:

    /////////////////////////////////////////////////////////////////////////////
    // Static members

    static const Vector Epsilon;
    static const Vector One;
    static const Vector Max;
    static const Vector Min;
    static const Vector Zero;

    /////////////////////////////////////////////////////////////////////////////
    // Constructors

    inline Vector(const T& x, const T& y, const T& z)
    {
      e[0] = x;
      e[1] = y;
      e[2] = z;
    }

    inline Vector(const Vector& v)
    {
      e[0] = v.e[0];
      e[1] = v.e[1];
      e[2] = v.e[2];
    }

    inline Vector()
    {
      e[0] = 0;
      e[1] = 0;
      e[2] = 0;
    }

    // RGB<T> --> Vector<T>
    inline explicit Vector(const rtCore::RGB<T>& rgb)
    {
      e[0] = rgb.e[0];
      e[1] = rgb.e[1];
      e[2] = rgb.e[2];
    }

    // RGB<U> --> Vector<T>
    template<typename U>
    inline explicit Vector(const rtCore::RGB<U>& rgb)
    {
      e[0] = T(rgb.e[0]);
      e[1] = T(rgb.e[1]);
      e[2] = T(rgb.e[2]);
    }

    // Vector<U> --> Vector<T>
    template<typename U>
    inline explicit Vector(const Vector<U>& v)
    {
      e[0] = T(v.e[0]);
      e[1] = T(v.e[1]);
      e[2] = T(v.e[2]);
    }

    /////////////////////////////////////////////////////////////////////////////
    // Destructor

    inline ~Vector()
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

    inline bool operator==(const Vector& v) const
    {
      return (e[0] == v.e[0] && e[1] == v.e[1] && e[2] == v.e[2]);
    }

    inline Vector operator-() const
    {
      return Vector(-e[0], -e[1], -e[2]);
    }

    inline Vector operator<<(uint) const
    {
      // NOTE(cpg) - effectively a no-op for anything but Vector<int>; see
      //             template specializations below

      FatalError("Vector<" << typeid(T).name()
                 << ">::operator<<(uint) - should not be called!" << endl);

      // To quiet warnings...
      return *this;
    }

    inline Vector operator>>(uint) const
    {
      // NOTE(cpg) - effectively a no-op for anything but Vector<int>; see
      //             template specializations below

      FatalError("Vector<" << typeid(T).name()
                 << ">::operator<<(uint) - should not be called!" << endl);

      // To quiet warnings...
      return *this;
    }

    inline Vector& operator<<=(uint)
    {
      // NOTE(cpg) - effectively a no-op for anything but Vector<int>; see
      //             template specializations below

      FatalError("Vector<" << typeid(T).name()
                 << ">::operator<<(uint) - should not be called!" << endl);

      // To quiet warnings...
      return *this;
    }

    inline Vector& operator>>=(uint)
    {
      // NOTE(cpg) - effectively a no-op for anything but Vector<int>; see
      //             template specializations below

      FatalError("Vector<" << typeid(T).name()
                 << ">::operator<<(uint) - should not be called!" << endl);

      // To quiet warnings...
      return *this;
    }

    inline T length2() const
    {
      return (e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
    }

    inline T length() const
    {
      return Sqrt(length2());
    }

    inline T normalize()
    {
      const T len = length();
      const T inv = (len > 0 ? 1/len : 0);
      e[0] *= inv;
      e[1] *= inv;
      e[2] *= inv;

      return len;
    }

    inline Vector<T> normal() const
    {
      const T len = Sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
      const T inv = (len > 0 ? 1/len : 0);
      return Vector<T>(inv*e[0], inv*e[1], inv*e[2]);
    }

    inline Vector<T> orthogonal() const
    {
      // if |z| < |x| or |z| < |y|
      //   <x, y, z> = <y, -x, 0>
      // else
      //   <x, y, z> = <0, z, -y>
      const T& abs_z = Abs(e[2]);
      if (abs_z < Abs(e[0]) || abs_z < Abs(e[1]))
        return Vector<T>(e[1], -e[0], 0).normal();

      return Vector<T>(0, e[2], e[1]).normal();
    }

    ////////////////////////////////////////////////////////////////////////////
    // Vector<T>-Vector<T> operators

    inline Vector operator+(const Vector& v) const
    {
      return Vector(e[0] + v.e[0], e[1] + v.e[1], e[2] + v.e[2]);
    }

    inline Vector operator-(const Vector& v) const
    {
      return Vector(e[0] - v.e[0], e[1] - v.e[1], e[2] - v.e[2]);
    }

    inline Vector& operator+=(const Vector& v)
    {
      e[0] += v.e[0];
      e[1] += v.e[1];
      e[2] += v.e[2];

      return *this;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Vector<T>-T interoperability

    inline Vector operator*(const T& s) const
    {
      return Vector(s*e[0], s*e[1], s*e[2]);
    }

    inline Vector& operator*=(const T& s)
    {
      e[0] *= s;
      e[1] *= s;
      e[2] *= s;

      return *this;
    }

    /////////////////////////////////////////////////////////////////////////////
    // Friend classes, functions

    friend class Point<T>;
    
    template<typename U>
    friend class Vector;

    friend Vector<T> Math::operator* <> (const T&,         const Vector<T>&);
    friend Vector<T> Math::operator/ <> (const T&,         const Vector<T>&);
    friend T         Math::Dot       <> (const Vector<T>&, const Vector<T>&);
    friend Vector<T> Math::Cross     <> (const Vector<T>&, const Vector<T>&);

    ////////////////////////////////////////////////////////////////////////////
    // Stream I/O

    inline friend istream& operator>>(istream& in, Vector& v)
    {
      char junk;
      in >> junk;
      in >> v.e[0] >> v.e[1] >> v.e[2];
      in >> junk;

      return in;
    }

    inline friend ostream& operator<<(ostream& out, const Vector& v)
    {
      out << '(' << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2] << ')';
      return out;
    }

  private:
    T e[3];
  };

  //////////////////////////////////////////////////////////////////////////////
  // Static members

  template<typename T>
  const Vector<T> Vector<T>::Epsilon(::Constants<T>::Epsilon,
                                     ::Constants<T>::Epsilon,
                                     ::Constants<T>::Epsilon);

  template<typename T>
  const Vector<T> Vector<T>::Max(numeric_limits<T>::max(),
                                 numeric_limits<T>::max(),
                                 numeric_limits<T>::max());

  template<typename T>
  const Vector<T> Vector<T>::Min(-numeric_limits<T>::max(),
                                 -numeric_limits<T>::max(),
                                 -numeric_limits<T>::max());

  template<typename T>
  const Vector<T> Vector<T>::One(1, 1, 1);

  template<typename T>
  const Vector<T> Vector<T>::Zero(0, 0, 0);

  ///////////////////////////////////////////////////////////////////////////////
  // Friend functions

  template<typename T>
  inline Vector<T> operator*(const T& s, const Vector<T>& v)
  {
    // NOTE(cpg) - Win32 doesn't recognize this definition as a friend of Vector
#if defined(WIN32)
    return Vector<T>(s*v[0], s*v[1], s*v[2]);
#else
    return Vector<T>(s*v.e[0], s*v.e[1], s*v.e[2]);
#endif // defined(WIN32)
  }

  template<typename T>
  inline Vector<T> operator/(const T& s, const Vector<T>& v)
  {
    // NOTE(cpg) - Win32 doesn't recognize this definition as a friend of Vector
#if defined(WIN32)
    return Vector<T>(s/v[0], s/v[1], s/v[2]);
#else
    return Vector<T>(s/v.e[0], s/v.e[1], s/v.e[2]);
#endif // defined(WIN32)
  }

  template<typename T>
  inline T Dot(const Vector<T>& v0, const Vector<T>& v1)
  {
    // NOTE(cpg) - Win32 doesn't recognize this definition as a friend of Vector
#if defined(WIN32)
    return (v0[0]*v1[0] + v0[1]*v1[1] + v0[2]*v1[2]);  
#else
    return (v0.e[0]*v1.e[0] + v0.e[1]*v1.e[1] + v0.e[2]*v1.e[2]);  
#endif // defined(WIN32)
  }

  template<typename T>
  inline Vector<T> Cross(const Vector<T>& v0, const Vector<T>& v1)
  {
    // NOTE(cpg) - Win32 doesn't recognize this definition as a friend of Vector
#if defined(WIN32)
    return Vector<T>(v0[1]*v1[2] - v0[2]*v1[1],
                     v0[2]*v1[0] - v0[0]*v1[2],
                     v0[0]*v1[1] - v0[1]*v1[0]);
#else
    return Vector<T>(v0.e[1]*v1.e[2] - v0.e[2]*v1.e[1],
                     v0.e[2]*v1.e[0] - v0.e[0]*v1.e[2],
                     v0.e[0]*v1.e[1] - v0.e[1]*v1.e[0]);
#endif // defined(WIN32)
  }

  ///////////////////////////////////////////////////////////////////////////////
  // Template specializations - Vector<float>

  template<>
    template<>
  inline Vector<float>::Vector(const rtCore::RGB<int>& rgb)
  {
    e[0] = FIXED16_TO_FLOAT(rgb.e[0]);
    e[1] = FIXED16_TO_FLOAT(rgb.e[1]);
    e[2] = FIXED16_TO_FLOAT(rgb.e[2]);
  }

  template<>
    template<>
  inline Vector<float>::Vector(const Vector<int>& v)
  {
    e[0] = FIXED31_TO_FLOAT(v.e[0]);
    e[1] = FIXED31_TO_FLOAT(v.e[1]);
    e[2] = FIXED31_TO_FLOAT(v.e[2]);
  }

  ///////////////////////////////////////////////////////////////////////////////
  // Template specializations - Vector<int>

  template<>
    template<>
  inline Vector<int>::Vector(const Vector<float>& v)
  {
    e[0] = FLOAT_TO_FIXED31(v.e[0]);
    e[1] = FLOAT_TO_FIXED31(v.e[1]);
    e[2] = FLOAT_TO_FIXED31(v.e[2]);
  }

  template<>
  inline int Vector<int>::length2() const
  {
    return (MUL31(e[0], e[0]) + MUL31(e[1], e[1]) + MUL31(e[2], e[2]));
  }

  template<>
  inline int Vector<int>::normalize()
  {
    const int len = length() + NORMALIZE_ADJUST;
    e[0] = DIV31(e[0], len);
    e[1] = DIV31(e[1], len);
    e[2] = DIV31(e[2], len);

    return len;
  }

  template<>
  inline Vector<int> Vector<int>::normal() const
  {
    const int len = length() + NORMALIZE_ADJUST;
    return Vector<int>(DIV31(e[0], len), 
                       DIV31(e[1], len),
                       DIV31(e[2], len));
  }

  template<>
  inline Vector<int> Vector<int>::operator<<(const uint i) const 
  {
    return Vector<int>(e[0] << i, e[1] << i, e[2] << i);
  }

  template<>
  inline Vector<int> Vector<int>::operator>>(const uint i) const 
  {
    return Vector<int>(e[0] >> i, e[1] >> i, e[2] >> i);
  }

  template<>
  inline Vector<int>& Vector<int>::operator<<=(const uint i) 
  {
    e[0] <<= i;
    e[1] <<= i;
    e[2] <<= i;

    return *this;
  }

  template<>
  inline Vector<int>& Vector<int>::operator>>=(const uint i) 
  {
    e[0] >>= i;
    e[1] >>= i;
    e[2] >>= i;

    return *this;
  }

  template<>
  inline Vector<int> Vector<int>::operator*(const int& s) const
  {
    return Vector<int>(MUL31(s, e[0]), MUL31(s, e[1]), MUL31(s, e[2]));
  }

  template<>
  inline Vector<int>& Vector<int>::operator*=(const int& s)
  {
    e[0] = MUL31(s, e[0]);
    e[1] = MUL31(s, e[1]);
    e[2] = MUL31(s, e[2]);

    return *this;
  }

  template<>
  inline Vector<int> operator*(const int& s, const Vector<int>& v)
  {
    // NOTE(cpg) - WIN32 doesn't recognize this specialization as
    //             a friend of Vector
#if defined(WIN32)
    return Vector<int>(MUL31(s, v[0]),
                       MUL31(s, v[1]),
                       MUL31(s, v[2]));
#else
    return Vector<int>(MUL31(s, v.e[0]),
                       MUL31(s, v.e[1]),
                       MUL31(s, v.e[2]));
#endif // defined(WIN32)
  }

  template<>
  inline Vector<int> operator/(const int& s, const Vector<int>& v)
  {
    // NOTE(cpg) - WIN32 doesn't recognize this specialization as
    //             a friend of Vector
#if defined(WIN32)
    return Vector<int>(DIV31(s, v[0]),
                       DIV31(s, v[1]),
                       DIV31(s, v[2]));
#else
    return Vector<int>(DIV31(s, v.e[0]),
                       DIV31(s, v.e[1]),
                       DIV31(s, v.e[2]));
#endif // defined(WIN32)
  }

  template<>
  inline int Dot(const Vector<int>& v0, const Vector<int>& v1)
  {
    // NOTE(cpg) - WIN32 doesn't recognize this specialization as
    //             a friend of Vector
#if defined(WIN32)
    return (MUL31(v0[0], v1[0]) +
            MUL31(v0[1], v1[1]) +
            MUL31(v0[2], v1[2]));
#else
    return (MUL31(v0.e[0], v1.e[0]) +
            MUL31(v0.e[1], v1.e[1]) +
            MUL31(v0.e[2], v1.e[2]));
#endif // defined(WIN32)
  }

  template<>
  inline Vector<int> Cross(const Vector<int>& v0, const Vector<int>& v1)
  {
    // NOTE(cpg) - WIN32 doesn't recognize this specialization as
    //             a friend of Vector
#if defined(WIN32)
    return Vector<int>(MUL31(v0[1], v1[2]) - MUL31(v0[2], v1[1]),
                       MUL31(v0[2], v1[0]) - MUL31(v0[0], v1[2]),
                       MUL31(v0[0], v1[1]) - MUL31(v0[1], v1[0]));
#else
    return Vector<int>(MUL31(v0.e[1], v1.e[2]) - MUL31(v0.e[2], v1.e[1]),
                       MUL31(v0.e[2], v1.e[0]) - MUL31(v0.e[0], v1.e[2]),
                       MUL31(v0.e[0], v1.e[1]) - MUL31(v0.e[1], v1.e[0]));
#endif // defined(WIN32)
  }

} // namespace Math

#endif // Math_Vector_t
