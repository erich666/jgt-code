
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

#ifndef Common_rtCore_RGB_t
#define Common_rtCore_RGB_t

#include <cmath>

#include <limits>
using std::numeric_limits;

#include <iosfwd>
using std::istream;
using std::ostream;

#include <typeinfo>

#include <Common/Utility/OutputCC.h>
#include <Common/Types.h>

#include <Math/FixedPoint.h>

namespace Math
{

  template<typename T>
  class Vector;

} // namespace rtCore

namespace rtCore
{

  ///////////////////////////////////////////////////////////////////////////////
  // Forward declarations

  template<typename T>
  class RGB;

  template<typename T>
  RGB<T> operator*(const T&, const RGB<T>&);

  ///////////////////////////////////////////////////////////////////////////////
  // Class template definition

  template<typename T>
  class RGB
  {
  public:

    /////////////////////////////////////////////////////////////////////////////
    // Static members

    static const RGB One;
    static const RGB Zero;

    /////////////////////////////////////////////////////////////////////////////
    // Constructors

    inline RGB(const T& r, const T& g, const T& b)
    {
      e[0] = r;
      e[1] = g;
      e[2] = b;
    }

    inline RGB()
    {
      e[0] = 0;
      e[1] = 0;
      e[2] = 0;
    }

    template<typename U>
    inline RGB(const RGB<U>& rgb)
    {
      e[0] = T(rgb.e[0]);
      e[1] = T(rgb.e[1]);
      e[2] = T(rgb.e[2]);
    }

    /////////////////////////////////////////////////////////////////////////////
    // Destructor

    inline ~RGB()
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

    inline bool operator==(const RGB<T>& rgb) const
    {
      return (e[0] == rgb.e[0] && e[1] == rgb.e[1] && e[2] == rgb.e[2]);
    }

    inline bool operator!=(const RGB<T>& rgb) const
    {
      return (e[0] != rgb.e[0] || e[1] != rgb.e[1] || e[2] != rgb.e[2]);
    }

    // XXX(cpg) - include shift operators only in RGB<int>?  (would require
    //            a specialized RGB<int> class definition...)

    inline RGB operator<<(uint i) const
    {
      // NOTE(cpg) - effectively a no-op for anything but RGB<int>; see
      //             template specializations below

      FatalError("RGB<" << typeid(T).name()
                 << ">::operator<<(uint) - should not be called!" << endl);

      // To quiet warnings...
      return *this;
    }

    inline RGB operator>>(uint i) const
    {
      // NOTE(cpg) - effectively a no-op for anything but RGB<int>; see
      //             template specializations below

      FatalError("RGB<" << typeid(T).name()
                 << ">::operator<<(uint) - should not be called!" << endl);

      // To quiet warnings...
      return *this;
    }

    ////////////////////////////////////////////////////////////////////////////
    // RGB<T>-RGB<T> operators

    inline RGB operator+(const RGB& rgb) const
    {
      return RGB(e[0]+rgb.e[0], e[1]+rgb.e[1], e[2]+rgb.e[2]);
    }

    inline RGB operator-(const RGB& rgb) const
    {
      return RGB(e[0]-rgb.e[0], e[1]-rgb.e[1], e[2]-rgb.e[2]);
    }

    inline RGB operator*(const RGB& rgb) const
    {
      return RGB(e[0]*rgb.e[0], e[1]*rgb.e[1], e[2]*rgb.e[2]);
    }

    inline RGB& operator+=(const RGB& rgb)
    {
      e[0] += rgb.e[0];
      e[1] += rgb.e[1];
      e[2] += rgb.e[2];

      return *this;
    }

    inline RGB& operator*=(const RGB& rgb)
    {
      e[0] *= rgb.e[0];
      e[1] *= rgb.e[1];
      e[2] *= rgb.e[2];

      return *this;
    }

    inline void clamp(const RGB& rgb0, const RGB& rgb1)
    {
      e[0] = (e[0] < rgb0.e[0] ? rgb0.e[0] : (e[0] > rgb1.e[0] ? rgb1.e[0] : e[0]));
      e[1] = (e[1] < rgb0.e[1] ? rgb0.e[1] : (e[1] > rgb1.e[1] ? rgb1.e[1] : e[1]));
      e[2] = (e[2] < rgb0.e[2] ? rgb0.e[2] : (e[2] > rgb1.e[2] ? rgb1.e[2] : e[2]));
    }

    ////////////////////////////////////////////////////////////////////////////
    // RGB<T>-T interoperability

    inline RGB operator*(const T& s) const
    {
      return RGB(s*e[0], s*e[1], s*e[2]);
    }

    inline RGB operator/(const T& s) const
    {
      const T inv = 1/s;
      return RGB(e[0]*inv, e[1]*inv, e[2]*inv);
    }

    inline RGB operator/=(const T& s)
    {
      *this = *this / s;
      return *this;
    }

    inline void clamp(const T& s0, const T& s1)
    {
      e[0] = (e[0] < s0 ? s0 : (e[0] > s1 ? s1 : e[0]));
      e[1] = (e[1] < s0 ? s0 : (e[1] > s1 ? s1 : e[1]));
      e[2] = (e[2] < s0 ? s0 : (e[2] > s1 ? s1 : e[2]));
    }

    ////////////////////////////////////////////////////////////////////////////
    // Friend classes, functions

    template<typename U>
    friend class RGB;

    friend class Math::Vector<T>;

    template<typename U>
    friend class Math::Vector;

    friend RGB<T> rtCore::operator* <> (const T&, const RGB<T>&);

    ////////////////////////////////////////////////////////////////////////////
    // Stream I/O

    inline friend istream& operator>>(istream& in, RGB& rgb)
    {
      char junk;
      in >> junk;
      in >> rgb.e[0] >> rgb.e[1] >> rgb.e[2];
      in >> junk;

      return in;
    }

    inline friend ostream& operator<<(ostream& out, const RGB& rgb)
    {
      out << '(' << rgb.e[0] << ' ' << rgb.e[1] << ' ' << rgb.e[2] << ')';
      return out;
    }

  private:
    T e[3];
  };

  //////////////////////////////////////////////////////////////////////////////
  // Static members

  // NOTE(cpg) - specializations for float, int moved to  RGB.cc to resolve
  //             linker errors (multiply defined symbols)
  //
  // RGB<float>
  // template<>
  // const RGB<float> RGB<float>::One(1.f, 1.f, 1.f);
  // 
  // RGB<int>
  // template<>
  // const RGB<int> RGB<int>::One(1 << 16, 1 << 16, 1 << 16);

  template<typename T>
  const RGB<T> RGB<T>::Zero(0, 0, 0);

  //////////////////////////////////////////////////////////////////////////////
  // Friend function definitions

  template<typename T>
  inline RGB<T> operator*(const T& s, const RGB<T>& rgb)
  {
    // NOTE(cpg) - WIN32 doesn't recognize this definition as a friend of RGB
#if defined(WIN32)
    return RGB<T>(s*rgb[0], s*rgb[1], s*rgb[2]);
#else
    return RGB<T>(s*rgb.e[0], s*rgb.e[1], s*rgb.e[2]);
#endif // defined(WIN32)
  }

  //////////////////////////////////////////////////////////////////////////////
  // Template specializations - RGB<int>

  // RGB<float> --> RGB<int>
  template<>
    template<>
  inline RGB<int>::RGB(const RGB<float>& rgb)
  {
    e[0] = FLOAT_TO_FIXED16(rgb.e[0]);
    e[1] = FLOAT_TO_FIXED16(rgb.e[1]);
    e[2] = FLOAT_TO_FIXED16(rgb.e[2]);
  }

  template<>
  inline RGB<int> RGB<int>::operator<<(uint i) const
  {
    return RGB<int>(e[0] << i, e[1] << i, e[2] << i);
  }

  template<>
  inline RGB<int> RGB<int>::operator>>(uint i) const
  {
    return RGB<int>(e[0] >> i, e[1] >> i, e[2] >> i);
  }

  template<>
  inline RGB<int> RGB<int>::operator*(const RGB<int>& rgb) const
  {
    return RGB<int>(MUL16(e[0], rgb.e[0]),
                    MUL16(e[1], rgb.e[1]),
                    MUL16(e[2], rgb.e[2]));
  }

  template<>
  inline RGB<int> RGB<int>::operator*(const int& s) const
  {
    return RGB<int>(MUL16(s, e[0]), MUL16(s, e[1]), MUL16(s, e[2]));
  }

  template<>
  inline RGB<int> RGB<int>::operator/(const int& s) const
  {
    return RGB<int>(DIV16(e[0],s), DIV16(e[1],s), DIV16(e[2],s));
  }

  template<>
  inline RGB<int> RGB<int>::operator/=(const int& s)
  {
    *this = *this / s;
    return *this;
  }

  template<>
  inline RGB<int> operator*(const int& s, const RGB<int>& rgb)
  {
    // NOTE(cpg) - WIN32 doesn't recognize this specialization as a friend of RGB
#if defined(WIN32)
    return RGB<int>(MUL16(s, rgb[0]),
                    MUL16(s, rgb[1]),
                    MUL16(s, rgb[2]));
#else
    return RGB<int>(MUL16(s, rgb.e[0]),
                    MUL16(s, rgb.e[1]),
                    MUL16(s, rgb.e[2]));
#endif // defined(WIN32)
  }

} // namespace rtCore

namespace rtCoreF
{

  //////////////////////////////////////////////////////////////////////////////
  // Type definitions

  typedef rtCore::RGB<float> RGB;

} // namespace rtCoreF

namespace rtCoreI
{

  //////////////////////////////////////////////////////////////////////////////
  // Type definitions

  typedef rtCore::RGB<int> RGB;

} // namespace rtCoreI

#endif // Common_rtCore_RGB_t
