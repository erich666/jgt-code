/****************************************************************************/
/* Copyright (c) 2011, Markus Billeter, Ola Olsson, Erik Sintorn and Ulf Assarsson
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
/****************************************************************************/
#ifndef _chag_SmallVector4_inl
#define _chag_SmallVector4_inl


#include "SmallVector4.h"
#include <utils/Math.h>
#include "SmallVector2.h"
#include <cmath>

// 
namespace chag
{


template <typename T>
inline T& SmallVector4<T>::operator [] (size_t i)
{  
  return *(&x + i); 
}



template <typename T>
inline const T& SmallVector4<T>::operator [] (size_t i) const
{  
  return *(&x + i); 
}



template <typename T>
inline bool SmallVector4<T>::operator == (const SmallVector4<T>& v) const
{ 
  return (v.x == x) && (v.y == y) && (v.z == z) && (v.w == w); 
}



template <typename T>
inline bool SmallVector4<T>::operator != (const SmallVector4<T>& v) const 
{ 
  return !(v == *this); 
}



template <typename T>
inline const SmallVector4<T> SmallVector4<T>::operator - () const
{ 
  return make_vector<T>(-x, -y, -z, -w); 
}




template <typename T>
inline const SmallVector4<T>& SmallVector4<T>::operator += (const SmallVector4<T>& v)
{
  x += v.x; 
  y += v.y;
  z += v.z;
  w += v.w;
  return *this;
}



template <typename T>
inline const SmallVector4<T>& SmallVector4<T>::operator -= (const SmallVector4<T>& v)
{
  x -= v.x; 
  y -= v.y;
  z -= v.z;
  w -= v.w;
  return *this;
}



template <typename T>
template <typename S>
inline const SmallVector4<T>& SmallVector4<T>::operator *= (const S &s)
{
  x *= s; 
  y *= s;
  z *= s;
  w *= s;
  return *this;
}



template <typename T>
template <typename S>
inline const SmallVector4<T>& SmallVector4<T>::operator += (const S &s)
{
  x += s; 
  y += s;
  z += s;
  w += s;
  return *this;
}



template <typename T>
template <typename S>
inline const SmallVector4<T>& SmallVector4<T>::operator /= (const S& s)
{
  x /= s; 
  y /= s;
  z /= s;
  w /= s;
  return *this;
}



template <typename T>
inline const SmallVector4<T> SmallVector4<T>::operator + (const SmallVector4<T>& v) const 
{
  return make_vector<T>(x + v.x, y + v.y, z + v.z, w + v.w);
}



template <typename T>
inline const SmallVector4<T> SmallVector4<T>::operator - (const SmallVector4<T>& v) const 
{
  return make_vector<T>(x - v.x, y - v.y, z - v.z, w - v.w);
}



template <typename T>
template <typename S>
inline const SmallVector4<T> SmallVector4<T>::operator * (const S& s) const
{
  return make_vector<T>(x * s, y * s, z * s, w * s);
}



template <typename T>
template <typename S>
inline const SmallVector4<T> SmallVector4<T>::operator / (const S& s) const
{
  return (*this) * (1.0f / s);
}



template <typename T>
inline const T dot(const SmallVector4<T>& a, const SmallVector4<T>& b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}



template <typename T>
inline const T dot3(const SmallVector4<T>& a, const SmallVector4<T>& b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}



template <typename T>
inline const T dot3(const SmallVector4<T>& a, const SmallVector3<T>& b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}



template <typename T>
inline const T dot3(const SmallVector3<T>& a, const SmallVector3<T>& b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}



template <typename T>
inline const T length(const SmallVector4<T> &v)
{
  return (T)sqrt(dot(v,v));
}



template <typename T>
inline const T lengthSquared(const SmallVector4<T> &v)
{
  return dot(v,v);
}



template <typename T>
inline const T length3(const SmallVector4<T> &v)
{
  return (T)sqrt(dot3(v,v));
}


template <typename T>
inline const SmallVector4<T> normalize(const SmallVector4<T>& v)
{
  return v / length(v);
}



template <typename T>
inline const SmallVector4<T> pow(const SmallVector4<T>& v, const T &exp)
{
  return make_vector(std::pow(v.x, exp), std::pow(v.y, exp), std::pow(v.z, exp), pow(v.w, exp));
}



template <typename S, typename T>
inline const SmallVector4<T> operator * (const S &s, const SmallVector4<T>& v)
{
  return v * s;
}



template <typename T>
inline const SmallVector4<T> make_vector(const T& x, const T& y, const T& z, const T& w)
{
  SmallVector4<T> r;
  r.x = x;
  r.y = y;
  r.z = z;
  r.w = w;
  return r;
};



template <typename T>
inline const SmallVector4<T> make_vector4(const T *v)
{
  return make_vector(v[0], v[1], v[2], v[3]);
};


template <typename T>
inline const SmallVector4<T> make_vector4(const SmallVector3<T> &xyz, const T &w)
{
  return make_vector(xyz.x, xyz.y, xyz.z, w);
};


template <typename T>
inline const SmallVector4<T> make_vector4(const SmallVector2<T> &xy, const SmallVector2<T> &zw)
{
  return make_vector(xy.x, xy.y, zw.x, zw.y);
};


template <typename T>
const SmallVector4<T> min(const SmallVector4<T> &a, const SmallVector4<T> &b)
{
  return make_vector(chag::min(a.x, b.x), chag::min(a.y, b.y), chag::min(a.z, b.z), chag::min(a.w, b.w));
}



template <typename T>
const SmallVector4<T> max(const SmallVector4<T> &a, const SmallVector4<T> &b)
{
  return make_vector(chag::max(a.x, b.x), chag::max(a.y, b.y), chag::max(a.z, b.z), chag::max(a.w, b.w));
}



template <typename T>
const SmallVector3<T> cross3(const SmallVector4<T> &a, const SmallVector4<T> &b)
{
  return make_vector(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}



template <typename T>
const SmallVector4<T> cross(const SmallVector4<T> &a, const SmallVector4<T> &b, const SmallVector4<T> &c)
{
  return make_vector(c.w*b.y*a.z - b.w*c.y*a.z - c.w*a.y*b.z + a.w*c.y*b.z + b.w*a.y*c.z - a.w*b.y*c.z, 
					-c.w*b.x*a.z + b.w*c.x*a.z + c.w*a.x*b.z - a.w*c.x*b.z - b.w*a.x*c.z + a.w*b.x*c.z, 
					 c.w*b.x*a.y - b.w*c.x*a.y - c.w*a.x*b.y + a.w*c.x*b.y + b.w*a.x*c.y - a.w*b.x*c.y, 
					-c.x*b.y*a.z + b.x*c.y*a.z + c.x*a.y*b.z - a.x*c.y*b.z - b.x*a.y*c.z + a.x*b.y*c.z);
}



#if 0
template <typename T>
const SmallVector3<T> cross(const SmallVector4<T> &a, const SmallVector4<T> &b)
{
  return make_vector(y*v.z - z*v.y, z*v.x - x*v.z, x*v.y - y*v.x);
}
#endif


} // namespace chag


#endif // _chag_SmallVector4_inl
