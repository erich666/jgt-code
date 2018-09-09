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
#ifndef _chag_SmallVector2_inl
#define _chag_SmallVector2_inl


#include "SmallVector2.h"
#include <utils/Math.h>
#include <cmath>

// 
namespace chag
{


template <typename T>
inline T& SmallVector2<T>::operator [] (size_t i)
{  
  return *(&x + i); 
}



template <typename T>
inline const T& SmallVector2<T>::operator [] (size_t i) const
{  
  return *(&x + i); 
}



template <typename T>
inline const bool SmallVector2<T>::operator == (const SmallVector2<T>& v) const
{ 
  return (v.x == x) && (v.y == y); 
}



template <typename T>
inline const bool SmallVector2<T>::operator != (const SmallVector2<T>& v) const 
{ 
  return !(v == *this); 
}



template <typename T>
inline const SmallVector2<T> SmallVector2<T>::operator - () const
{ 
  return make_vector<T>(-x, -y); 
}



template <typename T>
inline const SmallVector2<T>& SmallVector2<T>::operator = (const SmallVector2<T>& v)
{
  x = v.x; 
  y = v.y;
  return *this;
}



template <typename T>
inline const SmallVector2<T>& SmallVector2<T>::operator += (const SmallVector2<T>& v)
{
  x += v.x; 
  y += v.y;
  return *this;
}



template <typename T>
inline const SmallVector2<T>& SmallVector2<T>::operator -= (const SmallVector2<T>& v)
{
  x -= v.x; 
  y -= v.y;
  return *this;
}



template <typename T>
template <typename S>
inline const SmallVector2<T>& SmallVector2<T>::operator *= (const S &s)
{
  x *= s; 
  y *= s;
  return *this;
}



template <typename T>
template <typename S>
inline const SmallVector2<T>& SmallVector2<T>::operator /= (const S& s)
{
  x /= s; 
  y /= s;
  return *this;
}



template <typename T>
inline const SmallVector2<T> SmallVector2<T>::operator + (const SmallVector2<T>& v) const 
{
  return make_vector<T>(x + v.x, y + v.y);
}




template <typename T>
inline const SmallVector2<T> SmallVector2<T>::operator + (const T& s) const 
{
  return make_vector<T>(x + s, y + s);
}



template <typename T>
inline const SmallVector2<T> SmallVector2<T>::operator - (const SmallVector2<T>& v) const 
{
  return make_vector<T>(x - v.x, y - v.y);
}


template <typename T>
inline const SmallVector2<T> SmallVector2<T>::operator - (const T& s) const 
{
  return make_vector<T>(x - s, y - s);
}



template <typename T>
template <typename S>
inline const SmallVector2<T> SmallVector2<T>::operator * (const S& s) const
{
  return make_vector<T>(x * s, y * s);
}



template <typename T>
inline const SmallVector2<T> SmallVector2<T>::operator * (const SmallVector2<T>& v) const
{
  return make_vector<T>(x * v.x, y * v.y);
}



template <typename T>
template <typename S>
inline const SmallVector2<T> SmallVector2<T>::operator / (const S& s) const
{
  T r = 1.0f / s;
  return make_vector<T>(x * r, y * r);
}

template <typename T>
inline const SmallVector2<T> SmallVector2<T>::operator / (const SmallVector2<T>& v) const 
{
  return make_vector<T>(x / v.x, y / v.y);
}

template <typename T>
inline const T dot(const SmallVector2<T>& a, const SmallVector2<T>& b)
{
  return a.x * b.x + a.y * b.y;
}



template <typename T>
inline const T length(const SmallVector2<T> &v)
{
  return (T)sqrt(dot(v,v));
}



template <typename T>
inline const T lengthSquared(const SmallVector2<T> &v)
{
  return dot(v,v);
}



template <typename T>
const SmallVector2<T> perpendicular(const SmallVector2<T> &v)
{
  return make_vector(-v.y, v.x);
}



template <typename T>
inline const SmallVector2<T> normalize(const SmallVector2<T>& v)
{
  return v / length(v);
}



template <typename T>
inline const SmallVector2<T> pow(const SmallVector2<T>& v, const T &exp)
{
  return make_vector(std::pow(v.x, exp), std::pow(v.y, exp));
}



template <typename S, typename T>
inline const SmallVector2<T> operator * (const S &s, const SmallVector2<T>& v)
{
  return v * s;
}



template <typename T>
inline const SmallVector2<T> make_vector(const T& x, const T& y)
{
  SmallVector2<T> r = { x, y };
  return r;
};



template <typename T, typename S>
inline const SmallVector2<T> make_vector(const SmallVector2<S> &v)
{
  SmallVector2<T> r = { T(v.x), T(v.y) };
  return r;
}



template <typename T>
inline const SmallVector2<T> make_vector2(const T *v)
{
  SmallVector2<T> r = { v[0], v[1] };
  return r;
};



template <typename T>
const SmallVector2<T> min(const SmallVector2<T> &a, const SmallVector2<T> &b)
{
  return make_vector(chag::min(a.x, b.x), chag::min(a.y, b.y));
}



template <typename T>
const SmallVector2<T> max(const SmallVector2<T> &a, const SmallVector2<T> &b)
{
  return make_vector(chag::max(a.x, b.x), chag::max(a.y, b.y));
}

template <typename T>
const SmallVector2<T> clamp(const SmallVector2<T> &value, const SmallVector2<T> &lowerInc, const SmallVector2<T> &upperInc)
{
  return min(max(value, lowerInc), upperInc);
}


} // namespace chag


#endif // _chag_SmallVector2_inl
