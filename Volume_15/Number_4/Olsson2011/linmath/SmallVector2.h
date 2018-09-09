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
#ifndef _chag_SmallVector2_h
#define _chag_SmallVector2_h

#include <math.h>

#ifdef min
  #undef min
#endif
#ifdef max
  #undef max
#endif

// 
namespace chag
{

template <typename T>
class SmallVector2
{
public:
  T x;
  T y;

  // Indexation
  T& operator [] (const size_t i);
  const T& operator [] (const size_t i) const;
 
  const bool operator == (const SmallVector2& v) const;
  const bool operator != (const SmallVector2& v) const;
  
  // Negation
  const SmallVector2 operator - () const;

  // Assignement 
  const SmallVector2& operator = (const SmallVector2& v);
  
  const SmallVector2& operator += (const SmallVector2& v);
  
  const SmallVector2& operator -= (const SmallVector2& v);
  
  template <typename S>
  const SmallVector2& operator *= (const S& s);

  template <typename S>
  const SmallVector2& operator /= (const S& s);

  const SmallVector2 operator + (const SmallVector2& v) const;
  const SmallVector2 operator + (const T& s) const;
  const SmallVector2 operator - (const SmallVector2& v) const;
  const SmallVector2 operator - (const T& s) const;
  
  template <typename S>
  const SmallVector2 operator * (const S& s) const;

  const SmallVector2 operator * (const SmallVector2& v) const;

  template <typename S>
  const SmallVector2 operator / (const S& s) const;

  const SmallVector2 operator / (const SmallVector2& v) const;
};



template <typename T>
const T dot(const SmallVector2<T>& a, const SmallVector2<T>& b);



template <typename T>
const T length(const SmallVector2<T>& v);



template <typename T>
const T lengthSquared(const SmallVector2<T> &v);


template <typename T>
const SmallVector2<T> perpendicular(const SmallVector2<T> &v);


template <typename T>
const SmallVector2<T> normalize(const SmallVector2<T>& v);



template <typename T>
const SmallVector2<T> pow(const SmallVector2<T>& v, const T &exp);



template <typename S, typename T>
const SmallVector2<T> operator * (const S &s, const SmallVector2<T>& v);



template <typename T>
const SmallVector2<T> make_vector(const T& x, const T& y);



template <typename T, typename S>
const SmallVector2<T> make_vector(const SmallVector2<S> &v);



template <typename T>
const SmallVector2<T> make_vector2(const T *v);



template <typename T>
const SmallVector2<T> min(const SmallVector2<T> &a, const SmallVector2<T> &b);



template <typename T>
const SmallVector2<T> max(const SmallVector2<T> &a, const SmallVector2<T> &b);



template <typename T>
const SmallVector2<T> clamp(const SmallVector2<T> &value, const SmallVector2<T> &lowerInc, const SmallVector2<T> &upperInc);

} // namespace chag

#include "SmallVector2.inl"

#endif // _chag_SmallVector2_h
