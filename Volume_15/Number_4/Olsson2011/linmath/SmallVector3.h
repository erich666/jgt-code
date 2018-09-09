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
#ifndef _chag_SmallVector3_h
#define _chag_SmallVector3_h

#include <math.h>
#include "SmallVector2.h"


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
class SmallVector3
{
public:
  T x;
  T y;
  T z;

  // Indexation
  T& operator [] (const size_t i);
  const T& operator [] (const size_t i) const;
 
  bool operator == (const SmallVector3& v) const;
  bool operator != (const SmallVector3& v) const;
  
  // Negation
  const SmallVector3 operator - () const;

  // Assignement 
  const SmallVector3& operator = (const SmallVector3& v);
  const SmallVector3& operator += (const SmallVector3& v);
  const SmallVector3& operator -= (const SmallVector3& v);
  const SmallVector3& operator *= (const SmallVector3& v);
  
  template <typename S>
  const SmallVector3& operator *= (const S& s);

  template <typename S>
  const SmallVector3& operator /= (const S& s);

  const SmallVector3 operator + (const SmallVector3& v) const;
  const SmallVector3 operator + (const T &s) const;
  const SmallVector3 operator - (const SmallVector3& v) const;
  const SmallVector3 operator - (const T &s) const;
  
  template <typename S>
  const SmallVector3 operator * (const S& s) const;
  const SmallVector3 operator * (const SmallVector3& v) const;

  template <typename S>
  const SmallVector3 operator / (const S& s) const;
  const SmallVector3 operator / (const SmallVector3& v) const;
};



template <typename T>
const T dot(const SmallVector3<T>& a, const SmallVector3<T>& b);



template <typename T>
const T length(const SmallVector3<T>& v);



template <typename T>
const T lengthSquared(const SmallVector3<T> &v);



template <typename T>
const SmallVector3<T> normalize(const SmallVector3<T>& v);



template <typename S, typename T>
const SmallVector3<T> operator * (const S &s, const SmallVector3<T>& v);



template <typename T>
const SmallVector3<T> make_vector(const T& x, const T& y, const T& z);



template <typename T>
const SmallVector3<T> make_vector3(const T *v);


inline const SmallVector3<float> make_vector3(const float &v) { return make_vector(v, v, v); }



template <typename T>
const SmallVector3<T> make_vector3(const SmallVector2<T> &xy, const T &z);




template <typename T>
const SmallVector3<T> min(const SmallVector3<T> &a, const SmallVector3<T> &b);



template <typename T>
const SmallVector3<T> max(const SmallVector3<T> &a, const SmallVector3<T> &b);



template <typename T>
const SmallVector3<T> cross(const SmallVector3<T> &a, const SmallVector3<T> &b);



template <typename T>
const SmallVector3<T> perpendicular(const SmallVector3<T> &v);



template <typename T>
const SmallVector3<T> normalize(const SmallVector3<T>& v);



template <typename T>
const SmallVector3<T> pow(const SmallVector3<T>& v, const T &exp);




} // namespace chag

#include "SmallVector3.inl"

#endif // _chag_SmallVector3_h
