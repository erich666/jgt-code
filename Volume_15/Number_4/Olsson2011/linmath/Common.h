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
#ifndef _chag_linmath_Common_h
#define _chag_linmath_Common_h


namespace chag
{

/**
 * Specialized for the matrix types to contruct identity matrix.
 */
template<typename T> 
const T make_identity();
template <typename T>
const T make_rotation_x(float angle);
template <typename T>
const T make_rotation_y(float angle);
template <typename T>
const T make_rotation_z(float angle);
template <typename T, typename U>
const T make_rotation(const U& axis, float angle);
template <typename T>
const T make_matrix(const float *e);
template <typename T, typename U>
const T make_scale(const U& scale);


class float3x3;
class float4x4;

template <typename T>
class SmallVector3;
typedef SmallVector3<float> float3;
typedef SmallVector3<int> int3;

template <typename T>
class SmallVector2;
typedef SmallVector2<float> float2;
typedef SmallVector2<int> int2;

template <typename T>
class SmallVector4;
typedef SmallVector4<float> float4;
typedef SmallVector4<int> int4;



template <typename S, typename T>
inline const S lerp(const S &a, const S &b, const T &t)
{
  return a + (b - a) * t;
}

template <typename T>
inline void swap(SmallVector2<T> &a, SmallVector2<T> &b)
{
  SmallVector2<T> tmp(a);
  a = b;
  b = tmp;
}

template <typename T>
inline void swap(SmallVector3<T> &a, SmallVector3<T> &b)
{
  SmallVector3<T> tmp(a);
  a = b;
  b = tmp;
}

template <typename T>
inline void swap(SmallVector4<T> &a, SmallVector4<T> &b)
{
  SmallVector4<T> tmp(a);
  a = b;
  b = tmp;
}


} // namespace chag

#endif // _chag_linmath_Common_h
