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
#ifndef _chag_float3x3_h
#define _chag_float3x3_h

#include "Common.h"
#include "float3.h" 

namespace chag
{

class float4x4;

class float3x3
{
public: 
  // The three columns of the matrix
  float3 c1;
  float3 c2;
  float3 c3;

  float3& operator [] (const size_t i);
  const float3& operator [] (const size_t i) const;

  const float &operator()(const size_t row, const size_t column) const { return *(&(&c1 + column - 1)->x + row - 1); } 
  float &operator()(const size_t row, const size_t column) { return *(&(&c1 + column - 1)->x + row - 1); }

  const bool operator == (const float3x3& m) const;
  const bool operator != (const float3x3& m) const;

  // Assignement 
  const float3x3 operator - (const float3x3 M);
  const float3 operator * (const float3& v) const;
  const float3x3 operator * (const float f) const;

  const float3 row(size_t i) const;
  const float3x3 operator * (const float3x3& M) const;
};


// Find the LU decomposition of the matrix
const void lu(const float3x3 &a, float3x3 &l, float3x3 &u);
const float3 lr(float3x3 &a);
void egenvektor(float3x3 &m, float3 *v1, float3 *v2, float3 *v3);


const float3x3 transpose(const float3x3 &m);


const float determinant(const float3x3 &m);

const float3 cramers(const float3x3 &m, const float3& u);


/**
 */
template <>
const float3x3 make_identity<float3x3>();

template <>
const float3x3 make_matrix<float3x3>(const float *e);

const float3x3 make_matrix3x3(const float4x4 &m);

const float3x3 make_matrix(const float3 &c1, const float3 &c2, const float3 &c3);

/**
 * The matrix is indexed on the form Mrow,column.
 */
const float3x3 make_matrix(float m11, float m12, float m13, float m21, float m22, float m23, float m31, float m32, float m33);

const float3x3 operator * (const float& f, const float3x3& M);

} // namespace chag

#endif // _chag_float3x3_h
