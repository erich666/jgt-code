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
#ifndef _chag_float4x4_h
#define _chag_float4x4_h

#include "Common.h"
#include "float4.h" 

namespace chag
{

/**
 * The float4x4 is designed to be POD, which depends on float4 being so, this means it can be statically initialized etc
 * but one can use the make_matrix functions to construct matrices more conventiently.
 *
 * layout (note that these are _columns_!):
 * c1 = { BXx, BXy, BXz, 0 }
 * c2 = { BYx, BYy, BYz, 0 }
 * c3 = { BZx, BZy, BZz, 0 }
 * c4 = { Tx,  Ty,  Tz,  1 }
 */
class float4x4
{
public: 
  // The four columns of the matrix
  float4 c1;
  float4 c2;
  float4 c3;
  float4 c4;

  /**
   */
  float4& operator [] (const size_t i);
  const float4& operator [] (const size_t i) const;

  /**
   * Index into matrix, valid ranges are [1,4], [1,4]
   */
  const float &operator()(const size_t row, const size_t column) const { return *(&(&c1 + column - 1)->x + row - 1); } 
  float &operator()(const size_t row, const size_t column) { return *(&(&c1 + column - 1)->x + row - 1); }

  /**
   */
  bool operator == (const float4x4& m) const;
  /**
   */
  bool operator != (const float4x4& m) const; 
  /**
   */
  const float4 row(int i) const;
  /**
   * Component wise addition.
   */
  const float4x4 operator + (const float4x4& m);
  /**
   * Component wise scale.
   */
  const float4x4 operator * (const float& s) const;
  /**
   * Multiplication by column vector.
   */
  const float4 operator * (const float4& v) const;
  /**
   */
  const float4x4 operator * (const float4x4& m) const;
  /**
   */
  const float3 &getTranslation() const { return *reinterpret_cast<const float3 *>(&c4); }
};


/**
 */
template <>
const float4x4 make_identity<float4x4>();

/**
 */
template <>
const float4x4 make_matrix<float4x4>(const float *e);

/**
 * The matrix is indexed on the form Mrow,column.
 */
const float4x4 make_matrix(float m11, float m12, float m13, float m14, 
                           float m21, float m22, float m23, float m24, 
                           float m31, float m32, float m33, float m34,
                           float m41, float m42, float m43, float m44);
   
/**
 */
const float4x4 make_matrix(const float4& c1, const float4& c2, const float4& c3, const float4& c4);

/**
 */
const float4x4 make_matrix(const float3x3 &r, const float3 &pos);

/**
 */
const float4x4 make_translation(const float3 &pos);

/**
 */
const float4x4 make_matrix_from_zAxis(const float3 &pos, const float3 &zAxis, const float3 &yAxis);
const float4x4 make_matrix_from_yAxis(const float3 &pos, const float3 &yAxis, const float3 &zAxis);

/**
 */
float determinant(const float4x4 &m);

/**
 */
const float4x4 transpose(const float4x4 &m);

/**
 */
const float4 cramers(const float4x4 &m, const float4& u);

/**
 */
const float4x4 inverse(const float4x4 &m); 

/**
 */
const float3 transformPoint(const float4x4 &m, const float3 &p);

/**
 */
const float3 transformDirection(const float4x4 &m, const float3 &d);



} // namespace chag

#endif // _chag_float4x4_h
