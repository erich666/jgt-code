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
#include "float4x4.h"
#include "float3x3.h"

#include <cassert>
#include <memory.h>

namespace chag
{



float4& float4x4::operator [] (const size_t i)
{ 
  assert( i < 4 );
  return *(&c1 + i);
}



const float4& float4x4::operator [] (const size_t i) const
{ 
	assert( i < 4 );
  return *(&c1 + i);
}



bool float4x4::operator == (const float4x4& m) const
{ 
  return (m.c1 == c1) && (m.c2 == c2) && (m.c3 == c3) && (m.c4 == c4); 
}



bool float4x4::operator != (const float4x4& m) const 
{ 
  return !(m == *this); 
}



const float4 float4x4::row(int i) const
{
  assert( i < 4 );

  switch(i)
  {
  case 0: 
    return make_vector(c1[0], c2[0], c3[0], c4[0]);
  case 1: 
    return make_vector(c1[1], c2[1], c3[1], c4[1]);
  case 2: 
    return make_vector(c1[2], c2[2], c3[2], c4[2]);
  /*case 3*/
  default: 
    return make_vector(c1[3], c2[3], c3[3], c4[3]);
  }
}



const float4x4 float4x4::operator + (const float4x4& m)
{
  return make_matrix(c1 + m.c1, 
		  c2 + m.c2, 
		  c3 + m.c3, 
		  c4 + m.c4);
}



const float4x4 float4x4::operator * (const float& s) const
{
  return make_matrix(s * c1, s * c2, s * c3, s * c4);
}



const float4 float4x4::operator * (const float4& v) const
{
  return make_vector(c1[0] * v.x + c2[0] * v.y + c3[0] * v.z + c4[0] * v.w, 
				c1[1] * v.x + c2[1] * v.y + c3[1] * v.z + c4[1] * v.w, 
				c1[2] * v.x + c2[2] * v.y + c3[2] * v.z + c4[2] * v.w, 
				c1[3] * v.x + c2[3] * v.y + c3[3] * v.z + c4[3] * v.w);
}



const float4x4 float4x4::operator * (const float4x4& b) const
{
#if 1
  // a textbook implementation...
  float4x4 c;
  const float4x4 &a = *this;

  for (int i = 1; i <= 4; ++i)
  {
      for (int k = 1; k <= 4; ++k)
      {
        c(i, k) = a(i, 1) * b(1, k) 
                + a(i, 2) * b(2, k) 
                + a(i, 3) * b(3, k) 
                + a(i, 4) * b(4, k);
      }
  }

  return c;
#else
  // assuredly slow, since row() does a lot of logic...
  return make_matrix(dot(row(0), m.c1), dot(row(0), m.c2), dot(row(0), m.c3), dot(row(0), m.c4),  
		  dot(row(1), m.c1), dot(row(1), m.c2), dot(row(1), m.c3), dot(row(1), m.c4), 
		  dot(row(2), m.c1), dot(row(2), m.c2), dot(row(2), m.c3), dot(row(2), m.c4), 
		  dot(row(3), m.c1), dot(row(3), m.c2), dot(row(3), m.c3), dot(row(3), m.c4));
#endif
}



template <>
const float4x4 make_identity<float4x4>()
{
  float4x4 m = 
  { 
    { 1.0f, 0.0f, 0.0f, 0.0f },
    { 0.0f, 1.0f, 0.0f, 0.0f },
    { 0.0f, 0.0f, 1.0f, 0.0f },
    { 0.0f, 0.0f, 0.0f, 1.0f }
  };
  return m;
}


template <>
const float4x4 make_matrix<float4x4>(const float *e)
{
  float4x4 m;
  memcpy(&m, e, sizeof(m));
  return m;
}



const float4x4 make_matrix(
  float m11, float m12, float m13, float m14, 
  float m21, float m22, float m23, float m24, 
  float m31, float m32, float m33, float m34,
  float m41, float m42, float m43, float m44)
{
  float4x4 m = 
  { 
    /*c1*/{m11, m21, m31, m41}, 
    /*c2*/{m12, m22, m32, m42}, 
    /*c3*/{m13, m23, m33, m43}, 
    /*c4*/{m14, m24, m34, m44} 
  };
  return m;
}

   
const float4x4 make_matrix(const float4& c1, const float4& c2, const float4& c3, const float4& c4)
{
  float4x4 m = { c1, c2, c3, c4 };
  return m;
}



const float4x4 make_matrix(const float3x3 &r, const float3 &pos)
{
  float4x4 m = 
  { 
    { r.c1.x, r.c1.y, r.c1.z, 0.0f }, 
    { r.c2.x, r.c2.y, r.c2.z, 0.0f },
    { r.c3.x, r.c3.y, r.c3.z, 0.0f },
    { pos.x,  pos.y,  pos.z,  1.0f }
  };
  return m;
}



const float4x4 make_translation(const float3 &pos)
{
  float4x4 m = make_identity<float4x4>();
  m.c4 = make_vector(pos.x, pos.y, pos.z, 1.0f);
  return m;
}


const float4x4 make_matrix_from_zAxis(const float3 &pos, const float3 &zAxis, const float3 &yAxis)
{
  float3 z = normalize(zAxis);
  float3 x = normalize(cross(yAxis, z));
  float3 y = cross(z, x);
  
  float4x4 m = 
  { 
    { x.x, x.y, x.z, 0.0f }, 
    { y.x, y.y, y.z, 0.0f },
    { z.x, z.y, z.z, 0.0f },
    { pos.x,  pos.y,  pos.z,  1.0f }
  };
  return m;
}



const float4x4 make_matrix_from_yAxis(const float3 &pos, const float3 &yAxis, const float3 &zAxis)
{
  float3 y = normalize(yAxis);
  float3 x = normalize(cross(y, zAxis));
  float3 z = cross(x, y);
  
  float4x4 m = 
  { 
    { x.x, x.y, x.z, 0.0f }, 
    { y.x, y.y, y.z, 0.0f },
    { z.x, z.y, z.z, 0.0f },
    { pos.x,  pos.y,  pos.z,  1.0f }
  };
  return m;
}


template <>
const float4x4 make_rotation_x<float4x4>(float angle)
{
  return make_matrix(make_rotation_x<float3x3>(angle), make_vector(0.0f, 0.0f, 0.0f));
}



template <>
const float4x4 make_rotation_y<float4x4>(float angle)
{
  return make_matrix(make_rotation_y<float3x3>(angle), make_vector(0.0f, 0.0f, 0.0f));
}



template <>
const float4x4 make_rotation_z<float4x4>(float angle)
{
  return make_matrix(make_rotation_z<float3x3>(angle), make_vector(0.0f, 0.0f, 0.0f));
}



template <>
const float4x4 make_rotation<float4x4>(const float3& axis, float angleRad)
{
  return make_matrix(make_rotation<float3x3>(axis, angleRad), make_vector(0.0f, 0.0f, 0.0f));
}



template <>
const float4x4 make_scale(const float3& scale)
{
  float4x4 m = 
  { 
    { scale.x, 0.0f, 0.0f, 0.0f },
    { 0.0f, scale.y, 0.0f, 0.0f },
    { 0.0f, 0.0f, scale.z, 0.0f },
    { 0.0f, 0.0f, 0.0f, 1.0f }
  };
  return m;
}



template <>
const float4x4 make_scale(const float& scale)
{
  float4x4 m = 
  { 
    { scale, 0.0f, 0.0f, 0.0f },
    { 0.0f, scale, 0.0f, 0.0f },
    { 0.0f, 0.0f, scale, 0.0f },
    { 0.0f, 0.0f, 0.0f, 1.0f }
  };
  return m;
}


float determinant(const float4x4 &m)
{
  float3x3 s1 = make_matrix(m.c2[1], m.c3[1], m.c4[1],
			 m.c2[2], m.c3[2], m.c4[2],
			 m.c2[3], m.c3[3], m.c4[3]);
  float3x3 s2 = make_matrix(m.c3[1], m.c4[1], m.c1[1],
			 m.c3[2], m.c4[2], m.c1[2],
			 m.c3[3], m.c4[3], m.c1[3]);
  float3x3 s3 = make_matrix(m.c4[1], m.c1[1], m.c2[1],
			 m.c4[2], m.c1[2], m.c2[2],
			 m.c4[3], m.c1[3], m.c2[3]);
  float3x3 s4 = make_matrix(m.c1[1], m.c2[1], m.c3[1],
			 m.c1[2], m.c2[2], m.c3[2],
			 m.c1[3], m.c2[3], m.c3[3]);
  
  float det1 = determinant(s1);
  float det2 = determinant(s2);
  float det3 = determinant(s3);
  float det4 = determinant(s4);

  return m.c1[0]*det1 - m.c2[0]*det2 + m.c3[0]*det3 - m.c4[0]*det4;
}



const float4x4 transpose(const float4x4 &m)
{
  return make_matrix(m.c1[0], m.c1[1], m.c1[2], m.c1[3], 
    m.c2[0], m.c2[1], m.c2[2], m.c2[3],
    m.c3[0], m.c3[1], m.c3[2], m.c3[3], 
    m.c4[0], m.c4[1], m.c4[2], m.c4[3]);
}



// Find the solution v to 
// u = Mv
const float4 cramers(const float4x4 &m, const float4& u)
{
  float4 a = { determinant(make_matrix(u, m.c2, m.c3, m.c4)), 
	   determinant(make_matrix(m.c1, u, m.c3, m.c4)), 
	   determinant(make_matrix(m.c1, m.c2, u, m.c4)), 
     determinant(make_matrix(m.c1, m.c2, m.c3, u)) };

  return (1.0f / determinant(m)) * a;
}



const float4x4 inverse(const float4x4 &m)
{
  const float *elements = &m.c1.x;
  float transpose[16];

  // Transpose matrix.
  for (int i = 0; i < 4; i++) 
  {
    transpose[i]      = elements[i * 4];
    transpose[i + 4]  = elements[i * 4 + 1];
    transpose[i + 8]  = elements[i * 4 + 2];
    transpose[i + 12] = elements[i * 4 + 3];
  }

  float pairs[12];
  // Calculate pairs for first 8 cofactors.
  pairs[0]  = transpose[10] * transpose[15];
  pairs[1]  = transpose[11] * transpose[14];
  pairs[2]  = transpose[ 9] * transpose[15];
  pairs[3]  = transpose[11] * transpose[13];
  pairs[4]  = transpose[ 9] * transpose[14];
  pairs[5]  = transpose[10] * transpose[13];
  pairs[6]  = transpose[ 8] * transpose[15];
  pairs[7]  = transpose[11] * transpose[12];
  pairs[8]  = transpose[ 8] * transpose[14];
  pairs[9]  = transpose[10] * transpose[12];
  pairs[10] = transpose[ 8] * transpose[13];
  pairs[11] = transpose[ 9] * transpose[12];

  float cofactors[16];
  // Calculate first 8 cofactors.
  cofactors[0]  = pairs[0] * transpose[5] + pairs[3] * transpose[6] + pairs[ 4] * transpose[7];
  cofactors[0] -= pairs[1] * transpose[5] + pairs[2] * transpose[6] + pairs[ 5] * transpose[7];
  cofactors[1]  = pairs[1] * transpose[4] + pairs[6] * transpose[6] + pairs[ 9] * transpose[7];
  cofactors[1] -= pairs[0] * transpose[4] + pairs[7] * transpose[6] + pairs[ 8] * transpose[7];
  cofactors[2]  = pairs[2] * transpose[4] + pairs[7] * transpose[5] + pairs[10] * transpose[7];
  cofactors[2] -= pairs[3] * transpose[4] + pairs[6] * transpose[5] + pairs[11] * transpose[7];
  cofactors[3]  = pairs[5] * transpose[4] + pairs[8] * transpose[5] + pairs[11] * transpose[6];
  cofactors[3] -= pairs[4] * transpose[4] + pairs[9] * transpose[5] + pairs[10] * transpose[6];
  cofactors[4]  = pairs[1] * transpose[1] + pairs[2] * transpose[2] + pairs[ 5] * transpose[3];
  cofactors[4] -= pairs[0] * transpose[1] + pairs[3] * transpose[2] + pairs[ 4] * transpose[3];
  cofactors[5]  = pairs[0] * transpose[0] + pairs[7] * transpose[2] + pairs[ 8] * transpose[3];
  cofactors[5] -= pairs[1] * transpose[0] + pairs[6] * transpose[2] + pairs[ 9] * transpose[3];
  cofactors[6]  = pairs[3] * transpose[0] + pairs[6] * transpose[1] + pairs[11] * transpose[3];
  cofactors[6] -= pairs[2] * transpose[0] + pairs[7] * transpose[1] + pairs[10] * transpose[3];
  cofactors[7]  = pairs[4] * transpose[0] + pairs[9] * transpose[1] + pairs[10] * transpose[2];
  cofactors[7] -= pairs[5] * transpose[0] + pairs[8] * transpose[1] + pairs[11] * transpose[2];

  // Calculate pairs for second 8 cofactors.
  pairs[ 0] = transpose[2] * transpose[7];
  pairs[ 1] = transpose[3] * transpose[6];
  pairs[ 2] = transpose[1] * transpose[7];
  pairs[ 3] = transpose[3] * transpose[5];
  pairs[ 4] = transpose[1] * transpose[6];
  pairs[ 5] = transpose[2] * transpose[5];
  pairs[ 6] = transpose[0] * transpose[7];
  pairs[ 7] = transpose[3] * transpose[4];
  pairs[ 8] = transpose[0] * transpose[6];
  pairs[ 9] = transpose[2] * transpose[4];
  pairs[10] = transpose[0] * transpose[5];
  pairs[11] = transpose[1] * transpose[4];

  // Calculate second 8 cofactors.
  cofactors[ 8]  = pairs[ 0] * transpose[13] + pairs[ 3] * transpose[14] + pairs[ 4] * transpose[15];
  cofactors[ 8] -= pairs[ 1] * transpose[13] + pairs[ 2] * transpose[14] + pairs[ 5] * transpose[15];
  cofactors[ 9]  = pairs[ 1] * transpose[12] + pairs[ 6] * transpose[14] + pairs[ 9] * transpose[15];
  cofactors[ 9] -= pairs[ 0] * transpose[12] + pairs[ 7] * transpose[14] + pairs[ 8] * transpose[15];
  cofactors[10]  = pairs[ 2] * transpose[12] + pairs[ 7] * transpose[13] + pairs[10] * transpose[15];
  cofactors[10] -= pairs[ 3] * transpose[12] + pairs[ 6] * transpose[13] + pairs[11] * transpose[15];
  cofactors[11]  = pairs[ 5] * transpose[12] + pairs[ 8] * transpose[13] + pairs[11] * transpose[14];
  cofactors[11] -= pairs[ 4] * transpose[12] + pairs[ 9] * transpose[13] + pairs[10] * transpose[14];
  cofactors[12]  = pairs[ 2] * transpose[10] + pairs[ 5] * transpose[11] + pairs[ 1] * transpose[ 9];
  cofactors[12] -= pairs[ 4] * transpose[11] + pairs[ 0] * transpose[ 9] + pairs[ 3] * transpose[10];
  cofactors[13]  = pairs[ 8] * transpose[11] + pairs[ 0] * transpose[ 8] + pairs[ 7] * transpose[10];
  cofactors[13] -= pairs[ 6] * transpose[10] + pairs[ 9] * transpose[11] + pairs[ 1] * transpose[ 8];
  cofactors[14]  = pairs[ 6] * transpose[ 9] + pairs[11] * transpose[11] + pairs[ 3] * transpose[ 8];
  cofactors[14] -= pairs[10] * transpose[11] + pairs[ 2] * transpose[ 8] + pairs[ 7] * transpose[ 9];
  cofactors[15]  = pairs[10] * transpose[10] + pairs[ 4] * transpose[ 8] + pairs[ 9] * transpose[ 9];
  cofactors[15] -= pairs[ 8] * transpose[ 9] + pairs[11] * transpose[10] + pairs[ 5] * transpose[ 8];

  // Calculate determinant.
  const float det = transpose[0] * cofactors[0] + transpose[1] * cofactors[1] + transpose[2] * cofactors[2] + transpose[3] * cofactors[3];
  // fudge to 0 instead of infinity, is this actually very clever?
  const float invDet = det ? 1.0f / det : 0.0f;

	float4x4 result; 
  float *res = &result.c1.x;
  for (int j = 0; j < 16; j++)
  {
    res[j] = cofactors[j] * invDet;
  }

	return result; 
}



const float3 transformPoint(const float4x4 &m, const float3 &p)
{
  float4 r = m * make_vector4(p, 1.0f);
  return make_vector(r.x, r.y, r.z);
}



const float3 transformDirection(const float4x4 &m, const float3 &d)
{
  float4 r = m * make_vector4(d, 0.0f);
  return make_vector(r.x, r.y, r.z);
}


} // namespace chag
