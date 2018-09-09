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
#include "float3x3.h"
#include "float4x4.h"
#include <cassert>
#include <limits>

namespace chag
{


float3& float3x3::operator [] (const size_t i)
{ 
  assert( i < 3 );
  return *(&c1 + i);
}



const float3& float3x3::operator [] (const size_t i) const
{ 
	assert( i < 3 );
  return *(&c1 + i);
}



const bool float3x3::operator == (const float3x3& m) const
{ 
  return (m.c1 == c1) && (m.c2 == c2) && (m.c3 == c3); 
}



const bool float3x3::operator != (const float3x3& m) const 
{ 
  return !(m == *this); 
}



const float3x3 float3x3::operator - (const float3x3 m)
{
  return make_matrix(c1 - m.c1, c2 - m.c2, c3 - m.c3);
} 

const float3 float3x3::operator * (const float3& v) const
{
  return make_vector(c1[0]*v.x + c2[0]*v.y + c3[0]*v.z,  
		c1[1]*v.x + c2[1]*v.y + c3[1]*v.z, 
		c1[2]*v.x + c2[2]*v.y + c3[2]*v.z);
}

const float3x3 float3x3::operator * (const float f) const
{
  return make_matrix(c1*f, c2*f, c3*f);
}



const float3 float3x3::row(size_t i) const
{
  assert( i < 3 );
  switch(i)
  {
  case 0: 
    return make_vector(c1[0], c2[0], c3[0]);
  case 1: 
    return make_vector(c1[1], c2[1], c3[1]);
  case 2: 
    return make_vector(c1[2], c2[2], c3[2]);
  }

  // fill with the most incorrect value we have...
  return make_vector(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN());
}




const float3x3 float3x3::operator * (const float3x3& m) const
{
  return make_matrix(dot(row(0), m.c1), dot(row(0), m.c2), dot(row(0), m.c3), 
		  dot(row(1), m.c1), dot(row(1), m.c2), dot(row(1), m.c3), 
		  dot(row(2), m.c1), dot(row(2), m.c2), dot(row(2), m.c3));
}



const float3x3 transpose(const float3x3 &m)
{
  return make_matrix(m.c1[0], m.c1[1], m.c1[2], 
		  m.c2[0], m.c2[1], m.c2[2], 
		  m.c3[0], m.c3[1], m.c3[2]);
}



const void lu(const float3x3 &a, float3x3 &l, float3x3 &u)
{
  float u1 = a[0][0];
  float u2 = a[1][0];
  float u3 = a[2][0];
  float l1 = a[0][1] / u1;
  float u4 = a[1][1] - l1*u2;
  float u5 = a[2][1] - l1*u3;
  float l2 = a[0][2] / u1;
  float l3 = (a[1][2] - l2*u2) / u4;
  float u6 = a[2][2] - l2*u3 - l3*u5;
  l[0][0] = 1; l[1][0] = 0; l[2][0] = 0;
  l[0][1] = l1; l[1][1] = 1; l[2][1] = 0;
  l[0][2] = l2; l[1][2] = l3; l[2][2] = 1;
  u[0][0] = u1; u[1][0] = u2; u[2][0] = u3;
  u[0][1] = 0; u[1][1] = u4; u[2][1] = u5;
  u[0][2] = 0; u[1][2] = 0; u[2][2] = u6;
}



const float3 lr(float3x3 &a)
{
  bool go_on = true;
  float3x3 l;
  float3x3 u;
      
  while(go_on)
  {
    lu(a, l, u);
    a = u * l;
    if((a[0][1] + a[0][2] + a[1][2]) < 0.000001f)
    {
      go_on = false;
    }
  }

  return make_vector(a[0][0], a[1][1], a[2][2]);
}



void egenvektor(float3x3 &m, float3 *v1, float3 *v2, float3 *v3)
{
  float3 ev = lr(m);
  float3x3 a = m;

  float3x3 i;
  float l1 = ev.x;
  float l2 = ev.y;
  float l3 = ev.z;
  float3 o = make_vector(0.0f, 0.0f, 0.0f);
  float3x3 a1 = (a-(l1*i));

  *v1 = cramers(a1, o);
  a1 = (a - (l2 * i));
  *v2 = cramers(a1, o);
  a1 = (a - (l3 * i));
  *v3 = cramers(a1, o);
}



const float determinant(const float3x3 &m)
{
  return m.c1[0]*m.c2[1]*m.c3[2] + 
    m.c2[0]*m.c3[1]*m.c1[2] + 
    m.c3[0]*m.c1[1]*m.c2[2] - 
    m.c1[0]*m.c3[1]*m.c2[2] -
    m.c2[0]*m.c1[1]*m.c3[2] - 
    m.c3[0]*m.c2[1]*m.c1[2]; 
}




const float3 cramers(const float3x3& m, const float3& u)
{
  float3 a = 
  {
    determinant(make_matrix(u, m.c2, m.c3)), 
    determinant(make_matrix(m.c1, u, m.c3)), 
    determinant(make_matrix(m.c1, m.c2, u))
  };
  
  return (1.0f / determinant(m)) * a;
}


template <>
const float3x3 make_identity<float3x3>()
{
  float3x3 m = 
  { 
    { 1.0f, 0.0f, 0.0f },
    { 0.0f, 1.0f, 0.0f },
    { 0.0f, 0.0f, 1.0f }
  };
  return m;
}




template <>
const float3x3 make_matrix<float3x3>(const float *e)
{
  float3x3 m;
  memcpy(&m, e, sizeof(m));
  return m;
}




const float3x3 make_matrix3x3(const float4x4 &m)
{
  return make_matrix(make_vector3(m.c1), make_vector3(m.c2), make_vector3(m.c3));
}



template <>
const float3x3 make_rotation_x<float3x3>(float angle)
{
  float3x3 result = make_identity<float3x3>();

  float s = sinf(angle);
  float c = cosf(angle);

  result[1][1] = c; 
  result[2][1] = -s;

  result[1][2] = s; 
  result[2][2] = c;

  return result;
}



template <>
const float3x3 make_rotation_y<float3x3>(float angle)
{
  float3x3 result = make_identity<float3x3>();

  float s = sinf(angle);
  float c = cosf(angle);

  result[0][0] = c;
  result[0][2] = -s;

  result[2][0] = s;
  result[2][2] = c;

  return result;
}



template <>
const float3x3 make_rotation_z<float3x3>(float angle)
{
  float3x3 result = make_identity<float3x3>();

  float s = sinf(angle);
  float c = cosf(angle);

  result[0][0] = c;
  result[0][1] = -s;

  result[1][0] = s;
  result[1][1] = c;

  return result;
}


template <>
const float3x3 make_rotation<float3x3>(const float3& axis, float angleRad)
{
	float3x3 res;
	float3 v = normalize(axis);

	float sinA = sinf(angleRad);
	float cosA = cosf(angleRad);
	float cosI = 1.0f - cosA;

	res[0][0] = (cosI * v.x * v.x) + (cosA);
	res[1][0] = (cosI * v.x * v.y) - (sinA * v.z );
	res[2][0] = (cosI * v.x * v.z) + (sinA * v.y );
		    
	res[0][1] = (cosI * v.y * v.x) + (sinA * v.z);
	res[1][1] = (cosI * v.y * v.y) + (cosA);
	res[2][1] = (cosI * v.y * v.z) - (sinA * v.x);
		    
	res[0][2] = (cosI * v.z * v.x) - (sinA * v.y);
	res[1][2] = (cosI * v.z * v.y) + (sinA * v.x);
	res[2][2] = (cosI * v.z * v.z) + (cosA);

	return res;
}




const float3x3 make_matrix(const float3 &c1, const float3 &c2, const float3 &c3)
{
  float3x3 m = { c1, c2, c3};
  return m;
}


const float3x3 make_matrix(
  float m11, float m12, float m13, 
  float m21, float m22, float m23, 
  float m31, float m32, float m33)
{
  float3x3 m = 
  { 
    /*c1*/{m11, m21, m31}, 
    /*c2*/{m12, m22, m32}, 
    /*c3*/{m13, m23, m33}
  };
  return m;
}


const float3x3 operator * (const float& f, const float3x3& M)
{
  return make_matrix(M.c1*f, M.c2*f, M.c3*f);
}


} // namespace chag
