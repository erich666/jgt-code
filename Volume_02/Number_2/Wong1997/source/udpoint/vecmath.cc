/*
 * vecmath.c
 *
 * Copyright (c) Tien-Tsin Wong, 1996
 * All right reserved.
 * A set of vector math functions.
 * It will be continuously updated in the future version.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TRUE  1
#define FALSE 0

/*************************** Float vector ********************************/
/*
 * In the following implemetation of vector functions. The data structure
 * of a vector is assumed to be a 3-elements array of float type. In C, it is
 *
 *   float vector[3];
 *
 */
void vzero(float *v)
{
  v[0] = 0.0;
  v[1] = 0.0;
  v[2] = 0.0;
}


void vset(float *v, float x, float y, float z)
{
  v[0] = x;
  v[1] = y;
  v[2] = z;
}


void vsub(const float *src1, const float *src2, float *dst)
{
  dst[0] = src1[0] - src2[0];
  dst[1] = src1[1] - src2[1];
  dst[2] = src1[2] - src2[2];
}


void vcopy(const float *v1, float *v2)
{
  register int i;
  for (i = 0 ; i < 3 ; i++)
    v2[i] = v1[i];
}


void vcross(const float *v1, const float *v2, float *cross)
{
  float temp[3];
  temp[0] = (v1[1] * v2[2]) - (v1[2] * v2[1]);
  temp[1] = (v1[2] * v2[0]) - (v1[0] * v2[2]);
  temp[2] = (v1[0] * v2[1]) - (v1[1] * v2[0]);
  vcopy(temp, cross);
}


float vlength(const float *v)
{
  return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}


void vscale(float *v, float scale)
{
  v[0] *= scale;
  v[1] *= scale;
  v[2] *= scale;
}


/* Normalize the vector */
void vnormal(float *v)
{
  vscale(v,1.0/vlength(v));
}


float vdot(const float *v1, const float *v2)
{
  return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
}


void vadd(const float *src1, const float *src2, float *dst)
{
  dst[0] = src1[0] + src2[0];
  dst[1] = src1[1] + src2[1];
  dst[2] = src1[2] + src2[2];
}


/*
 * linearly interpolate between 2 vectors
 * dst = src1*ratio + src2*(1-ratio)
 */
void vlerp(const float *src1, const float *src2, float ratio, float *dst)
{
  float recip = 1-ratio;
  dst[0] = src1[0]*ratio + src2[0]*recip;
  dst[1] = src1[1]*ratio + src2[1]*recip;
  dst[2] = src1[2]*ratio + src2[2]*recip;
}

/*
 * Compare 2 vectors
 * return TRUE if equal
 * return FALSE otherwise
 */
int vequal(const float *src1, const float *src2)
{
  register int i;
  for (i=0 ; i<3 ; i++)
    if (src1[i]!=src2[i])
      return FALSE;
  return TRUE;
}
