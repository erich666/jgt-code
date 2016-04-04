#ifndef __VECMATH_H
#define __VECMATH_H

/*
 *
 * vecmath.h
 *
 * Copyright (c)  Tien-Tsin Wong, 1996
 * All right reserved.
 * Define some useful basic vector mathematical functions
 * It will be continuous mantained in the future version
 *
 *
 */

//#ifdef  __cplusplus
//extern "C" {
//#endif

void vzero(float *v);
void vset(float *v, float x, float y, float z);
void vsub(const float *src1, const float *src2, float *dst);
void vcopy(const float *v1, float *v2);
void vcross(const float *v1, const float *v2, float *cross);
float vlength(const float *v);
void vscale(float *v, float scale);
void vnormal(float *v);
float vdot(const float *v1, const float *v2);
void vadd(const float *src1, const float *src2, float *dst);
void vlerp(const float *src1, const float *src2, float ratio, float *dst);
int vequal(const float *src1, const float *src2);

//#ifdef  __cplusplus
//}
//#endif

#endif

