// Geometric Tools, LLC
// Copyright (c) 1998-2011
// Distributed under the Boost Software License, Version 1.0.
// http://www.boost.org/LICENSE_1_0.txt
// http://www.geometrictools.com/License/Boost/LICENSE_1_0.txt
//
// History:
//   Created 03 April 2011.

#include <cmath>
#include <ctime>
#include <set>
#include <xmmintrin.h>
#include <emmintrin.h>

#if (defined(_WIN32) || defined(WIN32)) && defined(_MSC_VER)
// Disable the Microsoft warnings about not using the secure functions.
#pragma warning(disable : 4996)
#endif

//----------------------------------------------------------------------------
// A simple 4-tuple class for the sample implementations.
//----------------------------------------------------------------------------
class FTuple4
{
public:
    FTuple4 () { /**/ }

    FTuple4 (const FTuple4& v)
    {
        mTuple[0] = v.mTuple[0];
        mTuple[1] = v.mTuple[1];
        mTuple[2] = v.mTuple[2];
        mTuple[3] = v.mTuple[3];
    }

    FTuple4 (float x, float y, float z, float w)
    {
        mTuple[0] = x;
        mTuple[1] = y;
        mTuple[2] = z;
        mTuple[3] = w;
    }

    FTuple4& operator= (const FTuple4& v)
    {
        mTuple[0] = v.mTuple[0];
        mTuple[1] = v.mTuple[1];
        mTuple[2] = v.mTuple[2];
        mTuple[3] = v.mTuple[3];
        return *this;
    }

    const float& operator[] (int i) const { return mTuple[i]; }

    float& operator[] (int i) { return mTuple[i]; }

    FTuple4 operator- ()
    {
        return FTuple4(-mTuple[0], -mTuple[1], -mTuple[2], -mTuple[3]);
    }

    FTuple4 operator+ (const FTuple4& v) const
    {
        return FTuple4(
            mTuple[0] + v.mTuple[0],
            mTuple[1] + v.mTuple[1],
            mTuple[2] + v.mTuple[2],
            mTuple[3] + v.mTuple[3]);
    }

    FTuple4 operator* (float s) const
    {
        return FTuple4(s*mTuple[0], s*mTuple[1], s*mTuple[2], s*mTuple[3]);
    }

    float Dot (const FTuple4& v) const
    {
        return mTuple[0]*v.mTuple[0] + mTuple[1]*v.mTuple[1] +
            mTuple[2]*v.mTuple[2] + mTuple[3]*v.mTuple[3];
    }

    void Normalize ()
    {
        float length = sqrt(Dot(*this));
        if (length > 0.0f)
        {
            float invLength = 1.0f/length;
            for (int i = 0; i < 4; ++i)
            {
                mTuple[i] *= invLength;
            }
        }
        else
        {
            for (int i = 0; i < 4; ++i)
            {
                mTuple[i] = 0.0f;
            }
        }
    }

private:
    float mTuple[4];
};
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// Support for the Remez Algorithm.
//----------------------------------------------------------------------------
class Solution
{
public:
    bool operator< (const Solution& solution) const
    {
        return esum < solution.esum;
    }

    double esum, e, u;
};
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// The standard implementation of SLERP.
//----------------------------------------------------------------------------
FTuple4 Slerp (float t, FTuple4 q0, FTuple4 q1)
{
    FTuple4 slerp;

    float cosTheta = q0.Dot(q1);
    if (cosTheta < 0.0f)
    {
        q1 = -q1;
        cosTheta = -cosTheta;
    }

    if (cosTheta < 1.0f)
    {
        float theta = acos(cosTheta);
        float sinTheta = sin(theta);
        float invSinTheta = 1.0f/sinTheta;
        float tTimesTheta = t*theta;
        float c0 = sin(theta - tTimesTheta)*invSinTheta;
        float c1 = sin(tTimesTheta)*invSinTheta;
        slerp = q0*c0 + q1*c1;
    }
    else
    {
        slerp = q0;
    }

    return slerp;
}
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// The fast algorithm implemented for the floating-point unit (FPU).
//----------------------------------------------------------------------------
const float onePlusMuFPU = 1.90110745351730037f;

const float uFPU[8] =  // 1/[i(2i+1)] for i >= 1
{
    1.0f/(1.0f* 3.0f),  1.0f/(2.0f* 5.0f),  1.0f/(3.0f* 7.0f),
    1.0f/(4.0f* 9.0f),  1.0f/(5.0f*11.0f),  1.0f/(6.0f*13.0f),
    1.0f/(7.0f*15.0f),  onePlusMuFPU/(8.0f*17.0f)
};
const float vFPU[8] =  // i/(2i+1) for i >= 1
{
    1.0f/3.0f,  2.0f/ 5.0f,  3.0f/ 7.0f,
    4.0f/9.0f,  5.0f/11.0f,  6.0f/13.0f,
    7.0f/15.0f, onePlusMuFPU*8.0f/17.0f
};
//----------------------------------------------------------------------------
FTuple4 SlerpFPU (float t, FTuple4 q0, FTuple4 q1)
{
    float x = q0.Dot(q1);  // cos(theta)
    float sign = (x >= 0 ? 1.0f : (x = -x, -1.0f));
    float xm1 = x - 1.0f;

    float d = 1.0f - t, sqrT = t*t, sqrD = d*d;

    float bT[8], bD[8];
    for (int i = 7; i >= 0; --i)
    {
        bT[i] = (uFPU[i]*sqrT - vFPU[i])*xm1;
        bD[i] = (uFPU[i]*sqrD - vFPU[i])*xm1;
    }

    float cT = sign*t*(
        1.0f + bT[0]*(1.0f + bT[1]*(1.0f + bT[2]*(1.0f + bT[3]*(
        1.0f + bT[4]*(1.0f + bT[5]*(1.0f + bT[6]*(1.0f + bT[7]))))))));

    float cD = d*(
        1.0f + bD[0]*(1.0f + bD[1]*(1.0f + bD[2]*(1.0f + bD[3]*(
        1.0f + bD[4]*(1.0f + bD[5]*(1.0f + bD[6]*(1.0f + bD[7]))))))));

    FTuple4 slerp = q0*cD + q1*cT;
    return slerp;
}
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// The fast algorithm implemented for SIMD in Intel (R) processors.  The
// Dot function is used by SlerpSSE1, SlerpSSE2, and SlerpSSE4.  Coefficient1
// is used by SlerpSSE1.  Coefficient4 is used by SlerpSSE4.
//----------------------------------------------------------------------------
const float onePlusMu = 1.90110745351730037f;

const __m128 u0123 = _mm_setr_ps(
    1.0f/(1.0f*3.0f),
    1.0f/(2.0f*5.0f),
    1.0f/(3.0f*7.0f),
    1.0f/(4.0f*9.0f));

const __m128 u4567 = _mm_setr_ps(
    1.0f/(5.0f*11.0f),
    1.0f/(6.0f*13.0f),
    1.0f/(7.0f*15.0f),
    onePlusMu/(8.0f*17.0f));

const __m128 v0123 = _mm_setr_ps(
    1.0f/3.0f,
    2.0f/5.0f,
    3.0f/7.0f,
    4.0f/9.0f);

const __m128 v4567 = _mm_setr_ps(
    5.0f/11.0f,
    6.0f/13.0f,
    7.0f/15.0f,
    onePlusMu*8.0f/17.0f);

const __m128 signMask = _mm_set1_ps(-0.0f);
const __m128 one = _mm_set1_ps(1.0f);
//----------------------------------------------------------------------------
inline __m128 Dot (const __m128 tuple0, const __m128 tuple1)
{
    __m128 t0 = _mm_mul_ps(tuple0, tuple1);
    __m128 t1 = _mm_shuffle_ps(tuple1, t0, _MM_SHUFFLE(1,0,0,0));
    t1 = _mm_add_ps(t0, t1);
    t0 = _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(2,0,0,0));
    t0 = _mm_add_ps(t0, t1);
    t0 = _mm_shuffle_ps(t0, t0, _MM_SHUFFLE(3,3,3,3));
    return t0;
}
//----------------------------------------------------------------------------
inline __m128 Coefficient1 (const __m128 t, const __m128 xm1)
{
    __m128 sqrT = _mm_mul_ps(t, t);
    __m128 b0123, b4567, b, c;

    // (b4,b5,b6,b7) = (x-1)*(u4*t^2-v4, u5*t^2-v5, u6*t^2-v6, u7*t^2-v7)
    b4567 = _mm_mul_ps(u4567, sqrT);
    b4567 = _mm_sub_ps(b4567, v4567);
    b4567 = _mm_mul_ps(b4567, xm1);
    // (b7,b7,b7,b7)
    b = _mm_shuffle_ps(b4567, b4567, _MM_SHUFFLE(3,3,3,3));
    c = _mm_add_ps(b, one);
    // (b6,b6,b6,b6)
    b = _mm_shuffle_ps(b4567, b4567, _MM_SHUFFLE(2,2,2,2));
    c = _mm_mul_ps(b, c);
    c = _mm_add_ps(one, c);
    // (b5,b5,b5,b5)
    b = _mm_shuffle_ps(b4567, b4567, _MM_SHUFFLE(1,1,1,1));
    c = _mm_mul_ps(b, c);
    c = _mm_add_ps(one, c);
    // (b4,b4,b4,b4)
    b = _mm_shuffle_ps(b4567, b4567, _MM_SHUFFLE(0,0,0,0));
    c = _mm_mul_ps(b, c);
    c = _mm_add_ps(one, c);

    // (b0,b1,b2,b3) = (x-1)*(u0*t^2-v0, u1*t^2-v1, u2*t^2-v2, u3*t^2-v3)
    b0123 = _mm_mul_ps(u0123, sqrT);
    b0123 = _mm_sub_ps(b0123, v0123);
    b0123 = _mm_mul_ps(b0123, xm1);
    // (b3,b3,b3,b3)
    b = _mm_shuffle_ps(b0123, b0123, _MM_SHUFFLE(3,3,3,3));
    c = _mm_mul_ps(b, c);
    c = _mm_add_ps(one, c);
    // (b2,b2,b2,b2)
    b = _mm_shuffle_ps(b0123, b0123, _MM_SHUFFLE(2,2,2,2));
    c = _mm_mul_ps(b, c);
    c = _mm_add_ps(one, c);
    // (b1,b1,b1,b1)
    b = _mm_shuffle_ps(b0123, b0123, _MM_SHUFFLE(1,1,1,1));
    c = _mm_mul_ps(b, c);
    c = _mm_add_ps(one, c);
    // (b0,b0,b0,b0)
    b = _mm_shuffle_ps(b0123, b0123, _MM_SHUFFLE(0,0,0,0));
    c = _mm_mul_ps(b, c);
    c = _mm_add_ps(one, c);
    c = _mm_mul_ps(t, c);

    return c;
}
//----------------------------------------------------------------------------
void SlerpSSE1 (float t[1], const __m128 q0, const __m128 q1, __m128 slerp[1])
{
    __m128 x = Dot(q0, q1);
    __m128 sign = _mm_and_ps(signMask, x);
    x = _mm_xor_ps(sign, x);
    __m128 localQ1 = _mm_xor_ps(sign, q1);
    __m128 xm1 = _mm_sub_ps(x, one);

    __m128 splatT = _mm_set1_ps(t[0]);
    __m128 splatD = _mm_sub_ps(one, splatT);

    __m128 cT = Coefficient1(splatT, xm1);
    __m128 cD = Coefficient1(splatD, xm1);
    cT = _mm_mul_ps(cT, localQ1);
    cD = _mm_mul_ps(cD, q0);

    slerp[0] = _mm_add_ps(cT, cD);
}
//----------------------------------------------------------------------------
void SlerpSSE2 (float t[2], const __m128 q0, const __m128 q1, __m128 slerp[2])
{
    __m128 x = Dot(q0, q1);
    __m128 sign = _mm_and_ps(signMask, x);
    x = _mm_xor_ps(sign, x);
    __m128 localQ1 = _mm_xor_ps(sign, q1);
    __m128 xm1 = _mm_sub_ps(x, one);

    __m128 splatT[2], splatD[2];
    splatT[0] = _mm_set1_ps(t[0]);
    splatT[1] = _mm_set1_ps(t[1]);
    splatD[0] = _mm_sub_ps(one, splatT[0]);
    splatD[1] = _mm_sub_ps(one, splatT[1]);
    __m128 sqrT0 = _mm_mul_ps(splatT[0], splatT[0]);
    __m128 sqrT1 = _mm_mul_ps(splatT[1], splatT[1]);
    __m128 sqrD0 = _mm_mul_ps(splatD[0], splatD[0]);
    __m128 sqrD1 = _mm_mul_ps(splatD[1], splatD[1]);
    __m128 s0, s1, b, c;

    // (bT04,bT05,bT06,bT07) = (x-1)*(u4*t0^2-v4,u5*t0^2-v5,u6*t0^2-v6,u7*t0^2-v7)
    __m128 bT0_4567 = _mm_mul_ps(u4567, sqrT0);
    bT0_4567 = _mm_sub_ps(bT0_4567, v4567);
    bT0_4567 = _mm_mul_ps(bT0_4567, xm1);
    // (bT14,bT15,bT16,bT17) = (x-1)*(u4*t1^2-v4,u5*t1^2-v5,u6*t1^2-v6,u7*t1^2-v7)
    __m128 bT1_4567 = _mm_mul_ps(u4567, sqrT1);
    bT1_4567 = _mm_sub_ps(bT1_4567, v4567);
    bT1_4567 = _mm_mul_ps(bT1_4567, xm1);
    // (bD04,bD05,bD06,bD07) = (x-1)*(u4*d0^2-v4,u5*d0^2-v5,u6*d0^2-v6,u7*d0^2-v7)
    __m128 bD0_4567 = _mm_mul_ps(u4567, sqrD0);
    bD0_4567 = _mm_sub_ps(bD0_4567, v4567);
    bD0_4567 = _mm_mul_ps(bD0_4567, xm1);
    // (bD14,bD15,bD16,bD17) = (x-1)*(u4*d1^2-v4,u5*d1^2-v5,u6*d1^2-v6,u7*d1^2-v7)
    __m128 bD1_4567 = _mm_mul_ps(u4567, sqrD1);
    bD1_4567 = _mm_sub_ps(bD1_4567, v4567);
    bD1_4567 = _mm_mul_ps(bD1_4567, xm1);

    // (bT07,bT07,bD07,bD07)
    s0 = _mm_shuffle_ps(bT0_4567, bD0_4567, _MM_SHUFFLE(3,3,3,3));
    // (bT17,bT17,bD17,bD17)
    s1 = _mm_shuffle_ps(bT1_4567, bD1_4567, _MM_SHUFFLE(3,3,3,3));
    // (bT07,bD07,bT17,bD17)
    b = _mm_shuffle_ps(s0, s1, _MM_SHUFFLE(2,0,2,0));
    c = _mm_add_ps(b, one);

    // (bT06,bT06,bD06,bD06)
    s0 = _mm_shuffle_ps(bT0_4567, bD0_4567, _MM_SHUFFLE(2,2,2,2));
    // (bT16,bT16,bD16,bD16)
    s1 = _mm_shuffle_ps(bT1_4567, bD1_4567, _MM_SHUFFLE(2,2,2,2));
    // (bT06,bD06,bT16,bD16)
    b = _mm_shuffle_ps(s0, s1, _MM_SHUFFLE(2,0,2,0));
    c = _mm_mul_ps(b, c);
    c = _mm_add_ps(one, c);

    // (bT05,bT05,bD05,bD05)
    s0 = _mm_shuffle_ps(bT0_4567, bD0_4567, _MM_SHUFFLE(1,1,1,1));
    // (bT15,bT15,bD15,bD15)
    s1 = _mm_shuffle_ps(bT1_4567, bD1_4567, _MM_SHUFFLE(1,1,1,1));
    // (bT05,bD05,bT15,bD15)
    b = _mm_shuffle_ps(s0, s1, _MM_SHUFFLE(2,0,2,0));
    c = _mm_mul_ps(b, c);
    c = _mm_add_ps(one, c);

    // (bT04,bT04,bD04,bD04)
    s0 = _mm_shuffle_ps(bT0_4567, bD0_4567, _MM_SHUFFLE(0,0,0,0));
    // (bT14,bT14,bD14,bD14)
    s1 = _mm_shuffle_ps(bT1_4567, bD1_4567, _MM_SHUFFLE(0,0,0,0));
    // (bT04,bD04,bT14,bD14)
    b = _mm_shuffle_ps(s0, s1, _MM_SHUFFLE(2,0,2,0));
    c = _mm_mul_ps(b, c);
    c = _mm_add_ps(one, c);

    // (bT00,bT01,bT02,bT03) = (x-1)*(u0*t0^2-v0,u1*t0^2-v1,u2*t0^2-v2,u3*t0^2-v3)
    __m128 bT0_0123 = _mm_mul_ps(u0123, sqrT0);
    bT0_0123 = _mm_sub_ps(bT0_0123, v0123);
    bT0_0123 = _mm_mul_ps(bT0_0123, xm1);
    // (bT10,bT11,bT12,bT13) = (x-1)*(u0*t1^2-v0,u1*t1^2-v1,u2*t1^2-v2,u3*t1^2-v3)
    __m128 bT1_0123 = _mm_mul_ps(u0123, sqrT1);
    bT1_0123 = _mm_sub_ps(bT1_0123, v0123);
    bT1_0123 = _mm_mul_ps(bT1_0123, xm1);
    // (bD00,bD01,bD02,bD03) = (x-1)*(u0*d0^2-v0,u1*d0^2-v1,u2*d0^2-v2,u3*d0^2-v3)
    __m128 bD0_0123 = _mm_mul_ps(u0123, sqrD0);
    bD0_0123 = _mm_sub_ps(bD0_0123, v0123);
    bD0_0123 = _mm_mul_ps(bD0_0123, xm1);
    // (bD10,bD11,bD12,bD13) = (x-1)*(u0*d1^2-v0,u1*d1^2-v1,u2*d1^2-v2,u3*d1^2-v3)
    __m128 bD1_0123 = _mm_mul_ps(u0123, sqrD1);
    bD1_0123 = _mm_sub_ps(bD1_0123, v0123);
    bD1_0123 = _mm_mul_ps(bD1_0123, xm1);

    // (bT03,bT03,bD03,bD03)
    s0 = _mm_shuffle_ps(bT0_0123, bD0_0123, _MM_SHUFFLE(3,3,3,3));
    // (bT13,bT13,bD13,bD13)
    s1 = _mm_shuffle_ps(bT1_0123, bD1_0123, _MM_SHUFFLE(3,3,3,3));
    // (bT03,bD03,bT13,bD13)
    b = _mm_shuffle_ps(s0, s1, _MM_SHUFFLE(2,0,2,0));
    c = _mm_mul_ps(b, c);
    c = _mm_add_ps(one, c);

    // (bT02,bT02,bD02,bD02)
    s0 = _mm_shuffle_ps(bT0_0123, bD0_0123, _MM_SHUFFLE(2,2,2,2));
    // (bT12,bT12,bD12,bD12)
    s1 = _mm_shuffle_ps(bT1_0123, bD1_0123, _MM_SHUFFLE(2,2,2,2));
    // (bT02,bD02,bT12,bD12)
    b = _mm_shuffle_ps(s0, s1, _MM_SHUFFLE(2,0,2,0));
    c = _mm_mul_ps(b, c);
    c = _mm_add_ps(one, c);

    // (bT01,bT01,bD01,bD01)
    s0 = _mm_shuffle_ps(bT0_0123, bD0_0123, _MM_SHUFFLE(1,1,1,1));
    // (bT11,bT11,bD11,bD11)
    s1 = _mm_shuffle_ps(bT1_0123, bD1_0123, _MM_SHUFFLE(1,1,1,1));
    // (bT01,bD01,bT11,bD11)
    b = _mm_shuffle_ps(s0, s1, _MM_SHUFFLE(2,0,2,0));
    c = _mm_mul_ps(b, c);
    c = _mm_add_ps(one, c);

    // (bT00,bT00,bD00,bD00)
    s0 = _mm_shuffle_ps(bT0_0123, bD0_0123, _MM_SHUFFLE(0,0,0,0));
    // (bT10,bT10,bD10,bD10)
    s1 = _mm_shuffle_ps(bT1_0123, bD1_0123, _MM_SHUFFLE(0,0,0,0));
    // (bT00,bD00,bT10,bD10)
    b = _mm_shuffle_ps(s0, s1, _MM_SHUFFLE(2,0,2,0));
    c = _mm_mul_ps(b, c);
    c = _mm_add_ps(one, c);

    // (t0,t0,d0,d0)
    s0 = _mm_shuffle_ps(splatT[0], splatD[0], _MM_SHUFFLE(0,0,0,0));
    // (t1,t1,d1,d1)
    s1 = _mm_shuffle_ps(splatT[1], splatD[1], _MM_SHUFFLE(0,0,0,0));
    // (t0,d0,t1,d1)
    b = _mm_shuffle_ps(s0, s1, _MM_SHUFFLE(2,0,2,0));
    // c = (cT0,cD0,cT1,cD1)
    c = _mm_mul_ps(b, c);

    __m128 cT0 = _mm_shuffle_ps(c, c, _MM_SHUFFLE(0,0,0,0));
    __m128 cD0 = _mm_shuffle_ps(c, c, _MM_SHUFFLE(1,1,1,1));
    __m128 cT1 = _mm_shuffle_ps(c, c, _MM_SHUFFLE(2,2,2,2));
    __m128 cD1 = _mm_shuffle_ps(c, c, _MM_SHUFFLE(3,3,3,3));
    cT0 = _mm_mul_ps(cT0, localQ1);
    cD0 = _mm_mul_ps(cD0, q0);
    cT1 = _mm_mul_ps(cT1, localQ1);
    cD1 = _mm_mul_ps(cD1, q0);

    slerp[0] = _mm_add_ps(cT0, cD0);
    slerp[1] = _mm_add_ps(cT1, cD1);
}
//----------------------------------------------------------------------------
inline __m128 Coefficient4 (const __m128 t[4], const __m128 xm1)
{
    __m128 sqrT0 = _mm_mul_ps(t[0], t[0]);
    __m128 sqrT1 = _mm_mul_ps(t[1], t[1]);
    __m128 sqrT2 = _mm_mul_ps(t[2], t[2]);
    __m128 sqrT3 = _mm_mul_ps(t[3], t[3]);
    __m128 b0_0123, b0_4567, b1_0123, b1_4567;
    __m128 b2_0123, b2_4567, b3_0123, b3_4567;
    __m128 s01, s23, b, c;

    // (b04,b05,b06,b07) = (x-1)*(u4*t0^2-v4,u5*t0^2-v5,u6*t0^2-v6,u7*t0^2-v7)
    b0_4567 = _mm_mul_ps(u4567, sqrT0);
    b0_4567 = _mm_sub_ps(b0_4567, v4567);
    b0_4567 = _mm_mul_ps(b0_4567, xm1);
    // (b14,b15,b16,b17) = (x-1)*(u4*t1^2-v4,u5*t1^2-v5,u6*t1^2-v6,u7*t1^2-v7)
    b1_4567 = _mm_mul_ps(u4567, sqrT1);
    b1_4567 = _mm_sub_ps(b1_4567, v4567);
    b1_4567 = _mm_mul_ps(b1_4567, xm1);
    // (b24,b25,b26,b27) = (x-1)*(u4*t2^2-v4,u5*t2^2-v5,u6*t2^2-v6,u7*t2^2-v7)
    b2_4567 = _mm_mul_ps(u4567, sqrT2);
    b2_4567 = _mm_sub_ps(b2_4567, v4567);
    b2_4567 = _mm_mul_ps(b2_4567, xm1);
    // (b34,b35,b36,b37) = (x-1)*(u4*t3^2-v4,u5*t3^2-v5,u6*t3^2-v6,u7*t3^2-v7)
    b3_4567 = _mm_mul_ps(u4567, sqrT3);
    b3_4567 = _mm_sub_ps(b3_4567, v4567);
    b3_4567 = _mm_mul_ps(b3_4567, xm1);

    // (b07,b07,b17,b17)
    s01 = _mm_shuffle_ps(b0_4567, b1_4567, _MM_SHUFFLE(3,3,3,3));
    // (b27,b27,b37,b37)
    s23 = _mm_shuffle_ps(b2_4567, b3_4567, _MM_SHUFFLE(3,3,3,3));
    // (b07,b17,b27,b37)
    b = _mm_shuffle_ps(s01, s23, _MM_SHUFFLE(2,0,2,0));
    c = _mm_add_ps(b, one);
    // (b06,b06,b16,b16)
    s01 = _mm_shuffle_ps(b0_4567, b1_4567, _MM_SHUFFLE(2,2,2,2));
    // (b26,b26,b36,b36)
    s23 = _mm_shuffle_ps(b2_4567, b3_4567, _MM_SHUFFLE(2,2,2,2));
    // (b06,b16,b26,b36)
    b = _mm_shuffle_ps(s01, s23, _MM_SHUFFLE(2,0,2,0));
    c = _mm_mul_ps(b, c);
    c = _mm_add_ps(one, c);
    // (b05,b05,b15,b15)
    s01 = _mm_shuffle_ps(b0_4567, b1_4567, _MM_SHUFFLE(1,1,1,1));
    // (b25,b25,b35,b35)
    s23 = _mm_shuffle_ps(b2_4567, b3_4567, _MM_SHUFFLE(1,1,1,1));
    // (b05,b15,b25,b35)
    b = _mm_shuffle_ps(s01, s23, _MM_SHUFFLE(2,0,2,0));
    c = _mm_mul_ps(b, c);
    c = _mm_add_ps(one, c);
    // (b04,b04,b14,b14)
    s01 = _mm_shuffle_ps(b0_4567, b1_4567, _MM_SHUFFLE(0,0,0,0));
    // (b24,b24,b34,b34)
    s23 = _mm_shuffle_ps(b2_4567, b3_4567, _MM_SHUFFLE(0,0,0,0));
    // (b04,b14,b24,b34)
    b = _mm_shuffle_ps(s01, s23, _MM_SHUFFLE(2,0,2,0));
    c = _mm_mul_ps(b, c);
    c = _mm_add_ps(one, c);

    // (b00,b01,b02,b03) = (x-1)*(u0*t0^2-v0,u1*t0^2-v1,u2*t0^2-v2,u3*t0^2-v3)
    b0_0123 = _mm_mul_ps(u0123, sqrT0);
    b0_0123 = _mm_sub_ps(b0_0123, v0123);
    b0_0123 = _mm_mul_ps(b0_0123, xm1);
    // (b10,b11,b12,b13) = (x-1)*(u0*t1^2-v0,u1*t1^2-v1,u2*t1^2-v2,u3*t1^2-v3)
    b1_0123 = _mm_mul_ps(u0123, sqrT1);
    b1_0123 = _mm_sub_ps(b1_0123, v0123);
    b1_0123 = _mm_mul_ps(b1_0123, xm1);
    // (b20,b21,b22,b23) = (x-1)*(u0*t2^2-v0,u1*t2^2-v1,u2*t2^2-v2,u3*t2^2-v3)
    b2_0123 = _mm_mul_ps(u0123, sqrT2);
    b2_0123 = _mm_sub_ps(b2_0123, v0123);
    b2_0123 = _mm_mul_ps(b2_0123, xm1);
    // (b30,b31,b32,b33) = (x-1)*(u0*t3^2-v0,u1*t3^2-v1,u2*t3^2-v2,u3*t3^2-v3)
    b3_0123 = _mm_mul_ps(u0123, sqrT3);
    b3_0123 = _mm_sub_ps(b3_0123, v0123);
    b3_0123 = _mm_mul_ps(b3_0123, xm1);

    // (b03,b03,b13,b13)
    s01 = _mm_shuffle_ps(b0_0123, b1_0123, _MM_SHUFFLE(3,3,3,3));
    // (b23,b23,b33,b33)
    s23 = _mm_shuffle_ps(b2_0123, b3_0123, _MM_SHUFFLE(3,3,3,3));
    // (b03,b13,b23,b33)
    b = _mm_shuffle_ps(s01, s23, _MM_SHUFFLE(2,0,2,0));
    c = _mm_mul_ps(b, c);
    c = _mm_add_ps(one, c);
    // (b02,b02,b12,b12)
    s01 = _mm_shuffle_ps(b0_0123, b1_0123, _MM_SHUFFLE(2,2,2,2));
    // (b22,b22,b32,b32)
    s23 = _mm_shuffle_ps(b2_0123, b3_0123, _MM_SHUFFLE(2,2,2,2));
    // (b02,b12,b22,b32)
    b = _mm_shuffle_ps(s01, s23, _MM_SHUFFLE(2,0,2,0));
    c = _mm_mul_ps(b, c);
    c = _mm_add_ps(one, c);
    // (b01,b01,b11,b11)
    s01 = _mm_shuffle_ps(b0_0123, b1_0123, _MM_SHUFFLE(1,1,1,1));
    // (b21,b21,b31,b31)
    s23 = _mm_shuffle_ps(b2_0123, b3_0123, _MM_SHUFFLE(1,1,1,1));
    // (b01,b11,b21,b31)
    b = _mm_shuffle_ps(s01, s23, _MM_SHUFFLE(2,0,2,0));
    c = _mm_mul_ps(b, c);
    c = _mm_add_ps(one, c);
    // (b00,b00,b10,b10)
    s01 = _mm_shuffle_ps(b0_0123, b1_0123, _MM_SHUFFLE(0,0,0,0));
    // (b20,b20,b30,b30)
    s23 = _mm_shuffle_ps(b2_0123, b3_0123, _MM_SHUFFLE(0,0,0,0));
    // (b00,b10,b20,b30)
    b = _mm_shuffle_ps(s01, s23, _MM_SHUFFLE(2,0,2,0));
    c = _mm_mul_ps(b, c);
    c = _mm_add_ps(one, c);

    // (t0,t0,t1,t1)
    s01 = _mm_shuffle_ps(t[0], t[1], _MM_SHUFFLE(0,0,0,0));
    // (t2,t2,t3,t3)
    s23 = _mm_shuffle_ps(t[2], t[3], _MM_SHUFFLE(0,0,0,0));
    // (t0,t1,t2,t3)
    b = _mm_shuffle_ps(s01, s23, _MM_SHUFFLE(2,0,2,0));
    c = _mm_mul_ps(b, c);

    return c;
}
//----------------------------------------------------------------------------
void SlerpSSE4 (float t[4], const __m128 q0, const __m128 q1, __m128 slerp[4])
{
    __m128 x = Dot(q0, q1);
    __m128 sign = _mm_and_ps(_mm_set1_ps(-0.0f), x);
    x = _mm_xor_ps(sign, x);
    __m128 localQ1 = _mm_xor_ps(sign, q1);
    __m128 xm1 = _mm_sub_ps(x, one);

    __m128 splatT[4], splatD[4];
    splatT[0] = _mm_set1_ps(t[0]);
    splatT[1] = _mm_set1_ps(t[1]);
    splatT[2] = _mm_set1_ps(t[2]);
    splatT[3] = _mm_set1_ps(t[3]);
    splatD[0] = _mm_sub_ps(one, splatT[0]);
    splatD[1] = _mm_sub_ps(one, splatT[1]);
    splatD[2] = _mm_sub_ps(one, splatT[2]);
    splatD[3] = _mm_sub_ps(one, splatT[3]);

    // cT = (cT0,cT1,cT2,cT3), cD = (cD0,cD1,cD2,cD3)
    __m128 cT = Coefficient4(splatT, xm1);
    __m128 cD = Coefficient4(splatD, xm1);

    __m128 cT0 = _mm_shuffle_ps(cT, cT, _MM_SHUFFLE(0,0,0,0));
    __m128 cT1 = _mm_shuffle_ps(cT, cT, _MM_SHUFFLE(1,1,1,1));
    __m128 cT2 = _mm_shuffle_ps(cT, cT, _MM_SHUFFLE(2,2,2,2));
    __m128 cT3 = _mm_shuffle_ps(cT, cT, _MM_SHUFFLE(3,3,3,3));
    __m128 cD0 = _mm_shuffle_ps(cD, cD, _MM_SHUFFLE(0,0,0,0));
    __m128 cD1 = _mm_shuffle_ps(cD, cD, _MM_SHUFFLE(1,1,1,1));
    __m128 cD2 = _mm_shuffle_ps(cD, cD, _MM_SHUFFLE(2,2,2,2));
    __m128 cD3 = _mm_shuffle_ps(cD, cD, _MM_SHUFFLE(3,3,3,3));
    cT0 = _mm_mul_ps(cT0, localQ1);
    cT1 = _mm_mul_ps(cT1, localQ1);
    cT2 = _mm_mul_ps(cT2, localQ1);
    cT3 = _mm_mul_ps(cT3, localQ1);
    cD0 = _mm_mul_ps(cD0, q0);
    cD1 = _mm_mul_ps(cD1, q0);
    cD2 = _mm_mul_ps(cD2, q0);
    cD3 = _mm_mul_ps(cD3, q0);

    slerp[0] = _mm_add_ps(cT0, cD0);
    slerp[1] = _mm_add_ps(cT1, cD1);
    slerp[2] = _mm_add_ps(cT2, cD2);
    slerp[3] = _mm_add_ps(cT3, cD3);
}
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// Support for the Remez Algorithm.
//----------------------------------------------------------------------------
const double halfPi = 4.0*atan(1.0);
//----------------------------------------------------------------------------
double G (const int n, double t)
{
    double g = sin(halfPi*t) - t;
    double a = t, sign = 1.0;
    for (int i = 1; i <= n; ++i, sign = -sign)
    {
        a = a*(t*t - i*i)/(i*(2.0*i+1.0));
        g += sign*a;
    }
    return g;
}
//----------------------------------------------------------------------------
double GDer (const int n, double t)
{
    double gder = halfPi*cos(halfPi*t) - 1.0;
    double a = t, ader = 1.0, sign = 1.0;
    for (int i = 1; i <= n; ++i, sign = -sign)
    {
        ader = ader*(t*t - i*i)/(i*(2.0*i+1.0)) + a*2.0*t/(i*(2.0*i+1.0));
        a = a*(t*t - i*i)/(i*(2.0*i+1.0));
        gder += sign*ader;
    }
    return gder;
}
//----------------------------------------------------------------------------
double P (const int n, double t)
{
    double a = t, sign = 1.0;
    for (int i = 1; i <= n; ++i, sign = -sign)
    {
        a = a*(t*t - i*i)/(i*(2.0*i+1.0));
    }
    double p = sign*a;
    return p;
}
//----------------------------------------------------------------------------
double PDer (const int n, double t)
{
    double a = t, ader = 1.0, sign = 1.0;
    for (int i = 1; i <= n; ++i, sign = -sign)
    {
        ader = ader*(t*t - i*i)/(i*(2.0*i+1.0)) + a*2.0*t/(i*(2.0*i+1.0));
        a = a*(t*t - i*i)/(i*(2.0*i+1.0));
    }
    double pder = sign*ader;
    return pder;
}
//----------------------------------------------------------------------------
void RemezAlgorithm (const int n, double& u, double& e, double& t0,
    double& t1)
{
    double g0, g1, p0, p1;
    double tmin, tmax, fmin, fmax, tmid = 0.0, fmid = 0.0;
    int i, j;
    std::set<Solution> visited;
    Solution solution;

    t0 = 0.25;
    t1 = 0.75;
    for (i = 0; i < 128; ++i)
    {
        g0 = G(n, t0);
        g1 = G(n, t1);
        p0 = P(n, t0);
        p1 = P(n, t1);

        solution.u = (g0 + g1)/(p0 + p1);
        solution.e = (g0*p1 - g1*p0)/(p0 + p1);

        tmin = 0.0;
        tmax = 0.5;
        fmin = GDer(n, tmin) - solution.u*PDer(n, tmin);  // positive
        fmax = GDer(n, tmax) - solution.u*PDer(n, tmax);  // negative
        for (j = 0; j < 64; ++j)
        {
            tmid = 0.5*(tmin + tmax);
            if (tmid == tmin || tmid == tmax)
            {
                break;
            }
            fmid = GDer(n, tmid) - solution.u*PDer(n, tmid);
            if (fmid > 0.0)
            {
                fmin = fmid;
                tmin = tmid;
            }
            else if (fmid < 0.0)
            {
                fmax = fmid;
                tmax = tmid;
            }
            else
            {
                break;
            }
        }
        solution.esum = G(n, tmid) - solution.u*P(n, tmid);
        t0 = tmid;

        tmin = 0.5;
        tmax = 1.0;
        fmin = GDer(n, tmin) - solution.u*PDer(n, tmin);  // negative
        fmax = GDer(n, tmax) - solution.u*PDer(n, tmax);  // positive
        for (j = 0; j < 64; ++j)
        {
            tmid = 0.5*(tmin + tmax);
            if (tmid == tmin || tmid == tmax)
            {
                break;
            }
            fmid = GDer(n, tmid) - solution.u*PDer(n, tmid);
            if (fmid < 0.0)
            {
                fmin = fmid;
                tmin = tmid;
            }
            else if (fmid > 0.0)
            {
                fmax = fmid;
                tmax = tmid;
            }
            else
            {
                break;
            }
        }
        solution.esum += G(n, tmid) - solution.u*P(n, tmid);
        solution.esum = fabs(solution.esum);
        t1 = tmid;

        if (visited.find(solution) != visited.end())
        {
            break;
        }
        visited.insert(solution);
    }

    solution = *visited.begin();
    u = solution.u;
    e = solution.e;
}
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// A simple unit-test and performance program.
//----------------------------------------------------------------------------
float SymmetricRandom ()
{
    return 2.0f*((float)rand()/(float)RAND_MAX) - 1.0f;
}
//----------------------------------------------------------------------------
float UnitRandom ()
{
    return (float)rand()/(float)RAND_MAX;
}
//----------------------------------------------------------------------------
int main (int, char**)
{
    // Compute the parameters for the SLERP approximations.  The results are
    // listed in Table 4.1 of the paper.
    double u, e, t0, t1;
    for (int n = 1; n <= 16; ++n)
    {
        RemezAlgorithm(n, u, e, t0, t1);
    }

    // Generate two random unit-length quaternions.
    FTuple4 q0, q1;
    int i;
    for (i = 0; i < 4; ++i)
    {
        q0[i] = SymmetricRandom();
        q1[i] = SymmetricRandom();
    }
    q0.Normalize();
    q1.Normalize();

    // Generate four random times in [0,1].
    float t[4] = { UnitRandom(), UnitRandom(), UnitRandom(), UnitRandom() };

    FTuple4 slerpSTD[4], slerpFPU[4];
    __m128 sseQ0, sseQ1, sseSlerp1[1], sseSlerp2[2], sseSlerp4[4];

    // Test the code for quaternions that form an acute angle.
    if (q0.Dot(q1) < 0.0f)
    {
        q1 = -q1;
    }
    sseQ0 = _mm_setr_ps(q0[0], q0[1], q0[2], q0[3]);
    sseQ1 = _mm_setr_ps(q1[0], q1[1], q1[2], q1[3]);

    slerpSTD[0] = Slerp(t[0], q0, q1);
    slerpSTD[1] = Slerp(t[1], q0, q1);
    slerpSTD[2] = Slerp(t[2], q0, q1);
    slerpSTD[3] = Slerp(t[3], q0, q1);
    slerpFPU[0] = SlerpFPU(t[0], q0, q1);
    slerpFPU[1] = SlerpFPU(t[1], q0, q1);
    slerpFPU[2] = SlerpFPU(t[2], q0, q1);
    slerpFPU[3] = SlerpFPU(t[3], q0, q1);
    SlerpSSE1(t, sseQ0, sseQ1, sseSlerp1);
    SlerpSSE2(t, sseQ0, sseQ1, sseSlerp2);
    SlerpSSE4(t, sseQ0, sseQ1, sseSlerp4);

    // Test the code for quaternions that form an obtuse angle.
    q1 = -q1;
    sseQ1 = _mm_setr_ps(q1[0], q1[1], q1[2], q1[3]);

    slerpSTD[0] = Slerp(t[0], q0, q1);
    slerpSTD[1] = Slerp(t[1], q0, q1);
    slerpSTD[2] = Slerp(t[2], q0, q1);
    slerpSTD[3] = Slerp(t[3], q0, q1);
    slerpFPU[0] = SlerpFPU(t[0], q0, q1);
    slerpFPU[1] = SlerpFPU(t[1], q0, q1);
    slerpFPU[2] = SlerpFPU(t[2], q0, q1);
    slerpFPU[3] = SlerpFPU(t[3], q0, q1);
    SlerpSSE1(t, sseQ0, sseQ1, sseSlerp1);
    SlerpSSE2(t, sseQ0, sseQ1, sseSlerp2);
    SlerpSSE4(t, sseQ0, sseQ1, sseSlerp4);

    // Time the functions.  The writing of one of the SLERP components to
    // disk prevents the smart optimizing compiler from removing the loop
    // execution code.
    const int numIterations = (1 << 28);
    clock_t start, final, total;
    FILE* outFile = fopen("performance.txt", "wt");

    // The standard SLERP.
    start = clock();
    for (i = 0; i < numIterations; ++i)
    {
        slerpSTD[0] = Slerp(t[0], q0, q1);
    }
    final = clock();
    total = final - start;
    fprintf(outFile, "Slerp time = %d , dummy = %f\n",
        total, slerpSTD[0][0]);

    // The FPU SLERP.
    start = clock();
    for (i = 0; i < numIterations; ++i)
    {
        slerpFPU[0] = SlerpFPU(t[0], q0, q1);
    }
    final = clock();
    total = final - start;
    fprintf(outFile, "SlerpFPU time = %d , dummy = %f\n",
        total, slerpFPU[0][0]);

    // SIMD SLERP for 1 output.
    start = clock();
    for (i = 0; i < numIterations; ++i)
    {
        SlerpSSE1(t, sseQ0, sseQ1, sseSlerp1);
    }
    final = clock();
    total = final - start;
    fprintf(outFile, "SlerpSSE1 time = %d , dummy = %f\n",
        total, sseSlerp1[0].m128_f32[0]);

    // SIMD SLERP for 2 outputs.
    start = clock();
    for (i = 0; i < numIterations; ++i)
    {
        SlerpSSE2(t, sseQ0, sseQ1, sseSlerp2);
    }
    final = clock();
    total = final - start;
    fprintf(outFile, "SlerpSSE2 time = %d , dummy = %f\n",
        total, sseSlerp2[0].m128_f32[0]);

    // SIMD SLERP for 4 outputs.
    start = clock();
    for (i = 0; i < numIterations; ++i)
    {
        SlerpSSE4(t, sseQ0, sseQ1, sseSlerp4);
    }
    final = clock();
    total = final - start;
    fprintf(outFile, "SlerpSSE4 time = %d , dummy = %f\n",
        total, sseSlerp4[0].m128_f32[0]);

    fclose(outFile);
    return 0;
}
//----------------------------------------------------------------------------
