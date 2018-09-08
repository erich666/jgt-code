// David Eberly, Geometric Tools, Redmond WA 98052
// Copyright (c) 1998-2018
// Distributed under the Boost Software License, Version 1.0.
// http://www.boost.org/LICENSE_1_0.txt
// http://www.geometrictools.com/License/Boost/LICENSE_1_0.txt
// File Version: 3.0.0 (2016/06/19)

#pragma once

#include <Mathematics/GteVector4.h>
#include <Mathematics/GteMatrix.h>
#include <Mathematics/GteChebyshevRatio.h>

namespace gte
{

template <typename Real>
class Quaternion : public Vector<4,Real>
{
public:
    // The quaternions are of the form q = x*i + y*j + z*k + w.  In tuple
    // form, q = (x,y,z,w).

    // Construction.  The default constructor does not initialize the members.
    Quaternion();
    Quaternion(Quaternion const& q);
    Quaternion(Vector<4,Real> const& q);
    Quaternion(Real x, Real y, Real z, Real w);

    // Assignment.
    Quaternion& operator=(Quaternion const& q);
    Quaternion& operator=(Vector<4,Real> const& q);

    // Special quaternions.
    static Quaternion Zero();      // z = 0*i + 0*j + 0*k + 0
    static Quaternion I();         // i = 1*i + 0*j + 0*k + 0
    static Quaternion J();         // j = 0*i + 1*j + 0*k + 0
    static Quaternion K();         // k = 0*i + 0*j + 1*k + 0
    static Quaternion Identity();   // 1 = 0*i + 0*j + 0*k + 1
};

// Multiplication of quaternions.  This operation is not generally
// commutative; that is, q0*q1 and q1*q0 are not usually the same value.
// (x0*i + y0*j + z0*k + w0)*(x1*i + y1*j + z1*k + w1)
// =
// i*(+x0*w1 + y0*z1 - z0*y1 + w0*x1) +
// j*(-x0*z1 + y0*w1 + z0*x1 + w0*y1) +
// k*(+x0*y1 - y0*x1 + z0*w1 + w0*z1) +
// 1*(-x0*x1 - y0*y1 - z0*z1 + w0*w1)
template <typename Real>
Quaternion<Real> operator*(Quaternion<Real> const& q0,
    Quaternion<Real> const& q1);

// For a nonzero quaternion q = (x,y,z,w), inv(q) = (-x,-y,-z,w)/|q|^2, where
// |q| is the length of the quaternion.  When q is zero, the function returns
// zero, which is considered to be an improbable case.
template <typename Real>
Quaternion<Real> Inverse(Quaternion<Real> const& q);

// The conjugate of q = (x,y,z,w) is conj(q) = (-x,-y,-z,w).
template <typename Real>
Quaternion<Real> Conjugate(Quaternion<Real> const& q);

// Rotate a vector using quaternion multiplication.  The input quaternion must
// be unit length.
template <typename Real>
Vector<4,Real> Rotate(Quaternion<Real> const& q, Vector<4,Real> const& v);

// The spherical linear interpolation (slerp) of unit-length quaternions
// q0 and q1 for t in [0,1] is
//     slerp(t,q0,q1) = [sin(t*theta)*q0 + sin((1-t)*theta)*q1]/sin(theta)
// where theta is the angle between q0 and q1 [cos(theta) = Dot(q0,q1)].
// This function is a parameterization of the great spherical arc between
// q0 and q1 on the unit hypersphere.  Moreover, the parameterization is
// one of normalized arclength--a particle traveling along the arc through
// time t does so with constant speed.
//
// When using slerp in animations involving sequences of quaternions, it is
// typical that the quaternions are preprocessed so that consecutive ones
// form an acute angle A in [0,pi/2].  Other preprocessing can help with
// performance.  See the function comments below.
//
// See GteSlerpEstimate.{h,inl} for various approximations, including
// SLERP<Real>::EstimateRPH that gives good performance and accurate results
// for preprocessed quaternions.

// The angle between q0 and q1 is in [0,pi).  There are no angle restrictions
// restrictions and nothing is precomputed.
template <typename Real>
Quaternion<Real> Slerp(Real t, Quaternion<Real> const& q0,
    Quaternion<Real> const& q1);

// The angle between q0 and q1 must be in [0,pi/2].  The suffix R is for
// 'Restricted'.  The preprocessing code is
//   Quaternion<Real> q[n];  // assuming initialized
//   for (i0 = 0, i1 = 1; i1 < n; i0 = i1++)
//   {
//       cosA = Dot(q[i0], q[i1]);
//       if (cosA < 0)
//       {
//           q[i1] = -q[i1];  // now Dot(q[i0], q[i]1) >= 0
//       }
//   }
template <typename Real>
Quaternion<Real> SlerpR(Real t, Quaternion<Real> const& q0,
    Quaternion<Real> const& q1);

// The angle between q0 and q1 must be in [0,pi/2].  The suffix R is for
// 'Restricted' and the suffix P is for 'Preprocessed'.  The preprocessing
// code is
//   Quaternion<Real> q[n];  // assuming initialized
//   Real cosA[n-1], omcosA[n-1];  // to be precomputed
//   for (i0 = 0, i1 = 1; i1 < n; i0 = i1++)
//   {
//       cs = Dot(q[i0], q[i1]);
//       if (cosA[i0] < 0)
//       {
//           q[i1] = -q[i1];
//           cs = -cs;
//       }
//       cosA[n-1] = cs;  // for GeneralRP
//       omcosA[i0] = 1 - cs;  // for EstimateRP
//   }
template <typename Real>
Quaternion<Real> SlerpRP(Real t, Quaternion<Real> const& q0,
    Quaternion<Real> const& q1, Real cosA);


template <typename Real>
Quaternion<Real>::Quaternion()
{
    // Uninitialized.
}

template <typename Real>
Quaternion<Real>::Quaternion(Quaternion const& q)
{
    this->mTuple[0] = q[0];
    this->mTuple[1] = q[1];
    this->mTuple[2] = q[2];
    this->mTuple[3] = q[3];
}

template <typename Real>
Quaternion<Real>::Quaternion(Vector<4, Real> const& q)
{
    this->mTuple[0] = q[0];
    this->mTuple[1] = q[1];
    this->mTuple[2] = q[2];
    this->mTuple[3] = q[3];
}

template <typename Real>
Quaternion<Real>::Quaternion(Real x, Real y, Real z, Real w)
{
    this->mTuple[0] = x;
    this->mTuple[1] = y;
    this->mTuple[2] = z;
    this->mTuple[3] = w;
}

template <typename Real>
Quaternion<Real>& Quaternion<Real>::operator=(Quaternion const& q)
{
    Vector<4, Real>::operator=(q);
    return *this;
}

template <typename Real>
Quaternion<Real>& Quaternion<Real>::operator=(Vector<4, Real> const& q)
{
    Vector<4, Real>::operator=(q);
    return *this;
}

template <typename Real>
Quaternion<Real> Quaternion<Real>::Zero()
{
    return Quaternion((Real)0, (Real)0, (Real)0, (Real)0);
}

template <typename Real>
Quaternion<Real> Quaternion<Real>::I()
{
    return Quaternion((Real)1, (Real)0, (Real)0, (Real)0);
}

template <typename Real>
Quaternion<Real> Quaternion<Real>::J()
{
    return Quaternion((Real)0, (Real)1, (Real)0, (Real)0);
}

template <typename Real>
Quaternion<Real> Quaternion<Real>::K()
{
    return Quaternion((Real)0, (Real)0, (Real)1, (Real)0);
}

template <typename Real>
Quaternion<Real> Quaternion<Real>::Identity()
{
    return Quaternion((Real)0, (Real)0, (Real)0, (Real)1);
}

template <typename Real>
Quaternion<Real> operator*(Quaternion<Real> const& q0,
    Quaternion<Real> const& q1)
{
    // (x0*i + y0*j + z0*k + w0)*(x1*i + y1*j + z1*k + w1)
    // =
    // i*(+x0*w1 + y0*z1 - z0*y1 + w0*x1) +
    // j*(-x0*z1 + y0*w1 + z0*x1 + w0*y1) +
    // k*(+x0*y1 - y0*x1 + z0*w1 + w0*z1) +
    // 1*(-x0*x1 - y0*y1 - z0*z1 + w0*w1)

    return Quaternion<Real>
        (
        +q0[0] * q1[3] + q0[1] * q1[2] - q0[2] * q1[1] + q0[3] * q1[0],
        -q0[0] * q1[2] + q0[1] * q1[3] + q0[2] * q1[0] + q0[3] * q1[1],
        +q0[0] * q1[1] - q0[1] * q1[0] + q0[2] * q1[3] + q0[3] * q1[2],
        -q0[0] * q1[0] - q0[1] * q1[1] - q0[2] * q1[2] + q0[3] * q1[3]
        );
}

template <typename Real>
Quaternion<Real> Inverse(Quaternion<Real> const& q)
{
    Real sqrLen = Dot(q, q);
    if (sqrLen > (Real)0)
    {
        Real invSqrLen = ((Real)1) / sqrLen;
        Quaternion<Real> inverse = Conjugate(q)*invSqrLen;
        return inverse;
    }
    else
    {
        return Quaternion<Real>::Zero();
    }
}

template <typename Real>
Quaternion<Real> Conjugate(Quaternion<Real> const& q)
{
    return Quaternion<Real>(-q[0], -q[1], -q[2], +q[3]);
}

template <typename Real>
Vector<4, Real> Rotate(Quaternion<Real> const& q, Vector<4, Real> const& v)
{
    Vector<4, Real> u = q*Quaternion<Real>(v)*Conjugate(q);

    // Zero-out the w-component in remove numerical round-off error.
    u[3] = (Real)0;
    return u;
}

template <typename Real>
Quaternion<Real> Slerp(Real t, Quaternion<Real> const& q0,
    Quaternion<Real> const& q1)
{
    Real cosA = Dot(q0, q1);
    Real sign;
    if (cosA >= (Real)0)
    {
        sign = (Real)1;
    }
    else
    {
        cosA = -cosA;
        sign = (Real)-1;
    }

    Real f0, f1;
    ChebyshevRatio<Real>::Get(t, cosA, f0, f1);
    return q0 * f0 + q1 * (sign * f1);
}

template <typename Real>
Quaternion<Real> SlerpR(Real t, Quaternion<Real> const& q0,
    Quaternion<Real> const& q1)
{
    Real f0, f1;
    ChebyshevRatio<Real>::Get(t, Dot(q0, q1), f0, f1);
    return q0 * f0 + q1 * f1;
}

template <typename Real>
Quaternion<Real> SlerpRP(Real t, Quaternion<Real> const& q0,
    Quaternion<Real> const& q1, Real cosA)
{
    Real f0, f1;
    ChebyshevRatio<Real>::Get(t, cosA, f0, f1);
    return q0 * f0 + q1 * f1;
}


}
