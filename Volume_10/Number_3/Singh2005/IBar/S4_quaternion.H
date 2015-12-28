// -*-c++-*-
#ifndef IG_QUATERN_H
#define IG_QUATERN_H

// $Id: S4_quaternion.H,v 1.1 2004/12/22 02:19:25 cmg Exp $

// CwMtx matrix and vector math library
// Copyright (C) 1999-2001  Harry Kuiper
// Copyright (C) 2000  Will DeVore (template conversion)
// Copyright (C) 2000  Jiri Ecer (constructor from exponential form)

// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA

// 5/25/2002 - Made changes to default includes.
// Added routines from another quaternion class
// Replaced space vector with R3VectorTC --Cindy Grimm

#include "WINSystemDefines.H"

#include "Rn_Defs.H"

namespace CwMtx
{

  // prefix qtn
  template < class T = double >
  class CWTQuaternion : public R4VectorTC<T>
  {
  public:
	///
    CWTQuaternion(): R4VectorTC<T>() {};

    /// assumption: vec has four elements
    CWTQuaternion(const R4VectorTC<T> &vec): R4VectorTC<T>(vec) {};
    ///
    CWTQuaternion(const CWTQuaternion &qtn): R4VectorTC<T>(qtn) {};

    /// the space vector will become the quaternion's imaginary part, T
    /// will become its real part
    CWTQuaternion(const R3VectorTC<T> &, const T & = 0);

    WINbool SetFromFrame( const R3VectorTC<T> & in_vecFromZ, const R3VectorTC<T> & in_vecFromX,
                          const R3VectorTC<T> & in_vecToZ, const R3VectorTC<T> & in_vecToX );

    /// creates a quaternion, index runs from left to right
    CWTQuaternion(const T &, const T &, const T &, const T &);

    /// constructor from exponential form: q = r*exp(svec*angle), svec
    /// should be a unity vector, angle is in radians
    CWTQuaternion(const T &r, const R3VectorTC<T> &svec, const T &angle);

    ~CWTQuaternion() {};

    ///
    void Dimension() { R4VectorTC<T>::Dim(); };

    ///
    R4Matrix RotationMatrix( ) const;
    R4Matrix RotationMatrix2( ) const;

    ///
    CWTQuaternion operator +(const CWTQuaternion &) const;
    ///
    CWTQuaternion operator -(const CWTQuaternion &) const;
    ///
    CWTQuaternion operator -() const;
    ///
    CWTQuaternion operator *(const T &) const;
    ///
    R3VectorTC<T> operator *(const R3VectorTC<T> &) const;
    ///
    CWTQuaternion operator *(const CWTQuaternion &) const;
    ///
    CWTQuaternion operator /(const T &value) const
    {
      return (R4VectorTC<T>) (*this) * (1 / value);
    }
    ///
    CWTQuaternion operator /(const CWTQuaternion &) const;

    // not inherited
    CWTQuaternion & operator =(const CWTQuaternion &);
    CWTQuaternion & operator =(const R3VectorTC<T> &);
    CWTQuaternion & operator +=(const CWTQuaternion &);
    CWTQuaternion & operator -=(const CWTQuaternion &);
    CWTQuaternion & operator *=(const T &);
    CWTQuaternion & operator *=(const CWTQuaternion &);
    CWTQuaternion & operator /=(const T &);
    CWTQuaternion & operator /=(const CWTQuaternion &);

    // stores product of qtn*qtn in this
    void StoreProduct(const CWTQuaternion &, const CWTQuaternion &);
    // stores conjugate of argument in this
    void StoreConjugate(const CWTQuaternion &);
    // makes this its own conjugate
    void MakeConjugate();

    // Inverse rotation
    CWTQuaternion Inverse() const { return CWTQuaternion<T>( Axis() * -1.0, Angle() ); }
    // returns a unit vector with same direction as this
    CWTQuaternion Unit() const { return (*this) / ::Length(*this); }

    ///
    T Angle() const { return (*this)[3]; }
    ///
    T X() const { return (*this)[0]; }
    ///
    T Y() const { return (*this)[1]; }
    ///
    T Z() const { return (*this)[2]; }
    ///
    R3VectorTC<T> Axis() const { return R3VectorTC<T>( X(), Y(), Z() ); }
  };

  // Unity quaternion.
  //template <class T>
  //class CWTUnity< CWTQuaternion<T> >: public CWTQuaternion<T>
  //{
  //public:
  //  CWTUnity()
  //  {
  //    (*this)[0] = (*this)[1] = (*this)[2] = CWTZero<T>();
  //    (*this)[3] = CWTUnity<T>();
  //  }
  //};

  // Zero quaternion.
  //template <class T>
  //class CWTZero< CWTQuaternion<T> >: public CWTQuaternion<T>
  //{
  //public:
  //  CWTZero() { Fill(CWTZero<T>()); }
  //};

  //
  // Constructors
  //
  template < class T >
  inline CWTQuaternion<T>::CWTQuaternion(const T &elemIm0,
					 const T &elemIm1,
					 const T &elemIm2,
					 const T &elemReal)
    :
    R4VectorTC<T>()
  {
    (*this)[0] = elemIm0;
    (*this)[1] = elemIm1;
    (*this)[2] = elemIm2;
    (*this)[3] = elemReal;
  }

  template < class T >
  inline CWTQuaternion<T>::CWTQuaternion(const R3VectorTC<T> &svecIm,
					 const T &elemReal)
    :
    R4VectorTC<T>()
  {
    (*this)[0] = svecIm[0];
    (*this)[1] = svecIm[1];
    (*this)[2] = svecIm[2];
    (*this)[3] = elemReal;
  }

  // NOTE: This only works with template arguments that can be
  // converted to floating point types due to the use of sin(3) and
  // cos(3).
  template < class T >
  CWTQuaternion<T>::CWTQuaternion(const T &r,
				  const R3VectorTC<T> &svec,
				  const T &angle)
    :
    R4VectorTC<T>()
  {
    T rsina = r*sin(angle / 2.0);

    (*this)[0] = svec[0]*rsina;
    (*this)[1] = svec[1]*rsina;
    (*this)[2] = svec[2]*rsina;
    (*this)[3] = r*cos(angle / 2.0);
  }

  //
  // User Methods
  //

  ///
  template < class T >
  inline R4Matrix CWTQuaternion<T>::RotationMatrix( ) const
  {
    const double qw = Angle();
    const double qx = X();
    const double qy = Y();
    const double qz = Z();

    const double qwqw = qw * qw;
    const double qxqx = qx * qx;
    const double qyqy = qy * qy;
    const double qzqz = qz * qz;

    const double qLenSq = qwqw + qxqx + qyqy + qzqz;

    R4Matrix mat;

    mat(0,0) = 2.0 * (qLenSq / 2.0 - qyqy - qzqz) / qLenSq;
    mat(0,1) = (2.0 * (qx * qy + qw * qz)) / qLenSq;
    mat(0,2) = (2.0 * (qx * qz - qw * qy)) / qLenSq;
    mat(0,3) = 0.0;
    mat(1,0) = (2.0 * (qx * qy - qw * qz)) / qLenSq;
    mat(1,1) = (qLenSq - 2.0 * qxqx - 2.0 * qzqz) / qLenSq;
    mat(1,2) = (2.0 * (qw * qx + qy * qz)) / qLenSq;
    mat(1,3) = 0.0;
    mat(2,0) = (2.0 * (qw * qy + qx * qz)) / qLenSq;
    mat(2,1) = (2.0 * (qy * qz - qw * qx)) / qLenSq;
    mat(2,2) = (qLenSq - 2.0 * qxqx - 2.0 * qyqy) / qLenSq;
    mat(2,3) = 0.0;
    mat(3,0) = 0.0;
    mat(3,1) = 0.0;
    mat(3,2) = 0.0;
    mat(3,3) = qLenSq;

    return mat;
  }
  template < class T >
  inline R4Matrix CWTQuaternion<T>::RotationMatrix2( ) const
  {
    const double qLen = sqrt( Angle() * Angle() + X() * X() + Y() * Y() + Z() * Z() );
    const double qw = Angle() / qLen;
    const double qx = X() / qLen;
    const double qy = Y() / qLen;
    const double qz = Z() / qLen;

    const double qwqw = qw * qw;
    const double qxqx = qx * qx;
    const double qyqy = qy * qy;
    const double qzqz = qz * qz;


    R4Matrix mat;

    mat(0,0) = 2.0 * (1 / 2.0 - qyqy - qzqz);
    mat(0,1) = (2.0 * (qx * qy + qw * qz));
    mat(0,2) = (2.0 * (qx * qz - qw * qy));
    mat(0,3) = 0.0;
    mat(1,0) = (2.0 * (qx * qy - qw * qz));
    mat(1,1) = (1 - 2.0 * qxqx - 2.0 * qzqz);
    mat(1,2) = (2.0 * (qw * qx + qy * qz));
    mat(1,3) = 0.0;
    mat(2,0) = (2.0 * (qw * qy + qx * qz));
    mat(2,1) = (2.0 * (qy * qz - qw * qx));
    mat(2,2) = (1 - 2.0 * qxqx - 2.0 * qyqy);
    mat(2,3) = 0.0;
    mat(3,0) = 0.0;
    mat(3,1) = 0.0;
    mat(3,2) = 0.0;
    mat(3,3) = (1 / 2.0);
    return mat;
                  
  }

  // not inherited
  template < class T >
  inline CWTQuaternion<T> &
  CWTQuaternion<T>::operator =(const CWTQuaternion<T> &qtn)
  {
      (*this).R4VectorTC<T>::operator=( qtn );
      return *this;
  }

  template < class T >
  inline CWTQuaternion<T> &
  CWTQuaternion<T>::operator =(const R3VectorTC<T> &svec)
  {
    (*this)[0] = svec[0];
    (*this)[1] = svec[1];
    (*this)[2] = svec[2];
    (*this)[3] = 0;

    return *this;
  }

  template < class T >
  inline CWTQuaternion<T> &
  CWTQuaternion<T>::operator +=(const CWTQuaternion<T> &qtn)
  {
      R4VectorTC<T>::operator+=( qtn );
      return *this;
  }

  template < class T >
  inline CWTQuaternion<T> &
  CWTQuaternion<T>::operator -=(const CWTQuaternion<T> &qtn)
  {
      return R4VectorTC<T>::operator-=( qtn );

  }

  template < class T >
  inline CWTQuaternion<T> &
  CWTQuaternion<T>::operator *=(const T &value)
  {
      return R4VectorTC<T>::operator*=( value );
  }

  template < class T >
  inline CWTQuaternion<T> &
  CWTQuaternion<T>::operator *=(const CWTQuaternion<T> &qtn)
  {
    return R4VectorTC<T>::operator*=( qtn );
  }

  template < class T >
  inline CWTQuaternion<T> & CWTQuaternion<T>::operator /=(const T &value)
  {
      (*this) = (*this) / value;
      return *this;
  }

  template < class T >
  inline CWTQuaternion<T> &
  CWTQuaternion<T>::operator /=(const CWTQuaternion<T> &qtn)
  {
    return (*this) = (*this)/qtn;
  }

  template < class T >
  CWTQuaternion<T>
  CWTQuaternion<T>::operator +(const CWTQuaternion<T> &qtn) const
  {
    return CWTQuaternion(*this) += qtn;
  }

  template < class T >
  CWTQuaternion<T>
  CWTQuaternion<T>::operator -(const CWTQuaternion<T> &qtn) const
  {
    return (R4VectorTC<T>) (*this) - (R4VectorTC<T>) (qtn);
  }

  template < class T >
  CWTQuaternion<T> CWTQuaternion<T>::operator -() const
  {
      return R4VectorTC<T>::operator*( -1.0);
  }

  template < class T >
  CWTQuaternion<T> CWTQuaternion<T>::operator *(const T &value) const
  {
      return ::operator*( (R4VectorTC<T>) *this, value );
  }

  template < class T >
  R3VectorTC<T> CWTQuaternion<T>::operator *(const R3VectorTC<T> &in_vec) const
  {
      const CWTQuaternion<T> quat(in_vec[0], in_vec[1], in_vec[2], 0);
      
      const CWTQuaternion<T> quatConj = conj( (*this) );

      const CWTQuaternion<T> quatRes = quatConj * quat * (*this);

      const double d1 = quatRes[3];
      const double d2 = quatRes[0];
      const double d3 = quatRes[1];
      const double d4 = quatRes[2];


      return R3VectorTC<T>( quatRes[0], quatRes[1], quatRes[2] );
  }

  template < class T >
  void CWTQuaternion<T>::StoreProduct(const CWTQuaternion<T> &qtnLeft,
				      const CWTQuaternion<T> &qtnRight)
  {
    // to reduce the number of calls to the subscript operator
    T qtnLeft0 = qtnLeft[0];
    T qtnLeft1 = qtnLeft[1];
    T qtnLeft2 = qtnLeft[2];
    T qtnLeft3 = qtnLeft[3];

    T qtnRight0 = qtnRight[0];
    T qtnRight1 = qtnRight[1];
    T qtnRight2 = qtnRight[2];
    T qtnRight3 = qtnRight[3];

    (*this)[0] =
      qtnLeft0*qtnRight3 + qtnLeft1*qtnRight2
      - qtnLeft2*qtnRight1 + qtnLeft3*qtnRight0;

    (*this)[1] =
      -qtnLeft0*qtnRight2 + qtnLeft1*qtnRight3
      + qtnLeft2*qtnRight0 + qtnLeft3*qtnRight1;

    (*this)[2] =
      qtnLeft0*qtnRight1 - qtnLeft1*qtnRight0
      + qtnLeft2*qtnRight3 + qtnLeft3*qtnRight2;

    (*this)[3] =
      -qtnLeft0*qtnRight0 - qtnLeft1*qtnRight1
      - qtnLeft2*qtnRight2 + qtnLeft3*qtnRight3;
  }

  template < class T >
  CWTQuaternion<T>
  CWTQuaternion<T>::operator *(const CWTQuaternion<T> &qtn) const
  {
    CWTQuaternion qtnResult;
    qtnResult.StoreProduct(*this, qtn);
    return qtnResult;
  }

  template < class T >
  CWTQuaternion<T>
  CWTQuaternion<T>::operator /(const CWTQuaternion<T> &qtn) const
  {
    return (*this)*inv(qtn);
  }

  // stores conjugate of argument in this
  template < class T >
  void CWTQuaternion<T>::StoreConjugate(const CWTQuaternion<T> &qtn)
  {
    // copy argument into this
    (*this) = qtn;
    MakeConjugate();
  }

  template < class T >
  void CWTQuaternion<T>::MakeConjugate()
  {
    T tmp = -1.0;//static_cast<const T &>( -1);

    (*this)[0] *= tmp;
    (*this)[1] *= tmp;
    (*this)[2] *= tmp;
  }

  //
  // template functions designed work with the Quaternion class.
  //

  template < class T >
  inline CWTQuaternion<T> operator *(const T &value,
				     const CWTQuaternion<T> &qtn)
  {
    return qtn*value;
  }

  // return real part of a quaternion
  template < class T >
  inline T re(const CWTQuaternion<T> &qtn)
  {
    return qtn[3];
  }

  // returns imaginary (vector) part of a quaternion
  template < class T >
  R3VectorTC<T> im(const CWTQuaternion<T> &qtn)
  {
    return R3VectorTC<T>(qtn[0], qtn[1], qtn[2]);
  }

  // the conjugate
  template < class T >
  CWTQuaternion<T> conj(const CWTQuaternion<T> &qtn)
  {
    // copy input quaternion
    CWTQuaternion<T> qtnResult(qtn);
    qtnResult.MakeConjugate();
    return qtnResult;
  }

  // the inverse
  template < class T >
  CWTQuaternion<T> inv(const CWTQuaternion<T> &qtn)
  {
    T qtn0 = qtn[0];
    T qtn1 = qtn[1];
    T qtn2 = qtn[2];
    T qtn3 = qtn[3];
    return
      conj(qtn)
      /static_cast<const T &>(qtn0*qtn0 + qtn1*qtn1 + qtn2*qtn2 + qtn3*qtn3);
  }

  template < class T >
  WINbool CWTQuaternion<T>::SetFromFrame( const R3VectorTC<T> & in_vecFromZ, const R3VectorTC<T> & in_vecFromX,
                                          const R3VectorTC<T> & in_vecToZ, const R3VectorTC<T> & in_vecToX )
  {
      if ( RNIsZero( ::Length( in_vecFromX ) ) ) return FALSE;
      if ( RNIsZero( ::Length( in_vecFromZ ) ) ) return FALSE;
      if ( RNIsZero( ::Length( in_vecToX ) ) ) return FALSE;
      if ( RNIsZero( ::Length( in_vecToZ ) ) ) return FALSE;

      R3VectorTC<T> vecFromX = UnitSafe( in_vecFromX );
      R3VectorTC<T> vecFromZ = UnitSafe( in_vecFromZ );

      R3VectorTC<T> vecToX = UnitSafe( in_vecToX );
      R3VectorTC<T> vecToZ = UnitSafe( in_vecToZ );

      if ( RNIsZero( 1.0 - fabs( Dot( vecFromX, vecFromZ ) ) ) ) return FALSE;
      if ( RNIsZero( 1.0 - fabs( Dot( vecToX,   vecToZ ) ) ) ) return FALSE;


      R3VectorTC<T> vecFromY = Cross( vecFromZ, vecFromX );
      R3VectorTC<T> vecToY = Cross( vecToZ, vecToX );

    CWTQuaternion<T> q1;
    CWTQuaternion<T> q2;
    
    const double dZ1MY2 = vecFromZ[1] - vecFromY[2];
    const double dZ0MX0 = vecFromZ[0] - vecFromX[0];
    const double dZ0MX2 = vecFromZ[0] - vecFromX[2];
    const double dX1MY0 = vecFromX[1] - vecFromY[0];
    const double dX1Y0 = vecFromX[1] + vecFromY[0];
    const double dZ1Y2 = vecFromZ[1] + vecFromY[2];
    const double dZ0X2 = vecFromZ[0] + vecFromX[2];

    if ( RNIsZero( dX1MY0 ) ) {
        ASSERT( RNIsZero( dZ1MY2 ) || RNIsZero( dZ0MX0 ) );
        q1[0] = 0.0;
    } else {
        q1[0] = sqrt( dZ1MY2 * dZ0MX0 / (4.0 * dX1MY0) );
    }

    if ( RNIsZero( dZ1Y2 ) ) {
        ASSERT( RNIsZero( dX1Y0 ) || RNIsZero( dZ0X2 ) );
        q1[2] = 0.0;
    } else {
        q1[0] = sqrt( dX1Y0 * dZ0X2 / (4.0 * dZ1Y2) );
    }

    if ( RNIsZero( dX1Y0 ) ) {
        ASSERT( RNIsZero( dZ1Y2 ) || RNIsZero( dZ0MX2 ) );
        q1[2] = 0.0;
    } else {
        q1[2] = sqrt( dZ1Y2 * dZ0MX2 / (4.0 * dX1MY0) );
    }

    if ( RNIsZero( dZ1MY2 ) ) {
        ASSERT( RNIsZero( dZ0X2 ) || RNIsZero( dX1MY0 ) );
        q1[3] = 0.0;
    } else {
        q1[3] = sqrt( dZ0X2 * dX1MY0 ) / (4.0 * dZ1MY2);
    }

    const R3MatrixTC<T> mat = RotationMatrix();
    R3MatrixTC<T> m_mat;

    for (int i = 0; i < 3; i++) {
        m_mat(i,0) = vecFromX[i];
        m_mat(i,1) = vecFromY[i];
        m_mat(i,2) = vecFromZ[i];

        //f2.m_mat[i][0] = in_vToX[i];
        //f2.m_mat[i][1] = in_vToY[i];
        //f2.m_mat[i][2] = vToZ[i];
    }
    //WINbool bSuc = FALSE;
    //out_mat = f2 * f1.Inverse( bSuc);

    return TRUE;
  }

  typedef CWTQuaternion<double> S4Quaternion;
}

#endif // IG_QUATERN_H
