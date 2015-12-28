#if _MSC_VER > 1000
#pragma once
#endif
// _MSC_VER > 1000

#ifndef __BASETYPE_AFFINE3_H
#define __BASETYPE_AFFINE3_H

/**@name 3D Points, Vectors, Covectors
<hr>
  Copyright (C) 1999 Brown Univ.  All Rights Reserved. <br>
<br>
  Files: <br>
    <dd><li> include/utils/Rn_Affine3.H 
    <dd><li> include/utils/Rn_Point3_i.H<br>
    <dd><li> include/utils/Rn_Vector3_i.H<br>
    <dd><li> include/utils/Rn_CoVector3_i.H<br>
    <dd><li> include/utils/Rn_Matrix3_i.H<br>
<hr>
<br>
  Content:     3D affine - space classes <br>
  <dd><li>          R3VectorTC   - 3D column vectors <br>
  <dd><li>          R3CoVectorTC - 3D row vectors (normals) <br>
  <dd><li>          R3PointTC    - 3D points <br>
  <dd><li>          R3MatrixTC     - 3D matrix <br>
 <hr>
*/
//@{

#include "WINSystemDefines.H"

#include "Rn_Affine2.H"

template <class Coord> class R3CoVectorTC;
template <class Coord> class R3PointTC;
template <class Coord> class R3MatrixTC;

/// 3D Column vector
template<class Coord>
class R3VectorTC
{
public:
    ///
    R3VectorTC() {}
    ///
    inline R3VectorTC( const Coord& d );
    ///
    inline R3VectorTC( const Coord& dx, const Coord& dy, const Coord& dz );
    ///
    inline R3VectorTC( const R2VectorTC<Coord> & );

    /// Return dimension of the type
    int Dim() const { return 3; }

    /**@name Access grants */
    //@{
    ///
    Coord& operator[] ( const int i ) { return (&dx)[i]; }
    ///
    const Coord& operator[] ( const int i ) const { return (&dx)[i]; }
    //@}

    /**@name Assignment operators */
    //@{
    /// 
    inline R3VectorTC<Coord>& operator += ( const R3VectorTC<Coord>& );
    ///
    inline R3VectorTC<Coord>& operator -= ( const R3VectorTC<Coord>& );
    ///
    inline R3VectorTC<Coord>& operator *= ( const Coord& );
    ///
    inline R3VectorTC<Coord>& operator /= ( const Coord& );
    //@}

    /**@name Self-editing */
    //@{
    /// 
    inline double Normalize();
    //@}

    /**@name Unary operators */
    //@{
    /// 
    inline R3VectorTC<Coord>  operator + () const;
    ///
    inline R3VectorTC<Coord>  operator - () const;
    /// Return a vector that is perpendicular to this one
    inline R3VectorTC<Coord>  Perpendicular() const;
    //@}

    /**@name Read/write/print */
    //@{
    ///
    inline void    Write(ofstream &out) const;
    ///
    inline WINbool Read(ifstream &in) ;
    ///
    inline void    Print( const WINbool in_bDoReturn = FALSE ) const;

    //@}
private:
    ///
    Coord dx, dy, dz;
};

/// 3D Row vector
template<class Coord>
class R3CoVectorTC
{
public:
    ///
    R3CoVectorTC( ) {}
    ///
    R3CoVectorTC( const Coord& d );
    ///
    R3CoVectorTC( const Coord& dx, const Coord& dy, const Coord& dz );
    ///
    R3CoVectorTC( const R2CoVectorTC<Coord> & );

    /// Return dimension of the type
    int Dim() const { return 3; }

    /**@name Access grants */
    //@{
    ///
    Coord& operator[] ( const int i ) { return (&dx)[i]; }
    ///
    const Coord& operator[] ( const int i ) const { return (&dx)[i]; }
    //@}

    /**@name Assignment operators */
    //@{
    /// 
    R3CoVectorTC<Coord>& operator += ( const R3CoVectorTC<Coord>& );
    ///
    R3CoVectorTC<Coord>& operator -= ( const R3CoVectorTC<Coord>& );
    ///
    R3CoVectorTC<Coord>& operator *= ( const Coord& );
    ///
    R3CoVectorTC<Coord>& operator /= ( const Coord& );
    //@}

    /**@name Unary operators */
    //@{
    /// 
    R3CoVectorTC<Coord>  operator + () const;
    ///
    R3CoVectorTC<Coord>  operator - () const;
    //@}


    /**@name Read/write/print */
    //@{
    ///
    inline void    Write(ofstream &out) const;
    ///
    inline WINbool Read(ifstream &in) ;
    ///
    inline void    Print( const WINbool in_bDoReturn = FALSE ) const;

    //@}
private:
    ///
    Coord dx, dy, dz;
};


/// 3D Point
template<class Coord>
class R3PointTC
{
public:
    ///
    R3PointTC() { }
    ///
    inline R3PointTC( const Coord& x, const Coord& y, const Coord& z );
    ///
    inline R3PointTC( const Coord& x);
    ///
    inline R3PointTC( const R3PointTC<Coord>& in_pt);
    ///
    inline R3PointTC( const R2PointTC<Coord>& in_pt);

    /// Return dimension of the type
    int Dim() const { return 3; }

    /**@name Access grants */
    //@{
    ///
    Coord& operator[] ( const int i ) { return (&x)[i]; }
    ///
    const Coord& operator[] ( const int i ) const { return (&x)[i]; }
    //@}

    /**@name Assignment operators */
    //@{
    /// 
    inline R3PointTC<Coord>& operator += ( const R3VectorTC<Coord>& );
    ///
    inline R3PointTC<Coord>& operator -= ( const R3VectorTC<Coord>& );
    //@}


    /**@name Point dominance */
    //@{
    /// 
    inline WINbool operator < ( const R3PointTC<Coord>& ) const;
    ///
    inline WINbool operator <= ( const R3PointTC<Coord>& ) const;
    //@}

    /**@name Read/write/print */
    //@{
    ///
    inline void    Write(ofstream &out) const;
    ///
    inline WINbool Read(ifstream &in) ;
    ///
    inline void    Print( const WINbool in_bDoReturn = FALSE ) const;

    //@}
private:
    ///
    Coord x, y, z;
};


// =========================================================
// convenient typedefs for double-valued points, vectors, covectors
// =========================================================

/// Ray class
template< class Coord >
class R3RayTC
{
private:
    ///
    R3PointTC<Coord> pOrg;
    ///
    R3VectorTC<Coord> vDir;

public:
    /**@name Access grants */
    //@{
    ///
    const R3PointTC<Coord> &Pt() const { return pOrg; }
    ///
    const R3VectorTC<Coord> &Dir() const { return vDir; }
    //@}



    /**@name Assignment operators */
    //@{
    ///
    R3PointTC<Coord> &Pt() { return pOrg; }
    ///
    R3VectorTC<Coord> &Dir() { return vDir; }
    ///
    R3PointTC<Coord> operator()( const double in_dT ) const { return pOrg + vDir * in_dT; }
    //@}

    /**@name Constructor/destructor */
    //@{
    ///
    R3RayTC() { }
    ///
    R3RayTC( const R3PointTC<Coord> & _pOrg, const R3VectorTC<Coord>& _vDir )
        { pOrg = _pOrg; vDir = _vDir; }
    ///
    ~R3RayTC() {}
    //@}

    /**@name Read/write/print */
    //@{
    ///Write out in format: Pt Dir
    inline void    Write(ofstream &out) const { pOrg.Write(out); vDir.Write(out); }
    ///Read in format: Pt Dir
    inline WINbool Read(ifstream &in) { pOrg.Read(in); return vDir.Read(in); }
    ///
    inline void    Print( const WINbool in_bDoReturn = FALSE ) const { pOrg.Print( in_bDoReturn ); vDir.Print( in_bDoReturn ); }
    //@}

};

/// R3MatrixTC: 3D linear transform class
template< class Coord >
class R3MatrixTC
{
public:
    ///
    R3MatrixTC() {}
    ///
    inline R3MatrixTC( const R3VectorTC<Coord>& vCol1, const R3VectorTC<Coord>& vCol2, const R3VectorTC<Coord>& vCol3 );
    ///
    inline R3MatrixTC( const R3CoVectorTC<Coord>& cvRow1, const R3CoVectorTC<Coord>& cvRow2, const R3CoVectorTC<Coord>& cvRow3 );

    /**@name Access grants */
    //@{
    ///
    Coord & operator () ( const int i, const int j ) { return m_mat[i][j]; }
    ///
    const Coord & operator () ( const int i, const int j ) const { return m_mat[i][j]; }
    //@}

    /**@name Assignment operators */
    //@{
    /// 
    inline R3MatrixTC<Coord>& operator *= ( const R3MatrixTC<Coord>& );
    ///
    inline R3MatrixTC<Coord>& operator += ( const R3MatrixTC<Coord>& );
    ///
    inline R3MatrixTC<Coord>& operator -= ( const R3MatrixTC<Coord>& );
    ///
    inline R3MatrixTC<Coord>& operator *= ( const Coord );
    ///
    inline R3MatrixTC<Coord>& operator /= ( const Coord );
    //@}

    /**@name Unary operators */
    //@{
    /// 
    inline R3MatrixTC<Coord> operator + ( );
    ///
    inline R3MatrixTC<Coord> operator - ( );
    //@}

    /**@name Matrix construction operators */
    //@{
    /// Set to identity
    void MakeIdentity( );
    /// Identity matrix
    static R3MatrixTC<Coord> Identity( );
    ///
    static R3MatrixTC<Coord> Rotation(const int axis, const Coord angle);
    ///
    static R3MatrixTC<Coord> Rotation(const R3VectorTC<Coord> &in_vec, const Coord angle);
    ///
    static R3MatrixTC<Coord> Rotation2D( const Coord angle);
    ///
    static R3MatrixTC<Coord> Scaling( const Coord in_x, const Coord in_y, const Coord in_z );
    ///
    static R3MatrixTC<Coord> Scaling( const Coord in_scl );
    ///
    static R3MatrixTC<Coord> Translation( const R2VectorTC<Coord> & );

    /// Matrix taking the from vectors to the to vectors
    static WINbool MatrixVecToVec( R3MatrixTC<Coord>     & out_mat, 
                                   const R3VectorTC<Coord>  & in_vFromX, 
                                   const R3VectorTC<Coord>  & in_vFromY,
                                   const R3VectorTC<Coord>  & in_vToX,   
                                   const R3VectorTC<Coord>  & in_vToY );

    /// Matrix taking the from vector to the to vector, rotation around normal unspecified
    static WINbool MatrixVecToVec( R3MatrixTC<Coord>     & out_mat, 
                                   const R3VectorTC<Coord>  & in_vFromX, 
                                   const R3VectorTC<Coord>  & in_vToX );
    //@}

    /**@name Matrix operations */
    //@{
    /// 
    inline R3MatrixTC<Coord> Inverse( WINbool &out_bSuc ) const;
    /// 
    inline R3MatrixTC<Coord> Transpose(  ) const;
    //@}

    /**@name Read/write/print */
    //@{
    ///
    inline void    Write(ofstream &out) const;
    ///
    inline WINbool Read(ifstream &in) ;
    ///
    inline void    Print( const WINbool in_bDoReturn = FALSE ) const;
    //@}

private:
    ///
    Coord m_mat[3][3];
};

/**@name Non-method operations */
//@{

//@Include: Rn_Unary3_i.H
//@Include: Rn_Binary3_i.H
//@Include: Rn_Io3_i.H

//@}

//@}

#include "Rn_Vector3_i.H"
#include "Rn_CoVector3_i.H"
#include "Rn_Point3_i.H"
#include "Rn_Matrix3_i.H"
#include "Rn_Unary3_i.H"
#include "Rn_Binary3_i.H"
#include "Rn_Io3_i.H"

#endif
// #ifndef __BASETYPE_AFFINE3_H
