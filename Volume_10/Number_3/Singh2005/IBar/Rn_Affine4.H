#if _MSC_VER > 1000
#pragma once
#endif
// _MSC_VER > 1000

#ifndef __BASETYPE_AFFINE4_H
#define __BASETYPE_AFFINE4_H

/**@name 4D Points, Vectors, Covectors
<hr>
  Copyright (C) 1999 Brown Univ.  All Rights Reserved. <br>
<br>
  Files: <br>
    <dd><li> include/utils/Rn_Affine4.H 
    <dd><li> include/utils/Rn_Point4_i.H<br>
    <dd><li> include/utils/Rn_Vector4_i.H<br>
    <dd><li> include/utils/Rn_CoVector4_i.H<br>
    <dd><li> include/utils/Rn_Matrix4_i.H<br>
<hr>
<br>
  Content:     4D affine - space classes <br>
  <dd><li>          R4VectorTC   - 4D column vectors <br>
  <dd><li>          R4CoVectorTC - 4D row vectors (normals) <br>
  <dd><li>          R4PointTC    - 4D points <br>
  <dd><li>          R4MatrixTC   - 4D matrix <br>
 <hr>
*/
//@{

#include "WINSystemDefines.H"

template <class Coord> class R4CoVectorTC;
template <class Coord> class R4PointTC;
template <class Coord> class R4MatrixTC;

/// 4D Column vector
template<class Coord>
class R4VectorTC
{
public:
    ///
    R4VectorTC() {}
    ///
    inline R4VectorTC( const Coord& d );
    ///
    inline R4VectorTC( const Coord& dx, const Coord& dy, const Coord& dz, const Coord &dw = 1.0 );
    ///
    inline R4VectorTC( const R3VectorTC<Coord> & );

    /// Return dimension of the type
    int Dim() const { return 4; }


    /**@name Access grants */
    //@{
    ///
    Coord& operator[] ( int i ) { return (&dx)[i]; }
    ///
    const Coord& operator[] ( int i ) const { return (&dx)[i]; }
    //@}

    /**@name Assignment operators */
    //@{
    /// 
    inline R4VectorTC<Coord>& operator += ( const R4VectorTC<Coord>& );
    ///
    inline R4VectorTC<Coord>& operator -= ( const R4VectorTC<Coord>& );
    ///
    inline R4VectorTC<Coord>& operator *= ( const Coord& );
    ///
    inline R4VectorTC<Coord>& operator /= ( const Coord& );
    //@}

    /**@name Self-editing */
    //@{
    /// 
    inline double Normalize();
    //@}

    /**@name Unary operators */
    //@{
    /// 
    inline R4VectorTC<Coord>  operator + () const;
    ///
    inline R4VectorTC<Coord>  operator - () const;
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
    Coord dx, dy, dz, dw;
};

/// 4D Row vector
template<class Coord>
class R4CoVectorTC
{
public:
    ///
    R4CoVectorTC( ) {}
    ///
    R4CoVectorTC( const Coord& d );
    ///
    R4CoVectorTC( const Coord& dx, const Coord& dy, const Coord& dz, const Coord& dw = 1.0 );
    ///
    R4CoVectorTC( const R3CoVectorTC<Coord> & );

    /// Return dimension of the type
    int Dim() const { return 4; }

    /**@name Access grants */
    //@{
    ///
    Coord& operator[] ( int i ) { return (&dx)[i]; }
    ///
    const Coord& operator[] ( int i ) const { return (&dx)[i]; }
    //@}

    /**@name Assignment operators */
    //@{
    /// 
    R4CoVectorTC<Coord>& operator += ( const R4CoVectorTC<Coord>& );
    ///
    R4CoVectorTC<Coord>& operator -= ( const R4CoVectorTC<Coord>& );
    ///
    R4CoVectorTC<Coord>& operator *= ( const Coord& );
    ///
    R4CoVectorTC<Coord>& operator /= ( const Coord& );
    //@}

    /**@name Unary operators */
    //@{
    /// 
    R4CoVectorTC<Coord>  operator + () const;
    ///
    R4CoVectorTC<Coord>  operator - () const;
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
    Coord dx, dy, dz, dw;
};


/// 4D Point
template<class Coord>
class R4PointTC
{
public:
    ///
    R4PointTC() { }
    ///
    inline R4PointTC( const Coord& x, const Coord& y, const Coord& z, const Coord & w = 1.0 );
    ///
    inline R4PointTC( const Coord& x);
    ///
    inline R4PointTC( const R4PointTC<Coord>& in_pt);
    ///
    inline R4PointTC( const R3PointTC<Coord>& in_pt);

    /// Return dimension of the type
    int Dim() const { return 4; }

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
    inline R4PointTC<Coord>& operator += ( const R4VectorTC<Coord>& );
    ///
    inline R4PointTC<Coord>& operator -= ( const R4VectorTC<Coord>& );
    //@}


    /**@name Point dominance */
    //@{
    /// 
    inline WINbool operator < ( const R4PointTC<Coord>& ) const;
    ///
    inline WINbool operator <= ( const R4PointTC<Coord>& ) const;
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
    Coord x, y, z, w;
};


// =========================================================
// convenient typedefs for double-valued points, vectors, covectors
// =========================================================

/// Ray class
template< class Coord >
class R4RayTC
{
private:
    ///
    R4PointTC<Coord> pOrg;
    ///
    R4VectorTC<Coord> vDir;

public:
    /**@name Access grants */
    //@{
    ///
    const R4PointTC<Coord> &Pt() const { return pOrg; }
    ///
    const R4VectorTC<Coord> &Dir() const { return vDir; }
    //@}



    /**@name Assignment operators */
    //@{
    ///
    R4PointTC<Coord> &Pt() { return pOrg; }
    ///
    R4VectorTC<Coord> &Dir() { return vDir; }
    //@}

    /**@name Constructor/destructor */
    //@{
    ///
    R4RayTC() { }
    ///
    R4RayTC( const R4PointTC<Coord> & _pOrg, const R4VectorTC<Coord>& _vDir )
        { pOrg = _pOrg; vDir = _vDir; }
    ///
    ~R4RayTC() {}
    //@}

    /**@name Read/write/print */
    //@{
    ///
    inline void    Write(ofstream &out) const { pOrg.Write(out); vDir.Write(out); }
    ///
    inline WINbool Read(ifstream &in) { pOrg.Read(in); return vDir.Read(in); }
    ///
    inline void    Print( const WINbool in_bDoReturn = FALSE ) const { pOrg.Print( in_bDoReturn ); vDir.Print( in_bDoReturn ); }
    //@}

};

/// R4MatrixTC: 4D linear transform class
/// Data must be in column major order to match OpenGL
template< class Coord >
class R4MatrixTC
{
public:
    ///
    R4MatrixTC() {}
    ///
    inline R4MatrixTC( const R4VectorTC<Coord>& vCol1, 
                       const R4VectorTC<Coord>& vCol2, 
                       const R4VectorTC<Coord>& vCol3,
                       const R4VectorTC<Coord>& vCol4 );
    ///
    inline R4MatrixTC( const R4CoVectorTC<Coord>& cvRow1, 
                       const R4CoVectorTC<Coord>& cvRow2, 
                       const R4CoVectorTC<Coord>& cvRow3,
                       const R4CoVectorTC<Coord>& cvRow4 );
    ///
    inline R4MatrixTC( const R3MatrixTC<Coord> &in_oMat );

    /**@name Access grants */
    //@{
    ///
    Coord & operator () ( const int iRow, const int iCol ) { return m_mat[iCol][iRow]; }
    ///
    const Coord & operator () ( const int iRow, const int iCol ) const { return m_mat[iCol][iRow]; }
    //@}

    /**@name Assignment operators */
    //@{
    /// 
    inline R4MatrixTC<Coord>& operator *= ( const R4MatrixTC<Coord>& );
    ///
    inline R4MatrixTC<Coord>& operator += ( const R4MatrixTC<Coord>& );
    ///
    inline R4MatrixTC<Coord>& operator -= ( const R4MatrixTC<Coord>& );
    ///
    inline R4MatrixTC<Coord>& operator *= ( const Coord );
    ///
    inline R4MatrixTC<Coord>& operator /= ( const Coord );
    //@}

    /**@name Unary operators */
    //@{
    /// 
    inline R4MatrixTC<Coord> operator + ( );
    ///
    inline R4MatrixTC<Coord> operator - ( );
    //@}

    /**@name Matrix construction operators */
    //@{
    /// Set to identity
    void MakeIdentity( );
    /// Identity matrix
    static R4MatrixTC<Coord> Identity( );
    ///
    static R4MatrixTC<Coord> Rotation(const int axis, const Coord angle);
    ///
    static R4MatrixTC<Coord> Scaling( const Coord in_x, const Coord in_y, const Coord in_z, const Coord in_w );
    ///
    static R4MatrixTC<Coord> Scaling( const Coord in_scl );
    ///
    static R4MatrixTC<Coord> Translation( const R3VectorTC<Coord> &in_vec );
    ///
    static R4MatrixTC<Coord> Translation( const R3PointTC<Coord> &in_pt );
    //@}

    /**@name Matrix operations */
    //@{
    /// 
    inline R4MatrixTC<Coord> Inverse( WINbool &out_bSuccess ) const;
    /// 
    inline R4MatrixTC<Coord> Transpose(  ) const;
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
    /// column row
    Coord m_mat[4][4];
};

/**@name Non-method operations */
//@{

//@Include: Rn_Unary4_i.H
//@Include: Rn_Binary4_i.H
//@Include: Rn_Io4_i.H

//@}

//@}

#include "Rn_Vector4_i.H"
#include "Rn_CoVector4_i.H"
#include "Rn_Point4_i.H"
#include "Rn_Matrix4_i.H"
#include "Rn_Unary4_i.H"
#include "Rn_Binary4_i.H"
#include "Rn_Io4_i.H"

#endif
// #ifndef __BASETYPE_AFFINE4_H
