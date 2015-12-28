#ifndef __BASETYPE_AFFINE2_H
#define __BASETYPE_AFFINE2_H

/**@name 2D Points, Vectors, Covectors
<hr>
  Copyright (C) 1999 Brown Univ.  All Rights Reserved. <br>
<br>
  Files: <br>
    <dd><li> include/utils/Rn_Affine2.H 
    <dd><li> include/utils/Rn_Point2_i.H<br>
    <dd><li> include/utils/Rn_Vector2_i.H<br>
    <dd><li> include/utils/Rn_CoVector2_i.H<br>
    <dd><li> include/utils/Rn_Matrix2_i.H<br>
<hr>
<br>
  Content:     2D affine - space classes, templated on data type <br>
  <dd><li>          R2VectorTC   - 2D column vectors <br>
  <dd><li>          R2CoVectorTC - 2D row vectors (normals) <br>
  <dd><li>          R2PointTC    - 2D points <br>
 <hr>
*/
//@{

#include "WINSystemDefines.H"

template <class Coord> class R2CoVectorTC;
template <class Coord> class R2PointTC;
template <class Coord> class R2MatrixTC;

extern double RNEpsilon_d;
extern float RNEpsilon_f;


/// 2D Column vector
template<class Coord>
class R2VectorTC
{
public:
    /**@name Constructors */
    //@{
    ///
    inline R2VectorTC() {}
    ///
    inline R2VectorTC( const Coord& d );
    ///
    inline R2VectorTC( const Coord& du, const Coord& dv );
    //@}

    /**@name Access grants */
    //@{
    /// Return dimension of the type (always 2)
    inline int Dim() const { return 2; }

    ///
    inline Coord& operator[] ( int i ) { return (&du)[i]; }
    ///
    inline const Coord& operator[] ( int i ) const { return (&du)[i]; }
    //@}

    /**@name Self-editing */
    //@{
    /// 
    inline double Normalize();
    //@}

    /**@name Assignment operators */
    //@{
    /// 
    inline R2VectorTC<Coord>& operator += ( const R2VectorTC<Coord>& );
    ///
    inline R2VectorTC<Coord>& operator -= ( const R2VectorTC<Coord>& );
    ///
    inline R2VectorTC<Coord>& operator *= ( const Coord& );
    ///
    inline R2VectorTC<Coord>& operator /= ( const Coord& );
    //@}

    /**@name Unary operators */
    //@{
    /// 
    inline R2VectorTC<Coord>  operator + () const;
    ///
    inline R2VectorTC<Coord>  operator - () const;
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
    Coord du, dv;
};

/// 2D Row vector
template<class Coord>
class R2CoVectorTC
{
public:
    /**@name Constructors */
    //@{
    ///
    R2CoVectorTC( ) {}
    ///
    R2CoVectorTC( const Coord& d );
    ///
    R2CoVectorTC( const Coord &du, const Coord& dv );
    //@}

    /// Return dimension of the type
    int Dim() const { return 2; }

    /**@name Access grants */
    //@{
    ///
    Coord& operator[] ( const int i ) { return (&du)[i]; }
    ///
    const Coord& operator[] ( const int i ) const { return (&du)[i]; }
    //@}

    /**@name Assignment operators */
    //@{
    /// 
    R2CoVectorTC<Coord>& operator += ( const R2CoVectorTC<Coord>& );
    ///
    R2CoVectorTC<Coord>& operator -= ( const R2CoVectorTC<Coord>& );
    ///
    R2CoVectorTC<Coord>& operator *= ( const Coord& );
    ///
    R2CoVectorTC<Coord>& operator /= ( const Coord& );
    //@}

    /**@name Unary operators */
    //@{
    /// 
    R2CoVectorTC<Coord>  operator + () const;
    ///
    R2CoVectorTC<Coord>  operator - () const;
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
    Coord du, dv;
};


/// 2D point
template<class Coord>
class R2PointTC
{
public:
    /**@name Constructors */
    //@{
    ///
    R2PointTC() { }
    ///
    R2PointTC( const Coord u );
    ///
    R2PointTC(const Coord u, const Coord v);
    //@}

    /**@name Access grants */
    //@{
    /// Return dimension of the type (returns 2)
    int Dim() const { return 2; }

    ///
    Coord& operator[] ( const int i ) { return (&u)[i]; }
    ///
    const Coord& operator[] ( const int i ) const { return (&u)[i]; }
    //@}

    /**@name Assignment operators */
    //@{
    /// 
    inline R2PointTC<Coord>& operator += ( const R2VectorTC<Coord>& );
    ///
    inline R2PointTC<Coord>& operator -= ( const R2VectorTC<Coord>& );
    //@}

    /**@name Comparison operators */
    //@{
    /// 
    WINbool operator < ( const R2PointTC<Coord>& p ) const;
    ///
    WINbool operator <= ( const R2PointTC<Coord>& p ) const;
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
    Coord u, v;
};

/// Ray class
template< class Coord >
class R2RayTC
{
private:
    ///
    R2PointTC<Coord> pOrg;
    ///
    R2VectorTC<Coord> vDir;

public:
    /**@name Access grants */
    //@{
    ///
    const R2PointTC<Coord> &Pt() const { return pOrg; }
    ///
    const R2VectorTC<Coord> &Dir() const { return vDir; }
    //@}



    /**@name Assignment operators */
    //@{
    ///
    R2PointTC<Coord> &Pt() { return pOrg; }
    ///
    R2VectorTC<Coord> &Dir() { return vDir; }
    //@}

    /**@name Constructor/destructor */
    //@{
    ///
    R2RayTC() { }
    ///
    R2RayTC( const R2PointTC<Coord> & _pOrg, const R2VectorTC<Coord>& _vDir )
        { pOrg = _pOrg; vDir = _vDir; }
    ///
    ~R2RayTC() {}
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

/// R2MatrixTC: 2D linear transform class
template<class Coord>
class R2MatrixTC
{
public:
    /**@name Constructors */
    //@{
    ///
    inline R2MatrixTC() {}
    ///
    inline R2MatrixTC( const R2VectorTC<Coord>& vCol1, const R2VectorTC<Coord>& vCol2 );
    ///
    inline R2MatrixTC( const R2CoVectorTC<Coord>& covRow1, const R2CoVectorTC<Coord>& covRow2 );
    //@}

    /**@name Access grants */
    //@{
    ///
    inline Coord & operator () ( const int i, const int j ) { return mat[i][j]; }
    ///
    inline const Coord & operator () ( const int i, const int j ) const { return mat[i][j]; }
    //@}

    /**@name Assignment operators 
    */
    //@{
    /// 
    inline R2MatrixTC<Coord> & operator *= ( const R2MatrixTC<Coord>& );
    /// Componant-wise addition
    inline R2MatrixTC<Coord> & operator += ( const R2MatrixTC<Coord>& );
    /// Componant-wise subtraction
    inline R2MatrixTC<Coord> & operator -= ( const R2MatrixTC<Coord>& );
    /// Componant-wise multiplication
    inline R2MatrixTC<Coord> & operator *= ( const Coord );
    /// Componant-wise division (will not divide if input is zero)
    inline R2MatrixTC<Coord> & operator /= ( const Coord );
    //@}

    /**@name Unary operators */
    //@{
    /// 
    inline R2MatrixTC<Coord> operator + ( );
    ///
    inline R2MatrixTC<Coord> operator - ( );
    //@}

    /**@name Matrix operation entry points.
       Most of these have not been tested.
    */
    //@{
    ///
    inline void SetIdentity(  );

    ///
    inline R2MatrixTC<Coord> Inverse( Coord &out_dfDeterminant ) const;

    ///
    inline R2MatrixTC<Coord> Transpose(  ) const;

    ///
    inline WINbool SetRotation( const Coord fAngle );
    ///
    inline WINbool SetScale( const Coord fScaleU, const Coord fScaleV );
    ///
    static R2MatrixTC<Coord> Rotation(const Coord angle);
    ///
    static R2MatrixTC<Coord> Scaling(const Coord  fScaleU, const Coord fScaleV);
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
    Coord mat[2][2];
};

/**@name Non-method operations */
//@{

//@Include: Rn_Unary2_i.H
//@Include: Rn_Binary2_i.H
//@Include: Rn_Io2_i.H

//@}

//@}

#include "Rn_Vector2_i.H"
#include "Rn_CoVector2_i.H"
#include "Rn_Point2_i.H"
#include "Rn_Matrix2_i.H"
#include "Rn_Unary2_i.H"
#include "Rn_Binary2_i.H"
#include "Rn_Io2_i.H"

#endif
// #ifndef __BASETYPE_AFFINE2_H

