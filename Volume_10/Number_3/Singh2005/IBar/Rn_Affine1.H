#ifndef __BASETYPE_AFFINE1_H
#define __BASETYPE_AFFINE1_H

/**@name 1D Points, Vectors, Covectors
<hr>
  Copyright (C) 1999 Brown Univ.  All Rights Reserved. <br>
<br>
  Files: <br>
    <dd><li> include/utils/Rn_Affine1.H 
    <dd><li> include/utils/Rn_Point1_i.H<br>
    <dd><li> include/utils/Rn_Vector1_i.H<br>
    <dd><li> include/utils/Rn_CoVector1_i.H<br>
<hr>
<br>
  Content:     1D affine - space classes <br>
  <dd><li>          R1VectorTC   - 1D column vectors <br>
  <dd><li>          R1CoVectorTC - 1D row vectors (normals) <br>
  <dd><li>          R1PointTC    - 1D points <br>
 <hr>
*/
//@{

#include "WINSystemDefines.H"


template <class Coord> class R1CoVectorTC;
template <class Coord> class R1PointTC;

extern double RNEpsilon_d;
extern float RNEpsilon_f;


/** A 1D Column vector.

 Essentially a single number. Included for completeness'
 sake. Type-safe addition/subtraction/multiplication for points,
 vectors, and co-vectors. Operations are written as friend functions;
 unfortunately, some compilers balk if the friend functions are listed
 in the class definition, so they don't appear as class members except in the
 documentation.

 */
template<class Coord>
class R1VectorTC
{
public:
    /**@name Constructors */
    //@{
    ///
    inline R1VectorTC() {}
    ///
    inline R1VectorTC( const Coord d );
    //@}
    
    /**@name Access */
    //@{

    /// Return dimension (always returns 1)
    inline int Dim() const { return 1; }

    /// 
    inline Coord& operator[] ( const int DECLAREVAR(i) ) { ASSERT(i == 0); return d; }
    ///
    inline const Coord& operator[] ( const int DECLAREVAR(i) ) const { ASSERT(i == 0); return d; }
    //@}

    /**@name Assignment operators */
    //@{
    ///
    inline R1VectorTC<Coord>& operator += ( const R1VectorTC<Coord>& );
    ///
    inline R1VectorTC<Coord>& operator -= ( const R1VectorTC<Coord>& );
    ///
    inline R1VectorTC<Coord>& operator *= ( const Coord& );
    ///
    inline R1VectorTC<Coord>& operator /= ( const Coord& );
    //@}

    /**@name Self-editing */
    //@{
    /// 
    inline double Normalize();
    //@}

    /**@name Unary operators */
    //@{
    ///
    inline R1VectorTC<Coord>  operator + () const;
    ///
    inline R1VectorTC<Coord>  operator - () const;
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
    Coord d;
};


/// 1D CoVector
template<class Coord>
class R1CoVectorTC
{
public:
    ///
    inline R1CoVectorTC( ) {}
    ///
    inline R1CoVectorTC( const Coord d );

    /// Return dimension of the type
    inline int Dim() const { return 1; }

    /**@name Access grants */
    //@{
    ///
    inline Coord& operator[] ( const int DECLAREVAR(i) ) { ASSERT(i == 0); return d; }
    ///
    inline const Coord& operator[] ( const int DECLAREVAR(i) ) const { ASSERT(i == 0); return d; }
    //@}

    /**@name Assignment operators */
    //@{
    ///
    inline R1CoVectorTC<Coord>& operator += ( const R1CoVectorTC<Coord>& );
    ///
    inline R1CoVectorTC<Coord>& operator -= ( const R1CoVectorTC<Coord>& );
    ///
    inline R1CoVectorTC<Coord>& operator *= ( const Coord& );
    ///
    inline R1CoVectorTC<Coord>& operator /= ( const Coord& );
    //@}

    /**@name Unary operators */
    //@{
    ///
    inline R1CoVectorTC<Coord>  operator + () const;
    ///
    inline R1CoVectorTC<Coord>  operator - () const;
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
    Coord d;
};


/// A 1D point
template<class Coord>
class R1PointTC
{
public:
    ///
    R1PointTC() { }
    ///
    R1PointTC(Coord d);

    /// Return dimension of the type
    int Dim() const { return 1; }

    /**@name Access grants */
    //@{
    ///
    inline Coord& operator[] ( const int DECLAREVAR(i) ) { ASSERT( i == 0 ); return u; }
    ///
    inline const Coord& operator[] ( const int DECLAREVAR(i) ) const { ASSERT(i == 0); return u; }
    //@}

    /**@name Assignment operators */
    //@{
    ///
    inline R1PointTC<Coord>& operator += ( const R1VectorTC<Coord>& );
    ///
    inline R1PointTC<Coord>& operator -= ( const R1VectorTC<Coord>& );
    //@}

    /**@name Point dominance */
    //@{
    ///
    inline WINbool operator < ( const R1PointTC<Coord>& ) const;
    ///
    inline WINbool operator <= ( const R1PointTC<Coord>& ) const;
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
    Coord u;
};

/// Ray class
template< class Coord >
class R1RayTC
{
private:
    ///
    R1PointTC<Coord> pOrg;
    ///
    R1VectorTC<Coord> vDir;

public:
    /**@name Access grants */
    //@{
    ///
    const R1PointTC<Coord> &Pt() const { return pOrg; }
    ///
    const R1VectorTC<Coord> &Dir() const { return vDir; }
    //@}



    /**@name Assignment operators */
    //@{
    ///
    R1PointTC<Coord> &Pt() { return pOrg; }
    ///
    R1VectorTC<Coord> &Dir() { return vDir; }
    //@}

    /**@name Constructor/destructor */
    //@{
    ///
    R1RayTC() { }
    ///
    R1RayTC( const R1PointTC<Coord> & _pOrg, const R1VectorTC<Coord>& _vDir )
        { pOrg = _pOrg; vDir = _vDir; }
    ///
    ~R1RayTC() {}
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

/**@name Non-method operations */
//@{

//@Include: Rn_Unary1_i.H
//@Include: Rn_Binary1_i.H
//@Include: Rn_Io1_i.H

//@}

//@}


#include "Rn_Vector1_i.H"
#include "Rn_Point1_i.H"
#include "Rn_CoVector1_i.H"
#include "Rn_Unary1_i.H"
#include "Rn_Binary1_i.H"
#include "Rn_Io1_i.H"

#endif
// #ifndef __BASETYPE_AFFINE1_H

