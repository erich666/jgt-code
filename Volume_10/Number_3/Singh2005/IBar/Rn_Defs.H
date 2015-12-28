/* Copyright 1991, Brown Computer Graphics Group.  
   All Rights Reserved. */


/* -----------------------------------------------------------------
 *
 * Several objects in Euclidean space
 *
 * -------------------------------------------------------------- */

#ifndef _RN_DEFS_DEFS
#define _RN_DEFS_DEFS

#include "Rn_Affine1.H"
#include "Rn_Affine2.H"
#include "Rn_Affine3.H"
#include "Rn_Affine4.H"

/* -------------------  Constants  ------------------------------- */

/* -------------------  Classes  ------------------------------- */

/**@name Approximate numerical assertions
   Useful numerical routines
<br> Files: include/utils/Rn_Defs.H utils/geometry/RNDefs.cpp <br>
  */
//@{

/// Precision for double precision operations (default is $1e^{-15}$)
extern double RNEpsilon_d;
/// Precision for floating point operations (default is $1e^{-6}$)
extern float RNEpsilon_f;

/** Is zero within given epsilon.
    @param in_dEps default is $1e^{-15}$
    @returns TRUE/FALSE
  */
inline WINbool RNIsZero(double in_d, double in_dEps = RNEpsilon_d) 
{ return (in_d < in_dEps && in_d > -in_dEps)? TRUE : FALSE; }

/** Is zero within given epsilon.
    @param in_fEps default is $1e^{-6}$
    @returns TRUE/FALSE
  */
inline WINbool RNIsZero(float in_d, float in_dEps = RNEpsilon_f) 
{ return (in_d < in_dEps && in_d > -in_dEps)? TRUE : FALSE; }


/// Is the point/vector not at infinity?
template <class Affine>
inline
WINbool RNIsValid(const Affine &in_a) 
{ 
    for (int i = 0; i < in_a.Dim(); i++)
        if (in_a[i] < 1e30) return TRUE;
    return FALSE;
}

/// Zero out the point/vector
template <class Affine>
inline
Affine RNZero(Affine &io_a) 
{ 
    for (int i = 0; i < io_a.Dim(); i++)
        io_a[i] = 0;
    return io_a;
}


/** Are the numbers within epsilon of each other?
  @param in_dEpsilon default is $1e^{-15}$
  */
inline WINbool RNApproxEqual( double in_a1, double in_a2,
                              double in_dEpsilon = RNEpsilon_d)
{
    if (fabs(in_a1 - in_a2) > in_dEpsilon) return FALSE;
    return TRUE;
}

/** Are the numbers within epsilon of each other?
  @param in_fEpsilon default is $1e^{-06}$
  */
inline WINbool RNApproxEqual( float in_a1, float in_a2,
                              float in_dEpsilon = RNEpsilon_f)
{
    if (fabs(in_a1 - in_a2) > in_dEpsilon) return FALSE;
    return TRUE;
}

//@}

/**@name Constants
 */
//@{

#ifndef M_PI
/// Pi
const double M_PI =3.1415926535897932384626433832795028842;
#endif
//@}

/**@name Typedefs for points, vectors, matrices.
See the ND Points, Vectors, Covectors pages for class attributes (e.g., R?PointTC<>) */
//@{

///
typedef R4PointTC<double> R4Pt;
///
typedef R4PointTC<float>  R4Pt_f;
///
typedef R4PointTC<int>    R4Pt_i;
///
typedef R3PointTC<double> R3Pt;
///
typedef R3PointTC<float>  R3Pt_f;
///
typedef R3PointTC<int>    R3Pt_i;
///
typedef R2PointTC<double> R2Pt;
///
typedef R2PointTC<float>  R2Pt_f;
///
typedef R2PointTC<int>    R2Pt_i;
///
typedef R1PointTC<double> R1Pt;
///
typedef R1PointTC<float>  R1Pt_f;
///
typedef R1PointTC<int>    R1Pt_i;

///
typedef R4VectorTC<double> R4Vec;
///
typedef R4VectorTC<float>  R4Vec_f;
///
typedef R4VectorTC<int>    R4Vec_i;
///
typedef R3VectorTC<double> R3Vec;
///
typedef R3VectorTC<float>  R3Vec_f;
///
typedef R3VectorTC<int>    R3Vec_i;
///
typedef R2VectorTC<double> R2Vec;
///
typedef R2VectorTC<float>  R2Vec_f;
///
typedef R2VectorTC<int>    R2Vec_i;
///
typedef R1VectorTC<double> R1Vec;
///
typedef R1VectorTC<float>  R1Vec_f;
///
typedef R1VectorTC<int>    R1Vec_i;

///
typedef R4CoVectorTC<double> R4CoVec;
///
typedef R3CoVectorTC<double> R3CoVec;
///
typedef R2CoVectorTC<double> R2CoVec;
///
typedef R1CoVectorTC<double> R1CoVec;

///
typedef R4RayTC<double> R4Ray;
///
typedef R3RayTC<double> R3Ray;
///
typedef R2RayTC<double> R2Ray;
///
typedef R1RayTC<double> R1Ray;

///
typedef R4MatrixTC<double> R4Matrix;
///
typedef R3MatrixTC<double> R3Matrix;
///
typedef R2MatrixTC<double> R2Matrix;

//@}

#endif



