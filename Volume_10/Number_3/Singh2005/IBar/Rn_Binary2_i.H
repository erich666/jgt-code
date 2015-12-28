// =========================================================
//
//	Binary operators on affine2 classes
//
// =========================================================

/**@name Binary operators */
//@{


///
template<class Coord>
inline R2VectorTC<Coord>
operator + ( const R2VectorTC<Coord>& v1, const R2VectorTC<Coord>& v2 )
{
    return R2VectorTC<Coord>( v1[0] + v2[0], v1[1] + v2[1] );
}

///
template<class Coord>
inline R2VectorTC<Coord>
operator - ( const R2VectorTC<Coord>& v1, const R2VectorTC<Coord>& v2 )
{
    return R2VectorTC<Coord>( v1[0] - v2[0], v1[1] - v2[1] );
}

///
template<class Coord>
inline R2VectorTC<Coord>
operator * ( const Coord& s, const R2VectorTC<Coord>& v )
{
    return R2VectorTC<Coord>( v[0] * s, v[1] * s );
}

///
template<class Coord>
inline R2VectorTC<Coord>
operator * ( const R2VectorTC<Coord>& v, const Coord& s )
{
    return R2VectorTC<Coord>( v[0] * s, v[1] * s );
}

///
template<class Coord>
inline R2VectorTC<Coord>
operator / ( const R2VectorTC<Coord>& v, const Coord& s )
{
    ASSERT( !RNIsZero( s ) );

    return R2VectorTC<Coord>( v[0] / s, v[1] / s );
}


///
template<class Coord>
inline Coord
Dot( const R2VectorTC<Coord>& v1, const R2VectorTC<Coord>& v2 )
{
    return v1[0] * v2[0] + v1[1] * v2[1];
}

///
template<class Coord>
inline Coord
Dot( const R2VectorTC<Coord>& v1, const R2CoVectorTC<Coord>& v2 )
{
    return v1[0] * v2[0] + v1[1] * v2[1];
}

///
template<class Coord>
inline Coord
Dot( const R2CoVectorTC<Coord>& v1, const R2VectorTC<Coord>& v2 )
{
    return v1[0] * v2[0] + v1[1] * v2[1];
}

///
template<class Coord>
inline Coord
CCW( const R2VectorTC<Coord>& v1, const R2VectorTC<Coord>& v2 )
{
    return v1[0] * v2[1] - v1[1] * v2[0];
}

///
template<class Coord>
inline R2MatrixTC<Coord>
OuterProduct( const R2VectorTC<Coord>& v1, const R2VectorTC<Coord>& v2 )
{
    R2MatrixTC<Coord> ret;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            ret(i,j)=v1[i]*v2[j];
    return ret;
}

///
template<class Coord>
inline R2MatrixTC<Coord>
OuterProduct( const R2VectorTC<Coord>& v1, const R2CoVectorTC<Coord>& v2 )
{
    R2MatrixTC<Coord> ret;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            ret(i,j)=v1[i]*v2[j];
    return ret;
}

///
template<class Coord>
inline R2MatrixTC<Coord>
OuterProduct( const R2CoVectorTC<Coord>& v1, const R2VectorTC<Coord>& v2 )
{
    R2MatrixTC<Coord> ret;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            ret(i,j)=v1[i]*v2[j];
    return ret;
}
///
template<class Coord>
inline R2MatrixTC<Coord>
OuterProduct( const R2CoVectorTC<Coord>& v1, const R2CoVectorTC<Coord>& v2 )
{
    R2MatrixTC<Coord> ret;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            ret(i,j)=v1[i]*v2[j];
    return ret;
}

///
template<class Coord>
inline R2MatrixTC<Coord>
AlternatingProduct( const R2VectorTC<Coord>& v1, const R2VectorTC<Coord>& v2 )
{
    R2MatrixTC<Coord> ret;

    ret(0,0) = ret(1,1) = 0.0f;
    ret(1,0) = v1[1]*v2[0]-v1[0]*v2[1];
    ret(0,1) = -ret(1,0);

    return ret;
}

///
template<class Coord>
inline R2MatrixTC<Coord>
AlternatingProduct( const R2VectorTC<Coord>& v1, const R2CoVectorTC<Coord>& v2 )
{
    R2MatrixTC<Coord> ret;

    ret(0,0) = ret(1,1) = 0.0f;
    ret(1,0) = v1[1]*v2[0]-v1[0]*v2[1];
    ret(0,1) = -ret(1,0);

    return ret;
}

///
template<class Coord>
inline R2MatrixTC<Coord>
AlternatingProduct( const R2CoVectorTC<Coord>& v1, const R2VectorTC<Coord>& v2 )
{
    R2MatrixTC<Coord> ret;

    ret(0,0) = ret(1,1) = 0.0f;
    ret(1,0) = v1[1]*v2[0]-v1[0]*v2[1];
    ret(0,1) = -ret(1,0);

    return ret;
}

///
template<class Coord>
inline R2MatrixTC<Coord>
AlternatingProduct( const R2CoVectorTC<Coord>& v1, const R2CoVectorTC<Coord>& v2 )
{
    R2MatrixTC<Coord> ret;

    ret(0,0) = ret(1,1) = 0.0f;
    ret(1,0) = v1[1]*v2[0]-v1[0]*v2[1];
    ret(0,1) = -ret(1,0);

    return ret;
}

///
template<class Coord>
inline R2VectorTC<Coord>
operator-( const R2PointTC<Coord>& p1, const R2PointTC<Coord>& p2)
{
    return R2VectorTC<Coord>( p1[0] - p2[0], p1[1] - p2[1] );
}

///
template<class Coord>
inline R2CoVectorTC<Coord>
operator + ( const R2CoVectorTC<Coord>& v1, const R2CoVectorTC<Coord>& v2 )
{
    return R2CoVectorTC<Coord>( v1[0] + v2[0], v1[1] + v2[1] );
}

///
template<class Coord>
inline R2CoVectorTC<Coord>
operator - ( const R2CoVectorTC<Coord>& v1, const R2CoVectorTC<Coord>& v2 )
{
    return R2CoVectorTC<Coord>( v1[0] - v2[0], v1[1] - v2[1] );
}

///
template<class Coord>
inline R2CoVectorTC<Coord>
operator * ( const Coord& s, const R2CoVectorTC<Coord>& v )
{
    return R2CoVectorTC<Coord>( v[0] * s, v[1] * s );
}

///
template<class Coord>
inline R2CoVectorTC<Coord>
operator * ( const R2CoVectorTC<Coord>& v, const Coord& s )
{
    return R2CoVectorTC<Coord>( v[0] * s, v[1] * s );
}

///
template<class Coord>
inline R2CoVectorTC<Coord>
operator / ( const R2CoVectorTC<Coord>& v, const Coord& s )
{
    return R2CoVectorTC<Coord>(v[0] / s, v[1] / s);
}


///
template<class Coord>
inline Coord
Dot( const R2CoVectorTC<Coord>& v1, const R2CoVectorTC<Coord>& v2 )
{
    return v1[0] * v2[0] + v1[1] * v2[1];
}

///
template<class Coord>
inline Coord
CCW( const R2CoVectorTC<Coord>& v1, const R2CoVectorTC<Coord>& v2 )
{
    return v1[0] * v2[1] - v1[1] * v2[0];
}

// -------------------------------------
// binary operators
// -------------------------------------

///
template<class Coord>
inline R2PointTC<Coord>
operator + ( const R2PointTC<Coord>& p, const R2VectorTC<Coord>& vDelta )
{
    return R2PointTC<Coord>( p[0] + vDelta[0], p[1] + vDelta[1] );
}

///
template<class Coord>
inline R2PointTC<Coord>
operator + ( const R2VectorTC<Coord>& vDelta, const R2PointTC<Coord>& p )
{
    return R2PointTC<Coord>( p[0] + vDelta[0], p[1] + vDelta[1] );
}

///
template<class Coord>
inline R2PointTC<Coord>
operator - ( const R2PointTC<Coord>& p, const R2VectorTC<Coord>& vDelta )
{
    return R2PointTC<Coord>( p[0] - vDelta[0], p[1] - vDelta[1] );
}

// -------------------------------------
// friend functions
// -------------------------------------

///
template<class Coord>
inline R2PointTC<Coord>
Lerp( const R2PointTC<Coord>&p1, const R2PointTC<Coord>&p2, const Coord& s )
{
    R2VectorTC<Coord> p1p2 = p2 - p1;
    return R2PointTC<Coord>( p1 + s*p1p2 );
}

///
template<class Coord>
inline R2PointTC<Coord>
Barycentric( const R2PointTC<Coord>& p, 
             const R2PointTC<Coord>& q, 
             const R2PointTC<Coord>& r,
             Coord f, Coord g)
{
    return p+f*(q-p)+g*(r-p);
}

///
template<class Coord>
inline R2MatrixTC<Coord>
operator * ( const R2MatrixTC<Coord>& m1, const R2MatrixTC<Coord>& m2 )
{
    R2MatrixTC<Coord> ret;

    for (int i = 0; i < 2; i += 1)
        for (int j = 0; j < 2; j += 1)
            ret(i, j) = m1(i, 0) * m2(0,j) + m1(i,1) * m2(1, j);
    return ret;
}


///
template<class Coord>
inline R2MatrixTC<Coord>
operator + ( const R2MatrixTC<Coord>& m1, const R2MatrixTC<Coord>& m2 )
{
    R2MatrixTC<Coord> ret;

    for (int i = 0; i < 2; i += 1)
        for (int j = 0; j < 2; j += 1)
            ret(i, j) = m1(i, j) + m2(i, j);
    return ret;
}

///
template<class Coord>
inline R2MatrixTC<Coord>
operator - ( const R2MatrixTC<Coord>& m1, const R2MatrixTC<Coord>& m2 )
{
    R2MatrixTC<Coord> ret;

    for (int i = 0; i < 2; i += 1)
        for (int j = 0; j < 2; j += 1)
            ret(i, j) = m1(i, j) - m2(i, j);
    return ret;
}

///
template<class Coord>
inline R2MatrixTC<Coord>
operator * ( const R2MatrixTC<Coord>& m, Coord s )
{
    R2MatrixTC<Coord> ret;

    for (int i = 0; i < 2; i += 1)
        for (int j = 0; j < 2; j += 1)
            ret(i, j) = m(i, j) * s;
    return ret;
}

///
template<class Coord>
inline R2MatrixTC<Coord>
operator * ( Coord s, const R2MatrixTC<Coord>& m )
{
    R2MatrixTC<Coord> ret;

    for (int i = 0; i < 2; i += 1)
        for (int j = 0; j < 2; j += 1)
            ret(i, j) = m(i, j) * s;
    return ret;
}


///
template<class Coord>
inline R2MatrixTC<Coord>
operator / ( const R2MatrixTC<Coord>& m, Coord s )
{
    R2MatrixTC<Coord> ret;

    ASSERT( s != 0.0 );

    Coord s1 = 1.0 / s;

    for (int i = 0; i < 2; i += 1)
        for (int j = 0; j < 2; j += 1)
            ret(i, j) = m(i, j) * s1;
    return ret;
}

///
template<class Coord>
inline 
R2VectorTC<Coord>
operator * ( const R2MatrixTC<Coord> &m, const R2VectorTC<Coord> &vec )
{
    return R2VectorTC<Coord>( m(0, 0)*vec[0] + m(0, 1)*vec[1],
                              m(1, 0)*vec[0] + m(1, 1)*vec[1] );
}

///
template<class Coord>
inline 
R2CoVectorTC<Coord>
operator * ( const R2CoVectorTC<Coord> &cov, const R2MatrixTC<Coord>& m )
{
    return R2CoVectorTC<Coord>(cov[0]*m(0, 0) + cov[1]*m(1, 0),
                               cov[0]*m(0, 1) + cov[1]*m(1, 1));
}

///
template<class Coord>
inline 
R2PointTC<Coord>
operator * ( const R2MatrixTC<Coord> &m, const R2PointTC<Coord> &p )
{
    return R2PointTC<Coord>( m(0, 0)*p[0] + m(0, 1)*p[1],
                             m(1, 0)*p[0] + m(1, 1)*p[1] );
}



//@}

/**@name Comparison operators */
//@{


///
template<class Coord>
inline WINbool
operator==(const R2VectorTC<Coord>& v1, const R2VectorTC<Coord>& v2)
{
    return (v1[0] == v2[0] && v1[1] == v2[1]) ? TRUE : FALSE;
}

///
template<class Coord>
inline WINbool
operator!=(const R2VectorTC<Coord>& v1, const R2VectorTC<Coord>& v2)
{
    return (v1[0] == v2[0] && v1[1] == v2[1]) ? FALSE : TRUE;
}
///
inline WINbool
ApproxEqual(const R2VectorTC<double>& v1, const R2VectorTC<double>& v2)
{
    return ( fabs( v1[0] - v2[0] ) < RNEpsilon_d  &&
             fabs( v1[1] - v2[1] ) < RNEpsilon_d) ? TRUE : FALSE; 
}

///
inline WINbool
ApproxEqual(const R2VectorTC<double>& v1, const R2VectorTC<double>& v2, const double in_dEpsilon)
{
    return ( fabs( v1[0] - v2[0] ) < in_dEpsilon  &&
             fabs( v1[1] - v2[1] ) < in_dEpsilon) ? TRUE : FALSE; 
}


///
inline WINbool
ApproxEqual(const R2VectorTC<float>& v1, const R2VectorTC<float>& v2)
{
    return ( fabs( v1[0] - v2[0] ) < RNEpsilon_f  &&
             fabs( v1[1] - v2[1] ) < RNEpsilon_f) ? TRUE : FALSE; 
}
///

inline WINbool
ApproxEqual(const R2VectorTC<float>& v1, const R2VectorTC<float>& v2, const float in_fEpsilon)
{
    return ( fabs( v1[0] - v2[0] ) < in_fEpsilon  &&
             fabs( v1[1] - v2[1] ) < in_fEpsilon) ? TRUE : FALSE; 
}
///
template<class Coord>
inline WINbool
operator == ( const R2CoVectorTC<Coord>& v1, const R2CoVectorTC<Coord>& v2 )
{
    return (v1[0] == v2[0] && v1[1] == v2[1]) ? TRUE : FALSE;
}

///
template<class Coord>
inline WINbool
operator != ( const R2CoVectorTC<Coord>& v1, const R2CoVectorTC<Coord>& v2 )
{
    return (v1[0] == v2[0] && v1[1] == v2[1]) ? FALSE : TRUE;
}

///
inline WINbool
ApproxEqual(const R2CoVectorTC<double>& v1, const R2CoVectorTC<double>& v2)
{
    return ( fabs( v1[0] - v2[0] ) < RNEpsilon_d  &&
             fabs( v1[1] - v2[1] ) < RNEpsilon_d) ? TRUE : FALSE; 
}

///
inline WINbool
ApproxEqual(const R2CoVectorTC<float>& v1, const R2CoVectorTC<float>& v2)
{
    return ( fabs( v1[0] - v2[0] ) < RNEpsilon_f  &&
             fabs( v1[1] - v2[1] ) < RNEpsilon_f) ? TRUE : FALSE; 
}


///
template<class Coord>
inline WINbool
operator == ( const R2PointTC<Coord>& p1, const R2PointTC<Coord>& p2 )
{
    return (p1[0] == p2[0] && p1[1] == p2[1]) ? TRUE : FALSE;
}

///
template<class Coord>
inline WINbool
operator != ( const R2PointTC<Coord>& p1, const R2PointTC<Coord>& p2 )
{
    return (p1[0] == p2[0] && p1[1] == p2[1]) ? FALSE : TRUE;
}
///
inline WINbool
ApproxEqual(const R2PointTC<double>& p1, const R2PointTC<double>& p2)
{
    return ( fabs( p1[0] - p2[0] ) < RNEpsilon_d &&
             fabs( p1[1] - p2[1] ) < RNEpsilon_d ) ? TRUE : FALSE; 
}
///
inline WINbool
ApproxEqual(const R2PointTC<double>& p1, const R2PointTC<double>& p2, const double in_dEps)
{
    return ( fabs( p1[0] - p2[0] ) < in_dEps &&
             fabs( p1[1] - p2[1] ) < in_dEps ) ? TRUE : FALSE; 
}
///
inline WINbool
ApproxEqual(const R2PointTC<float>& p1, const R2PointTC<float>& p2, const float in_fEps)
{
    return ( fabs( p1[0] - p2[0] ) < in_fEps &&
             fabs( p1[1] - p2[1] ) < in_fEps ) ? TRUE : FALSE; 
}

///
template<class Coord>
inline WINbool
operator == ( const R2MatrixTC<Coord>& m1, const R2MatrixTC<Coord>& m2)
{
    for (int i = 0; i < 2; i += 1)
        for (int j = 0; j < 2; j += 1)
            if ( m1(i,j) != m2(i,j) )
                return FALSE;
    return TRUE;
}

///
template<class Coord>
inline WINbool
operator != ( const R2MatrixTC<Coord>& m1, const R2MatrixTC<Coord>& m2)
{
    for (int i = 0; i < 2; i += 1)
        for (int j = 0; j < 2; j += 1)
            if ( m1(i,j) != m2(i,j) )
                return TRUE;
    return FALSE;
}

///
inline
WINbool 
ApproxEqual ( const R2MatrixTC<double> &m1, const R2MatrixTC<double> &m2 )
{
    for (int i = 0; i < 2; i += 1)
        for (int j = 0; j < 2; j += 1)
            if ( fabs( m1(i, j) - m2(i, j) ) < RNEpsilon_d )
                return FALSE;
    return TRUE;
}

inline
WINbool 
ApproxEqual ( const R2MatrixTC<double> &m1, const R2MatrixTC<double> &m2, const double in_dEps )
{
    for (int i = 0; i < 2; i += 1)
        for (int j = 0; j < 2; j += 1)
            if ( fabs( m1(i, j) - m2(i, j) ) < in_dEps )
                return FALSE;
    return TRUE;
}

///
inline WINbool
ApproxEqual ( const R2MatrixTC<float>& m1, const R2MatrixTC<float>& m2)
{
    for (int i = 0; i < 2; i += 1)
        for (int j = 0; j < 2; j += 1)
            if ( fabs( m1(i,j) - m2(i,j) ) < RNEpsilon_f )
                return FALSE;
    return TRUE;
}

///
inline WINbool
ApproxEqual ( const R2MatrixTC<float>& m1, const R2MatrixTC<float>& m2, const float in_fEps)
{
    for (int i = 0; i < 2; i += 1)
        for (int j = 0; j < 2; j += 1)
            if ( fabs( m1(i,j) - m2(i,j) ) < in_fEps )
                return FALSE;
    return TRUE;
}

//@}
