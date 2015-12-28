// =========================================================
//
//	Binary operators on affine4 classes
//
// =========================================================

/**@name Binary operators */
//@{

// -------------------------------------
// binary operators
// -------------------------------------

///
template<class Coord>
inline R4VectorTC<Coord>
operator + ( const R4VectorTC<Coord>& v1, const R4VectorTC<Coord>& v2 )
{
    return R4VectorTC<Coord>( v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2], v1[3] + v2[3] );
}

///
template<class Coord>
inline R4VectorTC<Coord>
operator - ( const R4VectorTC<Coord>& v1, const R4VectorTC<Coord>& v2 )
{
    return R4VectorTC<Coord>( v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2], v1[3] - v2[3] );
}

///
template<class Coord>
inline R4VectorTC<Coord>
operator * ( const Coord& s, const R4VectorTC<Coord>& v )
{
    return R4VectorTC<Coord>( v[0] * s, v[1] * s, v[2] * s, v[3] * s );
}

///
template<class Coord>
inline R4VectorTC<Coord>
operator * ( const R4VectorTC<Coord>& v, const Coord& s )
{
    return R4VectorTC<Coord>( v[0] * s, v[1] * s, v[2] * s, v[3] * s );
}

///
template<class Coord>
inline R4VectorTC<Coord>
operator / ( const R4VectorTC<Coord>& v, const Coord& s )
{
    ASSERT( !RNIsZero( s ) );

    Coord s1 = 1.0f / s;
    return R4VectorTC<Coord>( v[0] * s1, v[1] * s1, v[2] * s1, v[3] * s1 );
}


///
template<class Coord>
inline Coord
Dot( const R4VectorTC<Coord>& v1, const R4VectorTC<Coord>& v2 )
{
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2] + v1[3] * v2[3];
}

///
template<class Coord>
inline Coord
Dot( const R4VectorTC<Coord>& v1, const R4CoVectorTC<Coord>& v2 )
{
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2] + v1[3] * v2[3];
}

///
template<class Coord>
inline Coord
Dot( const R4CoVectorTC<Coord>& v1, const R4VectorTC<Coord>& v2 )
{
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2] + v1[3] * v2[3];
}

///
template<class Coord>
inline R4MatrixTC<Coord>
OuterProduct( const R4VectorTC<Coord>& v1, const R4VectorTC<Coord>& v2 )
{
    R4MatrixTC<Coord> ret;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            ret(i,j)=v1[i]*v2[j];
    return ret;
}

///
template<class Coord>
inline R4MatrixTC<Coord>
OuterProduct( const R4CoVectorTC<Coord>& v1, const R4VectorTC<Coord>& v2 )
{
    R4MatrixTC<Coord> ret;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            ret(i,j)=v1[i]*v2[j];
    return ret;
}

///
template<class Coord>
inline R4MatrixTC<Coord>
OuterProduct( const R4VectorTC<Coord>& v1, const R4CoVectorTC<Coord>& v2 )
{
    R4MatrixTC<Coord> ret;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            ret(i,j)=v1[i]*v2[j];
    return ret;
}

///
template<class Coord>
inline R4MatrixTC<Coord>
OuterProduct( const R4CoVectorTC<Coord>& v1, const R4CoVectorTC<Coord>& v2 )
{
    R4MatrixTC<Coord> ret;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            ret(i,j)=v1[i]*v2[j];
    return ret;
}

///
template<class Coord>
inline R4MatrixTC<Coord>
AlternatingProduct( const R4VectorTC<Coord>& v1, const R4VectorTC<Coord>& v2 )
{
    R4MatrixTC<Coord> ret;

    ret(0,0) = ret(1,1) = ret(2,2) = 0.0f;
    ret(1,0) = v1[1]*v2[0]-v1[0]*v2[1];
    ret(2,0) = v1[2]*v2[0]-v1[0]*v2[2]; ret(2,1) = v1[2]*v2[1]-v1[1]*v2[2];
    ret(0,1) = -ret(1,0);
    ret(0,2) = -ret(2,0);   ret(1,2) = -ret(2,1);

    return ret;
}

///
template<class Coord>
inline R4MatrixTC<Coord>
AlternatingProduct( const R4VectorTC<Coord>& v1, const R4CoVectorTC<Coord>& v2 )
{
    R4MatrixTC<Coord> ret;

    ret(0,0) = ret(1,1) = ret(2,2) = ret(3,3) = 0.0f;

    ret(1,0) = v1[1]*v2[0]-v1[0]*v2[1];
    ret(2,0) = v1[2]*v2[0]-v1[0]*v2[2]; 
    ret(3,0) = v1[3]*v2[0]-v1[0]*v2[3]; 

    ret(2,1) = v1[2]*v2[1]-v1[1]*v2[2];
    ret(3,1) = v1[3]*v2[1]-v1[1]*v2[3];

    ret(3,2) = v1[3]*v2[2]-v1[2]*v2[3];

    ret(0,1) = -ret(1,0);
    ret(0,2) = -ret(2,0);   
    ret(0,3) = -ret(3,0);
    
    ret(1,2) = -ret(2,1);
    ret(1,3) = -ret(3,1);

    ret(3,2) = -ret(2,3);

    return ret;
}

///
template<class Coord>
inline R4MatrixTC<Coord>
AlternatingProduct( const R4CoVectorTC<Coord>& v1, const R4VectorTC<Coord>& v2 )
{
    R4MatrixTC<Coord> ret;

    ret(0,0) = ret(1,1) = ret(2,2) = ret(3,3) = 0.0f;

    ret(1,0) = v1[1]*v2[0]-v1[0]*v2[1];
    ret(2,0) = v1[2]*v2[0]-v1[0]*v2[2]; 
    ret(3,0) = v1[3]*v2[0]-v1[0]*v2[3]; 

    ret(2,1) = v1[2]*v2[1]-v1[1]*v2[2];
    ret(3,1) = v1[3]*v2[1]-v1[1]*v2[3];

    ret(3,2) = v1[3]*v2[2]-v1[2]*v2[3];

    ret(0,1) = -ret(1,0);
    ret(0,2) = -ret(2,0);   
    ret(0,3) = -ret(3,0);
    
    ret(1,2) = -ret(2,1);
    ret(1,3) = -ret(3,1);

    ret(3,2) = -ret(2,3);

    return ret;
}

///
template<class Coord>
inline R4MatrixTC<Coord>
AlternatingProduct( const R4CoVectorTC<Coord>& v1, const R4CoVectorTC<Coord>& v2 )
{
    R4MatrixTC<Coord> ret;

    ret(0,0) = ret(1,1) = ret(2,2) = ret(3,3) = 0.0f;

    ret(1,0) = v1[1]*v2[0]-v1[0]*v2[1];
    ret(2,0) = v1[2]*v2[0]-v1[0]*v2[2]; 
    ret(3,0) = v1[3]*v2[0]-v1[0]*v2[3]; 

    ret(2,1) = v1[2]*v2[1]-v1[1]*v2[2];
    ret(3,1) = v1[3]*v2[1]-v1[1]*v2[3];

    ret(3,2) = v1[3]*v2[2]-v1[2]*v2[3];

    ret(0,1) = -ret(1,0);
    ret(0,2) = -ret(2,0);   
    ret(0,3) = -ret(3,0);
    
    ret(1,2) = -ret(2,1);
    ret(1,3) = -ret(3,1);

    ret(3,2) = -ret(2,3);

    return ret;
}

// -------------------------------------
// binary operators
// -------------------------------------

///
template<class Coord>
inline R4CoVectorTC<Coord>
operator + ( const R4CoVectorTC<Coord>& v1, const R4CoVectorTC<Coord>& v2 )
{
    return R4CoVectorTC<Coord>( v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2], v1[3] + v2[3] );
}

///
template<class Coord>
inline R4CoVectorTC<Coord>
operator - ( const R4CoVectorTC<Coord>& v1, const R4CoVectorTC<Coord>& v2 )
{
    return R4CoVectorTC<Coord>( v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2], v1[3] - v2[3] );
}

///
template<class Coord>
inline R4CoVectorTC<Coord>
operator * ( const Coord& s, const R4CoVectorTC<Coord>& v )
{
    return R4CoVectorTC<Coord>( v[0] * s, v[1] * s, v[2] * s, v[3] * s );
}

///
template<class Coord>
inline R4CoVectorTC<Coord>
operator * ( const R4CoVectorTC<Coord>& v, const Coord& s )
{
    return R4CoVectorTC<Coord>( v[0] * s, v[1] * s, v[2] * s, v[3] * s );
}

///
template<class Coord>
inline R4CoVectorTC<Coord>
operator / ( const R4CoVectorTC<Coord>& v, const Coord& s )
{
    ASSERT( !RNIsZero( s ) );

    Coord s1 = Coord(1.0f) / s;
    return R4CoVectorTC<Coord>( v[0] * s1, v[1] * s1, v[2] * s1, v[3] * s1 );
}

///
template<class Coord>
tinline Coord
Dot( const R4CoVectorTC<Coord>& v1, const R4CoVectorTC<Coord>& v2 )
{
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2] + v1[3] * v2[3];
}



///
template<class Coord>
inline R4PointTC<Coord>
operator + ( const R4PointTC<Coord>& p, const R4VectorTC<Coord>& vDelta )
{
    return R4PointTC<Coord>( p[0] + vDelta[0], p[1] + vDelta[1], p[2] + vDelta[2], p[3] + vDelta[3] );
}

///
template<class Coord>
inline R4PointTC<Coord>
operator + ( const R4CoVectorTC<Coord>& vDelta, const R4PointTC<Coord>& p )
{
    return R4PointTC<Coord>( p[0] + vDelta[0], p[1] + vDelta[1], p[2] + vDelta[2], p[3] + vDelta[3] );
}

///
template<class Coord>
inline R4PointTC<Coord>
operator - ( const R4PointTC<Coord>& p, const R4VectorTC<Coord>& vDelta )
{
    return R4PointTC<Coord>( p[0] - vDelta[0], p[1] - vDelta[1], p[2] - vDelta[2], p[3] - vDelta[3] );
}

// -------------------------------------
// friend functions
// -------------------------------------

///
template<class Coord>
inline R4PointTC<Coord>
Lerp( const R4PointTC<Coord>&p1, const R4PointTC<Coord>&p2, Coord s )
{
    R4VectorTC<Coord> p1p2 = p2 - p1;
    return R4PointTC<Coord>( p1 + s*p1p2 );
}


///
template<class Coord>
inline R4MatrixTC<Coord>
operator + ( const R4MatrixTC<Coord>& m1, const R4MatrixTC<Coord>& m2 )
{
    R4MatrixTC<Coord> ret;

    for (int i = 0; i < 4; i += 1)
        for (int j = 0; j < 4; j += 1)
            ret(i, j) = m1(i, j) + m2(i, j);
    return ret;
}

///
template<class Coord>
inline R4MatrixTC<Coord>
operator - ( const R4MatrixTC<Coord>& m1, const R4MatrixTC<Coord>& m2 )
{
    R4MatrixTC<Coord> ret;

    for (int i = 0; i < 4; i += 1)
        for (int j = 0; j < 4; j += 1)
            ret(i, j) = m1(i, j) - m2(i, j);
    return ret;
}

///
template<class Coord>
inline R4MatrixTC<Coord>
operator * ( const R4MatrixTC<Coord>& m, Coord s )
{
    R4MatrixTC<Coord> ret;

    for (int i = 0; i < 4; i += 1)
        for (int j = 0; j < 4; j += 1)
            ret(i, j) = m(i, j) * s;
    return ret;
}

///
template<class Coord>
inline R4MatrixTC<Coord>
operator * ( Coord s, const R4MatrixTC<Coord>& m )
{
    R4MatrixTC<Coord> ret;

    for (int i = 0; i < 4; i += 1)
        for (int j = 0; j < 4; j += 1)
            ret(i, j) = m(i, j) * s;
    return ret;
}

///
template<class Coord>
inline R4MatrixTC<Coord> 
operator * ( const R4MatrixTC<Coord>& m1, const R4MatrixTC<Coord>& m2 )
{
    R4MatrixTC<Coord> ret;

    for (int i = 0; i < 4; i += 1)
        for (int j = 0; j < 4; j += 1)
            ret(i, j) = m1(i, 0) * m2(0,j) + m1(i,1) * m2(1, j) + m1(i,2) * m2(2,j) + m1(i,3) * m2(3,j);
    return ret;
}

///
template<class Coord>
inline R4MatrixTC<Coord>
operator / ( const R4MatrixTC<Coord>& m, Coord s )
{
    R4MatrixTC<Coord> ret;

    ASSERT( s != 0.0 );

    Coord s1 = 1.0 / s;

    for (int i = 0; i < 4; i += 1)
        for (int j = 0; j < 4; j += 1)
            ret(i, j) = m(i, j) * s1;
    return ret;
}

///
template<class Coord>
inline R4VectorTC<Coord>
operator * ( const R4MatrixTC<Coord>& m, const R4VectorTC<Coord>& v)
{
    return R4VectorTC<Coord>(v[0]*m(0,0) + v[1]*m(0,1) + v[2]*m(0,2) + v[3]*m(0,3),
                             v[0]*m(1,0) + v[1]*m(1,1) + v[2]*m(1,2) + v[3]*m(1,3),
                             v[0]*m(2,0) + v[1]*m(2,1) + v[2]*m(2,2) + v[3]*m(2,3),
                             v[0]*m(3,0) + v[1]*m(3,1) + v[2]*m(3,2) + v[3]*m(3,3)
                             );
}

/// Makes 4th coordinate 0, doesn't divide
template<class Coord>
inline R3VectorTC<Coord>
operator * ( const R4MatrixTC<Coord>& m, const R3VectorTC<Coord>& v)
{
    R4VectorTC<Coord>  vec(v[0]*m(0,0) + v[1]*m(0,1) + v[2]*m(0,2),
                           v[0]*m(1,0) + v[1]*m(1,1) + v[2]*m(1,2),
                           v[0]*m(2,0) + v[1]*m(2,1) + v[2]*m(2,2),
                           v[0]*m(3,0) + v[1]*m(3,1) + v[2]*m(3,2)
                           );

    return R3VectorTC<Coord>( vec[0], vec[1], vec[2] );
}

///
template<class Coord>
inline R4CoVectorTC<Coord>
operator * ( const R4CoVectorTC<Coord>& v, const R4MatrixTC<Coord>& m )
{
    return R4CoVectorTC<Coord>(v[0]*m(0,0) + v[1]*m(0,1) + v[2]*m(0,2) + v[3]*m(0,3),
                               v[0]*m(1,0) + v[1]*m(1,1) + v[2]*m(1,2) + v[3]*m(1,3),
                               v[0]*m(2,0) + v[1]*m(2,1) + v[2]*m(2,2) + v[3]*m(2,3),
                               v[0]*m(3,0) + v[1]*m(3,1) + v[2]*m(3,2) + v[3]*m(3,3)
                               );
}

///
template<class Coord>
inline R4PointTC<Coord>
operator * ( const R4MatrixTC<Coord>& m, const R4PointTC<Coord>& p )
{
    return R4PointTC<Coord>(p[0]*m(0,0) + p[1]*m(0,1) + p[2]*m(0,2) + p[3]*m(0,3),
                            p[0]*m(1,0) + p[1]*m(1,1) + p[2]*m(1,2) + p[3]*m(1,3),
                            p[0]*m(2,0) + p[1]*m(2,1) + p[2]*m(2,2) + p[3]*m(2,3),
                            p[0]*m(3,0) + p[1]*m(3,1) + p[2]*m(3,2) + p[3]*m(3,3)
                            );
}

/// Makes 4th coordinate 1, divides by w
template<class Coord>
inline R3PointTC<Coord>
operator * ( const R4MatrixTC<Coord>& m, const R3PointTC<Coord>& p )
{
    R4PointTC<Coord>  pt(p[0]*m(0,0) + p[1]*m(0,1) + p[2]*m(0,2) + m(0,3),
                         p[0]*m(1,0) + p[1]*m(1,1) + p[2]*m(1,2) + m(1,3),
                         p[0]*m(2,0) + p[1]*m(2,1) + p[2]*m(2,2) + m(2,3),
                         p[0]*m(3,0) + p[1]*m(3,1) + p[2]*m(3,2) + m(3,3)
                        );

    ASSERT( !RNIsZero( pt[3] ) );
    return R3PointTC<Coord>( pt[0] / pt[3], pt[1] / pt[3], pt[2] / pt[3] );
}


//@}

/**@name Comparison operators */
//@{

///
template<class Coord>
inline WINbool
operator == ( const R4VectorTC<Coord>& v1, const R4VectorTC<Coord>& v2 )
{
    return (v1[0] == v2[0] && v1[1] == v2[1] && v1[2] == v2[2] && v1[3] == v2[3]) ? TRUE : FALSE;
}

///
template<class Coord>
inline WINbool
operator != ( const R4VectorTC<Coord>& v1, const R4VectorTC<Coord>& v2 )
{
    return (v1[0] == v2[0] && v1[1] == v2[1] && v1[2] == v2[2] && v1[3] == v2[3]) ? FALSE : TRUE;
}

///
inline WINbool
ApproxEqual(const R4VectorTC<double>& v1, const R4VectorTC<double>& v2)
{
    return ( fabs( v1[0] - v2[0] ) < RNEpsilon_d  &&
             fabs( v1[1] - v2[1] ) < RNEpsilon_d  &&
             fabs( v1[2] - v2[2] ) < RNEpsilon_d  &&
             fabs( v1[3] - v2[3] ) < RNEpsilon_d) ? TRUE : FALSE; 
}


///
inline WINbool
ApproxEqual(const R4VectorTC<float>& v1, const R4VectorTC<float>& v2)
{
    return ( fabs( v1[0] - v2[0] ) < RNEpsilon_f  &&
             fabs( v1[1] - v2[1] ) < RNEpsilon_f  &&
             fabs( v1[2] - v2[2] ) < RNEpsilon_f  &&
             fabs( v1[3] - v2[3] ) < RNEpsilon_f) ? TRUE : FALSE; 
}
///
template<class Coord>
inline WINbool
operator == ( const R4CoVectorTC<Coord>& v1, const R4CoVectorTC<Coord>& v2 )
{
    return (v1[0] == v2[0] && v1[1] == v2[1] && v1[2] == v2[2]) ? TRUE : FALSE;
}

///
template<class Coord>
inline WINbool
operator != ( const R4CoVectorTC<Coord>& v1, const R4CoVectorTC<Coord>& v2 )
{
    return (v1[0] == v2[0] && v1[1] == v2[1] && v1[2] == v2[2] && v1[3] == v2[3]) ? FALSE : TRUE;
}

///
inline WINbool
ApproxEqual(const R4CoVectorTC<double>& v1, const R4CoVectorTC<double>& v2)
{
    return ( fabs( v1[0] - v2[0] ) < RNEpsilon_d  &&
             fabs( v1[1] - v2[1] ) < RNEpsilon_d  &&
             fabs( v1[2] - v2[2] ) < RNEpsilon_d  &&
             fabs( v1[3] - v2[3] ) < RNEpsilon_d) ? TRUE : FALSE; 
}


///
inline WINbool
ApproxEqual(const R4CoVectorTC<float>& v1, const R4CoVectorTC<float>& v2)
{
    return ( fabs( v1[0] - v2[0] ) < RNEpsilon_f  &&
             fabs( v1[1] - v2[1] ) < RNEpsilon_f  &&
             fabs( v1[2] - v2[2] ) < RNEpsilon_f  &&
             fabs( v1[3] - v2[3] ) < RNEpsilon_f) ? TRUE : FALSE; 
}

///
template<class Coord>
inline WINbool
operator == ( const R4PointTC<Coord>& p1, const R4PointTC<Coord>& p2 )
{
    return (p1[0] == p2[0] && p1[1] == p2[1] && p1[2] == p2[2] && p1[3] == p2[3]) ? TRUE : FALSE;
}

///
template<class Coord>
inline WINbool
operator != ( const R4PointTC<Coord>& p1, const R4PointTC<Coord>& p2 )
{
    return (p1[0] == p2[0] && p1[1] == p2[1] && p1[2] == p2[2] && p1[3] == p2[3]) ? FALSE : TRUE;
}

///
inline WINbool
ApproxEqual(const R4PointTC<double>& p1, const R4PointTC<double>& p2)
{
    return ( fabs( p1[0] - p2[0] ) < RNEpsilon_d &&
             fabs( p1[1] - p2[1] ) < RNEpsilon_d &&
             fabs( p1[2] - p2[2] ) < RNEpsilon_d &&
             fabs( p1[3] - p2[3] ) < RNEpsilon_d ) ? TRUE : FALSE; 
}

///
inline WINbool
ApproxEqual(const R4PointTC<float>& p1, const R4PointTC<float>& p2)
{
    return ( fabs( p1[0] - p2[0] ) < RNEpsilon_f &&
             fabs( p1[1] - p2[1] ) < RNEpsilon_f &&
             fabs( p1[2] - p2[2] ) < RNEpsilon_f &&
             fabs( p1[3] - p2[3] ) < RNEpsilon_f ) ? TRUE : FALSE; 
}

///
template<class Coord>
inline WINbool
operator == ( const R4MatrixTC<Coord>& m1, const R4MatrixTC<Coord>& m2)
{
    for (int i = 0; i < 4; i += 1)
        for (int j = 0; j < 4; j += 1)
            if ( m1(i,j) != m2(i,j) )
                return FALSE;
    return TRUE;
}

///
template<class Coord>
inline WINbool
operator != ( const R4MatrixTC<Coord>& m1, const R4MatrixTC<Coord>& m2)
{
    for (int i = 0; i < 4; i += 1)
        for (int j = 0; j < 4; j += 1)
            if ( m1(i,j) != m2(i,j) )
                return TRUE;
    return FALSE;
}


///
inline
WINbool 
ApproxEqual ( const R4MatrixTC<double> &m1, const R4MatrixTC<double> &m2, const double in_dEps = RNEpsilon_d )
{
    for (int i = 0; i < 4; i += 1)
        for (int j = 0; j < 4; j += 1)
            if ( fabs( m1(i,j) - m2(i,j) ) > in_dEps )
                return FALSE;
    return TRUE;
}


///
inline WINbool
ApproxEqual ( const R4MatrixTC<float>& m1, const R4MatrixTC<float>& m2, const float in_fEps = RNEpsilon_f)
{
    for (int i = 0; i < 4; i += 1)
        for (int j = 0; j < 4; j += 1)
            if ( fabs( m1(i,j) - m2(i,j) ) > in_fEps )
                return FALSE;
    return TRUE;
}

//@}
