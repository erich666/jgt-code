// =========================================================
//
//	Binary operators on affine3 classes
//
// =========================================================

/**@name Binary operators */
//@{

// -------------------------------------
// binary operators
// -------------------------------------

///
template<class Coord>
inline R3VectorTC<Coord>
operator + ( const R3VectorTC<Coord>& v1, const R3VectorTC<Coord>& v2 )
{
    return R3VectorTC<Coord>( v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2] );
}

///
template<class Coord>
inline R3VectorTC<Coord>
operator - ( const R3VectorTC<Coord>& v1, const R3VectorTC<Coord>& v2 )
{
    return R3VectorTC<Coord>( v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2] );
}

///
template<class Coord>
inline R3VectorTC<Coord>
operator * ( const Coord& s, const R3VectorTC<Coord>& v )
{
    return R3VectorTC<Coord>( v[0] * s, v[1] * s, v[2] * s );
}

///
template<class Coord>
inline R3VectorTC<Coord>
operator * ( const R3VectorTC<Coord>& v, const Coord& s )
{
    return R3VectorTC<Coord>( v[0] * s, v[1] * s, v[2] * s );
}

///
template<class Coord>
inline R3VectorTC<Coord>
operator / ( const R3VectorTC<Coord>& v, const Coord& s )
{
    ASSERT( !RNIsZero( s ) );

    Coord s1 = (Coord) 1.0 / s;
    return R3VectorTC<Coord>( v[0] * s1, v[1] * s1, v[2] * s1 );
}


///
template<class Coord>
inline Coord
Dot( const R3VectorTC<Coord>& v1, const R3VectorTC<Coord>& v2 )
{
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

///
template<class Coord>
inline Coord
Dot( const R3VectorTC<Coord>& v1, const R3CoVectorTC<Coord>& v2 )
{
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

///
template<class Coord>
inline Coord
Dot( const R3CoVectorTC<Coord>& v1, const R3VectorTC<Coord>& v2 )
{
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

///
template<class Coord>
inline R3VectorTC<Coord>
Cross( const R3VectorTC<Coord>& v1, const R3VectorTC<Coord>& v2 )
{
    return R3VectorTC<Coord>(v1[1]*v2[2] - v1[2]*v2[1],
                            v1[2]*v2[0] - v1[0]*v2[2],
                            v1[0]*v2[1] - v1[1]*v2[0]);
}

///Projection: the projection of v2 onto v1
//I choose this order because the math notation looks like:
//
//  proj  v2
//      v1
//
template<class Coord>
tinline R3VectorTC<Coord>
Projection( const R3VectorTC<Coord>& v1, const R3VectorTC<Coord>& v2 )
{
    return (Dot(v1,v2)/Dot(v1,v1)) * v1;
}

///Rejection: the rejection of v2 by v1
//I chose this order to match that of Projection
template<class Coord>
tinline R3VectorTC<Coord>
Rejection( const R3VectorTC<Coord>& v1, const R3VectorTC<Coord>& v2 )
{
    return v2 - Projection(v1,v2);
}

///
template<class Coord>
inline R3MatrixTC<Coord>
OuterProduct( const R3VectorTC<Coord>& v1, const R3VectorTC<Coord>& v2 )
{
    R3MatrixTC<Coord> ret;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            ret(i,j)=v1[i]*v2[j];
    return ret;
}

///
template<class Coord>
inline R3MatrixTC<Coord>
OuterProduct( const R3CoVectorTC<Coord>& v1, const R3VectorTC<Coord>& v2 )
{
    R3MatrixTC<Coord> ret;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            ret(i,j)=v1[i]*v2[j];
    return ret;
}

///
template<class Coord>
inline R3MatrixTC<Coord>
OuterProduct( const R3VectorTC<Coord>& v1, const R3CoVectorTC<Coord>& v2 )
{
    R3MatrixTC<Coord> ret;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            ret(i,j)=v1[i]*v2[j];
    return ret;
}

///
template<class Coord>
inline R3MatrixTC<Coord>
OuterProduct( const R3CoVectorTC<Coord>& v1, const R3CoVectorTC<Coord>& v2 )
{
    R3MatrixTC<Coord> ret;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            ret(i,j)=v1[i]*v2[j];
    return ret;
}

///
template<class Coord>
inline R3MatrixTC<Coord>
AlternatingProduct( const R3VectorTC<Coord>& v1, const R3VectorTC<Coord>& v2 )
{
    R3MatrixTC<Coord> ret;

    ret(0,0) = ret(1,1) = ret(2,2) = 0.0f;
    ret(1,0) = v1[1]*v2[0]-v1[0]*v2[1];
    ret(2,0) = v1[2]*v2[0]-v1[0]*v2[2]; ret(2,1) = v1[2]*v2[1]-v1[1]*v2[2];
    ret(0,1) = -ret(1,0);
    ret(0,2) = -ret(2,0);   ret(1,2) = -ret(2,1);

    return ret;
}

///
template<class Coord>
inline R3MatrixTC<Coord>
AlternatingProduct( const R3VectorTC<Coord>& v1, const R3CoVectorTC<Coord>& v2 )
{
    R3MatrixTC<Coord> ret;

    ret(0,0) = ret(1,1) = ret(2,2) = 0.0f;
    ret(1,0) = v1[1]*v2[0]-v1[0]*v2[1];
    ret(2,0) = v1[2]*v2[0]-v1[0]*v2[2]; ret(2,1) = v1[2]*v2[1]-v1[1]*v2[2];
    ret(0,1) = -ret(1,0);
    ret(0,2) = -ret(2,0);   ret(1,2) = -ret(2,1);

    return ret;
}

///
template<class Coord>
inline R3MatrixTC<Coord>
AlternatingProduct( const R3CoVectorTC<Coord>& v1, const R3VectorTC<Coord>& v2 )
{
    R3MatrixTC<Coord> ret;

    ret(0,0) = ret(1,1) = ret(2,2) = 0.0f;
    ret(1,0) = v1[1]*v2[0]-v1[0]*v2[1];
    ret(2,0) = v1[2]*v2[0]-v1[0]*v2[2]; ret(2,1) = v1[2]*v2[1]-v1[1]*v2[2];
    ret(0,1) = -ret(1,0);
    ret(0,2) = -ret(2,0);   ret(1,2) = -ret(2,1);

    return ret;
}

///
template<class Coord>
inline R3MatrixTC<Coord>
AlternatingProduct( const R3CoVectorTC<Coord>& v1, const R3CoVectorTC<Coord>& v2 )
{
    R3MatrixTC<Coord> ret;

    ret(0,0) = ret(1,1) = ret(2,2) = 0.0f;
    ret(1,0) = v1[1]*v2[0]-v1[0]*v2[1];
    ret(2,0) = v1[2]*v2[0]-v1[0]*v2[2]; ret(2,1) = v1[2]*v2[1]-v1[1]*v2[2];
    ret(0,1) = -ret(1,0);
    ret(0,2) = -ret(2,0);   ret(1,2) = -ret(2,1);

    return ret;
}

// -------------------------------------
// binary operators
// -------------------------------------

///
template<class Coord>
inline R3CoVectorTC<Coord>
operator + ( const R3CoVectorTC<Coord>& v1, const R3CoVectorTC<Coord>& v2 )
{
    return R3CoVectorTC<Coord>( v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2] );
}

///
template<class Coord>
inline R3CoVectorTC<Coord>
operator - ( const R3CoVectorTC<Coord>& v1, const R3CoVectorTC<Coord>& v2 )
{
    return R3CoVectorTC<Coord>( v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2] );
}

///
template<class Coord>
inline R3CoVectorTC<Coord>
operator * ( const Coord& s, const R3CoVectorTC<Coord>& v )
{
    return R3CoVectorTC<Coord>( v[0] * s, v[1] * s, v[2] * s );
}

///
template<class Coord>
inline R3CoVectorTC<Coord>
operator * ( const R3CoVectorTC<Coord>& v, const Coord& s )
{
    return R3CoVectorTC<Coord>( v[0] * s, v[1] * s, v[2] * s );
}

///
template<class Coord>
inline R3CoVectorTC<Coord>
operator / ( const R3CoVectorTC<Coord>& v, const Coord& s )
{
    Coord s1 = Coord(1.0f) / s;
    return R3CoVectorTC<Coord>( v[0] * s1, v[1] * s1, v[2] * s1 );
}

///
template<class Coord>
tinline Coord
Dot( const R3CoVectorTC<Coord>& v1, const R3CoVectorTC<Coord>& v2 )
{
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

///
template<class Coord>
tinline R3CoVectorTC<Coord>
Cross( const R3CoVectorTC<Coord>& v1, const R3CoVectorTC<Coord>& v2 )
{
    return R3CoVectorTC<Coord>(v1[1]*v2[2] - v1[2]*v2[1],
                              v1[2]*v2[0] - v1[0]*v2[2],
                              v1[0]*v2[1] - v1[1]*v2[0]);
}


///Projection: the projection of v2 onto v1
//I choose this order because the math notation looks like:
//
//  proj  v2
//      v1
//
template<class Coord>
tinline R3CoVectorTC<Coord>
Projection( const R3CoVectorTC<Coord>& v1, const R3CoVectorTC<Coord>& v2 )
{
    return (Dot(v1,v2)/Dot(v1,v1)) * v1;
}


///
template<class Coord>
inline R3PointTC<Coord>
operator + ( const R3PointTC<Coord>& p, const R3VectorTC<Coord>& vDelta )
{
    return R3PointTC<Coord>( p[0] + vDelta[0], p[1] + vDelta[1], p[2] + vDelta[2] );
}

///
template<class Coord>
inline R3PointTC<Coord>
operator + ( const R3CoVectorTC<Coord>& vDelta, const R3PointTC<Coord>& p )
{
    return R3PointTC<Coord>( p[0] + vDelta[0], p[1] + vDelta[1], p[2] + vDelta[2] );
}

///
template<class Coord>
inline R3PointTC<Coord>
operator - ( const R3PointTC<Coord>& p, const R3VectorTC<Coord>& vDelta )
{
    return R3PointTC<Coord>( p[0] - vDelta[0], p[1] - vDelta[1], p[2] - vDelta[2] );
}

// -------------------------------------
// friend functions
// -------------------------------------

///
template<class Coord>
inline R3PointTC<Coord>
Lerp( const R3PointTC<Coord>&p1, const R3PointTC<Coord>&p2, Coord s )
{
    R3VectorTC<Coord> p1p2 = p2 - p1;
    return R3PointTC<Coord>( p1 + s*p1p2 );
}

///
template<class Coord>
inline R3PointTC<Coord>
Barycentric( const R3PointTC<Coord>& p, 
             const R3PointTC<Coord>& q, 
             const R3PointTC<Coord>& r,
             Coord f, Coord g)
{
    return p+f*(q-p)+g*(r-p);
}


///
template<class Coord>
inline R3MatrixTC<Coord>
operator + ( const R3MatrixTC<Coord>& m1, const R3MatrixTC<Coord>& m2 )
{
    R3MatrixTC<Coord> ret;

    for (int i = 0; i < 3; i += 1)
        for (int j = 0; j < 3; j += 1)
            ret(i, j) = m1(i, j) + m2(i, j);
    return ret;
}

///
template<class Coord>
inline R3MatrixTC<Coord>
operator - ( const R3MatrixTC<Coord>& m1, const R3MatrixTC<Coord>& m2 )
{
    R3MatrixTC<Coord> ret;

    for (int i = 0; i < 3; i += 1)
        for (int j = 0; j < 3; j += 1)
            ret(i, j) = m1(i, j) - m2(i, j);
    return ret;
}

///
template<class Coord>
inline R3MatrixTC<Coord>
operator * ( const R3MatrixTC<Coord>& m, Coord s )
{
    R3MatrixTC<Coord> ret;

    for (int i = 0; i < 3; i += 1)
        for (int j = 0; j < 3; j += 1)
            ret(i, j) = m(i, j) * s;
    return ret;
}

///
template<class Coord>
inline R3MatrixTC<Coord>
operator * ( Coord s, const R3MatrixTC<Coord>& m )
{
    R3MatrixTC<Coord> ret;

    for (int i = 0; i < 3; i += 1)
        for (int j = 0; j < 3; j += 1)
            ret(i, j) = m(i, j) * s;
    return ret;
}

///
template<class Coord>
inline R3MatrixTC<Coord> 
operator * ( const R3MatrixTC<Coord>& m1, const R3MatrixTC<Coord>& m2 )
{
    R3MatrixTC<Coord> ret;

    for (int i = 0; i < 3; i += 1)
        for (int j = 0; j < 3; j += 1)
            ret(i, j) = m1(i, 0) * m2(0,j) + m1(i,1) * m2(1, j) + m1(i,2) * m2(2,j);
    return ret;
}

///
template<class Coord>
inline R3MatrixTC<Coord>
operator / ( const R3MatrixTC<Coord>& m, Coord s )
{
    R3MatrixTC<Coord> ret;

    ASSERT( s != 0.0 );

    Coord s1 = 1.0 / s;

    for (int i = 0; i < 3; i += 1)
        for (int j = 0; j < 3; j += 1)
            ret(i, j) = m(i, j) * s1;
    return ret;
}

///
template<class Coord>
inline R3VectorTC<Coord>
operator * ( const R3MatrixTC<Coord>& m, const R3VectorTC<Coord>& v)
{
    return R3VectorTC<Coord>(v[0]*m(0,0) + v[1]*m(0,1) + v[2]*m(0,2),
                             v[0]*m(1,0) + v[1]*m(1,1) + v[2]*m(1,2),
                             v[0]*m(2,0) + v[1]*m(2,1) + v[2]*m(2,2)
                             );
}

///
template<class Coord>
inline R2VectorTC<Coord>
operator * ( const R3MatrixTC<Coord>& m, const R2VectorTC<Coord>& v)
{
    R3VectorTC<Coord> vec (v[0]*m(0,0) + v[1]*m(0,1) + m(0,2),
                           v[0]*m(1,0) + v[1]*m(1,1) + m(1,2),
                           v[0]*m(2,0) + v[1]*m(2,1) + m(2,2)
                           );

    return R2VectorTC<Coord>( vec[0], vec[1] );
}

///
template<class Coord>
inline R3CoVectorTC<Coord>
operator * ( const R3CoVectorTC<Coord>& v, const R3MatrixTC<Coord>& m )
{
    return R3CoVectorTC<Coord>(v[0]*m(0,0) + v[1]*m(1,0) + v[2]*m(2,0),
                               v[0]*m(0,1) + v[1]*m(1,1) + v[2]*m(2,1),
                               v[0]*m(0,2) + v[1]*m(1,2) + v[2]*m(2,2)
                               );
}

///
template<class Coord>
inline R2CoVectorTC<Coord>
operator * ( const R2CoVectorTC<Coord>& v, const R3MatrixTC<Coord>& m )
{
    R3CoVectorTC<Coord> vec(v[0]*m(0,0) + v[1]*m(1,0) + m(2,0),
                            v[0]*m(0,1) + v[1]*m(1,1) + m(2,1),
                            v[0]*m(0,2) + v[1]*m(1,2) + m(2,2)
                            );

    return R2CoVectorTC<Coord>( vec[0], vec[1] );
}

///
template<class Coord>
inline R3PointTC<Coord>
operator * ( const R3MatrixTC<Coord>& m, const R3PointTC<Coord>& p )
{
    return R3PointTC<Coord>(p[0]*m(0,0) + p[1]*m(0,1) + p[2]*m(0,2),
                            p[0]*m(1,0) + p[1]*m(1,1) + p[2]*m(1,2),
                            p[0]*m(2,0) + p[1]*m(2,1) + p[2]*m(2,2)
                            );
}


///
template<class Coord>
inline R2PointTC<Coord>
operator * ( const R3MatrixTC<Coord>& m, const R2PointTC<Coord>& p )
{
    R3PointTC<Coord> pt( p[0]*m(0,0) + p[1]*m(0,1) + m(0,2),
                         p[0]*m(1,0) + p[1]*m(1,1) + m(1,2),
                         p[0]*m(2,0) + p[1]*m(2,1) + m(2,2)
                         );

    ASSERT( !RNIsZero( pt[2] ) );
    return R2PointTC<Coord>(pt[0] / pt[2], pt[1] / pt[2] );
}


//@}

/**@name Comparison operators */
//@{

///
template<class Coord>
inline WINbool
operator == ( const R3VectorTC<Coord>& v1, const R3VectorTC<Coord>& v2 )
{
    return (v1[0] == v2[0] && v1[1] == v2[1] && v1[2] == v2[2]) ? TRUE : FALSE;
}

///
template<class Coord>
inline WINbool
operator != ( const R3VectorTC<Coord>& v1, const R3VectorTC<Coord>& v2 )
{
    return (v1[0] == v2[0] && v1[1] == v2[1] && v1[2] == v2[2]) ? FALSE : TRUE;
}

///
template<class Coord>
inline WINbool
ApproxEqual(const R3VectorTC<Coord>& v1, const R3VectorTC<Coord>& v2, const Coord in_eps)
{
    return ( fabs( v1[0] - v2[0] ) < in_eps  &&
             fabs( v1[1] - v2[1] ) < in_eps  &&
             fabs( v1[2] - v2[2] ) < in_eps) ? TRUE : FALSE; 
}

///
inline WINbool
ApproxEqual(const R3VectorTC<double>& v1, const R3VectorTC<double>& v2)
{
    return ( fabs( v1[0] - v2[0] ) < RNEpsilon_d  &&
             fabs( v1[1] - v2[1] ) < RNEpsilon_d  &&
             fabs( v1[2] - v2[2] ) < RNEpsilon_d) ? TRUE : FALSE; 
}


///
inline WINbool
ApproxEqual(const R3VectorTC<float>& v1, const R3VectorTC<float>& v2)
{
    return ( fabs( v1[0] - v2[0] ) < RNEpsilon_f  &&
             fabs( v1[1] - v2[1] ) < RNEpsilon_f  &&
             fabs( v1[2] - v2[2] ) < RNEpsilon_f) ? TRUE : FALSE; 
}
///
template<class Coord>
inline WINbool
operator == ( const R3CoVectorTC<Coord>& v1, const R3CoVectorTC<Coord>& v2 )
{
    return (v1[0] == v2[0] && v1[1] == v2[1] && v1[2] == v2[2]) ? TRUE : FALSE;
}

///
template<class Coord>
inline WINbool
operator != ( const R3CoVectorTC<Coord>& v1, const R3CoVectorTC<Coord>& v2 )
{
    return (v1[0] == v2[0] && v1[1] == v2[1] && v1[2] == v2[2]) ? FALSE : TRUE;
}

///
inline WINbool
ApproxEqual(const R3CoVectorTC<double>& v1, const R3CoVectorTC<double>& v2)
{
    return ( fabs( v1[0] - v2[0] ) < RNEpsilon_d  &&
             fabs( v1[1] - v2[1] ) < RNEpsilon_d  &&
             fabs( v1[2] - v2[2] ) < RNEpsilon_d) ? TRUE : FALSE; 
}


///
inline WINbool
ApproxEqual(const R3CoVectorTC<float>& v1, const R3CoVectorTC<float>& v2)
{
    return ( fabs( v1[0] - v2[0] ) < RNEpsilon_f  &&
             fabs( v1[1] - v2[1] ) < RNEpsilon_f  &&
             fabs( v1[2] - v2[2] ) < RNEpsilon_f) ? TRUE : FALSE; 
}

///
template<class Coord>
inline WINbool
operator == ( const R3PointTC<Coord>& p1, const R3PointTC<Coord>& p2 )
{
    return (p1[0] == p2[0] && p1[1] == p2[1] && p1[2] == p2[2]) ? TRUE : FALSE;
}

///
template<class Coord>
inline WINbool
operator != ( const R3PointTC<Coord>& p1, const R3PointTC<Coord>& p2 )
{
    return (p1[0] == p2[0] && p1[1] == p2[1] && p1[2] == p2[2]) ? FALSE : TRUE;
}


///
template<class Coord>
inline WINbool
ApproxEqual(const R3PointTC<Coord>& p1, const R3PointTC<Coord>& p2, const Coord in_eps)
{
    return ( fabs( p1[0] - p2[0] ) < in_eps &&
             fabs( p1[1] - p2[1] ) < in_eps &&
             fabs( p1[2] - p2[2] ) < in_eps ) ? TRUE : FALSE; 
}

///
inline WINbool
ApproxEqual(const R3PointTC<double>& p1, const R3PointTC<double>& p2)
{
    return ( fabs( p1[0] - p2[0] ) < RNEpsilon_d &&
             fabs( p1[1] - p2[1] ) < RNEpsilon_d &&
             fabs( p1[2] - p2[2] ) < RNEpsilon_d ) ? TRUE : FALSE; 
}

///
inline WINbool
ApproxEqual(const R3PointTC<float>& p1, const R3PointTC<float>& p2)
{
    return ( fabs( p1[0] - p2[0] ) < RNEpsilon_f &&
             fabs( p1[1] - p2[1] ) < RNEpsilon_f &&
             fabs( p1[2] - p2[2] ) < RNEpsilon_f ) ? TRUE : FALSE; 
}

///
template<class Coord>
inline WINbool
operator == ( const R3MatrixTC<Coord>& m1, const R3MatrixTC<Coord>& m2)
{
    for (int i = 0; i < 3; i += 1)
        for (int j = 0; j < 3; j += 1)
            if ( m1(i,j) != m2(i,j) )
                return FALSE;
    return TRUE;
}

///
template<class Coord>
inline WINbool
operator != ( const R3MatrixTC<Coord>& m1, const R3MatrixTC<Coord>& m2)
{
    for (int i = 0; i < 3; i += 1)
        for (int j = 0; j < 3; j += 1)
            if ( m1(i,j) != m2(i,j) )
                return TRUE;
    return FALSE;
}


///
inline
WINbool 
ApproxEqual ( const R3MatrixTC<double> &m1, const R3MatrixTC<double> &m2 )
{
    for (int i = 0; i < 3; i += 1)
        for (int j = 0; j < 3; j += 1)
            if ( fabs( m1(i,j) - m2(i,j) ) > RNEpsilon_d )
                return FALSE;
    return TRUE;
}

///
inline
WINbool 
ApproxEqual ( const R3MatrixTC<double> &m1, const R3MatrixTC<double> &m2, const double in_dEps )
{
    for (int i = 0; i < 3; i += 1)
        for (int j = 0; j < 3; j += 1)
            if ( fabs( m1(i,j) - m2(i,j) ) > in_dEps )
                return FALSE;
    return TRUE;
}


///
inline WINbool
ApproxEqual ( const R3MatrixTC<float>& m1, const R3MatrixTC<float>& m2)
{
    for (int i = 0; i < 3; i += 1)
        for (int j = 0; j < 3; j += 1)
            if ( fabs( m1(i,j) - m2(i,j) ) > RNEpsilon_f )
                return FALSE;
    return TRUE;
}
///
inline WINbool
ApproxEqual ( const R3MatrixTC<float>& m1, const R3MatrixTC<float>& m2, const float in_fEps)
{
    for (int i = 0; i < 3; i += 1)
        for (int j = 0; j < 3; j += 1)
            if ( fabs( m1(i,j) - m2(i,j) ) > in_fEps )
                return FALSE;
    return TRUE;
}

//@}
