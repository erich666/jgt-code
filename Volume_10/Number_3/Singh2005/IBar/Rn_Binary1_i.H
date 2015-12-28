// =========================================================
//
//	Binary operators on affine1 classes
//
// =========================================================

/**@name Binary operators */
//@{


///
template<class Coord>
inline R1VectorTC<Coord>
operator + ( const R1VectorTC<Coord>& v1, const R1VectorTC<Coord>& v2 )
{
    return R1VectorTC<Coord>( v1[0] + v2[0] );
}

///
template<class Coord>
inline R1VectorTC<Coord>
operator - ( const R1VectorTC<Coord>& v1, const R1VectorTC<Coord>& v2 )
{
    return R1VectorTC<Coord>( v1[0] - v2[0] );
}

///
template<class Coord>
inline R1VectorTC<Coord>
operator * ( const Coord& s, const R1VectorTC<Coord>& v )
{
    return R1VectorTC<Coord>( v[0] * s );
}

///
template<class Coord>
inline R1VectorTC<Coord>
operator * ( const R1VectorTC<Coord>& v, const Coord& s )
{
    return R1VectorTC<Coord>( v[0] * s );
}

///
template<class Coord>
inline R1VectorTC<Coord>
operator / ( const R1VectorTC<Coord>& v, const Coord& s )
{
     ASSERT( !RNIsZero( s ) );

   return R1VectorTC<Coord>( v[0] / s );
}

///
template<class Coord>
inline Coord
Dot( const R1VectorTC<Coord>& v1, const R1VectorTC<Coord>& v2 )
{
    return v1[0] * v2[0];
}

///
template<class Coord>
inline Coord
Dot( const R1VectorTC<Coord>& v1, const R1CoVectorTC<Coord>& v2 )
{
    return v1[0] * v2[0];
}

///
template<class Coord>
inline Coord
Dot( const R1CoVectorTC<Coord>& v1, const R1VectorTC<Coord>& v2 )
{
    return v1[0] * v2[0];
}

///
template<class Coord>
inline Coord
CCW( const R1VectorTC<Coord>& v1, const R1VectorTC<Coord>& v2 )
{
    return v1[0] * v2[0];
}

///
template<class Coord>
inline R1VectorTC<Coord>
operator-( const R1PointTC<Coord>& p1, const R1PointTC<Coord>& p2)
{
    return R1VectorTC<Coord>( p1[0] - p2[0] );
}

///
template<class Coord>
inline R1CoVectorTC<Coord>
operator + ( const R1CoVectorTC<Coord>& v1, const R1CoVectorTC<Coord>& v2 )
{
    return R1CoVectorTC<Coord>( v1[0] + v2[0] );
}

///
template<class Coord>
inline R1CoVectorTC<Coord>
operator - ( const R1CoVectorTC<Coord>& v1, const R1CoVectorTC<Coord>& v2 )
{
    
    return R1CoVectorTC<Coord>( v1[0] - v2[0] );
}


///
template<class Coord>
inline R1CoVectorTC<Coord>
operator * ( const Coord& s, const R1CoVectorTC<Coord>& v )
{
    
    return R1CoVectorTC<Coord>( v[0] * s );
}


///
template<class Coord>
inline R1CoVectorTC<Coord>
operator * ( const R1CoVectorTC<Coord>& v, const Coord& s )
{
    
    return R1CoVectorTC<Coord>( v[0] * s );
}


///
template<class Coord>
inline R1CoVectorTC<Coord>
operator / ( const R1CoVectorTC<Coord>& v, const Coord& s )
{
    
    return R1CoVectorTC<Coord>(v[0] / s );
}

///
template<class Coord>
inline Coord
Dot( const R1CoVectorTC<Coord>& v1, const R1CoVectorTC<Coord>& v2 )
{
    
    return v1[0] * v2[0];
}


///
template<class Coord>
inline Coord
CCW( const R1CoVectorTC<Coord>& v1, const R1CoVectorTC<Coord>& v2 )
{
    
    return v1[0] * v2[0];
}

///
template<class Coord>
inline R1PointTC<Coord>
operator + ( const R1PointTC<Coord>& p, const R1VectorTC<Coord>& vDelta )
{
    
    return R1PointTC<Coord>( p[0] + vDelta[0] );
}


///
template<class Coord>
inline R1PointTC<Coord>
operator + ( const R1VectorTC<Coord>& vDelta, const R1PointTC<Coord>& p )
{
    
    return R1PointTC<Coord>( p[0] + vDelta[0] );
}


///
template<class Coord>
inline R1PointTC<Coord>
operator - ( const R1PointTC<Coord>& p, const R1VectorTC<Coord>& vDelta )
{
    
    return R1PointTC<Coord>( p[0] - vDelta[0] );
}


///
template<class Coord>
inline R1PointTC<Coord>
Lerp( const R1PointTC<Coord>&p1, const R1PointTC<Coord>&p2, const Coord& s )
{
    
    R1VectorTC<Coord> p1p2 = p2 - p1;
    
    return R1PointTC<Coord>( p1 + s*p1p2 );
}


//@}

/**@name Comparison operators */
//@{
///
template<class Coord>
inline WINbool
operator==(const R1VectorTC<Coord>& v1, const R1VectorTC<Coord>& v2)
{
    return (v1[0] == v2[0]) ? TRUE : FALSE;
}

///
template<class Coord>
inline WINbool
operator!=(const R1VectorTC<Coord>& v1, const R1VectorTC<Coord>& v2)
{
    return (v1[0] != v2[0]) ? TRUE : FALSE;
}

///
inline
WINbool
ApproxEqual(const R1VectorTC<double>& v1, const R1VectorTC<double>& v2)
{
    return ( fabs( v1[0] - v2[0] ) < RNEpsilon_d ) ? TRUE : FALSE; 
}

///
inline WINbool
ApproxEqual(const R1VectorTC<float>& v1, const R1VectorTC<float>& v2)
{
    return ( fabs( v1[0] - v2[0] ) < RNEpsilon_f ) ? TRUE : FALSE; 
}

///
template<class Coord>
inline WINbool
operator == ( const R1CoVectorTC<Coord>& v1, const R1CoVectorTC<Coord>& v2 )
{
    
    return (v1[0] == v2[0]) ? TRUE : FALSE;
}


///
template<class Coord>
inline WINbool
operator != ( const R1CoVectorTC<Coord>& v1, const R1CoVectorTC<Coord>& v2 )
{
    return (v1[0] == v2[0]) ? FALSE : TRUE;
}

///
inline WINbool
ApproxEqual( const R1CoVectorTC<double> & v1, const R1CoVectorTC<double> & v2)
{
    return ( fabs( v1[0] - v2[0] ) < RNEpsilon_d ) ? TRUE : FALSE; 
}

///
inline WINbool
ApproxEqual(const R1CoVectorTC<float>& v1, const R1CoVectorTC<float>& v2)
{
    return ( fabs( v1[0] - v2[0] ) < RNEpsilon_f ) ? TRUE : FALSE; 
}

///
template<class Coord>
inline WINbool
operator == ( const R1PointTC<Coord>& p1, const R1PointTC<Coord>& p2 )
{
    
    return (p1[0] == p2[0]) ? TRUE : FALSE;
}


///
template<class Coord>
inline WINbool
operator != ( const R1PointTC<Coord>& p1, const R1PointTC<Coord>& p2 )
{
    return (p1[0] == p2[0]) ? FALSE : TRUE;
}

///
inline WINbool
ApproxEqual(const R1PointTC<double>& p1, const R1PointTC<double>& p2)
{
    return ( fabs( p1[0] - p2[0] ) < RNEpsilon_d ) ? TRUE : FALSE; 
}

///
inline WINbool
ApproxEqual(const R1PointTC<float>& p1, const R1PointTC<float>& p2)
{
    return ( fabs( p1[0] - p2[0] ) < RNEpsilon_f ) ? TRUE : FALSE; 
}




//@}
