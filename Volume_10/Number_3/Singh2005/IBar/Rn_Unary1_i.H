// =========================================================
//
//	Unary operators on affine1 classes
//
// =========================================================

/**@name Unary operators */
//@{

///
template<class Coord>
inline Coord
Length( const R1VectorTC<Coord>& v)
{
    return (Coord) fabs(v[0]);
}

///
template<class Coord>
inline Coord
LengthSq( const R1VectorTC<Coord>& v)
{
    return v[0]*v[0];
}

///
template<class Coord>
inline R1VectorTC<Coord>
Unit( const R1VectorTC<Coord>& v)
{
    return v / Length(v);
}

/// Return given vector if length is zero
template<class Coord>
inline R1VectorTC<Coord>
UnitSafe( const R1VectorTC<Coord>& v, const R1VectorTC<Coord>& out_vIfBad )
{
    Coord dLength = Length( v );

    if ( fabs( dLength ) < (Coord) 1e-12 )
        return out_vIfBad;

    return v / dLength;
}

/// Return 0 vector if length is zero
template<class Coord>
inline R1VectorTC<Coord>
UnitSafe( const R1VectorTC<Coord>& v )
{
    Coord dLength = Length( v );

    if ( fabs( dLength ) < (Coord) 1e-12 )
        return v;

    return v / dLength;
}

///
template<class Coord>
inline Coord
MaximumNorm( const R1VectorTC<Coord>&v)
{
    // L-infinity norm
    return v[0];
}





///
template<class Coord>
inline Coord
Length( const R1CoVectorTC<Coord>& v)
{
    
    return (Coord) fabs(v[0]);
}


///
template<class Coord>
inline Coord
LengthSq( const R1CoVectorTC<Coord>& v)
{
    
    return v[0]*v[0];
}


///
template<class Coord>
inline R1CoVectorTC<Coord>
Unit( const R1CoVectorTC<Coord>& v)
{
    
    return v / Length(v);
}


///
template<class Coord>
inline Coord
MaximumNorm( const R1CoVectorTC<Coord>& v )
{
    /// L-infinity norm
    return v[0];
}



//@}

/**@name Conversion operators */
//@{
///
template<class Coord>
inline R1CoVectorTC<Coord>
VecToCov( const R1VectorTC<Coord>& v )
{
    return R1CoVectorTC<Coord>( v[0] );
}

///
template<class Coord>
inline R1VectorTC<Coord>
CovToVec( const R1CoVectorTC<Coord>& cv )
{
    return R1VectorTC<Coord>( cv[0] );
}


//@}
