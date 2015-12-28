// =========================================================
//
//	Unary operators on affine2 classes
//
// =========================================================

/**@name Unary operators */
//@{


///
template<class Coord>
inline Coord
Length( const R2VectorTC<Coord>& v)
{
    return (Coord) sqrt(v[0]*v[0] + v[1]*v[1]);
}

///
template<class Coord>
inline Coord
LengthSq( const R2VectorTC<Coord>& v)
{
    return v[0]*v[0] + v[1] * v[1];
}

///
template<class Coord>
inline R2VectorTC<Coord>
Unit( const R2VectorTC<Coord>& v)
{
    return v / Length(v);
}

/// Return given vector if length is zero
template<class Coord>
inline R2VectorTC<Coord>
UnitSafe( const R2VectorTC<Coord>& v, const R2VectorTC<Coord>& out_vIfBad )
{
    Coord dLength = Length( v );

    if ( fabs( dLength ) < (Coord) 1e-12 )
        return out_vIfBad;

    return v / dLength;
}

/// Return 0 vector if length is zero
template<class Coord>
inline R2VectorTC<Coord>
UnitSafe( const R2VectorTC<Coord>& v )
{
    Coord dLength = Length( v );

    if ( fabs( dLength ) < (Coord) 1e-12 )
        return v;

    return v / dLength;
}

///
template<class Coord>
inline Coord
MaximumNorm( const R2VectorTC<Coord>&v)
{
    // L-infinity norm
    Coord max = (Coord) fabs(v[0]);
    if ((Coord) fabs(v[1]) > max)
        max = (Coord) fabs(v[1]);
    return max;
}


///
template<class Coord>
inline Coord
Length( const R2CoVectorTC<Coord>& v)
{
    return (Coord) sqrt(v[0]*v[0] + v[1]*v[1]);
}

///
template<class Coord>
inline Coord
LengthSq( const R2CoVectorTC<Coord>& v)
{
    return v[0]*v[0] + v[1] * v[1];
}

///
template<class Coord>
inline R2CoVectorTC<Coord>
Unit( const R2CoVectorTC<Coord>& v)
{
    return v / Length(v);
}

///
template<class Coord>
inline Coord
MaximumNorm( const R2CoVectorTC<Coord>& v )
{
    // L-infinity norm
    Coord max = (Coord) fabs(v[0]);
    if ( (Coord) fabs(v[1]) > max)
        max = (Coord) fabs(v[1]);
    return max;
}

//@}

/**@name Conversion operators */
//@{


///
template<class Coord>
inline R2CoVectorTC<Coord>
VecToCov( const R2VectorTC<Coord>& v )
{
    return R2CoVectorTC<Coord>( v[0], v[1] );
}

///
template<class Coord>
inline R2VectorTC<Coord>
CovToVec( const R2CoVectorTC<Coord>& cv )
{
    return R2VectorTC<Coord>( cv[0], cv[1] );
}

//@}

