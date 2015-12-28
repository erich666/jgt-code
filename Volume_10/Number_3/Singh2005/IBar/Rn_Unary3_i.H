// =========================================================
//
//	Unary operators on affine3 classes
//
// =========================================================

/**@name Unary operators */
//@{

// -------------------------------------
// friend functions
// -------------------------------------

///
template<class Coord>
inline Coord
Length( const R3VectorTC<Coord>& v )
{
    return (Coord) sqrt( v[0]*v[0] + v[1]*v[1] + v[2]*v[2] );
}

///
template<class Coord>
inline Coord
LengthSq( const R3VectorTC<Coord>& v )
{
    return v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
}

///
template<class Coord>
inline R3VectorTC<Coord>
Unit( const R3VectorTC<Coord>& v )
{
    return v / Length(v);
}

/// Return given vector if length is zero
template<class Coord>
inline R3VectorTC<Coord>
UnitSafe( const R3VectorTC<Coord>& v, const R3VectorTC<Coord>& out_vIfBad )
{
    Coord dLength = Length( v );

    if ( fabs( dLength ) < (Coord) 1e-12 )
        return out_vIfBad;

    return v / dLength;
}

/// Return 0 vector if length is zero
template<class Coord>
inline R3VectorTC<Coord>
UnitSafe( const R3VectorTC<Coord>& v )
{
    Coord dLength = Length( v );

    if ( fabs( dLength ) < (Coord) 1e-12 )
        return v;

    return v / dLength;
}

///
template<class Coord>
inline Coord
MaximumNorm( const R3VectorTC<Coord>& v )
{
    // L-infinity norm
    Coord max = (Coord) fabs(v[0]);
    if ((Coord) fabs(v[1]) > max)
        max = (Coord) fabs(v[1]);
    if ((Coord) fabs(v[2]) > max)
        max = (Coord) fabs(v[2]);
    return max;
}

///
template<class Coord>
inline R3VectorTC<Coord>  R3VectorTC<Coord>::Perpendicular() const
{
    if ( fabs( (*this)[0] ) >= fabs( (*this)[1] ) && fabs( (*this)[0] ) >= fabs( (*this)[2] ) ) {
        if ( fabs( (*this)[1] ) > fabs( (*this)[2] ) ) {
            return R3VectorTC<Coord>( -(*this)[1], (*this)[0], 0 );
        } else {
            return R3VectorTC<Coord>( -(*this)[2], 0, (*this)[0] );
        }
    } else if ( fabs( (*this)[1] ) >= fabs( (*this)[2] ) ) {
        if ( fabs( (*this)[0] ) > fabs( (*this)[2] ) ) {
            return R3VectorTC<Coord>( -(*this)[1], (*this)[0], 0 );
        } else {
            return R3VectorTC<Coord>( 0, -(*this)[1], (*this)[2] );
        }
    }
    if ( fabs( (*this)[1] ) > fabs( (*this)[0] ) ) {
        return R3VectorTC<Coord>( 0, (*this)[2], -(*this)[1] );
    } else {
        return R3VectorTC<Coord>( (*this)[2], 0, -(*this)[0] );
    }

    ASSERT(FALSE);

    return R3VectorTC<Coord>(0,0,0);
}


// -------------------------------------
// friend functions
// -------------------------------------

///
template<class Coord>
tinline Coord
Length( const R3CoVectorTC<Coord>& v )
{
    return (Coord) sqrt( v[0]*v[0] + v[1]*v[1] + v[2]*v[2] );
}

///
template<class Coord>
tinline Coord
LengthSq( const R3CoVectorTC<Coord>& v )
{
    return v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
}

///
template<class Coord>
tinline R3CoVectorTC<Coord>
Unit( const R3CoVectorTC<Coord>& v )
{
    return v / Length(v);
}

///
template<class Coord>
tinline Coord
MaximumNorm( const R3CoVectorTC<Coord>& v )
{
    // L-infinity norm
    Coord max = (Coord) fabs(v[0]);
    if ((Coord) fabs(v[1]) > max)
        max = (Coord) fabs(v[1]);
    if ((Coord) fabs(v[2]) > max)
        max = (Coord) fabs(v[2]);
    return max;
}


///
template<class Coord>
inline R3MatrixTC<Coord> R3MatrixTC<Coord>::Inverse( WINbool &out_bRes ) const
{
    R3MatrixTC<Coord> m;
    const R3MatrixTC<Coord> &t = *this;

      // In general case, invert it directly

    const double dDet
        = (-t(0,2) * t(1,1) * t(2,0) + 
            t(0,1) * t(1,2) * t(2,0) + 
            t(0,2) * t(1,0) * t(2,1) - 
            t(0,0) * t(1,2) * t(2,1) - 
            t(0,1) * t(1,0) * t(2,2) + 
           t(0,0) * t(1,1) * t(2,2));

    out_bRes = TRUE;
    if (fabs(dDet) < 1e-25) {
        out_bRes = FALSE;
        m = R3MatrixTC<Coord>::Identity();
        return m;
    }
    m(0,0) = (-t(1,2) * t(2,1) + t(1,1) * t(2,2)) / dDet;
    m(0,1) = ( t(0,2) * t(2,1) - t(0,1) * t(2,2)) / dDet;
    m(0,2) = (-t(0,2) * t(1,1) + t(0,1) * t(1,2)) / dDet; 
    m(1,0) = ( t(1,2) * t(2,0) - t(1,0) * t(2,2)) / dDet;
    m(1,1) = (-t(0,2) * t(2,0) + t(0,0) * t(2,2)) / dDet;
    m(1,2) = ( t(0,2) * t(1,0) - t(0,0) * t(1,2)) / dDet;
    m(2,0) = (-t(1,1) * t(2,0) + t(1,0) * t(2,1)) / dDet;
    m(2,1) = ( t(0,1) * t(2,0) - t(0,0) * t(2,1)) / dDet;
    m(2,2) = (-t(0,1) * t(1,0) + t(0,0) * t(1,1)) / dDet;

    return m;
}

//@}

/**@name Conversion operators */
//@{

///
template<class Coord>
inline R3CoVectorTC<Coord>
VecToCov( const R3VectorTC<Coord>& v )
{
    return R3CoVectorTC<Coord>( v[0], v[1], v[2] );
}

///
template<class Coord>
inline R3VectorTC<Coord>
CovToVec( const R3CoVectorTC<Coord>& cv )
{
    return R3VectorTC<Coord>( cv[0], cv[1], cv[2] );
}

///
template<class Coord>
inline R3VectorTC<Coord>
operator - ( const R3PointTC<Coord>& p1, const R3PointTC<Coord>& p2 )
{
    return R3VectorTC<Coord>( p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2] );
}

//@}
