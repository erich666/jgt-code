// =========================================================
//
//	Methods for R4VectorTC template class
//
// =========================================================


// -------------------------------------
// constructors
// -------------------------------------

template<class Coord>
inline
R4VectorTC<Coord>::R4VectorTC( const Coord& _d )
{
    dx = dy = dz = dw = _d;
}

template<class Coord>
inline
R4VectorTC<Coord>::R4VectorTC( const Coord& _dx, const Coord& _dy, const Coord& _dz, const Coord& _dw )
{
    dx = _dx; dy = _dy; dz = _dz; dw = _dw;
}

template<class Coord>
inline
R4VectorTC<Coord>::R4VectorTC( const R3VectorTC<Coord>& vec )
{
    dx = vec[0]; dy = vec[1]; dz = vec[2]; 
    dw = 1;
}

// -------------------------------------
// assignment operators
// -------------------------------------

template<class Coord>
inline R4VectorTC<Coord>&
R4VectorTC<Coord>::operator += ( const R4VectorTC<Coord>& v)
{
    dx += v.dx;
    dy += v.dy;
    dz += v.dz;
    dw += v.dw;
    return *this;
}

template<class Coord>
inline R4VectorTC<Coord>&
R4VectorTC<Coord>::operator -= ( const R4VectorTC<Coord>& v )
{
    dx -= v.dx;
    dy -= v.dy;
    dz -= v.dz;
    dw -= v.d2;
    return *this;
}

template<class Coord>
inline R4VectorTC<Coord>&
R4VectorTC<Coord>::operator *= ( const Coord& s )
{
    dx *= s;
    dy *= s;
    dz *= s;
    dw *= s;
    return *this;
}

template<class Coord>
inline R4VectorTC<Coord>&
R4VectorTC<Coord>::operator /= ( const Coord& s )
{
    ASSERT( !RNIsZero( s ) );
    Coord s1 = 1.0f / s;
    dx *= s1;
    dy *= s1;
    dz *= s1;
    dw *= s1;
    return *this;
}

// -------------------------------------
// unary operators
// -------------------------------------

template<class Coord>
inline R4VectorTC<Coord>
R4VectorTC<Coord>::operator + () const
{
    return *this;
}

template<class Coord>
inline R4VectorTC<Coord>
R4VectorTC<Coord>::operator - () const
{
    return R4VectorTC<Coord>( -dx, -dy, -dz, -dw );
}

// -------------------------------------
// Self-editing functions
// -------------------------------------
template<class Coord>
inline double R4VectorTC<Coord>::Normalize()
{
    const double dLen = Length( *this );
    if ( ! RNIsZero( dLen ) ) {
        dx = dx / dLen;
        dy = dy / dLen;
        dz = dz / dLen;
        dw = dw / dLen;
    }

    return dLen;
}



// -------------------------------------
// Read/write/print functions
// -------------------------------------
template<class Coord>
inline void R4VectorTC<Coord>::Write(ofstream &out) const
{
    out << dx << " " << dy << " " << dz << " " << dw << " ";
}

template<class Coord>
inline WINbool R4VectorTC<Coord>::Read(ifstream &in) 
{
    in >> dx >> dy >> dz >> dw;

    return in.good() ? TRUE : FALSE;
}

template<class Coord>
inline void R4VectorTC<Coord>::Print( const WINbool in_bDoReturn ) const
{
    cout << dx << " " << dy << " " << dz << " " << dw;
    if ( in_bDoReturn == TRUE )
        cout << "\n";
    else
        cout << " ";
}



