// =========================================================
//
//	Methods for R3VectorTC template class
//
// =========================================================


// -------------------------------------
// constructors
// -------------------------------------

template<class Coord>
inline
R3VectorTC<Coord>::R3VectorTC( const Coord& _d )
{
    dx = dy = dz = _d;
}

template<class Coord>
inline
R3VectorTC<Coord>::R3VectorTC( const Coord& _dx, const Coord& _dy, const Coord& _dz )
{
    dx = _dx; dy = _dy; dz = _dz;
}

template<class Coord>
inline
R3VectorTC<Coord>::R3VectorTC( const R2VectorTC<Coord>& vec )
{
    dx = vec[0]; dy = vec[1]; 
    dz = 1;
}

// -------------------------------------
// assignment operators
// -------------------------------------

template<class Coord>
inline R3VectorTC<Coord>&
R3VectorTC<Coord>::operator += ( const R3VectorTC<Coord>& v)
{
    dx += v.dx;
    dy += v.dy;
    dz += v.dz;
    return *this;
}

template<class Coord>
inline R3VectorTC<Coord>&
R3VectorTC<Coord>::operator -= ( const R3VectorTC<Coord>& v )
{
    dx -= v.dx;
    dy -= v.dy;
    dz -= v.dz;
    return *this;
}

template<class Coord>
inline R3VectorTC<Coord>&
R3VectorTC<Coord>::operator *= ( const Coord& s )
{
    dx *= s;
    dy *= s;
    dz *= s;
    return *this;
}

template<class Coord>
inline R3VectorTC<Coord>&
R3VectorTC<Coord>::operator /= ( const Coord& s )
{
    Coord s1 = 1.0f / s;
    dx *= s1;
    dy *= s1;
    dz *= s1;
    return *this;
}

// -------------------------------------
// unary operators
// -------------------------------------

template<class Coord>
inline R3VectorTC<Coord>
R3VectorTC<Coord>::operator + () const
{
    return *this;
}

template<class Coord>
inline R3VectorTC<Coord>
R3VectorTC<Coord>::operator - () const
{
    return R3VectorTC<Coord>( -dx, -dy, -dz );
}


// -------------------------------------
// miscellaneous methods
// -------------------------------------

// -------------------------------------
// Self-editing functions
// -------------------------------------
template<class Coord>
inline double R3VectorTC<Coord>::Normalize()
{
    const double dLen = sqrt(dx * dx + dy * dy + dz * dz);
    if ( ! RNIsZero( dLen ) ) {
        dx = dx / dLen;
        dy = dy / dLen;
        dz = dz / dLen;
    }

    return dLen;
}


// -------------------------------------
// Read/write/print functions
// -------------------------------------
template<class Coord>
inline void R3VectorTC<Coord>::Write(ofstream &out) const
{
    out << dx << " " << dy << " " << dz << " ";
}

template<class Coord>
inline WINbool R3VectorTC<Coord>::Read(ifstream &in) 
{
    in >> dx >> dy >> dz;

    return in.good() ? TRUE : FALSE;
}

template<class Coord>
inline void R3VectorTC<Coord>::Print( const WINbool in_bDoReturn ) const
{
    cout << dx << " " << dy << " " << dz;
    if ( in_bDoReturn == TRUE )
        cout << "\n";
    else
        cout << " ";
}



