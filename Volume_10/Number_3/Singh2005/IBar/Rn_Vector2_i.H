// =========================================================
//
//	Methods for Vector2TC template class
//
// =========================================================


// -------------------------------------
// constructors
// -------------------------------------

template<class Coord>
inline
R2VectorTC<Coord>::R2VectorTC( const Coord& _d )
{
    du = dv = _d;
}

template<class Coord>
inline
R2VectorTC<Coord>::R2VectorTC( const Coord& _du, const Coord& _dv )
{
    du = _du; dv = _dv;
}

// -------------------------------------
// assignment operators
// -------------------------------------

template<class Coord>
inline R2VectorTC<Coord>&
R2VectorTC<Coord>::operator += ( const R2VectorTC<Coord>& v )
{
    du += v.du;
    dv += v.dv;
    return *this;
}

template<class Coord>
inline R2VectorTC<Coord>&
R2VectorTC<Coord>::operator -= ( const R2VectorTC<Coord>& v )
{
    du -= v.du;
    dv -= v.dv;
    return *this;
}

template<class Coord>
inline R2VectorTC<Coord>&
R2VectorTC<Coord>::operator *= ( const Coord& s )
{
    du *= s;
    dv *= s;
    return *this;
}

template<class Coord>
inline R2VectorTC<Coord>&
R2VectorTC<Coord>::operator /= ( const Coord& s )
{
    du /= s;
    dv /= s;
    return *this;
}

// -------------------------------------
// unary operators
// -------------------------------------

template<class Coord>
inline R2VectorTC<Coord>
R2VectorTC<Coord>::operator + () const
{
    return *this;
}

template<class Coord>
inline R2VectorTC<Coord>
R2VectorTC<Coord>::operator - () const
{
    return R2VectorTC<Coord>( -du, -dv );
}


// -------------------------------------
// Self-editing functions
// -------------------------------------

template<class Coord>
inline double R2VectorTC<Coord>::Normalize()
{
    const double dLen = sqrt(du * du + dv * dv);
    if ( ! RNIsZero( dLen ) ) {
        du = du / dLen;
        dv = dv / dLen;
    }
    return dLen;
}



// -------------------------------------
// Read/write/print functions
// -------------------------------------
template<class Coord>
inline void R2VectorTC<Coord>::Write(ofstream &out) const
{
    out << du << " " << dv << " ";
}

template<class Coord>
inline WINbool R2VectorTC<Coord>::Read(ifstream &in) 
{
    in >> du >> dv;

    return in.good() ? TRUE : FALSE;
}

template<class Coord>
inline void R2VectorTC<Coord>::Print( const WINbool in_bDoReturn ) const
{
    cout << du << " " << dv;
    if ( in_bDoReturn == TRUE )
        cout << "\n";
    else
        cout << " ";
}

