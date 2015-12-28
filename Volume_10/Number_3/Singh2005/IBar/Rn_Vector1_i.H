// =========================================================
//
//	Methods for R1VectorTC template class
//
// =========================================================

// -------------------------------------
// constructors
// -------------------------------------

template<class Coord>
inline
R1VectorTC<Coord>::R1VectorTC( const Coord _d )
{
    d = _d;
}

template<class Coord>
inline R1VectorTC<Coord>
R1VectorTC<Coord>::operator + () const
{
    return *this;
}

template<class Coord>
inline R1VectorTC<Coord>
R1VectorTC<Coord>::operator - () const
{
    return R1VectorTC<Coord>( -d );
}


// -------------------------------------
// assignment operators
// -------------------------------------

template<class Coord>
inline R1VectorTC<Coord>&
R1VectorTC<Coord>::operator += ( const R1VectorTC<Coord>& v )
{
    d += v.d;
    return *this;
}

template<class Coord>
inline R1VectorTC<Coord>&
R1VectorTC<Coord>::operator -= ( const R1VectorTC<Coord>& v )
{
    d -= v.d;
    return *this;
}


template<class Coord>
inline R1VectorTC<Coord>&
R1VectorTC<Coord>::operator *= ( const Coord& s )
{
    d *= s;
    return *this;
}

template<class Coord>
inline R1VectorTC<Coord>&
R1VectorTC<Coord>::operator /= ( const Coord& s )
{
    d /= s;
    return *this;
}

// -------------------------------------
// Self-editing functions
// -------------------------------------

template<class Coord>
inline double R1VectorTC<Coord>::Normalize()
{
    const double dLen = d;

    d = 1.0;
    return dLen;
}

// -------------------------------------
// Read/write/print functions
// -------------------------------------
template<class Coord>
inline void R1VectorTC<Coord>::Write(ofstream &out) const
{
    out << d << " ";
}

template<class Coord>
inline WINbool R1VectorTC<Coord>::Read(ifstream &in) 
{
    in >> d;

    return in.good() ? TRUE : FALSE;
}

template<class Coord>
inline void R1VectorTC<Coord>::Print( const WINbool in_bDoReturn ) const
{
    cout << d;
    if ( in_bDoReturn == TRUE )
        cout << "\n";
    else
        cout << " ";
}
