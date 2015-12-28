// =========================================================
//
//	Methods for R1CoVectorTC template class
//
// =========================================================


// -------------------------------------
// constructors
// -------------------------------------

template<class Coord>
inline
R1CoVectorTC<Coord>::R1CoVectorTC( const Coord _d )
{
    d = _d;
}


// -------------------------------------
// assignment operators
// -------------------------------------

template<class Coord>
inline R1CoVectorTC<Coord>&
R1CoVectorTC<Coord>::operator += ( const R1CoVectorTC<Coord>& v )
{
    d += v.d;
    return *this;
}

//
template<class Coord>
inline R1CoVectorTC<Coord>&
R1CoVectorTC<Coord>::operator -= ( const R1CoVectorTC<Coord>& v )
{
    d -= v.d;
    return *this;
}

template<class Coord>
inline R1CoVectorTC<Coord>&
R1CoVectorTC<Coord>::operator *= ( const Coord& s )
{
    d *= s;
    return *this;
}

template<class Coord>
inline R1CoVectorTC<Coord>&
R1CoVectorTC<Coord>::operator /= ( const Coord& s )
{
    d /= s;
    return *this;
}



template<class Coord>
inline R1CoVectorTC<Coord>
R1CoVectorTC<Coord>::operator + () const
{
    return *this;
}


template<class Coord>
inline R1CoVectorTC<Coord>
R1CoVectorTC<Coord>::operator - () const
{
    return R1CoVectorTC<Coord>( -d );
}


// -------------------------------------
// Read/write/print functions
// -------------------------------------
template<class Coord>
inline void R1CoVectorTC<Coord>::Write(ofstream &out) const
{
    out << d << " ";
}

template<class Coord>
inline WINbool R1CoVectorTC<Coord>::Read(ifstream &in) 
{
    in >> d;

    return in.good() ? TRUE : FALSE;
}

template<class Coord>
inline void R1CoVectorTC<Coord>::Print( WINbool in_bDoReturn ) const
{
    cout << d;
    if ( in_bDoReturn == TRUE )
        cout << "\n";
    else
        cout << " ";
}
