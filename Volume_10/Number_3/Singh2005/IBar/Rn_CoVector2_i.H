
// =========================================================
//
//	Methods for R2CoVectorTC template class
//
// =========================================================


// -------------------------------------
// constructors
// -------------------------------------

template<class Coord>
inline
R2CoVectorTC<Coord>::R2CoVectorTC( const Coord& _d )
{
    du = dv = _d;
}

template<class Coord>
inline
R2CoVectorTC<Coord>::R2CoVectorTC( const Coord& _du, const Coord& _dv )
{
    du = _du; dv = _dv;
}

// -------------------------------------
// assignment operators
// -------------------------------------

template<class Coord>
inline R2CoVectorTC<Coord>&
R2CoVectorTC<Coord>::operator += ( const R2CoVectorTC<Coord>& v )
{
    du += v.du;
    dv += v.dv;
    return *this;
}

template<class Coord>
inline R2CoVectorTC<Coord>&
R2CoVectorTC<Coord>::operator -= ( const R2CoVectorTC<Coord>& v )
{
    du -= v.du;
    dv -= v.dv;
    return *this;
}

template<class Coord>
inline R2CoVectorTC<Coord>&
R2CoVectorTC<Coord>::operator *= ( const Coord& s )
{
    du *= s;
    dv *= s;
    return *this;
}

template<class Coord>
inline R2CoVectorTC<Coord>&
R2CoVectorTC<Coord>::operator /= ( const Coord& s )
{
    du /= s;
    dv /= s;
    return *this;
}

// -------------------------------------
// unary operators
// -------------------------------------

template<class Coord>
inline R2CoVectorTC<Coord>
R2CoVectorTC<Coord>::operator + () const
{
    return *this;
}

template<class Coord>
inline R2CoVectorTC<Coord>
R2CoVectorTC<Coord>::operator - () const
{
    return R2CoVectorTC<Coord>( -du, -dv );
}




// -------------------------------------
// Read/write/print functions
// -------------------------------------
template<class Coord>
inline void R2CoVectorTC<Coord>::Write(ofstream &out) const
{
    out << du << " " << dv << " ";
}

template<class Coord>
inline WINbool R2CoVectorTC<Coord>::Read(ifstream &in) 
{
    in >> du >> dv;

    return in.good() ? TRUE : FALSE;
}

template<class Coord>
inline void R2CoVectorTC<Coord>::Print( WINbool in_bDoReturn ) const
{
    cout << du << dv;
    if ( in_bDoReturn == TRUE )
        cout << "\n";
    else
        cout << " ";
}
