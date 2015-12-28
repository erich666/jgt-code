
// =========================================================
//
//	Methods for R2PointTC template class
//
// =========================================================


// -------------------------------------
// constructors
// -------------------------------------

template<class Coord>
inline
R2PointTC<Coord>::R2PointTC ( const Coord _u )
{
    u = _u; v = _u;
}


template<class Coord>
inline
R2PointTC<Coord>::R2PointTC ( const Coord _u, const Coord _v )
{
    u = _u; v = _v;
}

// -------------------------------------
// Unary operators
// -------------------------------------

template<class Coord>
inline R2PointTC<Coord>&
R2PointTC<Coord>::operator += (const R2VectorTC<Coord>& vDelta )
{
    u += vDelta[0];
    v += vDelta[1];
    return *this;
}

template<class Coord>
inline R2PointTC<Coord>&
R2PointTC<Coord>::operator -= (const R2VectorTC<Coord>& vDelta )
{
    u -= vDelta[0];
    v -= vDelta[1];
    return *this;
}

// -------------------------------------
// point dominance
// -------------------------------------

///
template<class Coord>
inline WINbool
R2PointTC<Coord>::operator < ( const R2PointTC<Coord>& p ) const
{
    return (u < p[0] && v < p[1]) ? TRUE : FALSE;
}

///
template<class Coord>
inline WINbool
R2PointTC<Coord>::operator<= ( const R2PointTC<Coord>& p ) const
{
    return (u <= p[0] && v <= p[1]) ? TRUE : FALSE;
}



// -------------------------------------
// Read/write/print functions
// -------------------------------------
template<class Coord>
inline void R2PointTC<Coord>::Write(ofstream &out) const
{
    out << u << " " << v << " ";
}

template<class Coord>
inline WINbool R2PointTC<Coord>::Read(ifstream &in) 
{
    in >> u >> v;

    return in.good() ? TRUE : FALSE;
}

template<class Coord>
inline void R2PointTC<Coord>::Print( const WINbool in_bDoReturn ) const
{
    TRACE("(%f, %f)", u, v);
    if ( in_bDoReturn == TRUE )
        TRACE("\n");
    else
        TRACE("\n");
}
