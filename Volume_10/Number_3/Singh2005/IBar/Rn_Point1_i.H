// =========================================================
//
//	Methods for R1PointTC template class
//
// =========================================================


// -------------------------------------
// constructors
//------------------------------------

template<class Coord>
inline
R1PointTC<Coord>::R1PointTC ( const Coord _u )
{
    
    u = _u;
}

// -------------------------------------
// constructors
//------------------------------------

template<class Coord>
inline R1PointTC<Coord>&
R1PointTC<Coord>::operator += (const R1VectorTC<Coord>& vDelta )
{
    
    u += vDelta[0];
    
    return *this;
}


template<class Coord>
inline R1PointTC<Coord>&
R1PointTC<Coord>::operator -= (const R1VectorTC<Coord>& vDelta )
{
    
    u -= vDelta.d;
    
    return *this;
}

// -------------------------------------
// point dominance
// ------------------------------------

///
template<class Coord>
inline WINbool
R1PointTC<Coord>::operator < ( const R1PointTC<Coord>& p ) const
{
    
    return (u < p[0]) ? TRUE : FALSE;
}


///
template<class Coord>
inline WINbool
R1PointTC<Coord>::operator<= ( const R1PointTC<Coord>& p ) const
{
    
    return (u <= p[0]) ? TRUE : FALSE;
}


// -------------------------------------
// Read/write/print functions
// -------------------------------------
template<class Coord>
inline void R1PointTC<Coord>::Write(ofstream &out) const
{
    out << u;
    out << " ";
}

template<class Coord>
inline WINbool R1PointTC<Coord>::Read(ifstream &in) 
{
    in >> u;

    return in.good() ? TRUE : FALSE;
}

template<class Coord>
inline void R1PointTC<Coord>::Print( const WINbool in_bDoReturn ) const
{
    TRACE("%f", u);
    if ( in_bDoReturn == TRUE )
        TRACE("\n");
    else
        TRACE("\n");
}



