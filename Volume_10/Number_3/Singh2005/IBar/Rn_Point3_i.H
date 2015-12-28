// =========================================================
//
//	Methods for R3PointTC template class
//
// =========================================================


// -------------------------------------
// constructors
// -------------------------------------

template<class Coord>
inline
R3PointTC<Coord>::R3PointTC ( const Coord& _x, const Coord& _y, const Coord& _z )
{
    x = _x; y = _y; z = _z;
}

template<class Coord>
inline R3PointTC<Coord>::R3PointTC ( const Coord& _x )
{
    x = y = z = _x;
}

template<class Coord>
R3PointTC<Coord>::R3PointTC ( const R3PointTC<Coord> & in_p )
{
    x = in_p.x; y = in_p.y; z = in_p.z;
}

template<class Coord>
R3PointTC<Coord>::R3PointTC ( const R2PointTC<Coord> & in_p )
{
    x = in_p[0]; y = in_p[1]; 
    z = 1.0;
}


// -------------------------------------
// constructors
// -------------------------------------

template<class Coord>
inline R3PointTC<Coord>&
R3PointTC<Coord>::operator += ( const R3VectorTC<Coord>& vDelta )
{
    x += vDelta[0];
    y += vDelta[1];
    z += vDelta[2];
    return *this;
}

template<class Coord>
inline R3PointTC<Coord>&
R3PointTC<Coord>::operator -= (const R3VectorTC<Coord>& vDelta )
{
    x -= vDelta[0];
    y -= vDelta[1];
    z -= vDelta[2];
    return *this;
}

// -------------------------------------
// binary operators
// -------------------------------------



// -------------------------------------
// point dominance
// -------------------------------------

template<class Coord>
inline WINbool
R3PointTC<Coord>::operator < ( const R3PointTC<Coord>& p ) const
{
    return (x < p.x && y < p.y && z < p.z) ? TRUE : FALSE;
}

template<class Coord>
inline WINbool
R3PointTC<Coord>::operator<= ( const R3PointTC<Coord>& p ) const
{
    return (x <= p.x && y <= p.y && z <= p.z) ? TRUE : FALSE;
}


// -------------------------------------
// Read/write/print functions
// -------------------------------------
template<class Coord>
inline void R3PointTC<Coord>::Write(ofstream &out) const
{
    out << x << " " << y << " " << z << " ";
}

template<class Coord>
inline WINbool R3PointTC<Coord>::Read(ifstream &in) 
{
    in >> x >> y >> z;

    return in.good() ? TRUE : FALSE;
}

template<class Coord>
inline void R3PointTC<Coord>::Print( const WINbool in_bDoReturn ) const
{
    TRACE("(%f, %f, %f)", x,y,z);

    if ( in_bDoReturn == TRUE )
        TRACE("\n");
    else
        TRACE("\n");
}
