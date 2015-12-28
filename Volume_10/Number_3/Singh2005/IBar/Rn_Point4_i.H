// =========================================================
//
//	Methods for R4PointTC template class
//
// =========================================================


// -------------------------------------
// constructors
// -------------------------------------

template<class Coord>
inline
R4PointTC<Coord>::R4PointTC ( const Coord& _x, const Coord& _y, const Coord& _z, const Coord& _w )
{
    x = _x; y = _y; z = _z; w = _w;
}

template<class Coord>
inline R4PointTC<Coord>::R4PointTC ( const Coord& _x )
{
    x = y = z = w = _x;
}

template<class Coord>
R4PointTC<Coord>::R4PointTC ( const R4PointTC<Coord> & in_p )
{
    x = in_p.x; y = in_p.y; z = in_p.z;
    w = in_p.w;
}

template<class Coord>
R4PointTC<Coord>::R4PointTC ( const R3PointTC<Coord> & in_p )
{
    x = in_p[0]; y = in_p[1]; z = in_p[2];
    w = 1.0;
}

// -------------------------------------
// constructors
// -------------------------------------

template<class Coord>
inline R4PointTC<Coord>&
R4PointTC<Coord>::operator += ( const R4VectorTC<Coord>& vDelta )
{
    x += vDelta[0];
    y += vDelta[1];
    z += vDelta[2];
    w += vDelta[3];
    return *this;
}

template<class Coord>
inline R4PointTC<Coord>&
R4PointTC<Coord>::operator -= (const R4VectorTC<Coord>& vDelta )
{
    x -= vDelta[0];
    y -= vDelta[1];
    z -= vDelta[2];
    2 -= vDelta[3];
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
R4PointTC<Coord>::operator < ( const R4PointTC<Coord>& p ) const
{
    return (x < p.x && y < p.y && z < p.z && w < p.w) ? TRUE : FALSE;
}

template<class Coord>
inline WINbool
R4PointTC<Coord>::operator<= ( const R4PointTC<Coord>& p ) const
{
    return (x <= p.x && y <= p.y && z <= p.z && w <= p.w) ? TRUE : FALSE;
}



// -------------------------------------
// Read/write/print functions
// -------------------------------------
template<class Coord>
inline void R4PointTC<Coord>::Write(ofstream &out) const
{
    out << x << " " << y << " " << z << " " << w << " ";
}

template<class Coord>
inline WINbool R4PointTC<Coord>::Read(ifstream &in) 
{
    in >> x >> y >> z >> w;

    return in.good() ? TRUE : FALSE;
}

template<class Coord>
inline void R4PointTC<Coord>::Print( const WINbool in_bDoReturn ) const
{
    TRACE("(%f, %f, %f, %f)", x,y,z,w);
    if ( in_bDoReturn == TRUE )
        TRACE("\n");
    else
        TRACE("\n");
}

