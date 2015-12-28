// =========================================================
//
//	IO operators on affine1 classes
//
// =========================================================

/**@name IO operators */
//@{

///
template<class Coord>
inline ostream& operator<<( ostream &out, const R4VectorTC<Coord> &v )
{
    out << v[0] << " " << v[1] << " " << v[2] << " " << v[3] << " ";
    return out;
}

///
template<class Coord>
inline istream& operator>>( istream &in, R4VectorTC<Coord> &v )
{
    in >> v[0] >> v[1] >> v[2] >> v[3];
    return in;
}


///
template<class Coord>
inline ostream& operator<<( ostream &out, const R4CoVectorTC<Coord> &v )
{
    out << v[0] << " " << v[1] << " " << v[2] << " " << v[3] << " ";
    return out;
}

///
template<class Coord>
inline istream& operator>>( istream &in, R4CoVectorTC<Coord> &v )
{
    in >> v[0] >> v[1] >> v[2] >> v[3];
    return in;
}



///
template<class Coord>
inline ostream& operator<<( ostream &out, const R4PointTC<Coord> &p )
{
    out << p[0] << " " << p[1] << " " << p[2] << " " << p[3] << " ";
    return out;
}

///
template<class Coord>
inline istream& operator>>( istream &in, R4PointTC<Coord> &p )
{
    in >> p[0] >> p[1] >> p[2] >> p[3];
    return in;
}



///
template<class Coord>
inline ostream& operator<<( ostream &out, const R4MatrixTC<Coord> &m )
{
    for (int i = 0; i < 4; i++) {
        for ( int j = 0; j < 4; j++) {
            out << m(i, j) << " ";
        }
        out << "  ";
    }
    
    return out;
}

///
template<class Coord>
inline istream& operator>>( istream &in, R4MatrixTC<Coord> &m )
{
    for (int i = 0; i < 4; i++)
        for ( int j = 0; j < 4; j++)
            in >> m(i, j);

    return in;
}



//@}
