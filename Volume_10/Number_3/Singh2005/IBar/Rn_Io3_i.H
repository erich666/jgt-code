// =========================================================
//
//	IO operators on affine1 classes
//
// =========================================================

/**@name IO operators */
//@{

///
template<class Coord>
inline ostream& operator<<( ostream &out, const R3VectorTC<Coord> &v )
{
    out << v[0] << " " << v[1] << " " << v[2] << " ";
    return out;
}

///
template<class Coord>
inline istream& operator>>( istream &in, R3VectorTC<Coord> &v )
{
    in >> v[0] >> v[1] >> v[2];
    return in;
}


///
template<class Coord>
inline ostream& operator<<( ostream &out, const R3CoVectorTC<Coord> &v )
{
    out << v[0] << " " << v[1] << " " << v[2] << " ";
    return out;
}

///
template<class Coord>
inline istream& operator>>( istream &in, R3CoVectorTC<Coord> &v )
{
    in >> v[0] >> v[1] >> v[2];
    return in;
}



///
template<class Coord>
inline ostream& operator<<( ostream &out, const R3PointTC<Coord> &p )
{
    out << p[0] << " " << p[1] << " " << p[2] << " " ;
    return out;
}

///
template<class Coord>
inline istream& operator>>( istream &in, R3PointTC<Coord> &p )
{
    in >> p[0] >> p[1] >> p[2];
    return in;
}



///
template<class Coord>
inline ostream& operator<<( ostream &out, const R3MatrixTC<Coord> &m )
{
    for (int i = 0; i < 3; i++) {
        for ( int j = 0; j < 3; j++) {
            out << m(i, j) << " ";
        }
        out << "  ";
    }
    
    return out;
}

///
template<class Coord>
inline istream& operator>>( istream &in, R3MatrixTC<Coord> &m )
{
    for (int i = 0; i < 3; i++)
        for ( int j = 0; j < 3; j++)
            in >> m(i, j);

    return in;
}



//@}
