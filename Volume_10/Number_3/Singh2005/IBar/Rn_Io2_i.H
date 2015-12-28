// =========================================================
//
//	IO operators on affine1 classes
//
// =========================================================

/**@name IO operators */
//@{

///
template<class Coord>
inline ostream& operator<<( ostream &out, const R2VectorTC<Coord> &v )
{
    out << v[0] << " " << v[1] << " ";
    return out;
}

///
template<class Coord>
inline istream& operator>>( istream &in, R2VectorTC<Coord> &v )
{
    in >> v[0] >> v[1];
    return in;
}



///
template<class Coord>
inline ostream& operator<<( ostream &out, const R2CoVectorTC<Coord> &v )
{
    out << v[0] << " " << v[1] << " ";
    return out;
}

///
template<class Coord>
inline istream& operator>>( istream &in, R2CoVectorTC<Coord> &v )
{
    in >> v[0] >> v[1];
    return in;
}



///
template<class Coord>
inline ostream& operator<<( ostream &out, const R2PointTC<Coord> &p )
{
    out << p[0] << " " << p[1] << " ";
    return out;
}

///
template<class Coord>
inline istream& operator>>( istream &in, R2PointTC<Coord> &p )
{
    in >> p[0] >> p[1];
    return in;
}




template<class Coord>
inline ostream& operator<<( ostream &out, const R2MatrixTC<Coord> &m )
{
    for (int i = 0; i < 2; i++) {
        for ( int j = 0; j < 2; j++) {
            out << m(i,j) << " ";
        }
        out << "  ";
    }
    
    return out;
}

template<class Coord>
inline istream& operator>>( istream &in, R2MatrixTC<Coord> &m )
{
    for (int i = 0; i < 2; i++)
        for ( int j = 0; j < 2; j++)
            in >> m(i,j);

    return in;
}



//@}

