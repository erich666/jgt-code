// =========================================================
//
//	Methods for R4MatrixTC<Coord>TC template class
//
// =========================================================


template<class Coord>
inline
R4MatrixTC<Coord>::R4MatrixTC( const R4VectorTC<Coord>& vCol1, 
                               const R4VectorTC<Coord>& vCol2, 
                               const R4VectorTC<Coord>& vCol3,
                               const R4VectorTC<Coord>& vCol4)
{
    m_mat[0][0] = vCol1[0]; m_mat[1][0] = vCol2[0]; m_mat[2][0] = vCol3[0]; m_mat[3][0] = vCol4[0];
    m_mat[0][1] = vCol1[1]; m_mat[1][1] = vCol2[1]; m_mat[2][1] = vCol3[1]; m_mat[3][1] = vCol4[1];
    m_mat[0][2] = vCol1[2]; m_mat[1][2] = vCol2[2]; m_mat[2][2] = vCol3[2]; m_mat[3][2] = vCol4[2];
    m_mat[0][3] = vCol1[3]; m_mat[1][3] = vCol2[3]; m_mat[2][3] = vCol3[3]; m_mat[3][3] = vCol4[3];
}

template<class Coord>
inline
R4MatrixTC<Coord>::R4MatrixTC( const R4CoVectorTC<Coord>& cvRow1, 
                               const R4CoVectorTC<Coord>& cvRow2,
                               const R4CoVectorTC<Coord>& cvRow3,
                               const R4CoVectorTC<Coord>& cvRow4 )
{
    m_mat[0][0] = cvRow1[0]; m_mat[1][0] = cvRow1[1]; m_mat[2][0] = cvRow1[2]; m_mat[3][0] = cvRow1[3];
    m_mat[0][1] = cvRow2[0]; m_mat[1][1] = cvRow2[1]; m_mat[2][1] = cvRow2[2]; m_mat[3][1] = cvRow2[3];
    m_mat[0][2] = cvRow3[0]; m_mat[1][2] = cvRow3[1]; m_mat[2][2] = cvRow3[2]; m_mat[3][2] = cvRow3[3];
    m_mat[0][3] = cvRow4[0]; m_mat[1][3] = cvRow4[1]; m_mat[2][3] = cvRow4[2]; m_mat[3][3] = cvRow4[3];
}

template<class Coord>
inline
R4MatrixTC<Coord>::R4MatrixTC( const R3MatrixTC<Coord>& mat )
{
    for (int i = 0; i < 3; i++) {
        m_mat[i][3] = 0;
        m_mat[3][i] = 0;
        for (int j = 0; j < 3; j++) {
            (*this)(i,j) = mat(i, j);
        }
    }
    m_mat[3][3] = 1.0;
}

template<class Coord>
inline R4MatrixTC<Coord>&
R4MatrixTC<Coord>::operator *= ( const R4MatrixTC<Coord>& m )
{
    (*this) = (*this) * m;
    return (*this);
}

template<class Coord>
inline R4MatrixTC<Coord>&
R4MatrixTC<Coord>::operator += ( const R4MatrixTC<Coord>& m )
{
    for (int i = 0; i < 4; i += 1)
        for (int j = 0; j < 4; j += 1)
            (*this)(i,j) += m(i, j);
    return *this;
}

template<class Coord>
inline R4MatrixTC<Coord>&
R4MatrixTC<Coord>::operator -= ( const R4MatrixTC<Coord>& m )
{
    for (int i = 0; i < 4; i += 1)
        for (int j = 0; j < 4; j += 1)
            (*this)(i,j) -= m(i, j);
    return *this;
}

template<class Coord>
inline R4MatrixTC<Coord>&
R4MatrixTC<Coord>::operator *= ( const Coord s )
{
    for (int i = 0; i < 4; i += 1)
        for (int j = 0; j < 4; j += 1)
            (*this)(i,j) *= s;
    return *this;
}

template<class Coord>
inline R4MatrixTC<Coord>&
R4MatrixTC<Coord>::operator /= ( const Coord s )
{
    Coord s1 = 1.0f / s;
    for (int i = 0; i < 4; i += 1)
        for (int j = 0; j < 4; j += 1)
            (*this)(i,j) *= s1;
    return *this;
}

template<class Coord>
inline R4MatrixTC<Coord>
R4MatrixTC<Coord>::operator + ( )
{
    return *this;
}

template<class Coord>
inline R4MatrixTC<Coord>
R4MatrixTC<Coord>::operator - ( )
{
    R4MatrixTC<Coord> ret;
    
    for (int i = 0; i < 4; i += 1)
        for (int j = 0; j < 4; j += 1)
            ret(i, j) = -(*this)(i,j);
    return ret;
}

// -------------------------------------
// Read/write/print functions
// -------------------------------------
template<class Coord>
inline void R4MatrixTC<Coord>::Write(ofstream &out) const
{
    for (int i = 0; i < 4; i++) {
        for ( int j = 0; j < 4; j++) {
            out << (*this)(i,j) << " ";
        }
        out << "\n";
    }
    out << "\n";
}

template<class Coord>
inline WINbool R4MatrixTC<Coord>::Read(ifstream &in) 
{
    for (int i = 0; i < 4; i++) {
        for ( int j = 0; j < 4; j++) {
            in >> (*this)(i,j);
        }
    }

    return in.good() ? TRUE : FALSE;
}

template<class Coord>
inline void R4MatrixTC<Coord>::Print( const WINbool in_bDoReturn ) const
{
    for (int i = 0; i < 4; i++) {
        for ( int j = 0; j < 4; j++) {
            cout << (*this)(i,j) << " ";
        }
        if ( in_bDoReturn == TRUE )
            cout << "\n";
        else
            cout << " ";
    }
}

template<class Coord>
inline void R4MatrixTC<Coord>::MakeIdentity(  )
{
    for (int i = 0; i < 4; i++) {
        for ( int j = 0; j < 4; j++) {
            if ( i == j )
                (*this)(i,j) = 1;
            else
                (*this)(i,j) = 0;
        }
    }
}


template<class Coord>
inline R4MatrixTC<Coord> R4MatrixTC<Coord>::Identity( )
{
    R4MatrixTC<Coord> out_mat;

    out_mat.MakeIdentity();
    return out_mat;
}


/// 
template<class Coord>
inline R4MatrixTC<Coord> R4MatrixTC<Coord>::Rotation(const int axis, const Coord angle)
{
    ASSERT( axis >= 0 && axis <= 2 );

    R4MatrixTC<Coord> out_mat;
    out_mat.MakeIdentity();

    double c = cos(angle), s = sin(angle);
    switch (axis) {
    case 0:
        out_mat(0,0)=1; out_mat(1,1)=c; out_mat(1,2)=s; out_mat(2,1)=-s; out_mat(2,2)=c;
        break;
    case 1:
        out_mat(1,1)=1; out_mat(2,2)=c; out_mat(2,0)=s; out_mat(0,2)=-s; out_mat(0,0)=c;
        break;
    case 2:
        out_mat(2,2)=1; out_mat(0,0)=c; out_mat(0,1)=s; out_mat(1,0)=-s; out_mat(1,1)=c;
        break;
    }

    return out_mat;
}

///
template<class Coord>
inline R4MatrixTC<Coord> R4MatrixTC<Coord>::Scaling( const Coord in_x, const Coord in_y, const Coord in_z, const Coord in_w )
{
    R4MatrixTC<Coord> out_mat;
    out_mat.MakeIdentity(  );

    out_mat(0,0) = in_x;
    out_mat(1,1) = in_y;
    out_mat(2,2) = in_z;
    out_mat(3,3) = in_w;

    return out_mat;
}

template<class Coord>
inline R4MatrixTC<Coord> R4MatrixTC<Coord>::Scaling( const Coord in_scl )
{
    return Scaling( in_scl, in_scl, in_scl, 1.0 );
}

template<class Coord>
inline R4MatrixTC<Coord> R4MatrixTC<Coord>::Translation( const R3VectorTC<Coord> &in_vec )
{
    R4MatrixTC<Coord> out_mat;
    out_mat.MakeIdentity( );

    out_mat(0,3) = in_vec[0];
    out_mat(1,3) = in_vec[1];
    out_mat(2,3) = in_vec[2];

    return out_mat;
}


template<class Coord>
inline R4MatrixTC<Coord> R4MatrixTC<Coord>::Translation( const R3PointTC<Coord> &in_pt )
{
    R4MatrixTC<Coord> out_mat;
    out_mat.MakeIdentity( );

    out_mat(0,3) = in_pt[0];
    out_mat(1,3) = in_pt[1];
    out_mat(2,3) = in_pt[2];

    return out_mat;
}

template<class Coord>
inline R4MatrixTC<Coord> R4MatrixTC<Coord>::Transpose(  ) const
{
	R4MatrixTC<Coord> out_mat;

    for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			out_mat(i,j) = (*this)(j,i);

	return out_mat;
}

