// =========================================================
//
//	Methods for Point2TC template class
//
// =========================================================


// -------------------------------------
// constructors
// -------------------------------------

template<class Coord>
inline
R2MatrixTC<Coord>::R2MatrixTC( const R2VectorTC<Coord>& vCol1, const R2VectorTC<Coord>& vCol2 )
{
    mat[0][0] = vCol1[0]; mat[0][1] = vCol2[0];
    mat[1][0] = vCol1[1]; mat[1][1] = vCol2[1];
}

template<class Coord>
inline
R2MatrixTC<Coord>::R2MatrixTC( const R2CoVectorTC<Coord>& covRow1, 
                               const R2CoVectorTC<Coord>& covRow2 )
{
    mat[0][0] = covRow1.du; mat[0][1] = covRow1.dv;
    mat[1][0] = covRow2.du; mat[1][1] = covRow2.dv;
}

template<class Coord>
inline R2MatrixTC<Coord>&
R2MatrixTC<Coord>::operator *= ( const R2MatrixTC<Coord>& m )
{
    (*this) = (*this) * m;
    return *this;
}

template<class Coord>
inline R2MatrixTC<Coord>&
R2MatrixTC<Coord>::operator += ( const R2MatrixTC<Coord>& m )
{
    for (int i = 0; i < 2; i += 1)
        for (int j = 0; j < 2; j += 1)
            mat[i][j] += m(i, j);
    return *this;
}

template<class Coord>
inline R2MatrixTC<Coord>&
R2MatrixTC<Coord>::operator -= ( const R2MatrixTC<Coord>& m )
{
    for (int i = 0; i < 2; i += 1)
        for (int j = 0; j < 2; j += 1)
            mat[i][j] -= m(i, j);
    return *this;
}

template<class Coord>
inline R2MatrixTC<Coord>&
R2MatrixTC<Coord>::operator *= ( Coord s )
{
    for (int i = 0; i < 2; i += 1)
        for (int j = 0; j < 2; j += 1)
            mat[i][j] *= s;
    return *this;
}

template<class Coord>
inline R2MatrixTC<Coord>&
R2MatrixTC<Coord>::operator /= ( Coord s )
{
    ASSERT( s != 0.0 );

    Coord s1 = 1.0 / s;
    for (int i = 0; i < 2; i += 1)
        for (int j = 0; j < 2; j += 1)
            mat[i][j] *= s1;
    return *this;
}

template<class Coord>
inline R2MatrixTC<Coord>
R2MatrixTC<Coord>::operator + ( )
{
    return *this;
}

template<class Coord>
inline R2MatrixTC<Coord>
R2MatrixTC<Coord>::operator - ( )
{
    R2MatrixTC<Coord> ret;
    
    for (int i = 0; i < 2; i += 1)
        for (int j = 0; j < 2; j += 1)
            ret(i, j) = -mat[i][j];
    return ret;
}


template<class Coord>
inline void
R2MatrixTC<Coord>::SetIdentity( )
{
    for (int i = 0; i < 2; i += 1)
        for (int j = 0; j < 2; j += 1)
            mat[i][j] = 0;
    mat[0][0] = 1.0;
    mat[1][1] = 1.0;
}




template<class Coord>
inline R2MatrixTC<Coord>
R2MatrixTC<Coord>::Inverse( Coord &out_dDeterminant ) const
{
    R2MatrixTC<Coord> out_m;

    out_dDeterminant = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];
    ASSERT( out_dDeterminant != 0.0 );

    out_m.mat[0][0] = mat[1][1] / out_dDeterminant;
    out_m.mat[1][1] = mat[0][0] / out_dDeterminant;
    out_m.mat[0][1] = -mat[0][1] / out_dDeterminant;
    out_m.mat[1][0] = -mat[1][0] / out_dDeterminant;

    return out_m;
}

template<class Coord>
inline WINbool
R2MatrixTC<Coord>::SetRotation( const Coord in_fAngle )
{
    mat[0][0] = mat[1][1] = cos( in_fAngle );
    mat[1][0] = -  sin( in_fAngle );
    mat[0][1] = sin( in_fAngle );

    return TRUE;
}


template<class Coord>
inline R2MatrixTC<Coord>
R2MatrixTC<Coord>::Rotation( const Coord in_fAngle )
{
    R2MatrixTC<Coord> mat;
    mat.SetRotation( in_fAngle );

    return mat;
}

template<class Coord>
inline R2MatrixTC<Coord>
R2MatrixTC<Coord>::Scaling( const Coord in_fScaleU, const Coord in_fScaleV )
{
    R2MatrixTC<Coord> mat;
    mat.SetScale( in_fScaleU, in_fScaleV );

    return mat;
}



template<class Coord>
inline WINbool
R2MatrixTC<Coord>::SetScale( const Coord in_fScaleU, const Coord in_fScaleV )
{
    mat[0][0] = in_fScaleU;
    mat[1][1] = in_fScaleV;
    mat[1][0] = mat[0][1] = 0;

    return TRUE;
}

// -------------------------------------
// Read/write/print functions
// -------------------------------------
template<class Coord>
inline void R2MatrixTC<Coord>::Write(ofstream &out) const
{
    out << mat[0][0] << " " << mat[0][1] << "\n";
    out << mat[1][0] << " " << mat[1][1] << "\n";
}

template<class Coord>
inline WINbool R2MatrixTC<Coord>::Read(ifstream &in) 
{
    in >> mat[0][0] >> mat[0][1] >> mat[1][0] >> mat[1][1];

    return in.good() ? TRUE : FALSE;
}

template<class Coord>
inline void R2MatrixTC<Coord>::Print( const WINbool in_bDoReturn ) const
{
    cout << mat[0][0] << " " << mat[0][1];
    if ( in_bDoReturn == TRUE )
        cout << "\n";
    else
        cout << " ";

    cout << mat[1][0] << " " << mat[1][1];
    if ( in_bDoReturn == TRUE )
        cout << "\n";
    else
        cout << " ";
}

template<class Coord>
inline R2MatrixTC<Coord> R2MatrixTC<Coord>::Transpose(  ) const
{
	R2MatrixTC<Coord> out_mat;

    for (int i = 0; i < 2; i++)
		for (int j = 0; j < 2; j++)
			out_mat(i,j) = (*this)(j,i);

	return out_mat;
}

