// =========================================================
//
//	Methods for R3MatrixTC<Coord>TC template class
//
// =========================================================


template<class Coord>
inline
R3MatrixTC<Coord>::R3MatrixTC( const R3VectorTC<Coord>& vCol1, 
                               const R3VectorTC<Coord>& vCol2, 
                               const R3VectorTC<Coord>& vCol3)
{
    m_mat[0][0] = vCol1[0]; m_mat[0][1] = vCol2[0]; m_mat[0][2] = vCol3[0];
    m_mat[1][0] = vCol1[1]; m_mat[1][1] = vCol2[1]; m_mat[1][2] = vCol3[1];
    m_mat[2][0] = vCol1[2]; m_mat[2][1] = vCol2[2]; m_mat[2][2] = vCol3[2];
}

template<class Coord>
inline
R3MatrixTC<Coord>::R3MatrixTC( const R3CoVectorTC<Coord>& cvRow1, 
                               const R3CoVectorTC<Coord>& cvRow2,
                               const R3CoVectorTC<Coord>& cvRow3 )
{
    m_mat[0][0] = cvRow1[0]; m_mat[0][1] = cvRow1[1]; m_mat[0][2] = cvRow1[2];
    m_mat[1][0] = cvRow2[0]; m_mat[1][1] = cvRow2[1]; m_mat[1][2] = cvRow2[2];
    m_mat[2][0] = cvRow3[0]; m_mat[2][1] = cvRow3[1]; m_mat[2][2] = cvRow3[2];
}

template<class Coord>
inline R3MatrixTC<Coord>&
R3MatrixTC<Coord>::operator *= ( const R3MatrixTC<Coord>& m )
{
    (*this) = (*this) * m;
    return (*this);
}

template<class Coord>
inline R3MatrixTC<Coord>&
R3MatrixTC<Coord>::operator += ( const R3MatrixTC<Coord>& m )
{
    for (int i = 0; i < 3; i += 1)
        for (int j = 0; j < 3; j += 1)
            m_mat[i][j] += m(i, j);
    return *this;
}

template<class Coord>
inline R3MatrixTC<Coord>&
R3MatrixTC<Coord>::operator -= ( const R3MatrixTC<Coord>& m )
{
    for (int i = 0; i < 3; i += 1)
        for (int j = 0; j < 3; j += 1)
            m_mat[i][j] -= m(i, j);
    return *this;
}

template<class Coord>
inline R3MatrixTC<Coord>&
R3MatrixTC<Coord>::operator *= ( const Coord s )
{
    for (int i = 0; i < 3; i += 1)
        for (int j = 0; j < 3; j += 1)
            m_mat[i][j] *= s;
    return *this;
}

template<class Coord>
inline R3MatrixTC<Coord>&
R3MatrixTC<Coord>::operator /= ( const Coord s )
{
    Coord s1 = 1.0f / s;
    for (int i = 0; i < 3; i += 1)
        for (int j = 0; j < 3; j += 1)
            m_mat[i][j] *= s1;
    return *this;
}

template<class Coord>
inline R3MatrixTC<Coord>
R3MatrixTC<Coord>::operator + ( )
{
    return *this;
}

template<class Coord>
inline R3MatrixTC<Coord>
R3MatrixTC<Coord>::operator - ( )
{
    R3MatrixTC<Coord> ret;
    
    for (int i = 0; i < 3; i += 1)
        for (int j = 0; j < 3; j += 1)
            ret(i, j) = -m_mat[i][j];
    return ret;
}

// -------------------------------------
// Read/write/print functions
// -------------------------------------
template<class Coord>
inline void R3MatrixTC<Coord>::Write(ofstream &out) const
{
    for (int i = 0; i < 3; i++) {
        for ( int j = 0; j < 3; j++) {
            out << (*this)(i,j) << " ";
        }
        out << "\n";
    }
    out << "\n";
}

template<class Coord>
inline WINbool R3MatrixTC<Coord>::Read(ifstream &in) 
{
    for (int i = 0; i < 3; i++) {
        for ( int j = 0; j < 3; j++) {
            in >> (*this)(i,j);
        }
    }

    return in.good() ? TRUE : FALSE;
}

template<class Coord>
inline void R3MatrixTC<Coord>::Print( const WINbool in_bDoReturn ) const
{
    for (int i = 0; i < 3; i++) {
        for ( int j = 0; j < 3; j++) {
            TRACE("%f ", (*this)(i,j) );
        }
        if ( in_bDoReturn == TRUE )
            TRACE("\n");
        else
            TRACE(" ");
    }
}

template<class Coord>
inline void R3MatrixTC<Coord>::MakeIdentity(  )
{
    for (int i = 0; i < 3; i++) {
        for ( int j = 0; j < 3; j++) {
            if ( i == j )
                m_mat[i][j] = 1;
            else
                m_mat[i][j] = 0;
        }
    }
}


template<class Coord>
inline R3MatrixTC<Coord> R3MatrixTC<Coord>::Identity( )
{
    R3MatrixTC<Coord> out_mat;

    out_mat.MakeIdentity();
    return out_mat;
}


/// Matrix taking the from vectors to the to vectors
template<class Coord>
inline WINbool R3MatrixTC<Coord>::MatrixVecToVec( R3MatrixTC<Coord>        & out_mat, 
                                                  const R3VectorTC<Coord>  & in_vFromX, 
                                                  const R3VectorTC<Coord>  & in_vFromY,
                                                  const R3VectorTC<Coord>  & in_vToX,   
                                                  const R3VectorTC<Coord>  & in_vToY )
{
    if ( RNIsZero( Length( in_vFromX ) ) ) return FALSE;
    if ( RNIsZero( Length( in_vFromY ) ) ) return FALSE;
    if ( RNIsZero( Length( in_vToX ) ) ) return FALSE;
    if ( RNIsZero( Length( in_vToY ) ) ) return FALSE;
    if ( RNIsZero( 1.0 - fabs( Dot( Unit( in_vFromX ), Unit( in_vFromY ) ) ) ) ) return FALSE;
    if ( RNIsZero( 1.0 - fabs( Dot( Unit( in_vToX ), Unit( in_vToY ) ) ) ) ) return FALSE;

    R3VectorTC<Coord> vFromZ = Cross( in_vFromX, in_vFromY );
    R3VectorTC<Coord> vToZ = Cross( in_vToX, in_vToY );

    R3MatrixTC<Coord> f1;
    R3MatrixTC<Coord> f2;
    
    for (int i = 0; i < 3; i++) {
        f1.m_mat[i][0] = in_vFromX[i];
        f1.m_mat[i][1] = in_vFromY[i];
        f1.m_mat[i][2] = vFromZ[i];

        f2.m_mat[i][0] = in_vToX[i];
        f2.m_mat[i][1] = in_vToY[i];
        f2.m_mat[i][2] = vToZ[i];
    }
    out_mat = f2 * f1.Transpose( );

    return TRUE;
}

/// Matrix taking the from vector to the to vector, rotation around normal unspecified
template<class Coord>
inline WINbool R3MatrixTC<Coord>::MatrixVecToVec( R3MatrixTC<Coord>        & out_mat, 
                                                  const R3VectorTC<Coord>  & in_vFrom, 
                                                  const R3VectorTC<Coord>  & in_vTo )
{
    if ( RNIsZero( Length( in_vFrom ) ) ) return FALSE;
    if ( RNIsZero( Length( in_vTo ) ) ) return FALSE;

    R3VectorTC<Coord> vFrom( 0.0, in_vFrom[2], -in_vFrom[1] );
    R3VectorTC<Coord> vTo(   0.0, in_vTo[2],   -in_vTo[1] );

    if ( fabs( in_vFrom[1] ) < 1e-6f && fabs( in_vFrom[2] ) < 1e-6f ) {
        if ( fabs( in_vTo[1] ) < 1e-6f && fabs( in_vTo[0] ) < 1e-6f ) {
            vFrom = R3VectorTC<Coord>( in_vFrom[2], -in_vFrom[0], 0.0 );
            vTo = R3VectorTC<Coord>( in_vTo[2], -in_vTo[0], 0.0 );
        } else {
            vFrom = R3VectorTC<Coord>( in_vFrom[1], -in_vFrom[0], 0.0 );
            vTo = R3VectorTC<Coord>( in_vTo[1], -in_vTo[0], 0.0 );
        }
    }
             
    return MatrixVecToVec( out_mat, 
                           in_vFrom, UnitSafe( vFrom ),
                           in_vTo, UnitSafe( vTo ) );
}

///
template<class Coord>
inline R3MatrixTC<Coord> R3MatrixTC<Coord>::Rotation(const int axis, const Coord angle)
{
    ASSERT( axis >= 0 && axis <= 2 );

    R3MatrixTC<Coord> out_mat;
    out_mat.MakeIdentity();

    Coord c = cos(angle), s = sin(angle);
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
inline R3MatrixTC<Coord> R3MatrixTC<Coord>::Rotation(const R3VectorTC<Coord> &in_vec, const Coord angle)
{
    const R3VectorTC<Coord> vecAng = UnitSafe( in_vec );
    const Coord yzAng = atan2( -in_vec[2], in_vec[0] );
    const R3MatrixTC<Coord> yRot = R3MatrixTC<Coord>::Rotation(1, yzAng);
    const R3MatrixTC<Coord> yRotInv = yRot.Transpose();
    const R3VectorTC<Coord> vecRot = yRot * vecAng;
    const Coord xzAng = atan2( vecRot[1], vecRot[0] );
    const R3MatrixTC<Coord> xRot = R3MatrixTC<Coord>::Rotation(2, xzAng );
    const R3MatrixTC<Coord> xRotInv = xRot.Transpose();
    const R3VectorTC<Coord> vecRot2 = xRot * vecRot;
    
    return yRotInv * xRotInv * R3MatrixTC<Coord>::Rotation(0, angle) * xRot * yRot;    
}

template<class Coord>
inline R3MatrixTC<Coord> R3MatrixTC<Coord>::Rotation2D( const Coord angle)
{
    R3MatrixTC<Coord> out_mat;
    out_mat.MakeIdentity();

    double c = cos(angle), s = sin(angle);
    out_mat(0,0) = c; 
    out_mat(0,1) = s;
    out_mat(1,0) = -s; 
    out_mat(1,1) = c;

    return out_mat;
}

///
template<class Coord>
inline R3MatrixTC<Coord> R3MatrixTC<Coord>::Scaling( const Coord in_x, const Coord in_y, const Coord in_z )
{
    R3MatrixTC<Coord> out_mat;
    out_mat.MakeIdentity(  );

    out_mat(0,0) = in_x;
    out_mat(1,1) = in_y;
    out_mat(2,2) = in_z;

    return out_mat;
}

template<class Coord>
inline R3MatrixTC<Coord> R3MatrixTC<Coord>::Scaling( const Coord in_scl )
{
    return Scaling( in_scl, in_scl, in_scl );
}

template<class Coord>
inline R3MatrixTC<Coord> R3MatrixTC<Coord>::Translation( const R2VectorTC<Coord> & in_vec )
{
    R3MatrixTC<Coord> mat;
    mat.MakeIdentity( );

    mat(0,2) = in_vec[0];
    mat(1,2) = in_vec[1];

    return mat;
}

template<class Coord>
inline R3MatrixTC<Coord> R3MatrixTC<Coord>::Transpose(  ) const
{
	R3MatrixTC<Coord> out_mat;

    for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			out_mat(i,j) = (*this)(j,i);

	return out_mat;
}
