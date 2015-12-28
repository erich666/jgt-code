// =========================================================
//
//	Unary operators on affine4 classes
//
// =========================================================

/**@name Unary operators */
//@{

// -------------------------------------
// friend functions
// -------------------------------------

///
template<class Coord>
inline Coord
Length( const R4VectorTC<Coord>& v )
{
    return (Coord) sqrt( v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3] * v[3] );
}

///
template<class Coord>
inline Coord
LengthSq( const R4VectorTC<Coord>& v )
{
    return v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3] * v[3];
}

///
template<class Coord>
inline R4VectorTC<Coord>
Unit( const R4VectorTC<Coord>& v )
{
    return v / Length(v);
}

/// Return given vector if length is zero
template<class Coord>
inline R4VectorTC<Coord>
UnitSafe( const R4VectorTC<Coord>& v, const R4VectorTC<Coord>& out_vIfBad )
{
    Coord dLength = Length( v );

    if ( fabs( dLength ) < (Coord) 1e-12 )
        return out_vIfBad;

    return v / dLength;
}

/// Return 0 vector if length is zero
template<class Coord>
inline R4VectorTC<Coord>
UnitSafe( const R4VectorTC<Coord>& v )
{
    Coord dLength = Length( v );

    if ( fabs( dLength ) < (Coord) 1e-12 )
        return v;

    return v / dLength;
}

///
template<class Coord>
inline Coord
MaximumNorm( const R4VectorTC<Coord>& v )
{
    // L-infinity norm
    Coord max = (Coord) fabs(v[0]);
    if ((Coord) fabs(v[1]) > max)
        max = (Coord) fabs(v[1]);
    if ((Coord) fabs(v[2]) > max)
        max = (Coord) fabs(v[2]);
    if ((Coord) fabs(v[3]) > max)
        max = (Coord) fabs(v[3]);
    return max;
}

// -------------------------------------
// friend functions
// -------------------------------------

///
template<class Coord>
tinline Coord
Length( const R4CoVectorTC<Coord>& v )
{
    return (Coord) sqrt( v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3]*v[3] );
}

///
template<class Coord>
tinline Coord
LengthSq( const R4CoVectorTC<Coord>& v )
{
    return v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3]*v[3];
}

///
template<class Coord>
tinline R4CoVectorTC<Coord>
Unit( const R4CoVectorTC<Coord>& v )
{
    return v / Length(v);
}

///
template<class Coord>
tinline Coord
MaximumNorm( const R4CoVectorTC<Coord>& v )
{
    // L-infinity norm
    Coord max = (Coord) fabs(v[0]);
    if ((Coord) fabs(v[1]) > max)
        max = (Coord) fabs(v[1]);
    if ((Coord) fabs(v[2]) > max)
        max = (Coord) fabs(v[2]);
    if ((Coord) fabs(v[3]) > max)
        max = (Coord) fabs(v[3]);
    return max;
}


///
template<class Coord>
inline R4MatrixTC<Coord> R4MatrixTC<Coord>::Inverse( WINbool &out_bSuccess ) const
{
    R4MatrixTC<Coord> mo;
    const R4MatrixTC<Coord> &mi = *this;
    
    Coord t[4][8], save;
    int i,j;
    // copy mi into t
    // set t's last four rows to the identity matrix
    for (i=0;i<4;i++)
        for (j=0;j<4;j++) {
            t[i][j]=mi(i, j);
            t[i][j+4]=0;
        }
    for (i=0;i<4;i++)
        t[i][i+4]=1;
    
    mo.MakeIdentity();
    for (i=0;i<4;i++) {
        if (i<3) {	// swap row with largest front coefficient
            Coord a=fabs(t[i][i]),ab;
            int m=i;
            for (int l=i+1;l<4;l++) {
                if ((ab=fabs(t[l][i]))>a) {
                    a=ab,m=l;
                }
            }
            if (m!=i) {
                for (j=0;j<8;j++) {
                    save = t[i][j];
                    t[i][j] = t[m][j];
                    t[m][j] = save;
                }
            }
        }
        if (!t[i][i]) {
            out_bSuccess = FALSE ;
            return mo;
        }
        for (j=0;j<4;j++) {
			if (i==j) {
                   Coord a=1/t[i][i];
                   for (int k=0;k<8;k++)
                       t[j][k]*=a;
			} else {
                   Coord a=-t[j][i]/t[i][i];
                   for (int k=0;k<8;k++)
                       t[j][k]+=a*t[i][k];
			}
        }
    }
    for (i=0;i<4;i++)
        for (j=0;j<4;j++)
            mo(i,j)=t[i][j+4];
    
    out_bSuccess = TRUE ;
    return mo;
}

//@}

/**@name Conversion operators */
//@{

///
template<class Coord>
inline R4CoVectorTC<Coord>
VecToCov( const R4VectorTC<Coord>& v )
{
    return R4CoVectorTC<Coord>( v[0], v[1], v[2], v[3] );
}

///
template<class Coord>
inline R4VectorTC<Coord>
CovToVec( const R4CoVectorTC<Coord>& cv )
{
    return R4VectorTC<Coord>( cv[0], cv[1], cv[2], cv[3] );
}

///
template<class Coord>
inline R4VectorTC<Coord>
operator - ( const R4PointTC<Coord>& p1, const R4PointTC<Coord>& p2 )
{
    return R4VectorTC<Coord>( p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2], p1[3] - p2[3] );
}

//@}

