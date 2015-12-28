/********************************* Axes *********************************/
inline
void OGLObjsCamera::PosXAxis()
{
    SetTranslationMatrix( At() + R3Vec(1,0,0) * m_dFocusDist );
    SetRotationMatrix( M_PI / 2.0, R3Vec(0,1,0) );

    SetFinalMatrices();
}

inline
void OGLObjsCamera::PosYAxis()
{
    SetTranslationMatrix( At() + R3Vec(0,1,0) * m_dFocusDist );
    SetRotationMatrix( -M_PI / 2.0, R3Vec(1,0,0) );
    SetFinalMatrices();
}

inline
void OGLObjsCamera::PosZAxis()
{
    SetTranslationMatrix( At() + R3Vec(0,0,1) * m_dFocusDist );
    SetRotationMatrix( 0, R3Vec(0,1,0) );
    SetFinalMatrices();
}

inline
void OGLObjsCamera::NegXAxis()
{
    SetTranslationMatrix( At() - R3Vec(1,0,0) * m_dFocusDist );
    SetRotationMatrix( -M_PI / 2.0, R3Vec(0,1,0) );
    SetFinalMatrices();
}

inline
void OGLObjsCamera::NegYAxis()
{
    SetTranslationMatrix( At() - R3Vec(0,1,0) * m_dFocusDist );
    SetRotationMatrix( M_PI / 2.0, R3Vec(1,0,0) );
    SetFinalMatrices();
}

inline
void OGLObjsCamera::NegZAxis()
{
    SetTranslationMatrix( At() - R3Vec(0,0,1) * m_dFocusDist );
    SetRotationMatrix( -M_PI, R3Vec(0,1,0) );
    SetFinalMatrices();
}

inline
void OGLObjsCamera::RotateLeft( const double in_dAngle )
{
	RotateAroundAt(1, -in_dAngle);
}

inline
void OGLObjsCamera::RotateRight( const double in_dAngle )
{
	RotateAroundAt(1, in_dAngle);
}

inline
void OGLObjsCamera::RotateUp( const double in_dAngle )
{
	RotateAroundAt(0, -in_dAngle);
}

inline
void OGLObjsCamera::RotateDown( const double in_dAngle )
{
	RotateAroundAt(0, in_dAngle);
}

inline
void OGLObjsCamera::SpinClockwise( const double in_dAngle )
{
	RotateAroundAt(2, in_dAngle);
}

inline
void OGLObjsCamera::SpinCounterClockwise( const double in_dAngle )
{
	RotateAroundAt(2, -in_dAngle);
}


/******************************** To and from points ***********/
inline
R2Pt OGLObjsCamera::ScreenPt( const R3Pt &in_pt ) const
{
    // (-1..1, -1..1, 0..1)
    const R2Pt pt = CameraPt(in_pt);

    // scale by width/height
    return CameraToScreen( pt );
}

inline
R2Pt OGLObjsCamera::FlTkToCamera( const int in_iX, const int in_iY ) const
{
    return ScreenToCamera( FlTkToScreen( in_iX, in_iY ) );
}

inline
R2Pt OGLObjsCamera::FlTkToScreen( const int in_iX, const int in_iY ) const
{
    return R2Pt( in_iX, Height() - in_iY - 1 );
}


inline
R2Pt OGLObjsCamera::CameraPt( const R3Pt &in_pt ) const
{
    // (-1..1, -1..1, 0..1)
    const R4Pt pt = Projection() * (R4Pt) in_pt;

    if ( RNIsZero( pt[3] ) )
        return R2Pt( 0.0, 0.0 );

    return R2Pt( pt[0] / pt[3], pt[1] / pt[3] );
}

inline
R3Vec    OGLObjsCamera::RayFromEye( const R2Pt &in_ptScreen ) const
{
    const R3Pt ptInv = m_matPerspInv * R3Pt( in_ptScreen[0], in_ptScreen[1], 1.0 );

    const R3Vec vecCam = ptInv - R3Pt(0,0,0);
    return CameraToWorld() * vecCam;
}

inline
R3Pt     OGLObjsCamera::PtOnScreen( const R2Pt &in_ptScreen ) const
{
    const R2Pt ptCam = ScreenToCamera( in_ptScreen );
    const R3Pt ptCamera( ptCam[0], ptCam[1], 1.0 );

    return ptCamera;
}

///
inline
R2Pt 
OGLObjsCamera::ScreenToCamera( const R2Pt &in_ptScreen ) const
{
    if ( m_iWidth == 0 || m_iHeight == 0 )
        return R2Pt(0,0);

    return R2Pt( -1.0 + 2.0 * (in_ptScreen[0]) / (double) m_iWidth,
                 -1.0 + 2.0 * (in_ptScreen[1]) / (double) m_iHeight );
                
}

inline
R2Vec
OGLObjsCamera::ScreenToCamera( const R2Vec &in_vecScreen ) const
{
    if ( m_iWidth == 0 || m_iHeight == 0 )
        return R2Vec(0,0);

    return R2Vec( 2.0 * (in_vecScreen[0]) / (double) m_iWidth,
                  2.0 * (in_vecScreen[1]) / (double) m_iHeight );
                
}

///
inline
R2Pt     
OGLObjsCamera::CameraToScreen( const R2Pt &in_ptCamera ) const
{
    return R2Pt( m_iWidth *  ( (in_ptCamera[0] + 1.0) / 2.0 ),
                 m_iHeight * ( (in_ptCamera[1] + 1.0) / 2.0 ) );
}

inline
R2Vec     
OGLObjsCamera::CameraToScreen( const R2Vec &in_vecCamera ) const
{
    return R2Vec( m_iWidth *  ( in_vecCamera[0] / 2.0 ),
                  m_iHeight * ( in_vecCamera[1] / 2.0 ) );
}

inline
R3Pt OGLObjsCamera::ProjectedPt( const R3Pt &in_pt ) const
{
    const R4Pt pt = Projection() * (R4Pt) (in_pt);

    if ( RNIsZero( pt[3] ) )
        return From();

    return R3Pt( pt[0] / pt[3], pt[1] / pt[3], pt[2] / pt[3] );
}

///
inline
R3Vec OGLObjsCamera::ProjectedNormal( const R3Vec &in_vec ) const
{
    return m_matProjNorm * in_vec;
}
