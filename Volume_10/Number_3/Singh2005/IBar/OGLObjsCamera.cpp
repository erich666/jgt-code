
#include "StdAfx.H"
#include "OGLObjs_Camera.H"

/******************************** Recalculation of internal data ***********/
void OGLObjsCamera::Check() const
{
    const R3Vec vLook = Look();
    const R3Vec vUp = Up();
    const R3Vec vRight = Right();

    ASSERT( RNApproxEqual( 1.0, Length( vLook ) ) );
    ASSERT( RNApproxEqual( 1.0, Length( vUp ) ) );
    ASSERT( RNApproxEqual( 1.0, Length( vRight ) ) );
    ASSERT( RNIsZero( Dot( vLook, vRight ) ) );
    ASSERT( RNIsZero( Dot( vLook, vUp ) ) );
    ASSERT( RNIsZero( Dot( vRight, vUp ) ) );

}

void OGLObjsCamera::SetTranslationMatrix( const R3Pt &in_pt )
{
    m_matTrans = R4Matrix::Translation( R3Pt(0,0,0) - in_pt );
    m_matTransInv = R4Matrix::Translation( in_pt - R3Pt(0,0,0) );
}

void OGLObjsCamera::SetRotationMatrix()
{
    m_matRot = m_quat.RotationMatrix();
    m_matRotInv = m_matRot.Transpose();
}

void OGLObjsCamera::SetRotationMatrix( const double &in_dAng, const R3Vec &in_vecAxis)
{
    m_quat = S4Quaternion( 1.0, in_vecAxis, in_dAng );
    SetRotationMatrix();
}


void OGLObjsCamera::SetScaleMatrix()
{
    if ( m_bPerspec == FALSE ) {
        m_matScl.MakeIdentity();
        m_matSclInv.MakeIdentity();
        return;
    }

    const double dAspect = (m_iHeight == 0 ) ? 1.0 : (double) m_iWidth / (double) m_iHeight;

    const double dWidth = 1.0 / tan( m_dZoom / 2.0 );
    const double dHeight = dAspect * dWidth;

    m_matScl = R4Matrix::Scaling( dWidth / m_dFar, dHeight / m_dFar, 1.0 / m_dFar, 1.0 );
    m_matSclInv = R4Matrix::Scaling( m_dFar / dWidth, m_dFar / dHeight, m_dFar, 1.0 );
}

void OGLObjsCamera::SetPerspective()
{
    m_bPerspec = TRUE;
    SetPerspectiveMatrix();
    SetScaleMatrix();
    SetFinalMatrices();
    m_matProj = m_matPersp * m_matWtoC;
}


void OGLObjsCamera::SetIdentity()
{
    Reset();

    m_matTrans.MakeIdentity();
    m_matTransInv.MakeIdentity();
    m_bPerspec = FALSE;
    m_ptOrthoSize = R2Pt(2.0, 2.0);
    m_dNear = 0.0;
    m_dFar = -1.0;
    m_dFocusDist = 1.0;
    m_ptFocus = R3Pt(0,0,1.0);

    SetAllMatrices();
}
///

void OGLObjsCamera::SetOrthogonal( const R3Pt &in_ptScale )
{
    m_bPerspec = FALSE;
    m_ptOrthoSize = in_ptScale;

    SetPerspectiveMatrix();
    SetScaleMatrix();
    SetFinalMatrices();

    m_matProj = m_matPersp * m_matWtoC;

}

void OGLObjsCamera::SetPerspectiveMatrix()
{
    // Don't reset to identity because we may have skew, center of proj

    const double dK = -m_dNear / m_dFar;
    if ( m_bPerspec == TRUE ) {

        m_matPersp.MakeIdentity();
        m_matPersp(0, 0) = m_dAspectRatio;
        m_matPersp(0, 2) = m_ptCOP[0];
        m_matPersp(1, 2) = m_ptCOP[1];
        m_matPersp(0, 1) = m_dSkew;
        m_matPersp(3, 3) = 0;
        m_matPersp(3, 2) = -1.0;
        m_matPersp(2, 3) = dK / (1.0 + dK);
        m_matPersp(2, 2) = -1.0 / (1.0 + dK);
    } else {
        m_matPersp.MakeIdentity();

        m_matPersp(0,0) = 2.0 / (m_ptOrthoSize[0]);
        m_matPersp(1,1) = 2.0 / (m_ptOrthoSize[1]);
        m_matPersp(0,3) = -m_ptCOP[0];
        m_matPersp(1,3) = -m_ptCOP[1];
        // a (-n) + b = 0
        // a (-f) + b = 1
        // a (-n + f) = -1
        // a = 1 / (n - f)
        // 1/(n-f) (-n) + b = 0
        // b = n / (n-f)
        //
        // 1 / (n-f) (-n) + n / (n-f) = (-n + n) / (n-f) = 0
        // 1 / (n-f) (-f) + n / (n-f) = (f- + n) / (n-f) = 1
        m_matPersp(2, 3) = m_dNear / (m_dNear - m_dFar);
        m_matPersp(2, 2) = 1.0 / (m_dNear - m_dFar);

    }
}

void OGLObjsCamera::SetFinalMatrices()
{
    m_matWtoC = m_matScl * m_matRot * m_matTrans;
    m_matCtoW = m_matTransInv * m_matRotInv * m_matSclInv;

    m_matCtoWTranspose = m_matCtoW.Transpose();

    m_matProj = m_matPersp * m_matWtoC;
    m_matProjNorm = m_matPersp * m_matCtoWTranspose;

    WINbool bSuc;
    m_matPerspInv = m_matPersp.Inverse( bSuc );
}

void OGLObjsCamera::SetAllMatrices()
{
    m_matRot = m_quat.RotationMatrix();
    m_matRotInv = m_matRot.Transpose();
    
    SetScaleMatrix();
    SetPerspectiveMatrix();
    SetFinalMatrices();
}

/******************* Set directly *****************************/
void OGLObjsCamera::Reset()
{
    m_matPersp.MakeIdentity();

    SetTranslationMatrix( R3Pt(0,0,5) );
    SetRotationMatrix(0, R3Vec(0,1,0));

    m_dZoom     = M_PI / 2.0;
    m_dFocusDist = 5.0;
    m_dNear     = 0.01;
    m_dFar      = 1000;
	m_bPerspec = TRUE;
    m_dSkew = 0.0;
    m_dAspectRatio = 1.0;
    m_ptCOP = R2Pt(0,0);
    m_ptOrthoSize = R2Pt(2.0, 2.0);

    SetAllMatrices();
    m_ptFocus = At();
}

void OGLObjsCamera::SetNearFar( const double in_dNear, const double in_dFar )
{
	m_dNear = WINmax( 1e-6, in_dNear );
	m_dFar = WINmax( 1.0, in_dFar );

    SetScaleMatrix();
    SetPerspectiveMatrix();
    SetFinalMatrices();
}

void OGLObjsCamera::SetZoom( const double in_dZoom )
{
	m_dZoom = WINminmax( in_dZoom, 0.01, M_PI - 0.01 );

    SetScaleMatrix();
    SetFinalMatrices();
}

void OGLObjsCamera::SetQuaternion( const S4Quaternion &in_quat )
{
    m_quat = in_quat;

    SetRotationMatrix();
    SetFinalMatrices();
}

void OGLObjsCamera::SetRotationMatrix( const double in_dRotX, const double in_dRotY, const double in_dRotZ )
{
	const S4Quaternion quatRotX = S4Quaternion( 1.0, R3Vec(1,0,0), -in_dRotX );
	const S4Quaternion quatRotY = S4Quaternion( 1.0, R3Vec(0,1,0), -in_dRotY );
	const S4Quaternion quatRotZ = S4Quaternion( 1.0, R3Vec(0,0,1), -in_dRotZ );

    m_quat = quatRotX * quatRotY * quatRotZ;
    m_quat.Normalize();

    const R4Matrix matRotX = quatRotX.RotationMatrix();
    const R4Matrix matCheckX = R3Matrix::Rotation(0, -in_dRotX);
    const R4Matrix matRotY = quatRotY.RotationMatrix();
    const R4Matrix matCheckY = R3Matrix::Rotation(1, -in_dRotY);
    const R4Matrix matRotZ = quatRotZ.RotationMatrix();
    const R4Matrix matCheckZ = R3Matrix::Rotation(2, -in_dRotZ);

    ASSERT(ApproxEqual( matRotX, matCheckX, 1e-5 ) );
    ASSERT(ApproxEqual( matRotY, matCheckY, 1e-5 ) );
    ASSERT(ApproxEqual( matRotZ, matCheckZ, 1e-5 ) );
    const R4Matrix matRot = m_quat.RotationMatrix();
    const R4Matrix matCheck = matRotZ * matRotY * matRotX;
    const R4Matrix matCheck2 = matRotZ * matRotY * matRotX;

    ASSERT(ApproxEqual( matRot, matCheck, 1e-5 ) );

	SetRotationMatrix();

    ASSERT(ApproxEqual( matRot, RotationToXYZ(), 1e-5 ) );

    SetFinalMatrices();

}

void OGLObjsCamera::SetTranslation( const R3Vec &in_vec )
{
    SetTranslationMatrix( R3Pt(0,0,0) + in_vec );
    SetFinalMatrices();
}

void OGLObjsCamera::SetAspectRatioScale( const double in_d )
{
    m_dAspectRatio = in_d;

    if ( RNIsZero( m_dAspectRatio ) )
        m_dAspectRatio = 1e-6;

    SetPerspectiveMatrix();
    SetFinalMatrices();
}

///
void OGLObjsCamera::SetSkew( const double in_dSkew )
{
    m_dSkew = in_dSkew;

    SetPerspectiveMatrix();
    SetFinalMatrices();
}
///
double OGLObjsCamera::GetSkew() const
{
    return m_dSkew;
}
///
void OGLObjsCamera::SetProjectionCenter( const R2Pt &in_pt )
{
    m_ptCOP = in_pt;

    SetPerspectiveMatrix();
    SetFinalMatrices();
}


///
void OGLObjsCamera::SetEye( const R3Pt &in_pt )
{
    SetTranslationMatrix( in_pt );
    SetFinalMatrices();
}


///
void OGLObjsCamera::SetProjectionDepth( const double in_d )
{
    m_matPersp(2,2) = in_d;
    SetFinalMatrices();
}

R3Vec  OGLObjsCamera::GetCOPOffset() const
{
    const double dScl = m_dFocusDist * tan( m_dZoom / 2.0 );
    const double dXOff = 1.0 / m_dAspectRatio * m_ptCOP[0] * dScl - m_dSkew * m_ptCOP[1];
    const double dYOff = m_ptCOP[1] * dScl;

    const R3Vec vecOff = Right() * dXOff + Up() * dYOff;

    return vecOff;
}

///
void OGLObjsCamera::SetFromAtUp( const R3Pt &in_ptFrom, const R3Pt &in_ptAt, const R3Vec &in_vecUp )
{
    SetTranslationMatrix( in_ptFrom );

	const R3Vec vecLook = UnitSafe(in_ptFrom - in_ptAt );
	const R3Vec vecUp = UnitSafe( in_vecUp - vecLook * Dot( vecLook, in_vecUp ) );

	// rotate into yz plane around y axis
	const double dAngY = atan2( vecLook[0], vecLook[2] );
	const S4Quaternion quatRotY = S4Quaternion( 1.0, R3Vec(0,1,0), dAngY );
	const R3Vec vecLookInYZ = quatRotY * vecLook;
	const R3Vec vecUpInYZ = quatRotY * vecUp;

	// Rotate around x axis to align view with z
	const double dAngX = atan2( -vecLookInYZ[1], vecLookInYZ[2] );
	const S4Quaternion quatRotX =  S4Quaternion( 1.0, R3Vec(1,0,0), dAngX );
	const R3Vec vecLookInZ = quatRotX * vecLookInYZ;
	const R3Vec vecUpInZ = quatRotX * vecUpInYZ;

	// Rotate around z to align up with y
	const double dAngZ = atan2( -vecUpInZ[0], vecUpInZ[1] );
	const S4Quaternion quatRotZ =  S4Quaternion( 1.0, R3Vec(0,0,1), dAngZ );

	//const R3Vec vecLookZ = quatRotZ * vecLookInZ;
	//const R3Vec vecUpY = quatRotZ * vecUpInZ;

	m_dFocusDist = Length( in_ptFrom - in_ptAt );

	m_quat = quatRotY * quatRotX * quatRotZ;

	//const R3Vec vecCheckZ = m_quat * vecLook;
	//const R3Vec vecCheckY = m_quat * vecUp;
	//const R3Vec vecCheckX = m_quat * Cross( vecLook, vecUp );

	SetRotationMatrix();
	SetFinalMatrices();
}

void OGLObjsCamera::SetSize ( const int in_iWidth, const int in_iHeight )
{
	m_iWidth = in_iWidth;
	m_iHeight = in_iHeight;

    SetScaleMatrix();
    SetFinalMatrices();
}

void OGLObjsCamera::CenterCamera( const R3Pt &in_ptCenter, const R3Pt &in_ptScale )
{
    Reset();

    const double dMax = WINmax( in_ptScale[0], WINmax( in_ptScale[1], in_ptScale[2] ) );
    m_dFocusDist = dMax / sin( m_dZoom / 2.0 );

    SetTranslationMatrix( in_ptCenter + R3Vec(0,0,1) * m_dFocusDist );

    SetFinalMatrices();

    m_ptFocus = At();
}

/********************** Change relative ****************************/

void OGLObjsCamera::RotateAroundAt( const int in_iAngle, const double in_dAngle )
{
	const R4Matrix matRot = R3Matrix::Rotation(in_iAngle, in_dAngle);
	RotateAroundPoint( At(), matRot );
}

void OGLObjsCamera::RotateAroundFocusPoint( const int in_iAngle, const double in_dAngle )
{
	const R4Matrix matRot = R4Matrix::Rotation(in_iAngle, in_dAngle);

    m_ptFocus = RotateAroundPoint( m_ptFocus, matRot );
}

R3Pt OGLObjsCamera::RotateAroundPoint( const R3Pt &in_pt, const R4Matrix &in_mat ) 
{
	const R4Matrix matAll = m_matRotInv * in_mat * m_matRot;

    const double dScl = m_dFocusDist * tan( m_dZoom / 2.0 );
    const double dXOff = 1.0 / m_dAspectRatio * m_ptCOP[0] * dScl - m_dSkew * m_ptCOP[1];
    const double dYOff = m_ptCOP[1] * dScl;

    // Undo center of projection pan to find true at point
    const R3Vec vecUndoCOPPan = dXOff * Right() +
                                dYOff * Up();

	// Should keep unit and ortho, but reset just to make sure
	const R3Vec vecFrom = matAll * (From() - in_pt);
	const R3Vec vecAt = matAll * (At() - in_pt);
	const R3Vec vecUp = UnitSafe( matAll * Up() );
	const R3Vec vecRight = UnitSafe( matAll * Right() );

    // Undo center of projection pan to find true at point

    const R3Vec vecRedoCOPPan = dXOff * vecRight +
                                dYOff * vecUp;

    // Find from point if we rotated around the correct at point, then fixed the pan
    const R3Pt ptFrom = (in_pt + vecUndoCOPPan) + vecFrom - vecRedoCOPPan;

    // Correct the at point for the COP pan, then add in the new COP pan
    const R3Pt ptAt = (in_pt + vecUndoCOPPan) + vecAt - vecRedoCOPPan;

    SetFromAtUp( ptFrom, ptAt, vecUp);

    return (in_pt + vecUndoCOPPan) - vecRedoCOPPan;
}

void OGLObjsCamera::RotateVirtualTrackball( const R2Pt &in_ptPrev, const R2Pt &in_ptNext,
                                            const R3Pt &in_ptCenter, const double in_dRadius)
{
    const R3Pt ptCenterCamera = WorldToCamera() * in_ptCenter;
    const R3Pt ptBBoxCamera = WorldToCamera() * ( in_ptCenter + Right() * in_dRadius + Up() * in_dRadius );
    const double dRadius = Length( ptCenterCamera - ptBBoxCamera );

    const R3Vec vecPrev( in_ptPrev[0], in_ptPrev[1], -1.0 );
    const R3Vec vecNext( in_ptNext[0], in_ptNext[1], -1.0 );

    const double dAPrev = pow( vecPrev[0], 2 ) + pow( vecPrev[1], 2 ) + pow( vecPrev[2], 2 );
    const double dBPrev = 2.0 * ( vecPrev[0] * (-ptCenterCamera[0]) + vecPrev[1] * (-ptCenterCamera[1]) + vecPrev[2] * (-ptCenterCamera[2]) );
    const double dCPrev = pow( ptCenterCamera[0], 2 ) + pow( ptCenterCamera[1], 2 ) + pow( ptCenterCamera[2], 2 ) - pow( dRadius, 2.0 );
    const double dDetPrev = pow( dBPrev, 2 ) - 4.0 * dAPrev * dCPrev;

    const double dANext = pow( vecNext[0], 2 ) + pow( vecNext[1], 2 ) + pow( vecNext[2], 2 );
    const double dBNext = 2.0 * ( vecNext[0] * (-ptCenterCamera[0]) + vecNext[1] * (-ptCenterCamera[1]) + vecNext[2] * (-ptCenterCamera[2]) );
    const double dCNext = pow( ptCenterCamera[0], 2 ) + pow( ptCenterCamera[1], 2 ) + pow( ptCenterCamera[2], 2 ) - pow( dRadius, 2.0 );
    const double dDetNext = pow( dBNext, 2 ) - 4.0 * dANext * dCNext;

    /*
    R3Pt ptPrev, ptNext;
    if ( dDetPrev <= 0.0 ) {
        const R2Sphere circ( R2Pt( ptCenterCamera[0], ptCenterCamera[1] ), dRadius );
        const R2Pt ptClosest = circ.Closest( in_ptPrev );
        ptPrev = R3Pt( ptClosest[0] * dRadius, ptClosest[1] * dRadius, ptCenterCamera[2] );
    } else {
        const double dT1 = (-dBPrev - dDetPrev) / (2.0 * dAPrev);
        const double dT2 = (-dBPrev + dDetPrev) / (2.0 * dAPrev);
        ptPrev = R3Pt(0,0,0) + WINmin( dT1, dT2 ) * vecPrev;
    }

    if ( dDetNext <= 0.0 ) {
        const R2Sphere circ( R2Pt( ptCenterCamera[0], ptCenterCamera[1] ), dRadius );
        const R2Pt ptClosest = circ.Closest( in_ptNext );
        ptNext = R3Pt( ptClosest[0] * dRadius, ptClosest[1] * dRadius, ptCenterCamera[2] );
    } else {
        const double dT1 = (-dBNext - dDetNext) / (2.0 * dANext);
        const double dT2 = (-dBNext + dDetNext) / (2.0 * dANext);
        ptNext = R3Pt(0,0,0) + WINmin( dT1, dT2 ) * vecNext;
    }

    const R3Vec vecDir = UnitSafe( ptNext - ptPrev );
    const R3Vec vecToOrigin = UnitSafe( ptNext - ptCenterCamera );
    const R3Vec vecAxis = Cross( vecDir, vecToOrigin );
    const double dDot = Dot( UnitSafe( ptNext - ptCenterCamera ), UnitSafe( ptPrev - ptCenterCamera ) );
    const double dTheta = acos( dDot );
    const S4Quaternion quat( 1.0, vecAxis, dTheta );
    const R4Matrix mat = quat.RotationMatrix();

	m_ptFocus = RotateAroundPoint( m_ptFocus, mat );
    */
}


void OGLObjsCamera::RotateSelf( const int in_iAngle, const double in_dAngle )
{
    m_quat = S4Quaternion( 1.0, R3Vec( (in_iAngle == 0) ? 1 : 0,
                                     (in_iAngle == 1) ? 1 : 0,
                                     (in_iAngle == 2) ? 1 : 0 ), in_dAngle ) * m_quat;
    SetRotationMatrix();             
    SetFinalMatrices();
}

void OGLObjsCamera::AddRotation( const R3Matrix &in_mat )
{
    const R4Matrix mat( in_mat );
    m_matRot = m_matRot * mat;

    SetFinalMatrices();
}

void OGLObjsCamera::ZoomIn( const double in_dAmount )
{
	m_dZoom -= in_dAmount;
	if ( m_dZoom < 0 )
		m_dZoom = in_dAmount / 2.0;
																							
    SetScaleMatrix();
    SetFinalMatrices();
}

void OGLObjsCamera::ZoomOut( const double in_dAmount )
{
	m_dZoom += in_dAmount;

    SetScaleMatrix();
    SetFinalMatrices();
}

void OGLObjsCamera::PanDown( const double in_dAmount )
{
	SetTranslationMatrix( From() + Up() * in_dAmount );

    m_ptFocus = m_ptFocus + Up() * in_dAmount;

    SetFinalMatrices();
}

void OGLObjsCamera::PanUp( const double in_dAmount )
{
	SetTranslationMatrix( From() - Up() * in_dAmount );

    m_ptFocus = m_ptFocus - Up() * in_dAmount;

    SetFinalMatrices();
}

void OGLObjsCamera::PanLeft( const double in_dAmount )
{
	SetTranslationMatrix( From() + Right() * in_dAmount );

    m_ptFocus = m_ptFocus + Right() * in_dAmount;

    SetFinalMatrices();
}

void OGLObjsCamera::PanRight( const double in_dAmount )
{
	SetTranslationMatrix( From() - Right() * in_dAmount );

    m_ptFocus = m_ptFocus - Right() * in_dAmount;

    SetFinalMatrices();
}

void OGLObjsCamera::PanIn( const double in_dAmount )
{
    const double dAmt = Length(Look()) * in_dAmount;
    m_dFocusDist += dAmt;
	SetTranslationMatrix( From() - Look() * in_dAmount );

    SetFinalMatrices();
}

void OGLObjsCamera::PanOut( const double in_dAmount )
{
    const double dAmt = Length(Look()) * in_dAmount;
    m_dFocusDist -= dAmt;
	SetTranslationMatrix( From() + Look() * in_dAmount );

    SetFinalMatrices();
}

void OGLObjsCamera::Translate( const R3Vec &in_vec )
{
	SetTranslationMatrix( From() + in_vec );

    SetFinalMatrices();
}


/******************** OpenGL *******************************/
#ifndef NO_OPENGL

void OGLObjsCamera::SetOpenGLCamera() const
{
    ::glViewport(0, 0, m_iWidth, m_iHeight);

    SetOpenGLCameraMatrices();
}

void OGLObjsCamera::SetOpenGLCameraMatrices() const
{
    SetOpenGLProjectionMatrix();

    SetOpenGLModelviewMatrix();
}

void OGLObjsCamera::SetOpenGLProjectionMatrix() const
{
    ::glMatrixMode( GL_PROJECTION );
    
	::glLoadMatrixd( &m_matPersp(0,0) );
	::glMultMatrixd( &m_matScl(0,0) );
}

void OGLObjsCamera::SetOpenGLModelviewMatrix() const
{
    ::glMatrixMode( GL_MODELVIEW );

	::glLoadMatrixd( &m_matRot(0,0) );
	::glMultMatrixd( &m_matTrans(0,0) );
}

#endif

/******************** Higher level data *******************************/
void OGLObjsCamera::GetFourCorners( R3Pt  & out_ptCamera,
                                    R3Vec   out_avec[4] ) const
{
    out_ptCamera = From();

    out_avec[0] = RayFromEye( R2Pt(0, Height() ) ); //R3Vec( -1, -1, -1 );
    out_avec[1] = RayFromEye( R2Pt(Width(), Height()) ); //R3Vec(  1, -1, -1 );
    out_avec[2] = RayFromEye( R2Pt( 0, 0 ) ); //R3Vec( -1,  1, -1 );
    out_avec[3] = RayFromEye( R2Pt( Width(), 0 ) ); //R3Vec(  1,  1, -1 );

}

/***************************** Constructor stuff *********************/
OGLObjsCamera &OGLObjsCamera::operator=( const OGLObjsCamera &in_oCam )
{
	m_bPerspec = in_oCam.m_bPerspec;
    m_dAspectRatio = in_oCam.m_dAspectRatio;
	m_dFar = in_oCam.m_dFar;
    m_dFocusDist = in_oCam.m_dFocusDist;
	m_dNear = in_oCam.m_dNear;
    m_quat = in_oCam.m_quat;
    m_matTrans = in_oCam.m_matTrans;
    m_matTransInv = in_oCam.m_matTransInv;
    m_ptFocus = in_oCam.m_ptFocus;
	m_dZoom = in_oCam.m_dZoom;
    m_dSkew = in_oCam.m_dSkew;
    m_ptCOP = in_oCam.m_ptCOP;

    m_iWidth = in_oCam.m_iWidth;
    m_iHeight = in_oCam.m_iHeight;
    SetProjectionCenter( in_oCam.GetProjectionCenter() );

    m_ptOrthoSize = in_oCam.m_ptOrthoSize;

    SetRotationMatrix();
    SetAllMatrices();

	return *this;
}

WINbool OGLObjsCamera::operator==( const OGLObjsCamera &in_oCam ) const
{
    if ( Width() != in_oCam.Width() )
        return FALSE;
    if ( Height() != in_oCam.Height() )
        return FALSE;
    if ( !ApproxEqual( From(), in_oCam.From() ) )
        return FALSE;

    if ( !ApproxEqual( At(), in_oCam.At() ) )
        return FALSE;

    if ( !ApproxEqual( m_ptCOP, in_oCam.m_ptCOP ) )
        return FALSE;

    if ( !ApproxEqual( Up(), in_oCam.Up() ) )
        return FALSE;

    if ( !RNApproxEqual( m_dZoom, in_oCam.GetZoom() ) )
        return FALSE;

    if ( !RNApproxEqual( m_dNear, in_oCam.GetNear() ) )
        return FALSE;

    if ( !RNApproxEqual( m_dFar, in_oCam.GetFar() ) )
        return FALSE;

    if ( !RNApproxEqual( m_dFocusDist, in_oCam.m_dFocusDist ) )
        return FALSE;

    if ( !RNApproxEqual( m_dSkew, in_oCam.m_dSkew ) )
        return FALSE;

    if ( !RNApproxEqual( m_dAspectRatio, in_oCam.m_dAspectRatio ) )
        return FALSE;

    if ( !ApproxEqual( GetProjectionCenter(), GetProjectionCenter() ) )
        return FALSE;

    return TRUE;
}


OGLObjsCamera::OGLObjsCamera() : m_bPerspec( TRUE )
{
    m_matPersp.MakeIdentity();
	m_iWidth = m_iHeight = 1;
	Reset();
}

OGLObjsCamera::OGLObjsCamera( const OGLObjsCamera &in_oCam ) : m_bPerspec( TRUE )
{
    m_matPersp.MakeIdentity();
	(*this) = in_oCam;
}

OGLObjsCamera::~OGLObjsCamera()
{
}


/*************************** read and write **********************/

void OGLObjsCamera::Read( ifstream &in )
{
    R3Pt ptFrom;
	in >> ptFrom >> m_ptFocus >> m_quat[0]  >> m_quat[1]  >> m_quat[2]  >> m_quat[3] >> m_dFocusDist >> m_dZoom >> m_dNear >> m_dFar;
    m_bPerspec = ::WINBoolRead(in);
    in >> m_ptCOP >> m_iWidth >> m_iHeight >> m_dAspectRatio >> m_dSkew;

    //char str[2];
	//in.getline( str, 1, '\n' );

    SetTranslationMatrix(ptFrom);
    SetRotationMatrix();
    SetAllMatrices();
}

///
void OGLObjsCamera::Write( ofstream &out ) const 
{
	out << From() << m_ptFocus << m_quat[0] << " " << m_quat[1] << " " << m_quat[2] << " " << m_quat[3] << " " << m_dFocusDist << " " << m_dZoom << " " << m_dNear << " " << m_dFar;
    if ( m_bPerspec == TRUE )
        out << " t ";
    else
        out << " f ";

    out << m_ptCOP << m_iWidth << " " << m_iHeight << " " << m_dAspectRatio << " " << m_dSkew << "\n";
}



void OGLObjsCamera::Print() const
{
	cout << "From " << From() << "  ";
    cout << "Quat " << m_quat[0] << " " << m_quat[1] << " " << m_quat[2] << " " << m_quat[3] << "\nfocus distance " << m_dFocusDist << " ";
    cout << "Zoom, n, f " << m_dZoom << " " << m_dNear << " " << m_dFar << "\n";

    cout << "COP " << m_ptCOP << m_iWidth << " " << m_iHeight << " " << m_dAspectRatio << " " << m_dSkew << "\n";
}



void OGLObjsCamera::SetFocusDistance( const double in_dFocusDist ) 
{ 
    m_dFocusDist = in_dFocusDist; 
    m_ptFocus = At();
}

void OGLObjsCamera::SetFocusPoint( const R3Pt &in_pt, const R3Pt &in_ptScale ) 
{ 
    const double dMaxScale = 0.75 * WINmax( in_ptScale[0], WINmax( in_ptScale[1], in_ptScale[2] ) );
    const R2Pt ptMin = CameraPt( in_pt - Right() * dMaxScale - Up() * dMaxScale );
    const R2Pt ptMax = CameraPt( in_pt + Right() * dMaxScale + Up() * dMaxScale );

    m_dFocusDist = Length( in_pt - From() ); 
    m_ptFocus = in_pt; 
}

