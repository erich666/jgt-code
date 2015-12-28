#include "StdAfx.H"
#include "Cam_IBar.H"
#include "R2_Line.H"
#include "R2_Line_seg.H"

static void OGLDrawCircle( const R2Pt &in_pt, const double in_dR )
{
    if ( RNIsZero( in_dR ) )
        return;

    const double dDiv = WINminmax( ( M_PI / 180.0 ) * in_dR, M_PI / 4.0, M_PI / 10000.0 );

    glPushMatrix();
    glTranslated( in_pt[0], in_pt[1], 0 );

    glBegin( GL_LINE_LOOP );
    for ( double dTheta = 0.0; dTheta < 2.0 * M_PI; dTheta += dDiv ) {
        glVertex2d( in_dR * cos( dTheta ), in_dR * sin( dTheta ) );
    }
    glEnd();

    glPopMatrix();

}

static void OGLBegin2D( )
{
    glMatrixMode( GL_PROJECTION );
    glPushMatrix();
    glLoadIdentity();

    glMatrixMode( GL_MODELVIEW );
    glPushMatrix();
    glLoadIdentity();

    glPushAttrib( GL_LIGHTING );
    glDisable( GL_LIGHTING );
}

static void OGLEnd2D( )
{
    glMatrixMode( GL_PROJECTION );
    glPopMatrix();

    glMatrixMode( GL_MODELVIEW );
    glPopMatrix();

    glPopAttrib( );
}


void CAMIBar::ToggleCenterObject() 
{ 
    m_bCenterObj = (m_bCenterObj == TRUE) ? FALSE : TRUE;

    if ( m_bCenterObj == FALSE ) {
        m_dCubeHeight = 0.5;
    } else {
        m_dCubeHeight = m_dObjCubeHeight;
    }
}


void CAMIBar::SetFocusPoint( const R3Pt &in_pt, const R3Pt &in_ptScale ) 
{ 
    const double dMaxScale = 0.75 * WINmax( in_ptScale[0], WINmax( in_ptScale[1], in_ptScale[2] ) );
    const R2Pt ptMin = CameraPt( in_pt - Right() * dMaxScale - Up() * dMaxScale );
    const R2Pt ptMax = CameraPt( in_pt + Right() * dMaxScale + Up() * dMaxScale );

    m_dObjCubeHeight = WINmin( 0.9, 0.5 / sqrt(2.0) * Length( ptMax - ptMin ) );

    m_dFocusDist = Length( in_pt - From() ); 
    m_ptFocus = in_pt; 
}


CAMIBar &CAMIBar::operator=( const CAMIBar &in_ibar )
{
    m_bCenterObj = in_ibar.m_bCenterObj;
    m_dClickDistance = in_ibar.m_dClickDistance;
    m_handle = in_ibar.m_handle;
    m_commit = in_ibar.m_commit;
    m_bCameraMode = in_ibar.m_bCameraMode;
    m_iClosest = in_ibar.m_iClosest;
    m_ptDown = in_ibar.m_ptDown;
    m_ptDrag = in_ibar.m_ptDrag;
    m_dObjCubeHeight = in_ibar.m_dObjCubeHeight;
        
    m_cameraOrig = in_ibar.m_cameraOrig;

    OGLObjsCamera::operator =( in_ibar );
    return *this;
}

CAMIBar::CAMIBar( const CAMIBar &in_ibar ) :
m_bCenterObj( in_ibar.m_bCenterObj )
{
    (*this) = in_ibar;
}

CAMIBar::CAMIBar() : 
m_bCenterObj(FALSE),
m_dClickDistance( 0.02 ),
m_handle( NONE ),
m_commit( UNCOMMITTED ),
m_bCameraMode( FALSE ),
m_iClosest( -1 ),
m_dCubeHeight(0.5),
m_dObjCubeHeight(0.4)
{
    m_cameraOrig = *this;
}

// Ordering of cube vertices
//  0  1  2  3  4  5  6
//        7  8  9 
//  10 11 12 13 14 15 16
std::vector<R3Pt> CAMIBar::MakeCube(  double &out_dTStem ) const 
{
    double dZoom = m_dZoom;
    double dFD = m_dFocusDist;
    R2Pt ptCOP = GetProjectionCenter();
    R3Pt ptFocus = m_ptFocus;
    R3Vec vecRight = Right();
    R3Vec vecUp = Up();
    R3Vec vecLook = Look();
    R3Pt ptFrom = From();
    R3Vec vecOffset = GetCOPOffset();

    std::vector<R3Pt> aptCube(17);

    out_dTStem = 0.5;

    if ( m_handle != NONE ) {
        dZoom = m_cameraOrig.GetZoom();
        dFD = m_cameraOrig.GetFocusDist();
        ptCOP = m_cameraOrig.GetProjectionCenter();
        ptFocus = m_cameraOrig.GetFocusPoint();
        vecRight = m_cameraOrig.Right();
        vecUp = m_cameraOrig.Up();
        vecLook = m_cameraOrig.Look();
        ptFrom = m_cameraOrig.From();
        vecOffset = m_cameraOrig.GetCOPOffset();
    }
    
    const double dScl = dFD * tan( dZoom / 2.0 );
    const double dSclHeight = dFD * m_dCubeHeight * tan( dZoom / 2.0 );
    const double dSclLimb   = dFD * m_dCubeHeight * tan( dZoom / 2.0 );

    // Vertical bar
    if ( m_bCenterObj == TRUE ) {
        aptCube[8] = ptFocus + vecOffset; //R3Pt( 0,  0.0, 0.0 );
    } else {
        aptCube[8] = ptFrom + vecLook * dFD + vecOffset; //R3Pt( 0,  0.0, 0.0 );
    }

    aptCube[3] = aptCube[8] + vecUp * dSclHeight; //R3Pt( 0,  dHeight,  0.0 );
    aptCube[13] = aptCube[8] - vecUp * dSclHeight; //R3Pt( 0,-dHeight,  0.0 );

    // limbs are broken into 3 pieces, in (+-, +-, -) direction
    const R3Pt ptUL = aptCube[3]  - vecRight * dSclLimb + vecLook * dSclLimb; 
    const R3Pt ptUR = aptCube[3]  + vecRight * dSclLimb + vecLook * dSclLimb; 
    const R3Pt ptLL = aptCube[13] - vecRight * dSclLimb + vecLook * dSclLimb; 
    const R3Pt ptLR = aptCube[13] + vecRight * dSclLimb + vecLook * dSclLimb; 
    for ( int i = 0; i < 3; i++ ) {
        const double dThird = i / 3.0;
        const double dThirdNext = (i+1.0) / 3.0;

        // top left, top right, bottom left, bottom right
        aptCube[i]      = Lerp( aptCube[3],  ptUL,  1.0 - dThird );
        aptCube[4 + i]  = Lerp( aptCube[3],  ptUR,  dThirdNext );
        aptCube[10 + i] = Lerp( aptCube[13], ptLL,  1.0 - dThird );
        aptCube[14 + i] = Lerp( aptCube[13], ptLR,  dThirdNext );
    }

    aptCube[7] = aptCube[8] - vecRight * 0.5 * dSclHeight;
    aptCube[9] = aptCube[8] + vecRight * 0.5 * dSclHeight;

    if ( ApproxEqual( aptCube[3], aptCube[0] ) )
        return aptCube;

    // This calculation is all done with the current camera
    double dDist;
    R2Pt ptIntersectL, ptIntersectR, ptClosest, ptCross;
    const R2Line line1( CameraPt( aptCube[3] ),  CameraPt( aptCube[0] ) );
    const R2Line line2( CameraPt( aptCube[13] ), CameraPt( aptCube[10] ) );
    const R2Line line3( CameraPt( aptCube[3] ),  CameraPt( aptCube[6] ) );
    const R2Line line4( CameraPt( aptCube[13] ), CameraPt( aptCube[16] ) );
    line1.Intersect( line2, ptIntersectL );
    line3.Intersect( line4, ptIntersectR );

    const R2Line_seg lineStem( CameraPt( aptCube[3] ), CameraPt( aptCube[13] ) );
    const R2Line_seg lineHorizon( ptIntersectL, ptIntersectR );

    lineStem.Intersect( lineHorizon, ptCross, out_dTStem, dDist );

    // Horizontal bar

    aptCube[7] = Lerp( aptCube[3], aptCube[13], out_dTStem ) - vecRight * 0.5 * dSclHeight;
    aptCube[9] = Lerp( aptCube[3], aptCube[13], out_dTStem ) + vecRight * 0.5 * dSclHeight;

    return aptCube;
}



void CAMIBar::SetOpenGLCamera() const
{
    if ( m_handle == NONE ) {
        OGLObjsCamera::SetOpenGLCamera();
    } else {
        if ( m_bCameraMode == TRUE )
            m_cameraOrig.SetOpenGLCamera();
        else
            OGLObjsCamera::SetOpenGLCamera();
    }
}

///
void CAMIBar::Draw( const WINbool in_bMouseOver ) const
{
    double dTStem = 0.5;
    std::vector<R3Pt> aptCube = MakeCube(dTStem);

    glPushAttrib( GL_ALL_ATTRIB_BITS );

    ::glMatrixMode( GL_PROJECTION );
    ::glPushMatrix();
    ::glMatrixMode( GL_MODELVIEW );
    ::glPushMatrix();
    
    ::glDisable( GL_LIGHTING );
    ::glDisable( GL_DEPTH_TEST );
    ::glEnable( GL_LINE_SMOOTH );


    OGLBegin2D();
    glColor3f(0,0,1);
    glLineWidth( 1.0f );
    OGLDrawCircle( CameraPt( Lerp( aptCube[3], aptCube[13], dTStem * 0.9 ) ), 0.02 );

    // This draws the black backgrounds behind the colored lines
    // All done in 2D
    // Also draws the highlited selection point
    if ( in_bMouseOver == TRUE ) {

        const OGLObjsCamera &cam = (m_bCameraMode == FALSE || m_handle == NONE) ? *this : m_cameraOrig;

        // Circle sizes
	    const double dScl1 = 0.01;
	    const double dScl2 = -dScl1 / 4.0;
	    std::vector<R2Pt> apt(10);
	    apt[0] = CameraPt( aptCube[0] );
	    apt[1] = CameraPt( aptCube[3] );
	    apt[2] = CameraPt( aptCube[3] );
	    apt[3] = CameraPt( aptCube[6] );
	    apt[4] = CameraPt( aptCube[7] );
	    apt[5] = CameraPt( aptCube[9] );
	    apt[6] = CameraPt( aptCube[10] );
	    apt[7] = CameraPt( aptCube[13] );
	    apt[8] = CameraPt( aptCube[13] );
	    apt[9] = CameraPt( aptCube[16] );

	    glColor3f(0.0,0.0,0.0);
	    for ( int iY = -2; iY < 3; iY++ ) {
		    glBegin( GL_LINES );
		    for ( int i = 0; i < apt.size(); i++ ) 
			    glVertex3d( apt[i][0], apt[i][1] + iY * 0.003, 0.0 );
		    glVertex3dv( &aptCube[16][0] );
		    glEnd();
	    }
	    
        // Where to draw the circle
        glColor3f(1,1,1);
        glLineWidth( 1.0f );
        switch (m_iClosest) {
        case 0 :
        case 1 :
        case 2 :
        case 10 :
        case 11 :
        case 12 :
            OGLDrawCircle( Lerp( CameraPt( aptCube[m_iClosest] ), CameraPt( aptCube[m_iClosest+1] ), 0.5 ), 0.025 ); 
            break;
        case 4 :
        case 5 :
        case 6 :
        case 14 :
        case 15 :
        case 16 :
            OGLDrawCircle( Lerp( CameraPt( aptCube[m_iClosest] ), CameraPt( aptCube[m_iClosest-1] ), 0.5 ), 0.025 ); 
            break;
        case 7 :
        case 9 :
            OGLDrawCircle( CameraPt( aptCube[m_iClosest] ), 0.025 ); 
            break;
        case 8 :
            OGLDrawCircle( CameraPt( Lerp( aptCube[3], aptCube[13], dTStem * 0.9 ) ), 0.025 ); 
            break;
        case 3 :
            OGLDrawCircle( CameraPt( Lerp( aptCube[3], aptCube[13], WINmin( dTStem * 0.5, 0.1) ) ), 0.025 ); 
            break;
        case 13 :
            OGLDrawCircle( CameraPt( Lerp( aptCube[3], aptCube[13], 0.9 ) ), 0.025 ); 
            break;
        case 17 :
            OGLDrawCircle( CameraPt( Lerp( aptCube[3], aptCube[13], dTStem * 0.5 ) ), 0.025 ); 
            break;
        case 18 :
            OGLDrawCircle( CameraPt( Lerp( aptCube[3], aptCube[13], dTStem * 1.1 ) ), 0.025 ); 
            break;
        case -1 :
            break;
        }

    }
    OGLEnd2D();

    // Perspective, focal length, and aspect ratio part of camera matrix
    SetOpenGLCamera();

    const float fDef = (m_handle == PAN) ? 3.0f : 2.0f;
    const float fSel = 2.0f;

    static float s_afGreen[4] = {0,1,0,1};

    // Draw the colored lines, one segment at a time
    ::glBegin( GL_LINES );
    ::glColor3f( 1.0,0,1.0 ); glMaterialfv( GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, s_afGreen );
    if ( m_handle == LEFT || m_handle == TOP )
        glLineWidth( fSel );
    else
        glLineWidth( fDef );

    ::glVertex3dv( &aptCube[0][0] );
    ::glVertex3dv( &aptCube[1][0] );

    if ( m_handle == LEFT || m_handle == BOTTOM )
        glLineWidth( fSel );
    else
        glLineWidth( fDef );

    ::glVertex3dv( &aptCube[10][0] );
    ::glVertex3dv( &aptCube[11][0] );

    ::glColor3f( 0,1.0,0 );
    if ( m_handle == TOP )
        glLineWidth( fSel );
    else
        glLineWidth( fDef );

    ::glVertex3dv( &aptCube[1][0] );
    ::glVertex3dv( &aptCube[2][0] );

    ::glVertex3dv( &aptCube[4][0] );
    ::glVertex3dv( &aptCube[5][0] );

    ::glColor3f( 1.0,0,0 );
    if ( m_handle == BOTTOM )
        glLineWidth( fSel );
    else
        glLineWidth( fDef );

    ::glVertex3dv( &aptCube[11][0] );
    ::glVertex3dv( &aptCube[12][0] );

    ::glVertex3dv( &aptCube[14][0] );
    ::glVertex3dv( &aptCube[15][0] );

    ::glColor3f( 0,1.0,1.0 );
    if ( m_handle == TOP || m_handle == RIGHT)
        glLineWidth( fSel );
    else
        glLineWidth( fDef );

    ::glVertex3dv( &aptCube[5][0] );
    ::glVertex3dv( &aptCube[6][0] );

    ::glVertex3dv( &aptCube[15][0] );
    ::glVertex3dv( &aptCube[16][0] );

    ::glColor3f( 0,0,1.0 );
    if ( m_handle == SKEW || m_handle == ASPECT_RATIO )
        glLineWidth( fSel );
    else
        glLineWidth( fDef );

    const R3Pt ptL = Lerp( aptCube[3], aptCube[13], dTStem * 0.5 ) + ( aptCube[9] - aptCube[7] ) * 0.1;
    const R3Pt ptR = Lerp( aptCube[3], aptCube[13], dTStem * 0.5 ) - ( aptCube[9] - aptCube[7] ) * 0.1;
    ::glVertex3dv( &ptL[0] );
    ::glVertex3dv( &ptR[0] );

    if ( m_handle == TOP || m_handle == LEFT )
        glLineWidth( fSel );
    else
        glLineWidth( fDef );

    ::glVertex3dv( &aptCube[2][0] );
    ::glVertex3dv( &aptCube[3][0] );

    if ( m_handle == TOP || m_handle == RIGHT )
        glLineWidth( fSel );
    else
        glLineWidth( fDef );

    ::glVertex3dv( &aptCube[3][0] );
    ::glVertex3dv( &aptCube[4][0] );

    if ( m_handle == BOTTOM || m_handle == LEFT )
        glLineWidth( fSel );
    else
        glLineWidth( fDef );

    ::glVertex3dv( &aptCube[12][0] );
    ::glVertex3dv( &aptCube[13][0] );

    if ( m_handle == BOTTOM || m_handle == RIGHT )
        glLineWidth( fSel );
    else
        glLineWidth( fDef );

    ::glVertex3dv( &aptCube[13][0] );
    ::glVertex3dv( &aptCube[14][0] );

    if ( m_handle == ZOOM )
        glLineWidth( fSel );
    else
        glLineWidth( fDef );

    ::glVertex3dv( &aptCube[3][0] );
    ::glVertex3dv( &aptCube[13][0] );

    ::glVertex3dv( &aptCube[7][0] );
    ::glVertex3dv( &aptCube[9][0] );

    ::glEnd();

    const double dScl = Length( At() - From() ) * tan( GetZoom() / 2.0 );
    const R3Pt ptAt = ( m_bCenterObj == TRUE ? m_ptFocus : At() ) + GetCOPOffset();

    const R3Pt ptLL = ptAt - Right() * 0.1 - Up() * 0.1;
    const R3Pt ptRR = ptAt + Right() * 0.1 - Up() * 0.1;
    const R3Pt ptUU = ptAt + Right() * 0.1 + Up() * 0.1;
    const R3Pt ptBB = ptAt - Right() * 0.1 + Up() * 0.1;
    ::glColor4f( 0.5, 0.5, 0.5, 0.5 );

    ::glEnable( GL_DEPTH_TEST );
    ::glEnable( GL_BLEND );
    ::glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );

    ::glBegin( GL_POLYGON );
    ::glVertex3dv( &ptLL[0] );
    ::glVertex3dv( &ptRR[0] );
    ::glVertex3dv( &ptUU[0] );
    ::glVertex3dv( &ptBB[0] );
    ::glEnd( );

    ::glMatrixMode( GL_PROJECTION );
    :: glPopMatrix();

    ::glMatrixMode( GL_MODELVIEW );
    :: glPopMatrix();

    
    glPopAttrib();

}

///
void CAMIBar::SetClickDistance( const double in_d ) 
{
    m_dClickDistance = in_d;
}

static int m_aiMapLimb[6] = {0, 6, 10, 15, 9, 13};
static int m_aiMapLimbBase[6] = {3, 3, 13, 13, 7, 3};

// Loop through the limbs and see which segment we're over. Much the
// same as MouseDown.
WINbool CAMIBar::IsMouseOver( const R2Pt_i &in_ipt, WINbool &out_bRedraw ) 
{
    double dTStem = 0.5;
    std::vector<R3Pt> aptCube = MakeCube( dTStem );

    // Convert to (-1,1) X (-1,1)
    const R2Pt ptClick = FlTkToCamera( in_ipt[0], in_ipt[1] );

    const int iClosestSave = m_iClosest;

    double dDist;
    m_iClosest = -1;
    R2Pt ptClosest;
    double dT, dClosest = 1e30;
    for ( int i = 0; i < 6; i++ ) {
        const R2Line_seg segLimb( CameraPt( aptCube[ m_aiMapLimbBase[i]] ),  CameraPt( aptCube[ m_aiMapLimb[i] ] ) );
        segLimb.FindPtOnSeg( ptClick, ptClosest, dT, dDist );

        dT = WINminmax( dT, 1e-16, 1.0 - 1e-16 );
        if ( dDist < dClosest ) {
            switch (i) {
            case 0 :
            case 2 :
                m_iClosest = (int) ( dT * m_aiMapLimb[i] + (1.0 - dT) * m_aiMapLimbBase[i] );
                break;
            case 1 :
            case 3 :
                m_iClosest = 1 + (int) ( dT * m_aiMapLimb[i] + (1.0 - dT) * m_aiMapLimbBase[i] );
                break;
            case 4 :
                m_iClosest = 7 + (int) ( dT * 3.0 );
                break;
            case 5 :
                if ( dT < ( 0.5 * dTStem - 0.05) ) {
                    m_iClosest = 3;  // Spin object-centric
                } else if ( dT < (0.5 * dTStem + 0.05) ) {
                    m_iClosest = 17; // skew/aspect ratio
                } else if ( dT < dTStem ) {
                    m_iClosest = 8; // Pan, either object or camera-centric
                } else if ( dT < 0.5 * (dTStem + 1.0) ) {
                    m_iClosest = 18; // Pan, either object or camera-centric
                } else  {
                    m_iClosest = 13; // Spin camera-centric
                }
                break;
            }
            dClosest = dDist;
        }
    }

    if ( m_iClosest != iClosestSave ) {
        out_bRedraw = FALSE;
    } else { 
        out_bRedraw = TRUE;
    }

    if ( dClosest < m_dClickDistance ) {
        return TRUE;
    }

    m_iClosest = -1;
    return FALSE;
}

WINbool CAMIBar::MouseDown( const R2Pt_i &in_ipt, const WINbool in_bShift ) 
{
    double dTStem = 0.5;
    std::vector<R3Pt> aptCube = MakeCube(dTStem);

    m_cameraOrig = *this;

    // Convert to (-1,1) X (-1,1)
    m_ptDown = FlTkToCamera( in_ipt[0], in_ipt[1] );
    m_commit = UNCOMMITTED;
    m_bCameraMode = FALSE;

    // Project each point 
    m_iClosest = -1;

    double dDist = 0.0, dT = 0.0, dClosest = 1e30;
    R2Pt ptClosest;
    for ( int i = 0; i < 6; i++ ) {
        const R2Line_seg segLimb( CameraPt( aptCube[ m_aiMapLimbBase[i]] ),  CameraPt( aptCube[ m_aiMapLimb[i] ] ) );
        segLimb.FindPtOnSeg( m_ptDown, ptClosest, dT, dDist );

        dT = WINminmax( dT, 1e-16, 1.0 - 1e-16 );

        if ( dDist < dClosest ) {
            switch (i) {
            case 0 :
            case 2 :
                m_iClosest = (int) ( dT * m_aiMapLimb[i] + (1.0 - dT) * m_aiMapLimbBase[i] );
                break;
            case 1 :
            case 3 :
                m_iClosest = 1 + (int) ( dT * m_aiMapLimb[i] + (1.0 - dT) * m_aiMapLimbBase[i] );
                break;
            case 4 :
                m_iClosest = 7 + (int) ( dT * 3.0 );
                break;
            case 5 :
                if ( dT < ( 0.5 * dTStem - 0.05) ) {
                    m_iClosest = 3;  // Spin object-centric
                } else if ( dT < (0.5 * dTStem + 0.05) ) {
                    m_iClosest = 17; // skew/aspect ratio
                } else if ( dT < dTStem ) {
                    m_iClosest = 8; // Pan, either object or camera-centric
                } else if ( dT < 0.5 * (dTStem + 1.0) ) {
                    m_iClosest = 18; // Pan, either object or camera-centric
                } else  {
                    m_iClosest = 13; // Spin camera-centric
                }
                break;
            }
            dClosest = dDist;
        }
    }

    if ( dClosest > m_dClickDistance )
        m_iClosest = -1;

    switch( m_iClosest ) {
    case -1 : m_handle = NONE; break;
    case 8 : m_handle = PAN; m_bCameraMode = FALSE; break;
    case 18 : m_handle = PAN; m_bCameraMode = TRUE; break;
    case 3 :  m_handle = ROTATE; break;
    case 13 : m_handle = ROTATE; m_bCameraMode = TRUE; break;
    case 2 : 
    case 4 : 
    case 14 : 
    case 12 : m_handle = ALL;   break;
    case 7 : m_bCameraMode = TRUE; 
    case 9 : m_handle = ZOOM; break;
    case 0 : 
    case 10 : m_handle = LEFT; break;
    case 1 : 
    case 5 : m_handle = TOP; break;
    case 6 : 
    case 16 : m_handle = RIGHT; break;
    case 11 : 
    case 15 : m_handle = BOTTOM; break;
    case 17 : 
            if ( in_bShift ) 
                m_handle = SKEW;
            else
                m_handle = PAN;
            break;

    default : ASSERT(FALSE);
    }

	if ( m_iClosest == -1 ) return FALSE;
	return TRUE;
}

R2Vec CAMIBar::Limb( const std::vector<R3Pt> &in_aptCube, const R4Matrix &in_mat ) const
{
    R3Vec vecProj(0,0,1);

    //  Vector directions (* indicates arrow head)
    //
    //      ---* ----*
    //          *
    //          |
    //      ---------*
    //          |
    //      ---* ----*

    switch ( m_iClosest ) {
    case 0 :
    case 1 :
    case 2 :
        vecProj = in_mat * in_aptCube[3] - in_mat * in_aptCube[0];
        break;
    case 4 :
    case 5 :
    case 6 :
        vecProj = in_mat * in_aptCube[6] - in_mat * in_aptCube[3];
        break;
    case 10 :
    case 11 :
    case 12 :
        vecProj = in_mat * in_aptCube[13] - in_mat * in_aptCube[10];
        break;
    case 14 :
    case 15 :
    case 16 :
        vecProj = in_mat * in_aptCube[16] - in_mat * in_aptCube[13];
        break;
    case 7 :
    case 8 :
    case 9 :
    case 18 :
        vecProj = in_mat * in_aptCube[9] - in_mat * in_aptCube[7];
        break;
    case 3 :
    case 13 :
        vecProj = in_mat * in_aptCube[3] - in_mat * in_aptCube[13];
        break;
    case 17 :
        vecProj = in_mat * in_aptCube[3] - in_mat * in_aptCube[8];
        break;
    default :
        ASSERT(FALSE);
        break;
    }

    return R2Vec( vecProj[0], vecProj[1] );
}

R2Pt CAMIBar::LimbBase( const std::vector<R3Pt> &in_aptCube, const R4Matrix &in_mat ) const
{
    R3Pt ptLimb;

    if ( m_iClosest < 7 )
        ptLimb = in_mat * in_aptCube[3];
    else if ( m_iClosest < 10 )
        ptLimb = in_mat * Lerp( in_aptCube[7], in_aptCube[9], 0.5 );
    else if ( m_iClosest < in_aptCube.size() )
        ptLimb = in_mat * in_aptCube[13];
    else
        ptLimb = in_mat * in_aptCube[8];

    return R2Pt( ptLimb[0], ptLimb[1] );
}



void CAMIBar::MouseMoveObject()  
{
    const R2Vec vec = m_ptDrag - m_ptDown;

    double dTStem = 0.5;
    std::vector<R3Pt> aptCube = MakeCube( dTStem );

    OGLObjsCamera::operator= (m_cameraOrig);

    const R2Vec vecMiddle = UnitSafe( Limb( aptCube, m_matProj ) );
    const double dLenOrig = 2.0 * Dot( vecMiddle, m_ptDown - LimbBase( aptCube,m_matProj ) );
    const double dLenNew  = 2.0 * Dot( vecMiddle, m_ptDrag - LimbBase( aptCube,m_matProj ) );
    const double dRatio = dLenOrig / dLenNew;

    if ( m_handle == PAN ) {
        // Since cube is at film plane, move by mouse movement in film plane
        const double dTrans = m_dFocusDist * tan( m_dZoom / 2.0 );
        switch (m_commit) {
        case SCALE_ONLY :
            PanLeft( -vec[0] * dTrans ); break;
        case ANGLE_ONLY :
            PanUp( vec[1] * dTrans ); break;
        case BOTH :
            PanLeft( -vec[0] * dTrans ); 
            PanUp( vec[1] * dTrans ); break;
        case UNCOMMITTED :
            break;
        }
    } else if ( m_handle == ZOOM ) {
        const double dFocusLen = m_iHeight / tan( m_dZoom / 2.0 );
        const double dNewFL = dFocusLen / dRatio;
		SetZoom( WINminmax( 2.0 * atan( m_iHeight / dNewFL ), 1e-4, M_PI - 1e-4 ) );

    } else if ( m_handle == ROTATE ) {
        // Angle of the current point to the center of the screen (minus 90 degrees)
        // is desired rotation of object
        const double dAng = atan2( m_ptDrag[1], m_ptDrag[0] ) + ((m_iClosest == 3) ? - M_PI / 2.0 : M_PI / 2.0);
        RotateAroundFocusPoint( 2, dAng );

    } else if ( m_handle == LEFT || m_handle == RIGHT ) {
        switch ( m_commit ) {
        case SCALE_ONLY :
            RotateAroundFocusPoint( 1, -vec[0] * M_PI / 2.0 );
            break;
        case ANGLE_ONLY :
            PanLeft( -vec[1] * m_dFocusDist * tan( m_dZoom / 2.0 ) );
            SetProjectionCenter( GetProjectionCenter() + R2Vec( vec[1], 0 ) );
            break;
        case BOTH :
            PanLeft( -vec[1] * m_dFocusDist * tan( m_dZoom / 2.0 ) );
            SetProjectionCenter( GetProjectionCenter() + R2Vec( vec[1], 0 ) );

            RotateAroundFocusPoint( 1, -vec[0] * M_PI / 2.0 );
            break;
        case UNCOMMITTED :
            break;
        }
    } else if ( m_handle == TOP || m_handle == BOTTOM ) {
        switch ( m_commit ) {
        case SCALE_ONLY :
            RotateAroundFocusPoint( 0, ((m_handle == BOTTOM) ? -1.0 : 1.0) * vec[0] * M_PI / 2.0 );
            break;
        case ANGLE_ONLY :
            PanUp( -vec[1] * GetAspectRatio() * m_dFocusDist * tan( m_dZoom / 2.0 ) );
            SetProjectionCenter( GetProjectionCenter() + R2Vec( 0, -vec[1] ) );
            break;
        case BOTH :
            RotateAroundFocusPoint( 0, ((m_handle == BOTTOM) ? -1.0 : 1.0) * vec[0] * M_PI / 2.0 );
            break;
        case UNCOMMITTED :
            break;
        }
    } else if ( m_handle == ALL ) {
        double dNewFD = m_dFocusDist;
        const double dScalePerc = fabs( m_ptDown[1] - m_ptDrag[1] ) / 0.5;
        if ( m_ptDown[1] > m_ptDrag[1] )
            dNewFD = WINmax(0.01 * m_dFocusDist, 1.0 - 0.5 * dScalePerc) * m_dFocusDist;
        else 
            dNewFD = (1.0 + 2.0 * dScalePerc) * m_dFocusDist;

        // Adjust the focal length so that the IBar still shows up the same size
        const double dF = 1.0 / tan( m_dZoom / 2.0 );
        const double dFNew = dF * dNewFD / m_dFocusDist;
        const double dAngNew = 2.0 * atan( 1.0 / dFNew );

        switch ( m_commit ) {
        case SCALE_ONLY :
            SetZoom( WINminmax( GetZoom() * dRatio, 0.0001, M_PI - 0.001 ) );
            break;
        case ANGLE_ONLY :
            PanIn( dNewFD - m_dFocusDist );
            break;
        case BOTH :
            SetZoom( dAngNew );
            PanIn( dNewFD - m_dFocusDist );
            break;
        case UNCOMMITTED :
            break;
        }
    } else if ( m_handle == ASPECT_RATIO ) {
        m_dAspectRatio *= dRatio;
        SetPerspectiveMatrix();
    } else if ( m_handle == SKEW ) {
        m_dSkew = m_dSkew + vec[0];
        SetPerspectiveMatrix();
    }

    SetAllMatrices();
}
///
void CAMIBar::MouseMove( const R2Pt_i &in_ipt, const WINbool in_bShift ) 
{
    m_ptDrag = FlTkToCamera( in_ipt[0], in_ipt[1] );


    if ( m_iClosest == -1 )
        return;

    const R2Vec vec = m_ptDrag - m_ptDown;

    if ( (m_handle == LEFT || m_handle == RIGHT) && in_bShift == TRUE ) {
        m_commit = BOTH;
    }
    if ( (m_handle == TOP || m_handle == BOTTOM ) && in_bShift == TRUE ) {
        m_commit = BOTH;
    }

    if ( m_commit == UNCOMMITTED ) {
        if ( fabs( vec[0] ) > fabs( vec[1] ) + 0.009 || fabs( vec[1] ) > fabs( vec[0] ) + 0.009 ) {
            if ( fabs( vec[0] ) > fabs( vec[1] ) + 0.002 ) {
                m_commit = SCALE_ONLY;
            } else  {
                m_commit = ANGLE_ONLY;
            }

            if (m_handle == ALL && in_bShift == FALSE)
                m_commit = BOTH;
            if ( m_handle == PAN && in_bShift == FALSE )
                m_commit = BOTH;
        }
    }
    if ( m_commit != UNCOMMITTED && in_bShift == TRUE && m_handle == SKEW ) {
        if ( m_commit == ANGLE_ONLY ) {
            m_handle = ASPECT_RATIO; 
        } else {
            m_handle = SKEW;
       }
    }

	MouseMoveObject();
}
///
void CAMIBar::MouseRelease( const R2Pt_i &in_ipt, const WINbool in_bShift ) 
{
    m_ptDrag = FlTkToCamera( in_ipt[0], in_ipt[1] );
	if ( m_bCameraMode == TRUE ) {
        switch (m_handle) {
        case PAN :
        case ZOOM :
            m_ptDrag = m_ptDown;
            m_ptDown = FlTkToCamera( in_ipt[0], in_ipt[1] );
            break;
        case ROTATE :
            m_ptDrag[0] = -m_ptDrag[0];
            break;
        }
	}
	MouseMoveObject();


    m_iClosest = -1;
    m_handle = NONE;
    m_commit = UNCOMMITTED;
}

