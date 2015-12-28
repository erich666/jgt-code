// JofGTIBar.cpp: implementation of the JofGTIBar class.
//
//////////////////////////////////////////////////////////////////////

#include <FL/GL.H>
#include "JofGTIBar.h"
#include <GL/glut.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

void JofGTIBar::Spin( const double in_d )
{
    const double dSpin = m_dSpinLast - in_d;
    m_dSpinLast = in_d;

    SetCamera().SpinClockwise( dSpin );
}

void JofGTIBar::Pan( const double in_d )
{
    const double dPan = m_dPanLast - in_d;
    m_dPanLast = in_d;

    SetCamera().RotateLeft( 2.0 * M_PI * dPan );
}

void JofGTIBar::Tilt( const double in_d )
{
    const double dTilt = m_dTiltLast - in_d;
    m_dTiltLast = in_d;

    SetCamera().RotateUp( 2.0 * M_PI * dTilt );
}

JofGTIBar::JofGTIBar(int X, int Y, int W, int H, const char *L) : 
Fl_Gl_Window(X, Y, W, H, L),
m_bMouseOverIBar(FALSE),
m_bMouseOverBoundary(FALSE),
m_iSelected(-1)
{	
    mode( FL_RGB | FL_DEPTH | FL_DOUBLE | FL_ALPHA );

    SetCamera().SetSize( W, H );
    SetCamera().SetNearFar( 0.001, 1000 );

    m_dSpinLast = 0.0;
    m_dTiltLast = 0.0;
    m_dPanLast = 0.0;
    m_dDotSize = 1.0;

    m_bShowIBar = TRUE;
}

JofGTIBar::~JofGTIBar()
{
}

void JofGTIBar::draw()
{
    static float s_afCol[4] = {0,0,0,1};
    static float s_afColSelected[4] = {1,0,0,1};
    static float s_afAmbientLight[4] = {0.1,0.1,0.1,1};
    static float s_afDiffuseLight[4] = {0.9,0.9,0.9,1};
    static float s_afPosition[4] = {0,2,5,1};

    glClearColor( 0.0f, 0.0f, 0.0f, 1.0f );

    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    glEnable( GL_LIGHTING );
    glEnable( GL_DEPTH_TEST );
    glShadeModel( GL_SMOOTH  );
    glEnable( GL_CULL_FACE );

    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();

    m_IBar.SetOpenGLCamera();

    glEnable( GL_LIGHT0 );
    glLightfv( GL_LIGHT0, GL_AMBIENT, s_afAmbientLight );
    glLightfv( GL_LIGHT0, GL_DIFFUSE, s_afDiffuseLight );
    glLightfv( GL_LIGHT0, GL_POSITION, s_afPosition );
    glLightf( GL_LIGHT0, GL_SPOT_EXPONENT, 1 );
    glLightf( GL_LIGHT0, GL_SPOT_CUTOFF, 180.0f );

    for ( int i = 0; i < 9; i++ ) {
        const float fX = (i/3) * 2.0 - 2.0;
        const float fY = (i%3) * 2.0 - 2.0;

        s_afCol[0] = 0.2 + (i/3) * 0.3;
        s_afCol[1] = 0.2 + (i%3) * 0.3;
        glPushMatrix();
        glTranslatef( fX, fY, 0.0f);

        if ( m_iSelected == i ) {
            glMaterialfv( GL_FRONT_AND_BACK, GL_AMBIENT, s_afColSelected );
            glMaterialfv( GL_FRONT_AND_BACK, GL_DIFFUSE, s_afColSelected );
        } else {
            glMaterialfv( GL_FRONT_AND_BACK, GL_AMBIENT, s_afCol );
            glMaterialfv( GL_FRONT_AND_BACK, GL_DIFFUSE, s_afCol );
        }
        if ( i == 4 )
            glutSolidCube( 1.0 );
        else
            glutSolidSphere( 0.5, 20, 20 );
        glPopMatrix();
    }

    if ( m_bShowIBar && !Fl::get_key( FL_Alt_L ) )
        m_IBar.Draw( m_bMouseOverIBar );
}



int JofGTIBar::handle(int event) 
{
    const R2Pt pt = GetCamera().FlTkToCamera( Fl::event_x(), Fl::event_y() );

    static R2Pt    s_ptDown, s_ptLast;
	static bool s_bDoingCamera = FALSE;

    static int s_iCount = 0;

    static int s_iSeq = 0;

    bool bRedraw = FALSE;

    switch(event) {
    case FL_PUSH:
        //mouse down event ...
        //position in Fl::event_x() and Fl::event_y()
        if ( m_bShowIBar == TRUE && !Fl::get_key( FL_Alt_L ) ) {
            s_bDoingCamera = m_IBar.MouseDown( R2Pt_i( Fl::event_x(), Fl::event_y() ), Fl::event_state(FL_SHIFT) ? TRUE : FALSE );
        } else {
            s_bDoingCamera = FALSE;
        }

		if ( s_bDoingCamera == FALSE ) {
            m_iSelected = -1;
            double dDistBest = 1e30;
            for ( int i = 0; i < 9; i++ ) {
                const double dX = (i/3) * 2.0 - 2.0;
                const double dY = (i%3) * 2.0 - 2.0;
                const R2Pt ptSphereCenter = m_IBar.CameraPt( R4Matrix::Translation( R3Vec( dX, dY, 0.0 ) ) * R3Pt(0,0,0) );
                const double dDist = Length( ptSphereCenter - pt );
                if ( dDist < 0.1 && dDist < dDistBest ) {
                    m_iSelected = i;
                    dDistBest = dDist;
                }
            }
            if ( m_iSelected != -1 ) {
                m_IBar.SetFocusPoint( R3Pt( (m_iSelected/3) * 2.0 - 2.0, (m_iSelected%3) * 2.0 - 2.0, 0.0 ), R3Pt(1,1,1) );
            } else {
                m_IBar.SetFocusPoint( R3Pt(0,0,0), R3Pt(1,1,1) );
            }

            TRACE("Selected %d\n", m_iSelected );
            s_iCount = 0;
		}
        s_ptDown = pt;
        s_ptLast = pt;

        redraw();
        return 1;

    case FL_MOVE : {

        m_bMouseOverIBar = FALSE;
        if ( m_bShowIBar == TRUE && !Fl::get_key( FL_Alt_L ) ) {
            m_bMouseOverIBar = m_IBar.IsMouseOver( R2Pt_i( Fl::event_x(), Fl::event_y() ), bRedraw );
        }

        // Force 2 redraws - fluid doesn't always do so, otherwise
        if ( bRedraw == TRUE ) {
            s_iCount = 0;
        } else {
            s_iCount++;
        }
        if ( s_iCount < 2 ) {
            redraw();
        }

        m_ptLast = pt;
        return 1;
                   }

    case FL_DRAG: {
        //mouse moved while down event ...
        if ( s_bDoingCamera == TRUE ) {
            m_IBar.MouseMove( R2Pt_i( Fl::event_x(), Fl::event_y() ), Fl::event_state(FL_SHIFT) ? TRUE : FALSE );

        } 

        // Virtual trackball and pan
        if ( m_bShowIBar == FALSE && Fl::event_button1() ) {
            const double dScl = GetCamera().GetFocusDist() * tan( GetCamera().GetZoom() / 2.0 );
            const R2Vec vec = pt - s_ptLast;
            if ( Fl::event_state( FL_SHIFT ) ) {

                SetCamera().PanLeft( -vec[0] * dScl );
                SetCamera().PanUp( vec[1] * dScl );

            } else if ( Fl::event_state( FL_CTRL ) ) {

                if ( fabs(pt[1] - s_ptDown[1]) > fabs(pt[0] - s_ptDown[0]) ) {
                    SetCamera().PanIn( vec[1] * dScl );
                } else if ( fabs(pt[1] - s_ptDown[1]) < fabs(pt[0] - s_ptDown[0]) ) {
                    SetCamera().ZoomIn( vec[0] * M_PI );
                }
            } else {

                SetCamera().RotateVirtualTrackball( s_ptLast, pt, GetCamera().At(), 1.0 );
            }
        }

        s_ptLast = pt;
        m_ptLast = pt;
        redraw();
        return 1;
        }
    case FL_RELEASE:   
        if ( s_bDoingCamera == TRUE ) {
            m_IBar.MouseRelease( R2Pt_i( Fl::event_x(), Fl::event_y() ), Fl::event_state(FL_SHIFT) ? TRUE : FALSE );

        }

        redraw();

        return 1;
    case FL_FOCUS :
    case FL_UNFOCUS :
        // Return 1 if you want keyboard events, 0 otherwise
        return 1;
    case FL_KEYUP:
        SetCamera().HandleKeystroke( Fl::event_key(), 
                                  Fl::event_state(FL_SHIFT) ? TRUE : FALSE,
                                  Fl::event_state(FL_CTRL) ? TRUE : FALSE );
    
        redraw();
        return 1;

    default:
        // pass other events to the base class...
        return Fl_Gl_Window::handle(event);
    }

    redraw();
}

void JofGTIBar::resize( int x, int y, int w, int h )
{
    Fl_Gl_Window::resize( x, y, w, h );
    SetCamera().SetSize( w, h );

    redraw();
}

OGLObjsCamera &JofGTIBar::SetCamera() 
{ 
    return m_IBar; 
}

