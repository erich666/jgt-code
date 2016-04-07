/**
 * @file    glui_draws.cpp
 * @author  Thomas Lewiner <thomas.lewiner@polytechnique.org>
 * @author  Math Dept, PUC-Rio
 * @version 0.3
 * @date    30/05/2006
 *
 * @brief   MarchingCubes Graphical interface: drawing commands
 */
//________________________________________________


#if !defined(WIN32) || defined(__CYGWIN__)
#pragma implementation "glui_defs.h"
#endif // WIN32


#include "glui_defs.h"


//_____________________________________________________________________________
// declarations of this file

// display element switches
int   wireframe  = 0 ;
int   fill       = 1 ;
int   show_cube  = 1 ;
int   show_grid  = 0 ;

// orthographic / perspective projection
int   ortho =   0   ;

// object transformation
float view_rotate[16] = { 1.0f,0.0f,0.0f,0.0f, 0.0f,1.0f,0.0f,0.0f, 0.0f,0.0f,1.0f,0.0f, 0.0f,0.0f,0.0f,1.0f };
float obj_pos    [3 ] = { 0.0f, 0.0f, 0.0f };

/// viewer position for drawing the grid on the right side of the cube
float viewer[3] = { 0.0f, 0.0f, 0.0f };

// lights
int   light0_enabled        =   1 ;
int   light1_enabled        =   1 ;
int   light0_intensity      = 100 ;
int   light1_intensity      =  40 ;
GLfloat light0_diffuse [ 4] = {.6f, .6f, 1.0f, 1.0f};
GLfloat light1_diffuse [ 4] = {.9f, .6f, 0.0f, 1.0f};
GLfloat light0_position[ 4] = {.5f, .5f, 1.0f, 0.0f};
GLfloat light1_position[ 4] = {-1.0f, -1.0f, 1.0f, 0.0f};
GLfloat light0_rotation[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 };
GLfloat light1_rotation[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 };

/// openGL display list
#if USE_GL_DISPLAY_LIST
int gllist = -1 ;
#endif // USE_GL_DISPLAY_LIST

//-----------------------------------------------------------------------------

/// draw the grid lines on one side of the cube
void draw_cube_grid(const float p0[3], const float p1[3], const float p2[3], const float p3[3], int g1,int g2) ;

/// draw the grid lines on the cube
void draw_grid() ;

/// draw the cube wireframe
void draw_cube() ;

//-----------------------------------------------------------------------------

// window resizing
void reshape( int x, int y ) ;

// draw the mesh and eventually regenerate the display list
void draw() ;

// main drawing function
void display() ;

//_____________________________________________________________________________






//_____________________________________________________________________________
// draw the grid lines on one side of the cube
void draw_cube_grid(const float p0[3], const float p1[3], const float p2[3], const float p3[3], int g1,int g2)
//-----------------------------------------------------------------------------
{
  int i ;
  float dq[3], dr[3], q[3], r[3] ;

  q[0] = p1[0] ;  q[1] = p1[1] ;  q[2] = p1[2] ;
  dq[0] = (p0[0]-p1[0]) / g1;
  dq[1] = (p0[1]-p1[1]) / g1;
  dq[2] = (p0[2]-p1[2]) / g1;

  r[0] = p3[0] ;  r[1] = p3[1] ;  r[2] = p3[2] ;
  dr[0] = (p2[0]-p3[0]) / g1;
  dr[1] = (p2[1]-p3[1]) / g1;
  dr[2] = (p2[2]-p3[2]) / g1;
  for (i = 0 ; i < g1-1 ; i++)
  {
    q[0] += dq[0] ;  q[1] += dq[1] ;  q[2] += dq[2] ;
    ::glVertex3fv(q);
    r[0] += dr[0] ;  r[1] += dr[1] ;  r[2] += dr[2] ;
    ::glVertex3fv(r);
  }

  q[0] = p2[0] ;  q[1] = p2[1] ;  q[2] = p2[2] ;
  dq[0] = (p0[0]-p2[0]) / g2;
  dq[1] = (p0[1]-p2[1]) / g2;
  dq[2] = (p0[2]-p2[2]) / g2;

  r[0] = p3[0] ;  r[1] = p3[1] ;  r[2] = p3[2] ;
  dr[0] = (p1[0]-p3[0]) / g2;
  dr[1] = (p1[1]-p3[1]) / g2;
  dr[2] = (p1[2]-p3[2]) / g2;
  for (i = 0 ; i < g2-1 ; i++)
  {
    q[0] += dq[0] ;  q[1] += dq[1] ;  q[2] += dq[2] ;
    ::glVertex3fv(q);
    r[0] += dr[0] ;  r[1] += dr[1] ;  r[2] += dr[2] ;
    ::glVertex3fv(r);
  }
}
//_____________________________________________________________________________





//_____________________________________________________________________________
// draw the grid lines on the cube
void draw_grid()
//-----------------------------------------------------------------------------
{
  float p0[3], p1[3], p2[3], p3[3] ;

  ::glColor3f(0.3f, 0.3f, 0.3f);
  ::glBegin(GL_LINES);
  {
    p0[0] = p1[0] = p2[0] = p3[0] = (viewer[0] < 0) ? 1 : -1 ;
    p0[1] = p1[1] = -1 ;  p2[1] = p3[1] = 1 ;
    p0[2] = p2[2] = -1 ;  p1[2] = p3[2] = 1 ;
    draw_cube_grid(p0,p1,p2,p3, size_z-1,size_y-1);

    p0[1] = p1[1] = p2[1] = p3[1] = (viewer[1] < 0) ? 1 : -1 ;
    p0[0] = p1[0] = -1 ;  p2[0] = p3[0] = 1 ;
    p0[2] = p2[2] = -1 ;  p1[2] = p3[2] = 1 ;
    draw_cube_grid(p0,p1,p2,p3, size_z-1,size_x-1);

    p0[2] = p1[2] = p2[2] = p3[2] = (viewer[2] < 0) ? 1 : -1 ;
    p0[0] = p1[0] = -1 ;  p2[0] = p3[0] = 1 ;
    p0[1] = p2[1] = -1 ;  p1[1] = p3[1] = 1 ;
    draw_cube_grid(p0,p1,p2,p3, size_y-1,size_x-1);
  }
  ::glEnd();
}
//_____________________________________________________________________________



//_____________________________________________________________________________
// draw the cube wireframe
void draw_cube()
//-----------------------------------------------------------------------------
{
  float col ;

  ::glBegin(GL_LINES);
  {
    col = (viewer[1]>0 && viewer[2]>0)? 0.4f : 1 ;
    ::glColor3f(col,0,0);
    ::glVertex3f( -1 ,-1,-1); // e0
    ::glVertex3f(1.1f,-1,-1); // e1

    col = (viewer[1]<0 && viewer[2]>0)? 0.4f : 1 ;
    ::glColor3f(col,0,0);
    ::glVertex3f(-1,1,-1); // e2
    ::glVertex3f( 1,1,-1); // e3

    col = (viewer[1]<0 && viewer[2]<0)? 0.4f : 1 ;
    ::glColor3f(col,0,0);
    ::glVertex3f(-1,1,1); // e6
    ::glVertex3f( 1,1,1); // e7

    col = (viewer[1]>0 && viewer[2]<0)? 0.4f : 1 ;
    ::glColor3f(col,0,0);
    ::glVertex3f(-1,-1,1); // e4
    ::glVertex3f( 1,-1,1); // e5

/*---------------------------------------------------------------*/

    col = (viewer[0]>0 && viewer[2]>0)? 0.4f : 1 ;
    ::glColor3f(0,col,0);
    ::glVertex3f(-1, -1 ,-1); // e0
    ::glVertex3f(-1,1.1f,-1); // e2

    col = (viewer[0]<0 && viewer[2]>0)? 0.4f : 1 ;
    ::glColor3f(0,col,0);
    ::glVertex3f(1,-1,-1); // e1
    ::glVertex3f(1, 1,-1); // e3

    col = (viewer[0]<0 && viewer[2]<0)? 0.4f : 1 ;
    ::glColor3f(0,col,0);
    ::glVertex3f(1,-1,1); // e5
    ::glVertex3f(1, 1,1); // e7

    col = (viewer[0]>0 && viewer[2]<0)? 0.4f : 1 ;
    ::glColor3f(0,col,0);
    ::glVertex3f(-1,-1,1); // e4
    ::glVertex3f(-1, 1,1); // e6

/*---------------------------------------------------------------*/

    col = (viewer[0]>0 && viewer[1]>0)? 0.4f : 1 ;
    ::glColor3f(0,0,col);
    ::glVertex3f(-1,-1, -1 ); // e0
    ::glVertex3f(-1,-1,1.1f); // e4

    col = (viewer[0]<0 && viewer[1]>0)? 0.4f : 1 ;
    ::glColor3f(0,0,col);
    ::glVertex3f(1,-1,-1); // e1
    ::glVertex3f(1,-1, 1); // e5

    col = (viewer[0]<0 && viewer[1]<0)? 0.4f : 1 ;
    ::glColor3f(0,0,col);
    ::glVertex3f(1,1,-1); // e3
    ::glVertex3f(1,1, 1); // e7

    col = (viewer[0]>0 && viewer[1]<0)? 0.4f : 1 ;
    ::glColor3f(0,0,col);
    ::glVertex3f(-1,1,-1); // e2
    ::glVertex3f(-1,1, 1); // e6
  }
  ::glEnd();
}
//_____________________________________________________________________________



//_____________________________________________________________________________
// draw the mesh and eventually regenerate the display list
void draw()
//-----------------------------------------------------------------------------
{
#if USE_GL_DISPLAY_LIST
  if (::glIsList(gllist) == GL_TRUE )
    ::glDeleteLists( gllist, 1 ) ;

  gllist = ::glGenLists(1) ;
  if( ::glIsList(gllist) != GL_TRUE )
  {
    printf( "glGenLists(1) error: %d\n%s\n!", gllist, ::gluErrorString(glGetError()) ) ;
    return;
  }

  ::glNewList(gllist,GL_COMPILE);
  {
#endif // USE_GL_DISPLAY_LIST
    ::glBegin( GL_TRIANGLES ) ;
    {
      for( int i = 0 ; i < mc.ntrigs() ; ++i )
      {
        const Triangle *t = mc.trig(i) ;
        const Vertex   *v ;
        v = mc.vert(t->v1) ;
        ::glNormal3f( v->nx, v->ny, v->nz ) ;
        ::glVertex3f( v->x , v->y , v->z  ) ;
        v = mc.vert(t->v2) ;
        ::glNormal3f( v->nx, v->ny, v->nz ) ;
        ::glVertex3f( v->x , v->y , v->z  ) ;
        v = mc.vert(t->v3) ;
        ::glNormal3f( v->nx, v->ny, v->nz ) ;
        ::glVertex3f( v->x , v->y , v->z  ) ;
      }
    }
    ::glEnd() ;
#if USE_GL_DISPLAY_LIST
  }
  ::glEndList();
#endif // USE_GL_DISPLAY_LIST
}
//_____________________________________________________________________________



//_____________________________________________________________________________
//_____________________________________________________________________________



//_____________________________________________________________________________
// window resizing
void reshape( int x, int y )
//-----------------------------------------------------------------------------
{
  ::glutSetWindow(main_window);

  // get viewport
  int tx, ty, tw, th;
  GLUI_Master.get_viewport_area( &tx, &ty, &tw, &th );
  ::glViewport( tx, ty, tw, th );

  // sets the window trackball
  mouse_rot.set_w( tw ) ;
  mouse_rot.set_h( th ) ;
  mouse_mv .set_w( tw ) ;
  mouse_mv .set_h( th ) ;
  mouse_zm .set_w( tw ) ;
  mouse_zm .set_h( th ) ;

  // sets the projection matrix
  ::glMatrixMode(GL_PROJECTION);
  ::glLoadIdentity();

  // sets the viewport
  if( ortho )
  {
    if( th > tw )
      ::glOrtho( -1.3, 1.3, -1.3*(double)th/tw, 1.3*(double)th/tw, 0.5, 10.0 ) ;
    else
      ::glOrtho( -1.3*(double)tw/th, 1.3*(double)tw/th, -1.3, 1.3, 0.5, 10.0 ) ;
  }
  else
    ::gluPerspective( 45.0, th>0?(double)tw/th:1.0, 0.5, 10.0 );

  ::gluLookAt( 0.0f,0.0f,3.0f, 0.0f,0.0f,0.0f, 0.0f,1.0f,0.0f ) ;

  // switch to the modelview matrix
  ::glMatrixMode( GL_MODELVIEW );
  ::glLoadIdentity();

  // redisplay
  ::glutPostRedisplay();
}
//_____________________________________________________________________________




//_____________________________________________________________________________
// main drawing function
void display()
//-----------------------------------------------------------------------------
{
  ::glutSetWindow(main_window);

  // clear screen
  ::glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
  ::glEnable (GL_LIGHTING);

  // light 0
  ::glLoadIdentity();
  ::glMultMatrixf( light0_rotation );
  ::glLightfv(GL_LIGHT0, GL_POSITION, light0_position);

  // light 1
  ::glLoadIdentity();
  ::glMultMatrixf( light1_rotation );
  ::glLightfv(GL_LIGHT1, GL_POSITION, light1_position);

  // transformation matrix
  ::glLoadIdentity();
  ::glTranslatef( obj_pos[0], obj_pos[1], -obj_pos[2] );
  if( ortho )
    ::glScalef( 1.0+obj_pos[2], 1.0+obj_pos[2], 1.0+obj_pos[2] ) ;
  ::glMultMatrixf( view_rotate );

  if( show_cube || show_grid )
  {
    float Modelview[16];
    ::glGetFloatv(GL_MODELVIEW_MATRIX ,(float *) Modelview );
    viewer[0] = Modelview[0*4+2] ;
    viewer[1] = Modelview[1*4+2] ;
    viewer[2] = Modelview[2*4+2] ;

    ::glDisable(GL_LIGHTING);
    if( show_cube ) draw_cube() ;
    if( show_grid ) draw_grid() ;
    ::glEnable (GL_LIGHTING);
  }

  if( fill )
  {
#if USE_GL_DISPLAY_LIST
    if( ::glIsList(gllist) == GL_TRUE )
      ::glCallList(gllist);
#else  // USE_GL_DISPLAY_LIST
    draw() ;
#endif // USE_GL_DISPLAY_LIST
  }

  if( wireframe )
  {
    ::glPolygonOffset( 0.5, -0.1f );
    ::glLineWidth(1.5) ;
    ::glPolygonMode( GL_FRONT_AND_BACK, GL_LINE ) ;

    ::glDisable(GL_LIGHTING);
    ::glColor3f(0.8f,0.8f,0.8f) ;
#if USE_GL_DISPLAY_LIST
    if( ::glIsList(gllist) == GL_TRUE )
      ::glCallList(gllist);
#else  // USE_GL_DISPLAY_LIST
    draw() ;
#endif // USE_GL_DISPLAY_LIST

    ::glLineWidth(1.0) ;
    ::glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
  }

  // next frame
  ::glutSwapBuffers();
}
//_____________________________________________________________________________
