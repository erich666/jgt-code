/**
 * @file    glui_main.cpp
 * @author  Thomas Lewiner <thomas.lewiner@polytechnique.org>
 * @author  Math Dept, PUC-Rio
 * @version 0.3
 * @date    30/05/2006
 *
 * @brief   MarchingCubes Graphical interface: main pipeline
 */
//________________________________________________


#if !defined(WIN32) || defined(__CYGWIN__)
#pragma implementation "glui_defs.h"
#endif // WIN32


#include "glui_defs.h"



//_____________________________________________________________________________
// declarations of this file

// main window id
int  main_window  = -1 ;

//-----------------------------------------------------------------------------

/// keyboard control
void keyboard(unsigned char key, int x, int y) ;


/// idle behaviour: redraw constantly
void idle() ;

//_____________________________________________________________________________



//_____________________________________________________________________________
// keyboard control
void keyboard(unsigned char key, int x, int y)
//-----------------------------------------------------------------------------
{
  if( key == 27 ) exit(0) ;
}
//_____________________________________________________________________________




//_____________________________________________________________________________
// idle
void idle()
//-----------------------------------------------------------------------------
{
  /* According to the GLUT specification, the current window is
     undefined during an idle callback.  So we need to explicitly change
     it if necessary */
  if( ::glutGetWindow() != main_window )
    ::glutSetWindow(main_window);

  /*  GLUI_Master.sync_live_all();  -- not needed - nothing to sync in this
                                       application  */

  ::glutPostRedisplay();
}
//_____________________________________________________________________________



//_____________________________________________________________________________
// main
int main(int argc, char* argv[])
//-----------------------------------------------------------------------------
{
  // create main window
  ::glutInit( &argc, argv ) ;
  ::glutInitDisplayMode( GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH );
  ::glutInitWindowPosition( 50, 50 );
  ::glutInitWindowSize( 800, 600 );

  main_window = glutCreateWindow( "Topological Marching Cubes" );


  //--------------------------------------------------//
  // Create the side and bottom subwindow
#ifdef WIN32
  create_bottom_panel() ;
  create_side_panel  () ;
#else  // WIN32
  create_side_panel  () ;
  create_bottom_panel() ;
#endif // WIN32

  //--------------------------------------------------//
  // set callback functions
  ::glutKeyboardFunc( keyboard ) ;
  ::glutDisplayFunc ( display );
  ::glutMotionFunc  ( motion );
  ::glutMouseFunc   ( mouse );
  GLUI_Master.set_glutIdleFunc   ( idle    );
  GLUI_Master.set_glutReshapeFunc( reshape );
  PRINT_GL_DEBUG ;

  //--------------------------------------------------//
  // init trackball
  init_trackballs() ;


  //--------------------------------------------------//
  // OpenGL inits

  ::glEnable (GL_LIGHTING);
  ::glDisable(GL_NORMALIZE);
  ::glEnable(GL_DEPTH_TEST);
  ::glClearColor( 1,1,1,1 );

  const GLfloat light0_ambient[4] =  {0.1f, 0.1f, 0.3f, 1.0f};
  ::glEnable(GL_LIGHT0);
  ::glLightfv(GL_LIGHT0, GL_AMBIENT, light0_ambient);
  ::glLightfv(GL_LIGHT0, GL_DIFFUSE, light0_diffuse);

  const GLfloat light1_ambient[4] =  {0.1f, 0.1f, 0.3f, 1.0f};
  ::glEnable(GL_LIGHT1);
  ::glLightfv(GL_LIGHT1, GL_AMBIENT, light1_ambient);
  ::glLightfv(GL_LIGHT1, GL_DIFFUSE, light1_diffuse);
  PRINT_GL_DEBUG ;

  v[2] = v[3] = -1 ; v[0] = v[1] = v[4] = v[5] = v[6] = v[7] = +1 ;

  //--------------------------------------------------//
  // Parse command line
  bool quit = parse_cmdline(argc, argv) ;

  //--------------------------------------------------//
  // GLUT main loop
  if( !quit ) ::glutMainLoop();

  //--------------------------------------------------//
  // cleaning, although there is no exit to glut main loop
/*
  mc.clean_all() ;
  delete csg_root ;
  fclose( isofile ) ;
*/

#if USE_GL_DISPLAY_LIST
  if(glIsList(gllist)==GL_TRUE)
    glDeleteLists( gllist, 1 ) ;
#endif // USE_GL_DISPLAY_LIST

  return 0 ;
}
//_____________________________________________________________________________
