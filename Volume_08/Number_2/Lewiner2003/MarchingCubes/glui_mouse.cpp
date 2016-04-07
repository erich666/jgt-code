/**
 * @file    glui_mouse.cpp
 * @author  Thomas Lewiner <thomas.lewiner@polytechnique.org>
 * @author  Math Dept, PUC-Rio
 * @version 0.3
 * @date    30/05/2006
 *
 * @brief   MarchingCubes Graphical interface: mouse controls
 */
//________________________________________________


#if !defined(WIN32) || defined(__CYGWIN__)
#pragma implementation "glui_defs.h"
#endif // WIN32



#include "glui_defs.h"

//_____________________________________________________________________________
// declarations of this file

// motion type (-1 -> no motion, 0 -> rotate, 1 -> zoom, 2 -> translate)
int motion_type = -1 ;

// panel and window trackballs and sliders
GLUI_Rotation     mouse_rot, *objects_rot ;
GLUI_Translation  mouse_mv , *objects_mv  ;
GLUI_Translation  mouse_zm , *objects_zm  ;

// number of calls for updating the GLUI control
int ncalls = 0 ;

//-----------------------------------------------------------------------------

// init mouse and window controls
void init_trackballs() ;

// mouse events tracking
void mouse(int button, int button_state, int x, int y ) ;

// mouse motion tracking
void motion(int x, int y ) ;

//_____________________________________________________________________________







//_____________________________________________________________________________
// init mouse and window controls
void init_trackballs()
//-----------------------------------------------------------------------------
{
  // init trackball

  int tx,ty,tw,th ;
  GLUI_Master.get_viewport_area( &tx, &ty, &tw, &th );

  mouse_rot.set_spin(0.05f) ;
  mouse_rot.set_w( tw ) ;
  mouse_rot.set_h( th ) ;
  mouse_rot.set_ptr_val( view_rotate );
  mouse_rot.init_live() ;

  mouse_mv.set_speed(0.005f) ;
  mouse_mv.set_w( tw ) ;
  mouse_mv.set_h( th ) ;
  mouse_mv.set_ptr_val( obj_pos );
  mouse_mv.init_live() ;

  mouse_mv.trans_type = GLUI_TRANSLATION_XY;
  mouse_mv.float_array_size = 2 ;
  mouse_mv.hidden = true ;

  mouse_zm.set_speed(0.01f) ;
  mouse_zm.set_w( tw ) ;
  mouse_zm.set_h( th ) ;
  mouse_zm.set_ptr_val( &obj_pos[2] );
  mouse_zm.init_live() ;

  mouse_zm.trans_type = GLUI_TRANSLATION_Z ;
  mouse_zm.float_array_size = 1 ;
  mouse_zm.hidden = true ;
}
//_____________________________________________________________________________



//_____________________________________________________________________________
// mouse events
void mouse(int button, int button_state, int x, int y )
//-----------------------------------------------------------------------------
{
  // determine motion type
  if     ( glutGetModifiers() & GLUT_ACTIVE_CTRL  ) motion_type = 1 ;
  else if( glutGetModifiers() & GLUT_ACTIVE_SHIFT ) motion_type = 2 ;
  else                                              motion_type = 0 ;

  switch( motion_type )
  {
  // rotation
  case 0 :
    if ( button == GLUT_LEFT_BUTTON && button_state == GLUT_DOWN )
    {
      mouse_rot.init_live() ;
      mouse_rot.mouse_down_handler(x,y) ;
    }
    if ( button_state != GLUT_DOWN )
    {
      mouse_rot.mouse_up_handler(x,y,1) ;
      motion_type = -1 ;
    }
    objects_rot->sync_live(0,1) ;
    break ;

  // zoom
  case 1 :
    if ( button == GLUT_LEFT_BUTTON && button_state == GLUT_DOWN )
    {
      mouse_zm.init_live() ;
      mouse_zm.glui = glui_side ;
      mouse_zm.mouse_down_handler(x,y) ;
      mouse_zm.glui = NULL ;
    }
    if ( button_state != GLUT_DOWN )
    {
      mouse_zm.glui = glui_side ;
      mouse_zm.mouse_up_handler(x,y,1) ;
      mouse_zm.glui = NULL ;
      motion_type = -1 ;
    }
    objects_zm->sync_live(0,1) ;
    break ;

  // translation
  case 2 :
    if ( button == GLUT_LEFT_BUTTON && button_state == GLUT_DOWN )
    {
      mouse_mv.init_live() ;
      mouse_mv.glui = glui_side ;
      mouse_mv.mouse_down_handler(x,y) ;
      mouse_mv.glui = NULL ;
    }
    if ( button_state != GLUT_DOWN )
    {
      mouse_mv.glui = glui_side ;
      mouse_mv.mouse_up_handler(x,y,1) ;
      mouse_mv.glui = NULL ;
      motion_type = -1 ;
    }
    objects_mv->sync_live(0,1) ;
    break ;

  // no movement
  default :
    break ;
  }
  ncalls = 0 ;
}
//_____________________________________________________________________________


//_____________________________________________________________________________
// motion
void motion(int x, int y )
//-----------------------------------------------------------------------------
{
  switch( motion_type )
  {
  // rotation
  case 0 :
    mouse_rot.glui = glui_side ;
    mouse_rot.iaction_mouse_held_down_handler(x,y,1);
    mouse_rot.glui = NULL ;
    if( ++ncalls > 10 ) { objects_rot->sync_live(0,1) ;  ncalls = 0 ; }
    break ;

  // zoom
  case 1 :
    mouse_zm.glui = glui_side ;
    mouse_zm.iaction_mouse_held_down_handler(x,y,1);
    mouse_zm.glui = NULL ;
    if( ++ncalls > 10 ) { objects_zm->sync_live(0,1) ;  ncalls = 0 ; }
    break ;

  // translation
  case 2 :
    mouse_mv.glui = glui_side ;
    mouse_mv.iaction_mouse_held_down_handler(x,y,1);
    mouse_mv.glui = NULL ;
    if( ++ncalls > 10 ) { objects_mv->sync_live(0,1) ;  ncalls = 0 ; }
    break ;

  // no movement
  default :
    break ;
  }
  ::glutPostRedisplay() ;
}
//_____________________________________________________________________________
