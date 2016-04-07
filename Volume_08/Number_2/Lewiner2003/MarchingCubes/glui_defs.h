/**
 * @file    glui_defs.h
 * @author  Thomas Lewiner <thomas.lewiner@polytechnique.org>
 * @author  Math Dept, PUC-Rio
 * @version 0.3
 * @date    30/05/2006
 *
 * @brief   MarchingCubes Graphical interface
 */
//________________________________________________


#ifndef _MC_GLUI_DEFS_H_
#define _MC_GLUI_DEFS_H_

#if !defined(WIN32) || defined(__CYGWIN__)
#pragma interface
#endif // WIN32


#include <GL/glui.h> // openGL user interface
#include <stdio.h>   // i/o functions
#include "MarchingCubes.h"


#ifdef _DEBUG
#define PRINT_GL_DEBUG  { if( ::glGetError() != GL_NO_ERROR ) printf( "openGL watch at line %d: %s\n", __LINE__, ::gluErrorString( ::glGetError() ) ) ; }
#else  // _DEBUG
#define PRINT_GL_DEBUG  {}
#endif // _DEBUG


/// setting for disaply lists
#define USE_GL_DISPLAY_LIST 0




//_____________________________________________________________________________
// Types

// forward declaration
class CSG_Node ;

//_____________________________________________________________________________



//_____________________________________________________________________________
// Marching Cubes component

  /// main marching cubes object
  extern MarchingCubes mc ;

  /// isovalue defining the isosurface
  extern float isoval ;

  /// original/topological MC switch
  extern int   originalMC ;

  /// grid left extension
  extern float xmin ;
  /// grid right extension
  extern float xmax ;
  /// grid near extension
  extern float ymin ;
  /// grid far extension
  extern float ymax ;
  /// grid bottom extension
  extern float zmin ;
  /// grid up extension
  extern float zmax ;

  /// grid horizontal size control
  extern int size_x ;
  /// grid depth size control
  extern int size_y ;
  /// grid vertical size control
  extern int size_z ;


  //-----------------------------------------------------------------------------
// input data

  /// implicit formula
  extern char  formula[1024] ;

  /// number of example implicit functions
  #define NFUNS 15
  /// implicit functions
  extern char *fun_list[NFUNS] ;
  /// implicit functions
  extern char *fun_def [NFUNS] ;
  /// chosen implicit function
  extern int   curr_string ;

  /// cube data
  extern float v[8] ;

  /// loaded iso grid
  extern FILE        *isofile  ;

  /// loaded CSG tree
  extern CSG_Node    *csg_root ;


//-----------------------------------------------------------------------------
// main functions

/// run the MC algorithm
bool run() ;

/// Command Line
bool parse_cmdline( int argc, char* argv[] ) ;


//-----------------------------------------------------------------------------
// I/O functions

  /// switch to export iso grid
  extern int  export_iso ;

/// set file extension of out_filename
int  set_ext( const char ext[3] ) ;

/// EPS export
void export_eps() ;

/// PPM export
void export_ppm() ;

/// TGA export
void export_tga() ;

//_____________________________________________________________________________



//_____________________________________________________________________________
// Interface components

//-----------------------------------------------------------------------------
// GLUI class

  /// main window id
  extern int  main_window ;

  /// main glui class: (right) side panel
  extern GLUI *glui_side   ;

  /// bottom panel
  extern GLUI *glui_bottom ;


/// create side panel
void create_side_panel() ;

/// create bottom panel
void create_bottom_panel() ;

/// control events callback
void control_cb( int control ) ;


//-----------------------------------------------------------------------------
// Lights

  /// enable blue light
  extern int   light0_enabled   ;

  /// enable orange light
  extern int   light1_enabled   ;

  /// blue light diffuse color
  extern float light0_diffuse[4] ;

  /// orange light diffuse color
  extern float light1_diffuse[4] ;

  /// blue light position
  extern float light0_rotation[16] ;

  /// orange light position
  extern float light1_rotation[16] ;

  /// blue light intensity
  extern int   light0_intensity ;

  /// orange light intensity
  extern int   light1_intensity ;

//-----------------------------------------------------------------------------
// mouse and object movements

  /// motion type (-1 -> no motion, 0 -> rotate, 1 -> zoom, 2 -> translate)
  extern int motion_type ;

  /// window trackball
  extern GLUI_Rotation     mouse_rot   ;
  /// panel trackball
  extern GLUI_Rotation    *objects_rot ;
  /// window translation
  extern GLUI_Translation  mouse_mv    ;
  /// panel translation
  extern GLUI_Translation *objects_mv  ;
  /// window zoom
  extern GLUI_Translation  mouse_zm    ;
  /// panel zoom
  extern GLUI_Translation *objects_zm  ;

  /// number of calls for updating the GLUI control
  extern int ncalls ;



/// init mouse and window controls
void init_trackballs() ;

/// mouse events tracking
void mouse(int button, int button_state, int x, int y ) ;

/// mouse motion tracking
void motion(int x, int y ) ;


//-----------------------------------------------------------------------------
// i/o filenames

  /// name of the import file
  extern GLUI_EditText *in_filename  ;

  /// name of the export file
  extern GLUI_EditText *out_filename ;

//-----------------------------------------------------------------------------
// drawing parameters

  /// display element switch: wireframed surface
  extern int   wireframe  ;
  /// display element switch: continuous surface
  extern int   fill       ;
  /// display element switch: bounding cube
  extern int   show_cube  ;
  /// display element switch: grid lines
  extern int   show_grid  ;

  /// orthographic / perspective projection switch
  extern int ortho ;

  /// object rotation
  extern float view_rotate[16] ;
  /// object translation
  extern float obj_pos    [3 ] ;

/// window resizing
void reshape( int x, int y ) ;

/// draw the mesh and eventually regenerate the display list
void draw() ;

/// main drawing function
void display() ;

//_____________________________________________________________________________




//_____________________________________________________________________________
/// Callback ids
enum
{
  LIGHT0_ENABLED_ID   ,
  LIGHT1_ENABLED_ID   ,
  LIGHT0_INTENSITY_ID ,
  LIGHT1_INTENSITY_ID ,

  SAVE_VIEWPORT_ID    ,
  LOAD_VIEWPORT_ID    ,

  FUN_ID              ,
  CASE_ID             ,
  CSG_ID              ,
  RUN_ID              ,

  ISO_ID              ,
  IPLY_ID             ,
  PLY_ID              ,
  IV_ID               ,
  EPS_ID              ,
  PPM_ID              ,

  RESET_ROTATION_ID   ,
  RESET_TRANSLATION_ID,
  RESET_ZOOM_ID       ,

  FLIP_ID             ,
  PROJ_ID             ,

  EXIT_ID             ,
};
//_____________________________________________________________________________




#endif // _MC_GLUI_DEFS_H_
