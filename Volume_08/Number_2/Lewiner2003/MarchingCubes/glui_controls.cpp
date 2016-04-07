/**
 * @file    gluid_controls.cpp
 * @author  Thomas Lewiner <thomas.lewiner@polytechnique.org>
 * @author  Math Dept, PUC-Rio
 * @version 0.3
 * @date    30/05/2006
 *
 * @brief   MarchingCubes Graphical interface: interface controls
 */
//________________________________________________



#if !defined(WIN32) || defined(__CYGWIN__)
#pragma implementation "glui_defs.h"
#endif // WIN32


#include "glui_defs.h"
#include "csg.h"


//_____________________________________________________________________________
// declarations of this file

// main glui class: (right) side panel
GLUI *glui_side   = NULL ;

// bottom panel
GLUI *glui_bottom = NULL ;

// name of the import file
GLUI_EditText *in_filename  ;

// name of the export file
GLUI_EditText *out_filename ;

//-----------------------------------------------------------------------------

/// implicit formula control
GLUI_EditText *formula_ctrl ;

/// csg open button
GLUI_Button   *open_csg_ctrl ;

/// iso open button
GLUI_Button   *open_iso_ctrl ;

/// blue light intensity control
GLUI_EditText *light0_ctrl ;

/// orange light intensity control
GLUI_EditText *light1_ctrl ;

/// grid horizontal size control
GLUI_EditText *xres_ctrl ;
/// grid depth size control
GLUI_EditText *yres_ctrl ;
/// grid vertical size control
GLUI_EditText *zres_ctrl ;


//-----------------------------------------------------------------------------

// control events callback
void control_cb( int control ) ;

// create side panel
void create_side_panel() ;

// create bottom panel
void create_bottom_panel() ;

//_____________________________________________________________________________




//_____________________________________________________________________________
// control events callback
void control_cb( int control )
//-----------------------------------------------------------------------------
{
  int i ;
  FILE *fp     ;
  float val[3] ;

  ::glutSetWindow(main_window);

  switch( control )
  {
  // run marching cubes
  case RUN_ID   :
    run() ;
    break ;

  // set implicit function
  case FUN_ID   :
    if( curr_string > 0 ) strcpy( formula, fun_def[curr_string] ) ;
    formula_ctrl->set_text(formula) ;
    break ;

  // set trilinear implicit function
  case CASE_ID  :
    sprintf( formula, "(%f)*(1-x)*(1-y)*(1-z)+(%f)*(1+x)*(1-y)*(1-z)+(%f)*(1-x)*(1+y)*(1-z)+(%f)*(1+x)*(1+y)*(1-z)+(%f)*(1-x)*(1-y)*(1+z)+(%f)*(1+x)*(1-y)*(1+z)+(%f)*(1-x)*(1+y)*(1+z)+(%f)*(1+x)*(1+y)*(1+z)", v[0], v[1], v[3], v[2], v[4], v[5], v[7], v[6]) ;
    formula_ctrl->set_text(formula) ;
    break ;

  // open CSG definition file
  case CSG_ID   :
    if( csg_root )
    {
      delete csg_root ;
      csg_root = (CSG_Node*)NULL ;
      open_csg_ctrl->set_name("Open CSG") ;
    }
    else
    {
      if( strlen(in_filename->get_text()) <= 0 ) break ;
      FILE *fp = fopen( in_filename->get_text(), "r" ) ;
      if( !fp ) break ;
      csg_root = CSG_Node::parse(fp) ;
      fclose( fp ) ;
      open_csg_ctrl->set_name("Close CSG") ;
      formula_ctrl->set_text("c-0") ;
    }
    break ;

  // open iso grid file
  case ISO_ID   :
    if( isofile )
    {
      fclose( isofile ) ;
      isofile = (FILE*)NULL ;
      xres_ctrl->enable() ;
      yres_ctrl->enable() ;
      zres_ctrl->enable() ;
      open_iso_ctrl->set_name("Open ISO") ;
    }
    else
    {
      if( strlen(in_filename->get_text()) <= 0 ) break ;
      isofile = fopen( in_filename->get_text(), "rb" ) ;
      if( !isofile ) break ;
      unsigned char buf[sizeof(float)] ;

      fread (buf, sizeof(float), 1, isofile);
      size_x = * (int*)buf ;
      fread (buf, sizeof(float), 1, isofile);
      size_y = * (int*)buf ;
      fread (buf, sizeof(float), 1, isofile);
      size_z = * (int*)buf ;

      fread (buf, sizeof(float), 1, isofile);
      xmin = * (float*)buf ;
      fread (buf, sizeof(float), 1, isofile);
      xmax = * (float*)buf ;
      fread (buf, sizeof(float), 1, isofile);
      ymin = * (float*)buf ;
      fread (buf, sizeof(float), 1, isofile);
      ymax = * (float*)buf ;
      fread (buf, sizeof(float), 1, isofile);
      zmin = * (float*)buf ;
      fread (buf, sizeof(float), 1, isofile);
      zmax = * (float*)buf ;

      xres_ctrl->sync_live(0,1) ;
      yres_ctrl->sync_live(0,1) ;
      zres_ctrl->sync_live(0,1) ;

      xres_ctrl->disable() ;
      yres_ctrl->disable() ;
      zres_ctrl->disable() ;
      open_iso_ctrl->set_name("Close ISO") ;
      formula_ctrl->set_text("i-0") ;
    }
    break ;


  // flip surface normals
  case FLIP_ID   :
    for( i = 0 ; i < mc.nverts() ; ++i )
    {
      mc.vertices()[i].nx *= -1 ;
      mc.vertices()[i].ny *= -1 ;
      mc.vertices()[i].nz *= -1 ;
    }
#if USE_GL_DISPLAY_LIST
    draw() ;
#endif // USE_GL_DISPLAY_LIST
    break ;

  // open PLY file for display
  case IPLY_ID   :
    if( strlen(in_filename->get_text()) > 0 )
    {
      mc.clean_all() ;
      mc.readPLY( in_filename->get_text() ) ;
    }
    break ;

  // export PLY file
  case PLY_ID   :
    set_ext( "ply" ) ;
    mc.writePLY( out_filename->get_text() ) ;
    break ;

  // export VRML file
  case IV_ID    :
    set_ext( "iv\0" ) ;
    mc.writeIV( out_filename->get_text() ) ;
    break ;

  // export EPS file
  case EPS_ID    :
    export_eps() ;
    break ;

  // export PPM file
  case PPM_ID    :
    export_eps() ;
    break ;

  // save viewpoint
  case SAVE_VIEWPORT_ID   :
    fp = fopen( "viewport.txt", "w" ) ;
    if( !fp ) break ;
    fprintf( fp, "rotate:\n\t%f %f %f %f\n\t%f %f %f %f\n\t%f %f %f %f\n\t%f %f %f %f\n\n",
              view_rotate[ 0], view_rotate[ 1], view_rotate[ 2], view_rotate[ 3],
              view_rotate[ 4], view_rotate[ 5], view_rotate[ 6], view_rotate[ 7],
              view_rotate[ 8], view_rotate[ 9], view_rotate[10], view_rotate[11],
              view_rotate[12], view_rotate[13], view_rotate[14], view_rotate[15] ) ;
    fprintf( fp, "translate:\t%f %f %f\n", obj_pos[0], obj_pos[1], obj_pos[2] ) ;
    fclose( fp ) ;
    break ;

  // load viewpoint
  case LOAD_VIEWPORT_ID   :
    fp = fopen( "viewport.txt", "r" ) ;
    if( !fp ) break ;
    fscanf( fp, "rotate: %f %f %f %f  %f %f %f %f  %f %f %f %f  %f %f %f %f ",
              view_rotate +  0, view_rotate +  1, view_rotate +  2, view_rotate +  3,
              view_rotate +  4, view_rotate +  5, view_rotate +  6, view_rotate +  7,
              view_rotate +  8, view_rotate +  9, view_rotate + 10, view_rotate + 11,
              view_rotate + 12, view_rotate + 13, view_rotate + 14, view_rotate + 15 ) ;
    fscanf( fp, "translate: %f %f %f ", obj_pos + 0, obj_pos + 1, obj_pos + 2 ) ;
    fclose( fp ) ;

    objects_rot->sync_live(0,1) ;
    objects_mv ->sync_live(0,1) ;
    objects_zm ->sync_live(0,1) ;
    mouse_rot  . sync_live(0,1) ;
    mouse_mv   . sync_live(0,1) ;
    mouse_zm   . sync_live(0,1) ;
    break ;

  // reset rotation
  case RESET_ROTATION_ID    :
    view_rotate[ 0] = view_rotate[ 5] = view_rotate[10] = view_rotate[15] = 1.0f ;
    view_rotate[ 1] = view_rotate[ 2] = view_rotate[ 3] = view_rotate[ 4] = 0.0f ;
    view_rotate[ 6] = view_rotate[ 7] = view_rotate[ 8] = view_rotate[ 9] = 0.0f ;
    view_rotate[11] = view_rotate[12] = view_rotate[13] = view_rotate[14] = 0.0f ;
    break ;
  // reset translation
  case RESET_TRANSLATION_ID :  obj_pos[0] = obj_pos[1] = 0.0f ;  break ;
  // reset zoom
  case RESET_ZOOM_ID        :  obj_pos[2] = 0.0f ;  break ;


  // enable light 0
  case LIGHT0_ENABLED_ID :
    if ( light0_enabled ) { glEnable( GL_LIGHT0 );  light0_ctrl->enable(); }
    else { glDisable( GL_LIGHT0 );  light0_ctrl->disable(); }
    break ;

  // enable light 1
  case LIGHT1_ENABLED_ID :
    if ( light1_enabled ) { glEnable( GL_LIGHT1 );  light1_ctrl->enable(); }
    else { glDisable( GL_LIGHT1 );  light1_ctrl->disable(); }
    break ;

  // light 0 instensity
  case LIGHT0_INTENSITY_ID :
    val[0] = light0_diffuse[0] * light0_intensity / 100 ;
    val[1] = light0_diffuse[1] * light0_intensity / 100 ;
    val[2] = light0_diffuse[2] * light0_intensity / 100 ;
    ::glLightfv(GL_LIGHT0, GL_DIFFUSE, val );
    break ;

  // light 1 instensity
  case LIGHT1_INTENSITY_ID :
    val[0] = light1_diffuse[0] * light1_intensity / 100 ;
    val[1] = light1_diffuse[1] * light1_intensity / 100 ;
    val[2] = light1_diffuse[2] * light1_intensity / 100 ;
    ::glLightfv(GL_LIGHT1, GL_DIFFUSE, val );
    break ;

  // orthographic/perspective projection
  case PROJ_ID            :  reshape   (0,0) ;  break ;

  // quit
  case EXIT_ID            :  exit(0) ;

  default :  break ;
  }

  ::glutPostRedisplay();
}
//_____________________________________________________________________________




//_____________________________________________________________________________
//_____________________________________________________________________________



//_____________________________________________________________________________
// create side panel
void create_side_panel()
//-----------------------------------------------------------------------------
{
  GLUI_Rollout     *roll, *roll2 ;
  GLUI_Panel       *pan  ;
  GLUI_Listbox     *list ;
  GLUI_EditText    *text ;

  glui_side   = GLUI_Master.create_glui_subwindow( main_window, GLUI_SUBWINDOW_RIGHT  );

  //--------------------------------------------------//
  // Input
  roll = glui_side->add_rollout( "Input", true );

  // Input : functions
  list = glui_side->add_listbox_to_panel( roll, "Implicit Functions:", &curr_string, FUN_ID, control_cb );
  for( int i=0; i<NFUNS; i++ ) list->add_item( i, fun_list[i] );

  // Input : trilinear function
  roll2 = glui_side->add_rollout_to_panel(roll, "MC Case", false );
  text = glui_side->add_edittext_to_panel( roll2, "v0", GLUI_EDITTEXT_FLOAT, v+0 ) ; text->set_w(4) ;
  text = glui_side->add_edittext_to_panel( roll2, "v1", GLUI_EDITTEXT_FLOAT, v+1 ) ; text->set_w(4) ;
  text = glui_side->add_edittext_to_panel( roll2, "v2", GLUI_EDITTEXT_FLOAT, v+2 ) ; text->set_w(4) ;
  text = glui_side->add_edittext_to_panel( roll2, "v3", GLUI_EDITTEXT_FLOAT, v+3 ) ; text->set_w(4) ;
  text = glui_side->add_edittext_to_panel( roll2, "v4", GLUI_EDITTEXT_FLOAT, v+4 ) ; text->set_w(4) ;
  text = glui_side->add_edittext_to_panel( roll2, "v5", GLUI_EDITTEXT_FLOAT, v+5 ) ; text->set_w(4) ;
  text = glui_side->add_edittext_to_panel( roll2, "v6", GLUI_EDITTEXT_FLOAT, v+6 ) ; text->set_w(4) ;
  text = glui_side->add_edittext_to_panel( roll2, "v7", GLUI_EDITTEXT_FLOAT, v+7 ) ; text->set_w(4) ;
  glui_side->add_button_to_panel( roll2, "Set Formula", CASE_ID, control_cb ) ;
/*
  glui_side->add_statictext_to_panel( roll, "" );
  text = glui_side->add_edittext_to_panel( roll, "v4", GLUI_EDITTEXT_FLOAT, &v4, CASE_ID, control_cb ) ; text->set_w(4) ;
  glui_side->add_statictext_to_panel( roll, "" );
  glui_side->add_statictext_to_panel( roll, "" );
  text = glui_side->add_edittext_to_panel( roll, "v0", GLUI_EDITTEXT_FLOAT, &v0, CASE_ID, control_cb ) ; text->set_w(4) ;
  glui_side->add_column_to_panel( roll, false ) ;
  text = glui_side->add_edittext_to_panel( roll, "v7", GLUI_EDITTEXT_FLOAT, &v7, CASE_ID, control_cb ) ; text->set_w(4) ;
  glui_side->add_statictext_to_panel( roll, "" );
  glui_side->add_statictext_to_panel( roll, "" );
  text = glui_side->add_edittext_to_panel( roll, "v3", GLUI_EDITTEXT_FLOAT, &v3, CASE_ID, control_cb ) ; text->set_w(4) ;
  glui_side->add_statictext_to_panel( roll, "" );
  glui_side->add_column_to_panel( roll, false ) ;
  glui_side->add_statictext_to_panel( roll, "" );
  text = glui_side->add_edittext_to_panel( roll, "v5", GLUI_EDITTEXT_FLOAT, &v5, CASE_ID, control_cb ) ; text->set_w(4) ;
  glui_side->add_statictext_to_panel( roll, "" );
  glui_side->add_statictext_to_panel( roll, "" );
  text = glui_side->add_edittext_to_panel( roll, "v1", GLUI_EDITTEXT_FLOAT, &v1, CASE_ID, control_cb ) ; text->set_w(4) ;
  glui_side->add_column_to_panel( roll, false ) ;
  text = glui_side->add_edittext_to_panel( roll, "v6", GLUI_EDITTEXT_FLOAT, &v6, CASE_ID, control_cb ) ; text->set_w(4) ;
  glui_side->add_statictext_to_panel( roll, "" );
  glui_side->add_statictext_to_panel( roll, "" );
  text = glui_side->add_edittext_to_panel( roll, "v2", GLUI_EDITTEXT_FLOAT, &v2, CASE_ID, control_cb ) ; text->set_w(4) ;
  glui_side->add_statictext_to_panel( roll, "" );
*/

  // Input : file
  in_filename = new GLUI_EditText( roll, "File name" ) ;
  in_filename->set_w(200) ;
  open_csg_ctrl = glui_side->add_button_to_panel( roll, "Open CSG", CSG_ID , control_cb ) ;
  open_iso_ctrl = glui_side->add_button_to_panel( roll, "Open ISO", ISO_ID , control_cb ) ;
  glui_side->add_button_to_panel( roll, "Open PLY", IPLY_ID, control_cb ) ;

  //--------------------------------------------------//
  // Resolution
  roll = glui_side->add_rollout( "Resolution", true );
  xres_ctrl = glui_side->add_edittext_to_panel( roll, "X", GLUI_EDITTEXT_INT, &size_x ) ;
  yres_ctrl = glui_side->add_edittext_to_panel( roll, "Y", GLUI_EDITTEXT_INT, &size_y ) ;
  zres_ctrl = glui_side->add_edittext_to_panel( roll, "Z", GLUI_EDITTEXT_INT, &size_z ) ;

  //--------------------------------------------------//
  // Run
  roll = glui_side->add_rollout( "Run", true );
  formula_ctrl = glui_side->add_edittext_to_panel( roll, "Formula", GLUI_EDITTEXT_TEXT, formula ) ;
  formula_ctrl->set_w(200) ;
  glui_side->add_statictext_to_panel( roll, " x,y,z: coordinates in [-1..1]" ) ;
  glui_side->add_statictext_to_panel( roll, " c: csg evalutation at x,y,z" ) ;
  glui_side->add_statictext_to_panel( roll, " i: isovalue of the ISO file" ) ;
  pan  = glui_side->add_panel_to_panel( roll, "", GLUI_PANEL_NONE ) ;
  glui_side->add_checkbox_to_panel( pan, "Original MC", &originalMC );
  glui_side->add_column_to_panel( pan, false ) ;
  glui_side->add_button_to_panel( pan, "Run MC", RUN_ID, control_cb ) ;

  //--------------------------------------------------//
  // Saving
  roll = glui_side->add_rollout( "Save mesh", true );
  out_filename = glui_side->add_edittext_to_panel( roll, "File name", GLUI_EDITTEXT_TEXT ) ;
  out_filename->set_w(200) ;
  glui_side->add_checkbox_to_panel( roll, "Export ISO at next MC run", &export_iso );
  pan  = glui_side->add_panel_to_panel( roll, "", GLUI_PANEL_NONE ) ;
  glui_side->add_button_to_panel( pan, "Save PLY", PLY_ID, control_cb ) ;
  glui_side->add_column_to_panel( pan, false ) ;
  glui_side->add_button_to_panel( pan, "Save VRML/IV", IV_ID, control_cb ) ;
  pan  = glui_side->add_panel_to_panel( roll, "", GLUI_PANEL_NONE ) ;
  glui_side->add_button_to_panel( pan, "Save EPS", EPS_ID, control_cb ) ;
  glui_side->add_column_to_panel( pan, false ) ;
  glui_side->add_button_to_panel( pan, "Save PPM", PPM_ID, control_cb ) ;
  glui_side->add_statictext( "" );


  //--------------------------------------------------//
  // quit
  glui_side->add_button( "Quit", EXIT_ID, control_cb );
}
//_____________________________________________________________________________





//_____________________________________________________________________________
// create bottom panel
void create_bottom_panel()
//-----------------------------------------------------------------------------
{
  GLUI_Rotation    *rot  ;

  glui_bottom = GLUI_Master.create_glui_subwindow( main_window, GLUI_SUBWINDOW_BOTTOM );

  //--------------------------------------------------//
  // position
  objects_rot = glui_bottom->add_rotation( "Objects", view_rotate );
  objects_rot->set_spin( 1.0f );
  glui_bottom->add_button( "Reset", RESET_ROTATION_ID, control_cb ) ;
  glui_bottom->add_column( false );

  objects_mv = glui_bottom->add_translation( "Objects XY", GLUI_TRANSLATION_XY, obj_pos );
  objects_mv->set_speed( .005f );
  glui_bottom->add_button( "Reset", RESET_TRANSLATION_ID, control_cb ) ;
  glui_bottom->add_column( false );

  objects_zm = glui_bottom->add_translation( "Objects Z", GLUI_TRANSLATION_Z, &obj_pos[2] );
  objects_zm->set_speed( .005f );
  glui_bottom->add_button( "Reset", RESET_ZOOM_ID, control_cb ) ;

  //--------------------------------------------------//
  // Blue Light
  glui_bottom->add_column( true );
  rot = glui_bottom->add_rotation( "Blue Light", light0_rotation );
  rot->set_spin( .82f );
  glui_bottom->add_checkbox( "On", &light0_enabled, LIGHT0_ENABLED_ID, control_cb );
  light0_ctrl = glui_bottom->add_edittext( "I (%)", GLUI_EDITTEXT_INT, &light0_intensity, LIGHT0_INTENSITY_ID, control_cb );
  light0_ctrl->set_int_limits( 0, 100 );
  light0_ctrl->set_w(4) ;

  // Orange Light
  glui_bottom->add_column( true );
  rot = glui_bottom->add_rotation( "Orange Light", light1_rotation );
  rot->set_spin( .82f );
  glui_bottom->add_checkbox( "On", &light1_enabled, LIGHT1_ENABLED_ID, control_cb );
  light1_ctrl = glui_bottom->add_edittext( "I (%)", GLUI_EDITTEXT_INT, &light1_intensity, LIGHT0_INTENSITY_ID, control_cb );
  light1_ctrl->set_int_limits( 0, 100 );
  light1_ctrl->set_w(4) ;


  //--------------------------------------------------//
  // display element
  glui_bottom->add_column( true );
  glui_bottom->add_checkbox( "Ortho", &ortho, PROJ_ID, control_cb );
  glui_bottom->add_checkbox( "Cube", &show_cube, -1, control_cb );
  glui_bottom->add_checkbox( "Grid", &show_grid, -1, control_cb );

  glui_bottom->add_column( true );
  glui_bottom->add_checkbox( "Fill", &fill, -1, control_cb );
  glui_bottom->add_checkbox( "Wireframe", &wireframe, -1, control_cb );
  glui_bottom->add_button( "Flip Normals", FLIP_ID, control_cb ) ;
}
//_____________________________________________________________________________
