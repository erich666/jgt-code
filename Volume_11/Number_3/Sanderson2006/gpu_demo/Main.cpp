/*****************************************************************************/
/*								             */
/*	Copyright (c) 2005	Allen R. Sanderson		             */
/*								             */
/*				Scientific Computing and Imaging Institute   */
/*				University of Utah		             */
/*				Salt Lake City, Utah		             */
/*								             */
/*            							             */
/*  Permission is granted to modify and/or distribute this program so long   */
/*  as the program is distributed free of charge and this header is retained */
/*  as part of the program.                                                  */
/*								             */
/*****************************************************************************/

#include <math.h>
#include <time.h>

#include <vector>
#include <iostream>

#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/glui.h>

#include <Cg/cgGL.h>
#include "ReactDiffuse.h"

using namespace std;

unsigned int main_win = 0;       // index of the main rendering window

enum { TURING=0, GRAY_SCOTT=1, BRUSSELATOR=2 };
enum { INITIALIZE, RUN, START, PAUSE, RESTART, UPDATE, SAVE };
unsigned int run_status = INITIALIZE;

// glui variables
GLUI *glui_win;   // glui window

GLUI_Spinner *time_step_spinner,
  *time_mult_spinner,
  *theta_spinner,
  *a_spinner, *b_spinner, 
  *ab_variance_spinner,
  *react_const0_spinner,
  *react_const1_spinner,
  *react_const2_spinner,
  *react_const_variance_spinner,
  *render_passes_spinner,
  *num_iterations_spinner,
  *mixing_rate_spinner,
  *react_rate_spinner,
  *rr_coef_1_spinner, *rr_coef_2_spinner,
  *diff_coef_1_spinner, *diff_coef_2_spinner,
  *a_diff_spinner, *b_diff_spinner, *c_diff_spinner, *d_diff_spinner,
  *cell_size_spinner,
  *mult_spinner;

GLUI_Checkbox *invert_diff_vals_checkbox;

GLUI_RadioGroup *display_color_radiogroup,
  *reaction_radiogroup, *solution_radiogroup, *boundary_radiogroup,
  *laplacian_radiogroup,
  *reaction_rate_radiogroup, *diffusion_rate_radiogroup;

GLUI_Panel *setup_panel,
  *reaction_panel, *display_panel, *solution_panel, *laplacian_panel,
  *mixing_panel, *iteration_panel, *control_panel,
  *param_panel,
  *var_panel, *react_const_panel,
  *reaction_rate_panel, *diffusion_rate_panel, *df_panel, *button_panel;

GLUI_EditText *iteration_edittext;

GLUI_Listbox *files_listbox;

// Reaction Diffusion local variables
int display_color = 0, invert_diff_vals = 1;
float time_step, time_mult;
float cn_theta;
float a, b, ab_variance;
float react_const0, react_const1, react_const2, react_const_variance;
float rr_coef_1, rr_coef_2;
unsigned int render_passes = 50;
int num_iterations = 50000;
float mixing_rate;
float diff_coef_1, diff_coef_2, a_diff, b_diff, c_diff, d_diff;
int reaction = TURING, solution = 0, boundary = 2;
int laplacian = 0;
int mixing = 0, save = 0;
int rr_var = 1, aniso = 1, gradient = 1, mult = 1;
float cell_size = 1.0;
int   iteration = 0;

// For dispalying the running time
clock_t start_time, stop_time;
bool time_flag = 0;

// CG stuff
CGcontext   cgContext;       // frag program context
CGprogram   cgTexProgram;    // pbuffer rendering shader
CGprofile   cgProfile;       // frag program profile
CGparameter cgTexParam;      // frag program texture variable
CGparameter cgMinParam;      // frag program texture variable
CGparameter cgScaleParam;    // frag program texture variable
CGparameter cgReactionParam; // frag program texture variable
CGparameter cgColormapParam; // frag program texture variable

GLint  glCurrentDrawbuffer = 0;
GLuint glColormapTexID = 0;
GLuint glDisplayList = 0;

GLenum errorCode;

// Local stuff
ReactDiffuse *rdGPU = new ReactDiffuse;

#define MAX_COLORS 28
float texColormap[MAX_COLORS*4] = {
  0,   0,   255, 0,
  0,   52,  255, 0,
  1,   80,  255, 0,
  3,   105, 255, 0,
  5,   132, 255, 0,
  9,   157, 243, 0,
  11,  177, 213, 0,
  15,  193, 182, 0,
  21,  210, 152, 0,
  30,  225, 126, 0,
  42,  237, 102, 0,
  60,  248, 82,  0,
  87,  255, 62,  0,
  116, 255, 49,  0,
  148, 252, 37,  0,
  178, 243, 27,  0,
  201, 233, 19,  0,
  220, 220, 14,  0,
  236, 206, 10,  0,
  247, 185, 8,   0,
  253, 171, 5,   0,
  255, 151, 3,   0,
  255, 130, 2,   0,
  255, 112, 1,   0,
  255, 94,  0,   0,
  255, 76,  0,   0,
  255, 55,  0,   0,
  255, 0,   0,   0 };


/******************************************************************************
reshape
******************************************************************************/
void reshape( int w, int h )
{
  glutReshapeWindow( w, h );

  glViewport( 0, 0, w, h );
  
  // 2d rendering mode
  glMatrixMode( GL_PROJECTION );
  glLoadIdentity();
  
  gluOrtho2D( 0, (GLfloat)w, 0, (GLfloat)h );
}


/******************************************************************************
updateMixing
******************************************************************************/
void updateMixing( int )
{
  rdGPU->setMixingRate( mixing * mixing_rate );
}


/******************************************************************************
saveImage
******************************************************************************/
void saveImage()
{
  // get the texture values
  unsigned int w = rdGPU->getTextureWidth(), h = rdGPU->getTextureHeight();

  rdGPU->getTexValues();

  // Get the min and max for normalizing the image.
  float min_value = 1.0e12, max_value = -1.0e12;

  float rc0_2 = w / 2;
  float rc1_2 = h / 2;

  for ( unsigned int j = 0; j < h; j++ ) 
    for ( unsigned int i = 0; i < w; i++ ) {
      float val = rdGPU->_texMorphigens[i*4 + display_color + j*w*4];

      if ( min_value > val ) min_value = val;
      if ( max_value < val ) max_value = val;
    }

  if (min_value == max_value) {
    min_value = max_value - 1;
    max_value = min_value + 2;
  }

  fprintf( stdout, "%f %f \n", min_value, max_value );

  float scale;

  if( MAP_TOBY_PUFFER )
    scale = 254.0 / (max_value-min_value);
  else
    scale = 255.0 / (max_value-min_value);

  unsigned char *values = new unsigned char[w*h];;

  float db;

  if( SPOTTED_PUFFER )
    db = 0.20; // White Blue Spotted Puffer
  else if( MAP_TOBY_PUFFER )
    db = 0.24; // Map Toby Fuffer
  else if( PAPUA_TOBY_PUFFER )
    db = 0.15; // Papua Toby Fuffer
  else
    db = 0.0;

  // Normalize the image.
  for ( unsigned int j = 0; j < h; j++ ) {
    for ( unsigned int i = 0; i < w; i++ ){ 
      float di = ( (float) i-rc0_2) / rc0_2;
      float dj = ( (float) j-rc1_2) / rc1_2;
      float dr = sqrt( di*di + dj*dj);

      if( (SPOTTED_PUFFER || MAP_TOBY_PUFFER) && dr < db )
	values[i + j*w] = 255;
      else if( PAPUA_TOBY_PUFFER && dr < db )
	values[i + j*w] = 0;

      else
	values[i + j*w] = (unsigned char)
	  ((rdGPU->_texMorphigens[i*4 + display_color + j*w*4] - min_value) * scale);
    }
  }

  // Write the values
  char iterstr[12];

  if( save )
    sprintf( iterstr, "%06d", iteration/render_passes );
  else
    sprintf( iterstr, "" );

  std::string filename = "demo.raw";
  
  cout << filename.c_str() << " width " << w << "  height " << h << "  ";

  FILE *pFile = fopen( filename.c_str(), "w" );
  fwrite( values, sizeof( unsigned char ), h*w, pFile );
  fclose (pFile);

  delete[] values;

  cout << "DONE" << endl;
}


/******************************************************************************
Display
******************************************************************************/
void display()
{
  // get the texture values
  unsigned int w = rdGPU->getTextureWidth(), h = rdGPU->getTextureHeight();

  // Get the min and max for normalizing the image.
  float min_value[4], max_value[4];

  rdGPU->getTexValues();

  for ( unsigned int i=0; i<4; i++ ) {
    min_value[i] =  1.0e12;
    max_value[i] = -1.0e12;
  }

  for ( unsigned int j=0; j<h; j++ ) {
    for ( unsigned int i=0; i<w; i++ ) {
      for ( unsigned int c=0; c<4; c++ ) {
	float val = rdGPU->_texMorphigens[i*4 + c + j*w*4];

	if ( min_value[c] > val ) min_value[c] = val;
	if ( max_value[c] < val ) max_value[c] = val;
      }
    }
  }

  float scale[4];

  for ( unsigned int i=0; i<4; i++ ) {
    if (min_value[i] == max_value[i]) {
      min_value[i] = max_value[i] - 1;
      max_value[i] = min_value[i] + 2;
    }

    scale[i] = 1.0 / (max_value[i]-min_value[i]);
  }

  glClear( GL_COLOR_BUFFER_BIT );
  glDrawBuffer( glCurrentDrawbuffer );

 // Save the current Draw buffer
  // we want to render the framebuffer to the screen
  cgGLEnableProfile( cgProfile );
  cgGLBindProgram( cgTexProgram );

  // bind the framebuffer as a texture and render a quad
  glEnable(GL_TEXTURE_RECTANGLE_NV);
  glBindTexture( GL_TEXTURE_RECTANGLE_NV, rdGPU->getTextureID() );

  cgGLSetParameter4f( cgMinParam,
		      min_value[0], min_value[1], min_value[2], min_value[3] );

  cgGLSetParameter4f( cgScaleParam,
		      scale[0], scale[1], scale[2], scale[3] );

  cgGLSetParameter1f( cgReactionParam, display_color );

  cgGLSetTextureParameter( cgColormapParam, glColormapTexID );
  cgGLEnableTextureParameter( cgColormapParam );

  cgGLSetTextureParameter( cgTexParam, rdGPU->getTextureID() );  
  cgGLEnableTextureParameter( cgTexParam );
     
  glCallList( glDisplayList ); 

  cgGLDisableTextureParameter( cgTexParam );
  cgGLDisableTextureParameter( cgColormapParam );

  glBindTexture(GL_TEXTURE_RECTANGLE_NV, 0);
  glDisable(GL_TEXTURE_RECTANGLE_NV);

  cgGLDisableProfile( cgProfile );
  
  glutSwapBuffers();
}


/******************************************************************************
Load
******************************************************************************/
void load_cb( int ID )
{
  // set the reaction diffusion variables to those from our gui
  rdGPU->setSolution( solution );
  rdGPU->setBoundary( boundary );
  rdGPU->setLaplacian( laplacian );
  rdGPU->setTheta( cn_theta );
  rdGPU->setReaction( reaction );

  rdGPU->setTimeStep( time_step );
  rdGPU->setTimeMult( time_mult );

  rdGPU->setABVariance( ab_variance );
  rdGPU->setReactionConstVariance( react_const_variance );
  rdGPU->setA( a );
  rdGPU->setB( b );
  rdGPU->setReaction_Const0( react_const0 );
  rdGPU->setReaction_Const1( react_const1 );
  rdGPU->setReaction_Const2( react_const2 );
  
  rdGPU->setMixingRate( mixing * mixing_rate );

  rdGPU->setRRCoef1( rr_coef_1 );
  rdGPU->setRRCoef2( rr_coef_2 );

  rdGPU->setDiffCoef1( diff_coef_1 );
  rdGPU->setDiffCoef2( diff_coef_2 );

  if( invert_diff_vals ) {
    rdGPU->setADiffRate( 1.0 / a_diff / (cell_size*cell_size) );
    rdGPU->setBDiffRate( 1.0 / b_diff / (cell_size*cell_size) );
    rdGPU->setCDiffRate( 1.0 / c_diff / (cell_size*cell_size) );
    rdGPU->setDDiffRate( 1.0 / d_diff / (cell_size*cell_size) );
  } else {
    rdGPU->setADiffRate( a_diff / (cell_size*cell_size) );
    rdGPU->setBDiffRate( b_diff / (cell_size*cell_size) );
    rdGPU->setCDiffRate( c_diff / (cell_size*cell_size) );
    rdGPU->setDDiffRate( d_diff / (cell_size*cell_size) );
  }
  rdGPU->setGradient( gradient );

  rdGPU->setMult( mult );

  if( ID == RESTART ) {

    rdGPU->deleteGrids();
    rdGPU->createVectorData();
    rdGPU->initFbuffer();
    rdGPU->generateConstantsTex();
    rdGPU->setInitalState();

    // reset the counter
    iteration = 0;
    iteration_edittext->set_int_val( iteration );
    
    mult = rdGPU->getMult();
    mult_spinner->set_int_val( mult );

    if ( glIsList( glDisplayList ) == GL_FALSE ) {
      // Create a short display list.
      glDisplayList = glGenLists( 1 );

      if( glDisplayList == 0 ) {
	if ((errorCode = glGetError()) != GL_NO_ERROR) 
	  fprintf( stderr, "RESTART - glGenLists(): ERROR: %s\n",
		   gluErrorString(errorCode) );
      }
    }

    unsigned int b, t, l, r;
    rdGPU->getTextureCoords( b, t, l, r );

    glNewList( glDisplayList, GL_COMPILE );

    glBegin( GL_QUADS );
    glTexCoord2f( l, b ); glVertex2f( l, b );
    glTexCoord2f( r, b ); glVertex2f( r, b );
    glTexCoord2f( r, t ); glVertex2f( r, t );
    glTexCoord2f( l, t ); glVertex2f( l, t );
    glEnd();

    glEndList();

    if( run_status == RUN )
      run_status = START;

  } else if( ID == UPDATE ) {

    rdGPU->createVectorData();

    rdGPU->generateConstantsTex();
  }

  // make sure that our window is the right size
  reshape( rdGPU->getTextureHeight(), rdGPU->getTextureWidth() );

  glutPostRedisplay();
}


/******************************************************************************
Idle loop where things get down
******************************************************************************/
void idle()
{ 
  if ( run_status == INITIALIZE ) {
//    load_cb( RESTART );

    run_status = PAUSE;

  } else if ( run_status == PAUSE ) {
    if( time_flag == 1 ) {

      time_flag = 0;

      stop_time = clock();

      if( stop_time < start_time )
	stop_time += (24 * 3600) * CLOCKS_PER_SEC;

      cerr << "Total time "
	   << (double) (stop_time - start_time) / (double) CLOCKS_PER_SEC
	   << " seconds" << endl;
    }

  } else if ( run_status == START ) {

    // make sure that stuff goes to the glut window, not the 
    // glui window
    glutSetWindow( main_win );
      
    if( solution && iteration == 0 ) {

      updateMixing( 0 );

      rdGPU->setSolution( 0 );
      rdGPU->updateStateExplicit( 1 );
      rdGPU->setSolution( solution );
      
      glutPostRedisplay();
    }

    iteration_edittext->set_int_val( iteration );

    run_status = RUN;

    if( time_flag == 0 ) {

      time_flag = 1;

      start_time = clock();
    }

  } else if ( run_status == RUN ) {

    // make sure that stuff goes to the glut window, not the 
    // glui window
    glutSetWindow( main_win );

    updateMixing( 0 );

    // check to see if the maximum number of iterations has been reached.
    if ( iteration < num_iterations )	{
      iteration += render_passes;
      if( solution )
	rdGPU->updateStateImplicit( render_passes );
      else
	rdGPU->updateStateExplicit( render_passes );
	
      glutPostRedisplay();

      if( save )
	saveImage();

      iteration_edittext->set_int_val( iteration );

    } else {
      run_status = PAUSE;
    }
  }
}


/******************************************************************************
Callbacks
******************************************************************************/
void button_cb( int ID )
{
  switch ( ID ){
  case START:
    run_status = START;
    break;
  case PAUSE:
    run_status = PAUSE;
    break;
  case SAVE:
    saveImage();
    break;
  }
}


void solution_cb( int ID )
{
  if( reaction != TURING && solution != 0 ) {
    solution_radiogroup->set_int_val( 0 );

    solution_cb( 0 );
  }

  if( solution )
    num_iterations = 2000;
  else
    num_iterations = 50000;

  num_iterations_spinner->set_int_val( num_iterations );

  rdGPU->setSolution( solution );
}


void boundary_cb( int ID )
{
  rdGPU->setBoundary( boundary );
}


void reaction_const_cb( int ID )
{
  if( reaction == TURING ) {
    a = react_const1 - react_const0;
    b = react_const1 / a;

  } else if( reaction == BRUSSELATOR ) {
    a = react_const0;
    b = react_const1 / react_const0;

  }

  a_spinner->set_float_val( a );
  b_spinner->set_float_val( b );
}


void reaction_rate_cb( int ID )
{
 if( reaction == TURING ) {
    if( rr_var == 0 ) {
      rr_coef_1 = 128;
      rr_coef_2 = 0;
    } else if( rr_var == 1 ) {
      rr_coef_1 = 100;
      rr_coef_2 = 250;
    }
  } else if( reaction == GRAY_SCOTT ) {
    if( rr_var == 0 ) {
      rr_coef_1 = 3.0;
      rr_coef_2 = 0;
    } else if( rr_var == 1 ) {
      rr_coef_1 =  4.5;
      rr_coef_2 = -4.0;
    }
  } else if( reaction == BRUSSELATOR ) {
    if( rr_var == 0 ) {
      rr_coef_1 = 1.0;
      rr_coef_2 = 0;
    } else if( rr_var == 1 ) {
      rr_coef_1 =  1.25;
      rr_coef_2 = -1.00;

      // Sweetlips with brusselator.
//      rr_coef_1 =  2.00;
//      rr_coef_2 = -1.50;
    }
  }

  rr_coef_1_spinner->set_float_val( rr_coef_1 );
  rr_coef_2_spinner->set_float_val( rr_coef_2 );
}


void diffusion_cb( int ID )
{
  if( aniso == 0 ) {
    diff_coef_1 = 1.0;
    diff_coef_2 = 1.0;
  } else {
    diff_coef_1 = 1.5;
    diff_coef_2 = 0.5;
  } 

  diff_coef_1_spinner->set_float_val( diff_coef_1 );
  diff_coef_2_spinner->set_float_val( diff_coef_2 );
}


void reaction_cb( int ID )
{
  reaction_rate_cb( ID );

  if( reaction == TURING ) {
    a_diff = 4.0;
    b_diff = 16.0;

    c_diff = 1.0;
    d_diff = 4.0;

    invert_diff_vals = 1;
  } else if( reaction == GRAY_SCOTT ) {
    a_diff = 25000;
    b_diff = 12500;

    c_diff = 25000;
    d_diff = 12500;
    invert_diff_vals = 1;
  } else if( reaction == BRUSSELATOR ) {
    a_diff = 16.7;
    b_diff = 36.4;

    c_diff = 49.5;
    d_diff = 117.6;

    // Sweetlips with brusselator.
//     a_diff = 12.6;
//     b_diff = 27.5;
    
//     c_diff = 47.5;
//     d_diff = 141.5;
    
    invert_diff_vals = 0;
  }

  a_diff_spinner->set_float_val( a_diff );
  b_diff_spinner->set_float_val( b_diff );
  c_diff_spinner->set_float_val( c_diff );
  d_diff_spinner->set_float_val( d_diff );
  invert_diff_vals_checkbox->set_int_val( invert_diff_vals );


  if( reaction == TURING ) {
    react_const0 = 16.0;
    react_const1 = 12.0;
    react_const2 =  0.0;
    react_const_variance = 0.001;
  } else if( reaction == GRAY_SCOTT ) {
    react_const0 = 0.0925;
    react_const1 = 0.0300;
    react_const2 = 0.0000;
    react_const_variance = 0.01;

  } else if( reaction == BRUSSELATOR ) {
    react_const0 = 3.0;
    react_const1 = 9.0;
    react_const2 = 0.0;
    react_const_variance = 0.00;
  }

  react_const0_spinner->set_float_val( react_const0 );
  react_const1_spinner->set_float_val( react_const1 );
  react_const2_spinner->set_float_val( react_const2 );
  react_const_variance_spinner->set_float_val( react_const_variance );

  if( reaction == TURING ) {
    a = 4.0;
    b = 4.0;
    ab_variance = 0.0;
  } else if( reaction == GRAY_SCOTT ) {
    a = 0.25;
    b = 0.50;
    ab_variance = 0.0;
  } else if( reaction == BRUSSELATOR ) {
    a = react_const0;
    b = react_const1 / react_const0;
    ab_variance = 0.01;
  }

  a_spinner->set_float_val( a );
  b_spinner->set_float_val( b );
  ab_variance_spinner->set_float_val( ab_variance );

  if( reaction == GRAY_SCOTT )
    cell_size = 0.01;
  else
    cell_size = 1.0;

  cell_size_spinner->set_float_val( cell_size );

  if( reaction == TURING ) {
    time_step = 0.50;
  } else if( reaction == GRAY_SCOTT ) {
    time_step = 0.1;
  } else if( reaction == BRUSSELATOR ) {
    time_step = 0.0015;
  }

  time_step_spinner->set_float_val( time_step );

  if( reaction == TURING ) {
    display_color = 0;
  } else if( reaction == GRAY_SCOTT ) {
    display_color = 1;
  } else if( reaction == BRUSSELATOR ) {
    display_color = 0;
  }

  display_color_radiogroup->set_int_val( display_color );

  if( reaction != TURING ) {
    solution_radiogroup->set_int_val( 0 );
    solution_cb( 0 );
  }
}


/******************************************************************************
Main
******************************************************************************/
int main( int argc, char* argv[] )
{
  unsigned int x = 48, y = 48;

  // Need to have glut initialized even if no window.
  glutInit( &argc, argv );

  glutInitWindowPosition( x, y );
  glutInitWindowSize( 128, 128 );
  glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE );

  // create the window and store its index
  main_win = glutCreateWindow( "Reaction Diffusion" );

  // register the callback functions
  glutDisplayFunc( display );   // called in response to display events
  glutReshapeFunc( reshape );	  // called in response to window resize

  // create the glui window
  glui_win = GLUI_Master.create_glui( "RD Controls", 0, 500, 500 );

  GLUI_Master.set_glutIdleFunc( idle );

  // Add the glui controls....

  // Set up  Panel
  setup_panel = glui_win->add_panel( "" );

  // Reaction Type
  reaction_panel = glui_win->add_panel_to_panel( setup_panel, "" );

  glui_win->add_statictext_to_panel( reaction_panel, "Reaction Type:" );

  reaction_radiogroup =
    glui_win->add_radiogroup_to_panel( reaction_panel, &reaction, 0,
				       reaction_cb);

  glui_win->add_radiobutton_to_group( reaction_radiogroup, "Turing                 ");
  glui_win->add_radiobutton_to_group( reaction_radiogroup, "Gray-Scott             ");
  glui_win->add_radiobutton_to_group( reaction_radiogroup, "Brusselator            ");

  // Display Control
  display_panel = reaction_panel;
  glui_win->add_column_to_panel( display_panel, false );

  glui_win->add_statictext_to_panel( reaction_panel, "Display:" );

  display_color_radiogroup =
    glui_win->add_radiogroup_to_panel( display_panel, &display_color);

  glui_win->add_radiobutton_to_group( display_color_radiogroup, "Show a");
  glui_win->add_radiobutton_to_group( display_color_radiogroup, "Show b");
  glui_win->add_radiobutton_to_group( display_color_radiogroup, "Show c");
  glui_win->add_radiobutton_to_group( display_color_radiogroup, "Show d");

  // Explicit/Implicit Solution
  solution_panel =
    glui_win->add_panel_to_panel( setup_panel, "" );

  glui_win->add_statictext_to_panel( solution_panel, "Solution Type:" );
  solution_radiogroup =
    glui_win->add_radiogroup_to_panel( solution_panel, &solution, 0,
				       solution_cb);

  glui_win->add_radiobutton_to_group( solution_radiogroup, "Explicit");
  glui_win->add_radiobutton_to_group( solution_radiogroup, "Semi-Implicit");
  glui_win->add_radiobutton_to_group( solution_radiogroup, "Theta-Implicit");

  glui_win->add_column_to_panel( solution_panel, false );


  glui_win->add_statictext_to_panel( solution_panel, "Boundary Conditions:" );
  boundary_radiogroup =
    glui_win->add_radiogroup_to_panel( solution_panel, &boundary, 0,
				       boundary_cb);

  glui_win->add_radiobutton_to_group( boundary_radiogroup, "Zero Flux Clamp");
  glui_win->add_radiobutton_to_group( boundary_radiogroup, "Zero Flux Var.");
  glui_win->add_radiobutton_to_group( boundary_radiogroup, "Periodic");


  glui_win->add_column_to_panel( solution_panel, false );

  time_step = rdGPU->getTimeStep();
  time_step_spinner =
    glui_win->add_spinner_to_panel( solution_panel, "time step:",
				    GLUI_SPINNER_FLOAT, &time_step );

  time_mult = rdGPU->getTimeMult();
  time_mult_spinner =
    glui_win->add_spinner_to_panel( solution_panel, "time multiplier:",
				    GLUI_SPINNER_FLOAT, &time_mult );

  cn_theta = rdGPU->getTheta();
  theta_spinner =
    glui_win->add_spinner_to_panel( solution_panel, "theta:",
				    GLUI_SPINNER_FLOAT, &cn_theta );

  // Laplacian
  laplacian_panel =
    glui_win->add_panel_to_panel( setup_panel, "Laplacian Type (Turing ONLY)" );

  laplacian_radiogroup =
    glui_win->add_radiogroup_to_panel( laplacian_panel, &laplacian );

  glui_win->add_radiobutton_to_group( laplacian_radiogroup, "Inhomogeneous");
  glui_win->add_radiobutton_to_group( laplacian_radiogroup, "Uniform");


  // Mixing
  mixing_panel = glui_win->add_panel_to_panel( setup_panel, "Mixing" );
  mixing = 0;
  glui_win->add_checkbox_to_panel( mixing_panel, "Use mixing", 
				   &mixing, 0, updateMixing );
  
  glui_win->add_column_to_panel( mixing_panel, false );

  mixing_rate = rdGPU->getMixingRate();
  mixing_rate_spinner =
    glui_win->add_spinner_to_panel(mixing_panel, "rate: ", 
				   GLUI_SPINNER_FLOAT, &mixing_rate,
				   1, updateMixing );

  // Iteration Control
  iteration_panel = glui_win->add_panel_to_panel( setup_panel, "Iterations" );
  num_iterations_spinner =
    glui_win->add_spinner_to_panel( iteration_panel, "Maximum:", 
				    GLUI_SPINNER_INT, 
				    &num_iterations );
  iteration_edittext =
    glui_win->add_edittext_to_panel( iteration_panel, "Current:", 
				     GLUI_EDITTEXT_INT, 
				     &iteration );
  glui_win->add_column_to_panel( iteration_panel, false );
  render_passes_spinner =
    glui_win->add_spinner_to_panel( iteration_panel, "render passes:", 
				    GLUI_SPINNER_INT, 
				    &render_passes );

  glui_win->add_checkbox_to_panel( iteration_panel, "save after rendering", 
				   &save );

  // Control Buttons
  control_panel = glui_win->add_panel_to_panel( setup_panel, "" );
  glui_win->add_button_to_panel( control_panel, "Start", START, button_cb );
  glui_win->add_button_to_panel( control_panel, "Pause", PAUSE, button_cb );
  glui_win->add_column_to_panel( control_panel, false );
  glui_win->add_button_to_panel( control_panel, "Save Image", SAVE, button_cb );
  glui_win->add_button_to_panel( control_panel, "Exit", 0, (GLUI_Update_CB)exit );


  // Parameter Panel
  glui_win->add_column( false );
  param_panel = glui_win->add_panel( "" );

  // Variables
  var_panel = glui_win->add_panel_to_panel( param_panel, "Variables" );

  a = rdGPU->getA();
  a_spinner =
    glui_win->add_spinner_to_panel( var_panel, "a:", GLUI_SPINNER_FLOAT, &a );
  glui_win->add_column_to_panel( var_panel, false );

 b = rdGPU->getB();
 b_spinner =
   glui_win->add_spinner_to_panel( var_panel, "b:", GLUI_SPINNER_FLOAT, &b );

  glui_win->add_column_to_panel( var_panel, false );

  ab_variance = rdGPU->getABVariance();
  ab_variance_spinner =
    glui_win->add_spinner_to_panel( var_panel, "variance:",
				    GLUI_SPINNER_FLOAT, &ab_variance );

  // Reaction Constants
  react_const_panel =
    glui_win->add_panel_to_panel( param_panel, "Reaction Constants" );

  react_const0 = rdGPU->getReaction_Const0();
  react_const0_spinner =
    glui_win->add_spinner_to_panel( react_const_panel, "constant 0:",
				    GLUI_SPINNER_FLOAT, &react_const0,
				    0, reaction_const_cb );

  react_const1 = rdGPU->getReaction_Const1();
  react_const1_spinner  =
    glui_win->add_spinner_to_panel( react_const_panel, "constant 1:",
				    GLUI_SPINNER_FLOAT, &react_const1,
				    0, reaction_const_cb );

  glui_win->add_column_to_panel( react_const_panel, false );


  react_const2 = rdGPU->getReaction_Const2();
  react_const2_spinner  =
    glui_win->add_spinner_to_panel( react_const_panel, "constant 2:",
				    GLUI_SPINNER_FLOAT, &react_const2,
				    0, reaction_const_cb );

  react_const_variance = rdGPU->getReactionConstVariance();
  react_const_variance_spinner =
    glui_win->add_spinner_to_panel( react_const_panel, "variance:",
			    GLUI_SPINNER_FLOAT, &react_const_variance );

  // Reaction Rate
  reaction_rate_panel =
    glui_win->add_panel_to_panel( param_panel, "Reaction Rate" );

  reaction_rate_radiogroup =
    glui_win->add_radiogroup_to_panel( reaction_rate_panel,
				       &rr_var, 0,
				       reaction_rate_cb);

  glui_win->add_radiobutton_to_group( reaction_rate_radiogroup, "Constant");
  glui_win->add_radiobutton_to_group( reaction_rate_radiogroup, "Variable");

  glui_win->add_column_to_panel( reaction_rate_panel, false );

  rr_coef_1 = rdGPU->getRRCoef1();
  rr_coef_2 = rdGPU->getRRCoef2();

  rr_coef_1_spinner = 
    glui_win->add_spinner_to_panel( reaction_rate_panel, "1.0 / ( ", 
				    GLUI_SPINNER_FLOAT, &rr_coef_1 );
  glui_win->add_column_to_panel( reaction_rate_panel, false );

  rr_coef_2_spinner =
    glui_win->add_spinner_to_panel( reaction_rate_panel, " + vmag * ", 
				    GLUI_SPINNER_FLOAT, &rr_coef_2 );

  glui_win->add_column_to_panel( reaction_rate_panel, false );
  glui_win->add_statictext_to_panel( reaction_rate_panel, ")" );


  // Diffusion Rate
  diffusion_rate_panel =
    glui_win->add_panel_to_panel( param_panel, "Diffusion" );

  diffusion_rate_radiogroup =
    glui_win->add_radiogroup_to_panel( diffusion_rate_panel,
				       &aniso, 0,
				       diffusion_cb);

  glui_win->add_radiobutton_to_group( diffusion_rate_radiogroup, "Isotropic");
  glui_win->add_radiobutton_to_group( diffusion_rate_radiogroup, "Anisotropic");


  gradient = rdGPU->getGradient();
  glui_win->add_checkbox_to_panel( diffusion_rate_panel, "Gradient", 
				   &gradient );

  glui_win->add_column_to_panel( diffusion_rate_panel, false );

  diff_coef_1 = rdGPU->getDiffCoef1();
  diff_coef_2 = rdGPU->getDiffCoef2();
  
  diff_coef_1_spinner =
    glui_win->add_spinner_to_panel( diffusion_rate_panel, "coef 1:",
				    GLUI_SPINNER_FLOAT, &diff_coef_1 );

  diff_coef_2_spinner =
    glui_win->add_spinner_to_panel( diffusion_rate_panel, "coef 2:",
				    GLUI_SPINNER_FLOAT, &diff_coef_2 );

  glui_win->add_column_to_panel( diffusion_rate_panel, false );

  if( invert_diff_vals ) {
    a_diff = 1.0 / rdGPU->getADiffRate() / (cell_size*cell_size);
    b_diff = 1.0 / rdGPU->getBDiffRate() / (cell_size*cell_size);
    c_diff = 1.0 / rdGPU->getCDiffRate() / (cell_size*cell_size);
    d_diff = 1.0 / rdGPU->getDDiffRate() / (cell_size*cell_size);
  } else {
    a_diff = rdGPU->getADiffRate() / (cell_size*cell_size);
    b_diff = rdGPU->getBDiffRate() / (cell_size*cell_size);
    c_diff = rdGPU->getCDiffRate() / (cell_size*cell_size);
    d_diff = rdGPU->getDDiffRate() / (cell_size*cell_size);
  }

  a_diff_spinner = 
    glui_win->add_spinner_to_panel( diffusion_rate_panel, "   a rate:     ",
				    GLUI_SPINNER_FLOAT, &a_diff );
  b_diff_spinner =
    glui_win->add_spinner_to_panel( diffusion_rate_panel, "   b rate:     ",
				    GLUI_SPINNER_FLOAT, &b_diff );

  c_diff_spinner = 
    glui_win->add_spinner_to_panel( diffusion_rate_panel, "   c rate:     ",
				    GLUI_SPINNER_FLOAT, &c_diff );
  d_diff_spinner =
    glui_win->add_spinner_to_panel( diffusion_rate_panel, "   d rate:     ",
				    GLUI_SPINNER_FLOAT, &d_diff );


  glui_win->add_column_to_panel( diffusion_rate_panel, false );

  invert_diff_vals_checkbox =
    glui_win->add_checkbox_to_panel( diffusion_rate_panel, "Invert", 
				     &invert_diff_vals );


  df_panel = glui_win->add_panel_to_panel( param_panel, "Data" );

  mult = rdGPU->getMult();
  mult_spinner = glui_win->add_spinner_to_panel( df_panel, "Gird Size 64 x", 
						 GLUI_SPINNER_INT, 
						 &mult );

  glui_win->add_column_to_panel( df_panel, false );

  cell_size_spinner =
    glui_win->add_spinner_to_panel( df_panel, "   cell size:  ",
				    GLUI_SPINNER_FLOAT, &cell_size );


  button_panel = glui_win->add_panel_to_panel( param_panel, "" );
  glui_win->add_button_to_panel( button_panel, "Restart", RESTART, load_cb );
  glui_win->add_column_to_panel( button_panel, false );
  glui_win->add_button_to_panel( button_panel, "Update", UPDATE, load_cb );

  // let our glui window know where the main gfx window is
  glui_win->set_main_gfx_window( main_win );

  //----------------------
  // GL initializations
  //----------------------
  glMatrixMode( GL_MODELVIEW );
  glLoadIdentity();
  gluLookAt( 0, 0, 1, 0, 0, 0, 0, 1, 0 );

  glClearColor( 1, 1, 1, 1 );

  GLint nbuf;
  glGetIntegerv( GL_AUX_BUFFERS, &nbuf );
  cerr << "Aux Buffers available: " << nbuf << endl;

  GLboolean stereo;
  glGetBooleanv( GL_STEREO, &stereo );
  if( stereo )
    cerr << "Stereo Buffers available" << endl;

  GLboolean dbuffer;
  glGetBooleanv( GL_DOUBLEBUFFER, &dbuffer );
  if( dbuffer )
    cerr << "Double Buffers available" << endl;


  //----------------------
  // CG initializations
  //----------------------
  rdGPU->initCG();
  
  // get the best profile for this hardware
  cgProfile = cgGLGetLatestProfile( CG_GL_FRAGMENT );

  if ( cgProfile == CG_PROFILE_UNKNOWN ) {
    fprintf( stderr, "%s\n%s\n",
	     "Fragment programming extensions (GL_ARB_fragment_program or ",
	     "GL_NV_fragment_program) not supported, exiting." );
    return -1;
  }
  
  cgGLSetOptimalOptions( cgProfile );

  // create the frag program context
  cgContext = cgCreateContext();

  //---------------------------
  // other Cg initializations for displaying the image
  //---------------------------
  string filename = "./textureRECT.cg";

  cgTexProgram = cgCreateProgramFromFile( cgContext, 
					  CG_SOURCE, filename.c_str(),
					  cgProfile, NULL, NULL );
  cgGLLoadProgram( cgTexProgram );

  cgTexParam      = cgGetNamedParameter( cgTexProgram, "texture" );
  cgMinParam      = cgGetNamedParameter( cgTexProgram, "min" );
  cgScaleParam    = cgGetNamedParameter( cgTexProgram, "scale" );
  cgColormapParam = cgGetNamedParameter( cgTexProgram, "colormap" );
  cgReactionParam = cgGetNamedParameter( cgTexProgram, "flag" );
  
  // generate and bind the texture object
  glGenTextures( 1, &glColormapTexID );
  glBindTexture( GL_TEXTURE_RECTANGLE_NV, glColormapTexID );

  // set up the default texture environment parameters
  glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  for( unsigned int i=0; i<MAX_COLORS*4; i++ )
    texColormap[i] /= 256;

  glTexImage2D( GL_TEXTURE_RECTANGLE_NV, 0, GL_FLOAT_RGBA32_NV,
                MAX_COLORS, 1,
                0, GL_RGBA, GL_FLOAT, texColormap );

  // check for ogl errors that may have occured during texture setup
  if ((errorCode = glGetError()) != GL_NO_ERROR) 
    fprintf( stderr, "init - colormap(): ERROR: %s\n",
	     gluErrorString(errorCode) );

 // Save the current Draw buffer
  glGetIntegerv(GL_DRAW_BUFFER, &glCurrentDrawbuffer);

  // Load the data up so the display callback has a texture to work with.
  load_cb( RESTART );

  glutPostRedisplay();

  // enter the GLUT main loop to catch events that
  // will be handled by the callback functions
  glutMainLoop( );

  return 1;
}
