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

#include "rd_gray_scott.h"
#include "rd_turing.h"

using namespace std;

unsigned int main_win = 0;       // index of the main rendering window

enum { TURING=0, GRAY_SCOTT=1 };
enum { INITIALIZE, RUN, START, PAUSE, RESTART, UPDATE, SAVE };
int run_status = INITIALIZE;

// glui variables
GLUI *glui_win;   // glui window

GLUI_Spinner *time_step_spinner,
  *time_mult_spinner,
  *theta_spinner,
  *a_spinner, *b_spinner, 
  *ab_variance_spinner,
  *react_const0_spinner,
  *react_const1_spinner,
  *react_const_variance_spinner,
  *render_passes_spinner,
  *num_iterations_spinner,
  *react_rate_spinner,
  *rr_coef_1_spinner, *rr_coef_2_spinner,
  *diff_coef_1_spinner, *diff_coef_2_spinner,
  *a_diff_spinner, *b_diff_spinner,
  *cell_size_spinner,
  *mult_spinner;

GLUI_RadioGroup *display_color_radiogroup,
  *reaction_radiogroup, *solution_radiogroup,
  *boundary_radiogroup, *relaxation_radiogroup, *laplacian_radiogroup,
  *reaction_rate_radiogroup, *diffusion_rate_radiogroup;

GLUI_Panel *setup_panel,
  *reaction_panel, *display_panel, *solution_panel,
  *relaxation_panel, *laplacian_panel,
  *iteration_panel, *control_panel,
  *param_panel,
  *var_panel, *react_const_panel,
  *reaction_rate_panel, *diffusion_rate_panel, *df_panel, *button_panel;

GLUI_EditText *iteration_edittext;

GLUI_Listbox *files_listbox;

// Reaction Diffusion local variables
int display_color = 0;
float time_step, time_mult;
float theta;
float a, b, ab_variance;
float react_const0, react_const1, react_const_variance;
float rr_coef_1, rr_coef_2;
int render_passes = 50;
int num_iterations = 50000;
float diff_coef_1, diff_coef_2, a_diff, b_diff;
int reaction=TURING, solution = 0, boundary = 1;
int relaxation = 2, laplacian = 0;
int mixing = 0, save=0;
int rr_var = 1, aniso = 1, gradient = 1, mult = 1;
float cell_size = 1.0;
int iteration = 0;

// For dispalying the running time
clock_t start_time, stop_time;
bool time_flag = 0;

// Local stuff
rd_base *rdCPU = NULL;
float ***vectorData = NULL;
float  *displayData = NULL;

/******************************************************************************
Create a vector field.
******************************************************************************/

int create_vector_data( unsigned int *dims, float ****vector )
{
  float mid = 64;

  dims[0] = dims[1] = 2 * (unsigned int) mid;

  (*vector) = new float**[ dims[0] ];

  register unsigned int i, j;

  (*vector) = (float***) malloc( sizeof(float***) * dims[0] );

  for ( j=0; j<dims[0]; j++) {
    (*vector)[j] = (float**) malloc( sizeof(float*) * dims[1] );
      
    for ( i=0; i<dims[1]; i++) {
      (*vector)[j][i] = (float*) malloc( sizeof(float) * 4 );

      (*vector)[j][i][0] = -((float) j - mid) / mid;
      (*vector)[j][i][1] =  ((float) i - mid) / mid;
      (*vector)[j][i][2] = 0;
      (*vector)[j][i][3] =
	sqrt( (*vector)[j][i][0] * (*vector)[j][i][0] +
	      (*vector)[j][i][1] * (*vector)[j][i][1] +
	      (*vector)[j][i][2] * (*vector)[j][i][2] );

      if( (*vector)[j][i][3] < MIN_FLOAT ) {
	(*vector)[j][i][0] = 0;
	(*vector)[j][i][1] = 1;
	(*vector)[j][i][3] = MIN_FLOAT;
      } else {
	(*vector)[j][i][0] /= (*vector)[j][i][3];
	(*vector)[j][i][1] /= (*vector)[j][i][3];
      }
    }
  }
}


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
  
  gluOrtho2D( 0, (GLfloat) w, 0, (GLfloat) h );
}


/******************************************************************************
saveImage
******************************************************************************/
void saveImage()
{
  // get the texture values
  int w, h;

   w = rdCPU->width_;
   h = rdCPU->height_;

  // Get the min and max for normalizing the image.
  float val, min_value = 1.0e12, max_value = -1.0e12;

  float rc0_2 = w / 2;
  float rc1_2 = h / 2;

  
  for ( int j = 0; j < h; j++ ) 
    for ( int i = 0; i < w; i++ ) {
      val = rdCPU->morphigen[display_color][j][i];

      if ( min_value > val ) min_value = val;
      if ( max_value < val ) max_value = val;
    }

  cout << min_value << "  " <<  max_value << endl;

  if (min_value == max_value) {
    min_value = max_value - 1;
    max_value = min_value + 2;
  }

  cout << min_value << "  " <<  max_value << endl;

  float scale = 255.0 / (max_value-min_value);

  unsigned char *values = new unsigned char[w*h];;

  float db = 0.0;

  // Normalize the image.
  for ( int j = 0; j < h; j++ ) {
    for ( int i = 0; i < w; i++ ){ 
      float di = ( (float) i-rc0_2) / rc0_2;
      float dj = ( (float) j-rc1_2) / rc1_2;
      float dr = sqrt( di*di + dj*dj);

      values[i + j*w] = (unsigned char)
	((rdCPU->morphigen[display_color][j][i] - min_value) * scale);
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
  int w, h;

  w = rdCPU->width_;
  h = rdCPU->height_;

  // Get the min and max for normalizing the image.
  float val, min_value[6], max_value[6];

  for ( unsigned int i=0; i<rdCPU->nMorphs; i++ ) {
    min_value[i] =  1.0e12;
    max_value[i] = -1.0e12;
  }

  for ( unsigned int j=0; j<h; j++ ) {
    for ( unsigned int i=0; i<w; i++ ) {
      for ( unsigned int c=0; c<rdCPU->nMorphs; c++ ) {
	val = rdCPU->morphigen[c][j][i];

	if ( min_value[c] > val ) min_value[c] = val;
	if ( max_value[c] < val ) max_value[c] = val;
      }
    }
  }

  for ( unsigned int c=rdCPU->nMorphs; c<6; c++ ) {
    min_value[c] = 0;
    max_value[c] = 0;
  }

  float scale[6];

  for ( unsigned int i=0; i<rdCPU->nMorphs; i++ ) {
    if (min_value[i] == max_value[i]) {
      min_value[i] = max_value[i] - 1;
      max_value[i] = min_value[i] + 2;
    }

    scale[i] = 1.0 / (max_value[i]-min_value[i]);
  }

  // Rescale to 0 - > 1
  int cc = 0;

  for ( unsigned int j=0; j<h; j++ ) {
    for ( unsigned int i=0; i<w; i++ ) {
      for ( unsigned int c=0; c<4; c++ ) {
	val = rdCPU->morphigen[display_color][j][i];

	displayData[cc++] = (val - min_value[display_color]) *
	  scale[display_color];
      }
    }
  }

  glRasterPos2i(0,0);
  glDrawPixels( w, h, GL_RGBA, GL_FLOAT, displayData );
  glFlush();
    
  glutSwapBuffers();
}


/******************************************************************************
updateGUIValues
******************************************************************************/
void updateGUIValues() {

  // set the reaction diffusion variables to those from our gui
  rdCPU->setSolution( solution );
  rdCPU->setBoundary( boundary );
  rdCPU->setRelaxation( relaxation );
  rdCPU->setLaplacian( laplacian );
  rdCPU->setTheta( theta );
  rdCPU->setReaction( reaction );

  rdCPU->setTimeStep( time_step );
  rdCPU->setTimeMult( time_mult );

  rdCPU->setABVariance( ab_variance );
  rdCPU->setReactionConstVariance( react_const_variance );
  rdCPU->setA( a );
  rdCPU->setB( b );
  rdCPU->setReaction_Const0( react_const0 );
  rdCPU->setReaction_Const1( react_const1 );
  
  rdCPU->setRRCoef1( rr_coef_1 );
  rdCPU->setRRCoef2( rr_coef_2 );

  rdCPU->setDiffCoef1( diff_coef_1 );
  rdCPU->setDiffCoef2( diff_coef_2 );

  rdCPU->setADiffRate( 1.0 / a_diff / (cell_size*cell_size) );
  rdCPU->setBDiffRate( 1.0 / b_diff / (cell_size*cell_size) );

  rdCPU->setGradient( gradient );

  rdCPU->setMult( mult );
}


/******************************************************************************
Load
******************************************************************************/
void load_cb( int ID )
{
  if( ID == RESTART ) {

    // do the Reaction Diffusion initializations  
    unsigned int dims[2];

    create_vector_data( dims, &vectorData );

    if( rdCPU )
      delete rdCPU;

    if( reaction == TURING ) {
      rdCPU = new rd_turing;
      rdCPU->mIndex = 0;

    } else if( reaction == GRAY_SCOTT ) {
      rdCPU = new rd_gray_scott;
      rdCPU->mIndex = 1;
    }

    rdCPU->height_ = dims[0];
    rdCPU->width_  = dims[1];

    rdCPU->alloc( dims, mult ); // cell_mult

    updateGUIValues();

    // set up the inital parameters
    rdCPU->initialize( vectorData, 1 );
    rdCPU->set_rates( rd_base::VARIABLE );

    rdCPU->setGradient( 0 );   
    rdCPU->set_diffusion( aniso );
    rdCPU->setGradient( gradient );

    // reset the counter
    iteration = 0;
    iteration_edittext->set_int_val( iteration );
    
    mult = rdCPU->getMult();
    mult_spinner->set_int_val( mult );

    if( displayData )
      delete displayData;

    // Alloc the memory for the system.
    displayData = new float[rdCPU->height_*rdCPU->width_*4];

    if( run_status == RUN )
      run_status = START;

  } else if( ID == UPDATE ) {

    unsigned int dims[2];

    create_vector_data( dims, &vectorData );

    if( rdCPU->height_ == dims[0] && rdCPU->width_ == dims[1] ) {

      updateGUIValues();

      // set up the inital parameters
      rdCPU->initialize( vectorData, 0 );
      rdCPU->set_rates( rd_base::VARIABLE );
      rdCPU->set_diffusion( aniso );
    } else {
      cerr << "Can not update - vector grids are of different sizes.";
    }
  }

  // make sure that our window is the right size
  reshape( rdCPU->width_, rdCPU->height_ );

  glutPostRedisplay();
}


/******************************************************************************
Idle loop where things get down
******************************************************************************/
void idle()
{ 
  if ( run_status == INITIALIZE ) {
    load_cb( RESTART );

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
      
    if( (solution == rd_base::IMPLICIT || solution == rd_base::THETA)
	&& iteration == 0 ) {

      // Do one forward step so that the inital values are all different.
      rdCPU->setSolution( rd_base::EXPLICIT );
      rdCPU->set_rates( rd_base::VARIABLE );
      rdCPU->next_step_explicit_euler( );

      // Reset the rates for an implicit solution
      rdCPU->setSolution( rd_base::IMPLICIT );
      rdCPU->set_rates( rd_base::VARIABLE );
    
      rdCPU->setGradient( 0 );   
      rdCPU->set_diffusion( aniso );
      rdCPU->setGradient( gradient );

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

    // check to see if the maximum number of iterations has been reached.
    if ( iteration < num_iterations )	{

      if( solution == rd_base::EXPLICIT ) {
	for( unsigned int j=0; j<render_passes; j++ )
	  rdCPU->next_step_explicit_euler( );

	if( gradient )
	  rdCPU->set_diffusion( aniso );

      } else /*if( solution == rd_base::IMPLICIT )*/ {
	for( unsigned int j=0; j<5; j++ ) {
	  for( unsigned int k=0; k<render_passes/5; k++ )
	    rdCPU->next_step_implicit_euler( );

	  if( gradient )
	    rdCPU->set_diffusion( aniso );
	}
      }
	
      glutPostRedisplay();

      if( save )
	saveImage();

      iteration += render_passes;

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
  if( solution )
    num_iterations = 2000;
  else
    num_iterations = 50000;

  num_iterations_spinner->set_int_val( num_iterations );

  rdCPU->setSolution( solution );

  if( solution == rd_base::EXPLICIT )
    theta = 0;
  else if( solution == rd_base::IMPLICIT )
    theta = 1.0;
  else if( solution == rd_base::THETA )
    theta = 0.5;

  theta_spinner->set_float_val( theta );
}


void boundary_cb( int ID )
{
  rdCPU->setBoundary( boundary );
}


void reaction_const_cb( int ID )
{
  if( reaction == TURING ) {
    a = react_const1 - react_const0;
    b = react_const1 / a;
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

  } else if( reaction == GRAY_SCOTT ) {
    a_diff = 25000;
    b_diff = 12500;
  }

  a_diff_spinner->set_float_val( a_diff );
  b_diff_spinner->set_float_val( b_diff );

  if( reaction == TURING ) {
    react_const0 = 16.0;
    react_const1 = 12.0;
    react_const_variance = 0.001;

  } else if( reaction == GRAY_SCOTT ) {
    react_const0 = 0.0925;
    react_const1 = 0.0300;
    react_const_variance = 0.01;
  }

  react_const0_spinner->set_float_val( react_const0 );
  react_const1_spinner->set_float_val( react_const1 );
  react_const_variance_spinner->set_float_val( react_const_variance );

  if( reaction == TURING ) {
    a = 4.0;
    b = 4.0;
    ab_variance = 0.0;

  } else if( reaction == GRAY_SCOTT ) {
    a = 0.25;
    b = 0.50;
    ab_variance = 0.0;
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
  }

  time_step_spinner->set_float_val( time_step );

  if( reaction == TURING ) {
    display_color = 0;
  } else if( reaction == GRAY_SCOTT ) {
    display_color = 1;
  }

  display_color_radiogroup->set_int_val( display_color );
}


/******************************************************************************
Main
******************************************************************************/
int main( int argc, char* argv[] )
{
  // Basically a dummy start for the initial values.
  rdCPU = new rd_turing;

  // Display a glut window or not.
  int x = 48, y = 48;

  // Need to have glut initialized even if no window.
  glutInit( &argc, argv );

  glutInitWindowPosition( x, y );
  glutInitWindowSize( 128, 128 );
  glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE );

  // create the window and store its index
  main_win = glutCreateWindow( "Reaction Diffusion" );

  // register the callback functions
  glutDisplayFunc( display );	  // called in response to display events
  glutReshapeFunc( reshape );	  // called in response to window resize

  // Create the glui window
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

  // Display Control
  display_panel = reaction_panel;
  glui_win->add_column_to_panel( display_panel, false );

  glui_win->add_statictext_to_panel( reaction_panel, "Display:" );

  display_color_radiogroup =
    glui_win->add_radiogroup_to_panel( display_panel, &display_color);

  glui_win->add_radiobutton_to_group( display_color_radiogroup, "Show a");
  glui_win->add_radiobutton_to_group( display_color_radiogroup, "Show b");


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
				       boundary_cb );

  glui_win->add_radiobutton_to_group( boundary_radiogroup, "Zero Flux Clamp");
  glui_win->add_radiobutton_to_group( boundary_radiogroup, "Periodic");


  glui_win->add_column_to_panel( solution_panel, false );

  time_step = rdCPU->getTimeStep();
  time_step_spinner =
    glui_win->add_spinner_to_panel( solution_panel, "time step:",
				    GLUI_SPINNER_FLOAT, &time_step );

  time_mult = rdCPU->getTimeMult();
  time_mult_spinner =
    glui_win->add_spinner_to_panel( solution_panel, "time multiplier:",
				    GLUI_SPINNER_FLOAT, &time_mult );

  theta = rdCPU->getTheta();
  theta_spinner =
    glui_win->add_spinner_to_panel( solution_panel, "theta:",
				    GLUI_SPINNER_FLOAT, &theta );

  // Relaxation
  relaxation_panel =
    glui_win->add_panel_to_panel( setup_panel, "" );

  glui_win->add_statictext_to_panel( relaxation_panel, "Relaxation Type:" );

  relaxation_radiogroup =
    glui_win->add_radiogroup_to_panel( relaxation_panel, &relaxation );

  glui_win->add_radiobutton_to_group( relaxation_radiogroup, "Jacobi");
  glui_win->add_radiobutton_to_group( relaxation_radiogroup, "Gauss-Siedel");
  glui_win->add_radiobutton_to_group( relaxation_radiogroup, "Red-Black");


  // Laplacian
  laplacian_panel = relaxation_panel;
  glui_win->add_column_to_panel( laplacian_panel, false );

  glui_win->add_statictext_to_panel( laplacian_panel, "Laplacian Type:" );

  laplacian_radiogroup =
    glui_win->add_radiogroup_to_panel( laplacian_panel, &laplacian );

  glui_win->add_radiobutton_to_group( laplacian_radiogroup, "Inhomogeneous");
  glui_win->add_radiobutton_to_group( laplacian_radiogroup, "Uniform");


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
  var_panel = glui_win->add_panel_to_panel( param_panel, "Morphogens" );

  a = rdCPU->getA();
  a_spinner =
    glui_win->add_spinner_to_panel( var_panel, "a:", GLUI_SPINNER_FLOAT, &a );
  glui_win->add_column_to_panel( var_panel, false );

  b = rdCPU->getB();
  b_spinner =
   glui_win->add_spinner_to_panel( var_panel, "b:", GLUI_SPINNER_FLOAT, &b );

  glui_win->add_column_to_panel( var_panel, false );

  ab_variance = rdCPU->getABVariance();
  ab_variance_spinner =
    glui_win->add_spinner_to_panel( var_panel, "variance:",
				    GLUI_SPINNER_FLOAT, &ab_variance );

  // Reaction Constants
  react_const_panel =
    glui_win->add_panel_to_panel( param_panel, "Reaction Constants" );

  react_const0 = rdCPU->getReaction_Const0();
  react_const0_spinner =
    glui_win->add_spinner_to_panel( react_const_panel, "constant 0:",
				    GLUI_SPINNER_FLOAT, &react_const0,
				    0, reaction_const_cb );

  glui_win->add_column_to_panel( react_const_panel, false );

  react_const1 = rdCPU->getReaction_Const1();
  react_const1_spinner  =
    glui_win->add_spinner_to_panel( react_const_panel, "constant 1:",
				    GLUI_SPINNER_FLOAT, &react_const1,
				    0, reaction_const_cb );

  glui_win->add_column_to_panel( react_const_panel, false );


  react_const_variance = rdCPU->getReactionConstVariance();
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

  rr_coef_1 = rdCPU->getRRCoef1();
  rr_coef_2 = rdCPU->getRRCoef2();

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


  gradient = rdCPU->getGradient();
  glui_win->add_checkbox_to_panel( diffusion_rate_panel, "Gradient", 
				   &gradient );

  glui_win->add_column_to_panel( diffusion_rate_panel, false );

  diff_coef_1 = rdCPU->getDiffCoef1();
  diff_coef_2 = rdCPU->getDiffCoef2();

  diff_coef_1_spinner =
    glui_win->add_spinner_to_panel( diffusion_rate_panel, "coef 1:",
				    GLUI_SPINNER_FLOAT, &diff_coef_1 );

  diff_coef_2_spinner =
    glui_win->add_spinner_to_panel( diffusion_rate_panel, "coef 2:",
				    GLUI_SPINNER_FLOAT, &diff_coef_2 );


  glui_win->add_column_to_panel( diffusion_rate_panel, false );

  a_diff = 1.0 / rdCPU->getADiffRate() / (cell_size*cell_size);
  b_diff = 1.0 / rdCPU->getBDiffRate() / (cell_size*cell_size);

  a_diff_spinner = 
    glui_win->add_spinner_to_panel( diffusion_rate_panel, "   a rate:   1/",
				    GLUI_SPINNER_FLOAT, &a_diff );
  b_diff_spinner =
    glui_win->add_spinner_to_panel( diffusion_rate_panel, "   b rate:   1/",
				    GLUI_SPINNER_FLOAT, &b_diff );


  df_panel = glui_win->add_panel_to_panel( param_panel, "Data " );

  mult = rdCPU->getMult();
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


  glutPostRedisplay();

  // enter the GLUT main loop to catch events that
  // will be handled by the callback functions
  glutMainLoop( );

  return 1;
}
