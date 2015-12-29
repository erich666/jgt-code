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

#include "ReactDiffuse.h"

#include <iostream>

#include <math.h>
#include <time.h>

#include <GL/glut.h>

#include "gammp.h"

using namespace std;

#define LERP( t, x0, x1 ) \
( (float) (x0) + (t) * ( (float) (x1) - (float) (x0)) )


#define SGN( x ) \
	( (x) < 0 ? -1.0 : 1.0 )


#define INDEX( x, range ) \
	( (int) (x*range) + range )


#define OGL_ERROR_CHECKING 1

#define MIN_FLOAT 1.0e-8
#define MAX_FLOAT 1.0e12

#define MAX_GAMMP 500

//------------------------------------------------------------------------
// Constructor and Destructor 
//------------------------------------------------------------------------
ReactDiffuse::ReactDiffuse() :
  _cg_Context(0),
  _pbWidth(0), _pbHeight(0),
  _fbuffer(0),

  _gl_RHSTexID(0),
  _gl_ResidualsTexID(0),

  _gl_dlAll(0),
  _gl_dlInterior(0),
  _gl_dlBoundary(0),
  _gl_dlBoundaryZeroFlux(0),
  _gl_dlBoundaryPeriodic(0),

  _reaction(TURING),
  _solution(EULER_EXPLICIT),
  _last_solution(MAX_SOLUTIONS),

  // initial reactant values
  _a(4.0), _b(4.0),
  _ab_variance(0.000),
  _reaction_const0(12.0),
  _reaction_const1(16.0),
  _reaction_const2(0.0),
  _reaction_const_variance(0.001),
  _time_step(0.50),
  _time_mult(25.0),
  _time_step_inv(1.0 / (_time_step * _time_mult)),
  _cn_theta(0.50),

  _mixing_rate(0.1),

  _rr_coef_1(100.0),
  _rr_coef_2(250.0),

  _diff_coef_1(1.5),
  _diff_coef_2(0.5),

  _a_diff_rate(1.0 / 16.0),
  _b_diff_rate(1.0 /  4.0),

  _c_diff_rate(1.0 / 4.0),
  _d_diff_rate(1.0 / 1.0),

  _gradient( 1 ),

  _texMorphigens(0), 
  _texReaction(0), _texDiffusion(0), _texDiffusion2(0),
  _texVariance(0),
  _texRHS(0),
  _texResiduals(0), 
  _texReduction(0), 

  _reaction_const0_data(0),
  _reaction_const1_data(0),
  _reaction_const2_data(0),
  _rr_data(0),
  _dt_data(0),

  _v_data(0), 

  gammp_table( 0 ),

  _mult(1)
{
  gammp_table = new float[2*MAX_GAMMP+1];

  gammp_table[MAX_GAMMP] = 0;

  for( unsigned int i=1; i<=MAX_GAMMP; i++ ) {
    float val = gammp( 3.0, 10.0 * (float) i / (float) MAX_GAMMP );

    gammp_table[MAX_GAMMP-i] = -0.25 * val;
    gammp_table[MAX_GAMMP+i] = +0.25 * val;
  }

  constant_names[REACTION  ] = string("reactionConsts");
  constant_names[DIFFUSION ] = string("diffusionConsts");
  constant_names[DIFFUSION2] = string("diffusion2Consts");
  constant_names[VARIANCE  ] = string("varianceConsts");

  _gl_MorphogenTexID[0]= _gl_MorphogenTexID[1] = 0;
  _gl_ReductionTexID[0]= _gl_ReductionTexID[1] = 0;
}

ReactDiffuse::~ReactDiffuse()
{
  if ( _fbuffer ) delete _fbuffer;

  if ( _cg_InitProgram ) cgDestroyProgram( _cg_InitProgram );

  for( unsigned int i=0; i<MAX_REACTIONS; i++ )
    if ( _cg_Explicit[i] ) cgDestroyProgram( _cg_Explicit[i] );

  if ( _cg_Context ) cgDestroyContext( _cg_Context );

  if ( gammp_table ) delete gammp_table;

  deleteGrids();
}

//------------------------------------------------------------------------
// function    : initCG() 
// description : initialize CG
//------------------------------------------------------------------------
void ReactDiffuse::initCG()
{
  // initialize glew --> set this so that you get all extensions,
  //   even the ones that are not exported by the extensions string
  glewExperimental = GL_TRUE; 
  unsigned int err = glewInit();
  if (GLEW_OK != err) {
    // problem: glewInit failed, something is seriously wrong
    fprintf(stderr, "GLEW Error: %s\n", glewGetErrorString(err));
    exit(-1);
  }
  
  //----------------------
  // Cg initializations
  //----------------------
  // get the best profile for this hardware
  _cg_Profile = cgGLGetLatestProfile( CG_GL_FRAGMENT );

  if ( _cg_Profile == CG_PROFILE_UNKNOWN ) {
    cerr << "Fragment programming extensions ";
    cerr << "(GL_ARB_fragment_program or GL_NV_fragment_program) ";
    cerr << "not supported, exiting..." << endl;
    exit(-1);
  } else if( _cg_Profile <= CG_PROFILE_FP20 ) {
    cerr << "Fragment programming extensions of FP20 or less " << endl;
    cerr << "do not support floating point, exiting..." << endl;
    exit(-1);
  } else {
    string profile;

    switch( _cg_Profile ) {
    case CG_PROFILE_FP30:
      profile = "FP 30";
      break;
    case CG_PROFILE_FP40:
      profile = "FP 40";
      break;
    case CG_PROFILE_ARBFP1:
      profile = "ARB Fragment";
      break;
    }

    cout << "Running profile: " << profile
 	 << " (" << _cg_Profile << ")" << endl;

    // query for the number of support texture units --> this number can
    //  help you determine how many texture units are available on your
    //  card for passing in texture coords to your frag progams
    GLint units;
    glGetIntegerv(GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS_ARB, &units);
    cout << "Vertex texture units: " << units << endl;

    glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS_ARB, &units);
    cout << "Fragment texture image units: " << units << endl;

    glGetIntegerv( GL_MAX_TEXTURE_UNITS_ARB, &units);
    cout << "Number of support texture units: " << units << endl;
  }

  cgGLSetOptimalOptions( _cg_Profile );

  // register the error printing function
  cgSetErrorCallback( cgErrorCallback );

  // create the CG context
  _cg_Context = cgCreateContext();

  // load the fbuffer initializing frag shader --> this is where we setup
  //  the domain initial values

  string filename = "./initializer.cg";
  _cg_InitProgram =
    cgCreateProgramFromFile( _cg_Context, 
			     CG_SOURCE, filename.c_str(),
			     _cg_Profile, 0, 0 );

  // load the interior explicit frag shader
  filename = "./turing_explicit.cg";
  _cg_Explicit[TURING] =
    cgCreateProgramFromFile( _cg_Context, 
			     CG_SOURCE, filename.c_str(),
			     _cg_Profile, 0, 0 );

  filename = "./turing_explicit_uniform.cg";
  _cg_Explicit[TURING+1] =
    cgCreateProgramFromFile( _cg_Context, 
			     CG_SOURCE, filename.c_str(),
			     _cg_Profile, 0, 0 );
  
  filename = "./gray_scott_explicit.cg";
  _cg_Explicit[GRAY_SCOTT] =
    cgCreateProgramFromFile( _cg_Context, 
			     CG_SOURCE, filename.c_str(),
			     _cg_Profile, 0, 0 );

  filename = "./brusselator_explicit.cg";
  _cg_Explicit[BRUSSELATOR] =
    cgCreateProgramFromFile( _cg_Context, 
			     CG_SOURCE, filename.c_str(),
			     _cg_Profile, 0, 0 );

  // load the interior semi implicit frag shaders
  filename = "./turing_implicit_rhs.cg";
  _cg_Implicit_RHS[EULER_SEMI_IMPLICIT] =
    cgCreateProgramFromFile( _cg_Context, 
			     CG_SOURCE, filename.c_str(),
			     _cg_Profile, 0, 0 );

  filename = "./turing_implicit_relax.cg";
  _cg_Implicit_Relax[EULER_SEMI_IMPLICIT] =
    cgCreateProgramFromFile( _cg_Context, 
			     CG_SOURCE, filename.c_str(),
			     _cg_Profile, 0, 0 );

  filename = "./turing_implicit_residuals.cg";
  _cg_Implicit_Residuals[EULER_SEMI_IMPLICIT] =
    cgCreateProgramFromFile( _cg_Context, 
			     CG_SOURCE, filename.c_str(),
			     _cg_Profile, 0, 0 );

  // load the interior theta implicit frag shaders
  filename = "./turing_implicit_theta_rhs.cg";
  _cg_Implicit_RHS[EULER_THETA_IMPLICIT] =
    cgCreateProgramFromFile( _cg_Context, 
			     CG_SOURCE, filename.c_str(),
			     _cg_Profile, 0, 0 );

  filename = "./turing_implicit_relax.cg";
  _cg_Implicit_Relax[EULER_THETA_IMPLICIT] =
    cgCreateProgramFromFile( _cg_Context, 
			     CG_SOURCE, filename.c_str(),
			     _cg_Profile, 0, 0 );

  filename = "./turing_implicit_residuals.cg";
  _cg_Implicit_Residuals[EULER_THETA_IMPLICIT] =
    cgCreateProgramFromFile( _cg_Context, 
			     CG_SOURCE, filename.c_str(),
			     _cg_Profile, 0, 0 );

  // load the implicit utility frag shaders
  filename = "./implicit_clamp.cg";
  _cg_Implicit_Clamp =
    cgCreateProgramFromFile( _cg_Context, 
			     CG_SOURCE, filename.c_str(),
			     _cg_Profile, 0, 0 );

  // load the utility frag shadders
  filename = "./summation.cg";
  _cg_Summation =
    cgCreateProgramFromFile( _cg_Context, 
			     CG_SOURCE, filename.c_str(),
			     _cg_Profile, 0, 0 );

  //---------------------------------------------------
  // initialize the reaction-diffusion frag programs
  //---------------------------------------------------

  // because we are going to be using finite differences to integrate
  // our equations, but we have taken care of dealing with the
  // boundaries by passing in texture coords --> NOTE: we use periodic
  // boundary conditions

  // load the CG program so that we can get the variables
  for( unsigned int i=0; i<MAX_REACTIONS; i++ )
    cgGLLoadProgram( _cg_Explicit[i] );

  for( unsigned int i=EULER_SEMI_IMPLICIT; i<=EULER_THETA_IMPLICIT; i++ ) {
    cgGLLoadProgram( _cg_Implicit_RHS[i] );
    cgGLLoadProgram( _cg_Implicit_Relax[i] );
    cgGLLoadProgram( _cg_Implicit_Residuals[i] );
  }

  cgGLLoadProgram( _cg_Summation );
  cgGLLoadProgram( _cg_Implicit_Clamp );
}

//------------------------------------------------------------------------
// function    : deleteGrids() 
// description : delete the grids used.
//------------------------------------------------------------------------
void ReactDiffuse::deleteGrids()
{
  if( _texMorphigens ) {

    delete[] _texMorphigens;
    delete[] _texReaction;
    delete[] _texDiffusion;
    delete[] _texDiffusion2;
    delete[] _texVariance;
    delete[] _texRHS;
    delete[] _texResiduals;
    delete[] _texReduction;

    for ( unsigned int j = 0; j < _pbHeight; j++ ) {
      for ( unsigned int i = 0; i < _pbWidth; i++ ) {
	delete[] _dt_data[j][i];
      }

      delete[] _reaction_const0_data[j];
      delete[] _reaction_const1_data[j];
      delete[] _reaction_const2_data[j];
      delete[] _rr_data[j];
      delete[] _dt_data[j];
    }

    delete[] _reaction_const0_data;
    delete[] _reaction_const1_data;
    delete[] _reaction_const2_data;
    delete[] _rr_data;
    delete[] _dt_data;

    if( glIsTexture( _gl_RHSTexID ) ) {
      glDeleteTextures( 1, &_gl_RHSTexID );
      _gl_RHSTexID = 0;
    }

    if( glIsTexture( _gl_ResidualsTexID ) ) {
      glDeleteTextures( 1, &_gl_ResidualsTexID );
      _gl_ResidualsTexID = 0;
    }

    for ( unsigned int i=0; i<2; i++ ) {
      if( glIsTexture( _gl_MorphogenTexID[i] ) ) {
	glDeleteTextures( 1, &_gl_MorphogenTexID[i] );
	_gl_MorphogenTexID[i] = 0;
      }

      if( glIsTexture( _gl_ReductionTexID[i] ) ) {
	glDeleteTextures( 1, &_gl_ReductionTexID[i] );
	_gl_ReductionTexID[i] = 0;
      }
    }
  }
}

//------------------------------------------------------------------------
// function    : initFbuffer() 
// description : initialize the fbuffer
//------------------------------------------------------------------------
void ReactDiffuse::initFbuffer()
{
  //----------------------
  // setup the fbuffer
  //----------------------

  // instantiate the fbuffer
  if ( _fbuffer )
    delete _fbuffer;

  _fbuffer = new Fbuffer( _pbWidth, _pbHeight );
  _fbuffer->create();

  // setup the rendering context for the first fbuffer --> we now want to
  //  use our loaded initialization shader to render to the fbuffer
  _fbuffer->enable();

  //----------------------
  // setup the inital view for grid compuation
  //----------------------
  glViewport( 0, 0, _pbWidth, _pbHeight );

  glDisable( GL_DEPTH_TEST );
  glDisable( GL_LIGHTING );

  // setup for 2D rendering
  glMatrixMode( GL_PROJECTION );
  glLoadIdentity();
  gluOrtho2D( 0, (GLfloat)_pbWidth, 0, (GLfloat)_pbHeight );

  glMatrixMode ( GL_MODELVIEW );
  glLoadIdentity();
  gluLookAt( 0, 0, 1, 0, 0, 0, 0, 1, 0 );

  _fbuffer->disable();

  //----------------------------
  // any other initializings...
  //----------------------------

  // generate the OGL display lists
  generateDisplayLists();
}


//------------------------------------------------------------------------
// function    : setInitalState() 
// description : set the fbuffer to hold the inital state.
//------------------------------------------------------------------------
void ReactDiffuse::setInitalState()
{
  // setup the inital values for the buffer
  _fbuffer->enable();
  _fbuffer->attach( _gl_MorphogenTexID[0], 0 );

  glDrawBuffer( GL_COLOR_ATTACHMENT0_EXT );

  cgGLLoadProgram( _cg_InitProgram );
  
  // set the initial state and render to the buffer
  cgGLBindProgram( _cg_InitProgram );
  cgGLEnableProfile( _cg_Profile );
  
  cgGLSetParameter4f( cgGetNamedParameter( _cg_InitProgram, "initVal" ),
		      _a, _b, _a, _b );
  
  // setup the links for the variance texture
  _cg_ConstsTex[VARIANCE] =
    cgGetNamedParameter( _cg_InitProgram, constant_names[VARIANCE].c_str() ); 
  cgGLSetTextureParameter( _cg_ConstsTex[VARIANCE], _gl_ConstsTexID[VARIANCE] );
  cgGLEnableTextureParameter( _cg_ConstsTex[VARIANCE] );
  
  // render using the variance to set the initial values.
  glCallList( _gl_dlAll ); 
    
  cgGLDisableTextureParameter( _cg_ConstsTex[VARIANCE] );
    
  cgGLDisableProfile( _cg_Profile );
    
  _fbuffer->disable();
}


//------------------------------------------------------------------------
// function    : updateTexture 
// description : update the texture that stores the values.
//------------------------------------------------------------------------
void ReactDiffuse::updateTexture(GLuint &texID, float *texData)
{
  // generate and bind the texture object
  if ( glIsTexture( texID ) == GL_TRUE )
    glDeleteTextures( 1, &texID );

  glGenTextures( 1, &texID );
  glEnable(GL_TEXTURE_RECTANGLE_NV);
  glBindTexture( GL_TEXTURE_RECTANGLE_NV, texID );

  // set up the default texture environment parameters
  glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  glTexImage2D( GL_TEXTURE_RECTANGLE_NV, 0, GL_FLOAT_RGBA32_NV, 
                _pbWidth, _pbHeight,
                0, GL_RGBA, GL_FLOAT, texData );

  glBindTexture(GL_TEXTURE_RECTANGLE_NV, 0);
  glDisable(GL_TEXTURE_RECTANGLE_NV);

  // check for ogl errors that may have occured during texture setup
#ifdef OGL_ERROR_CHECKING
  unsigned int i;

  for( i=REACTION; i<=DIFFUSION; i++)
    if( texID == _gl_ConstsTexID[i] )
      break;

  GLenum errorCode;
  if ((errorCode = glGetError()) != GL_NO_ERROR) 
    fprintf( stderr, "%d updateTexture(): ERROR: %s\n", i,
	     gluErrorString(errorCode) );
#endif
}


//------------------------------------------------------------------------
// function    : updateConstantsReact() 
// description : update the texture that stores reaction.
//               NOTE: modulate the reaction rates based on the vec data
//------------------------------------------------------------------------
void ReactDiffuse::updateConstantsReact()
{
  float ts;

  if( _solution == EULER_EXPLICIT )
    ts = _time_step;
  else
    ts = 1.0;

//  cerr << "update reaction numerator " << ts << endl;

  float v_scale = 1.0 / (_vmag_max -_vmag_min);

  float min_mag_norm=MAX_FLOAT, max_mag_norm=-MAX_FLOAT;

  unsigned int index = 3;

  float u;

  for ( unsigned int j=0; j<_pbHeight; j++ ) {
    for ( unsigned int i=0; i<_pbWidth; i++ ) {
      
      // reaction rate
      if( _v_data[j][i][3] > 0 ) {

	u = (_v_data[j][i][3]-_vmag_min) * v_scale;

	if( _reaction == TURING )
	  _texReaction[index] = ts / (_rr_coef_1 + _rr_coef_2 * u );
	else
	  _texReaction[index] = ts * (_rr_coef_1 + _rr_coef_2 * u );
      } else {
	_texReaction[index] = 0;
      }

      _rr_data[j][i] =_texReaction[index];

//       if( min_mag_norm > _texReaction[index] )
// 	min_mag_norm = _texReaction[index];
//       if( max_mag_norm < _texReaction[index] )
// 	max_mag_norm = _texReaction[index];

//       if( _pbHeight/2-1 <= j && j <= _pbHeight/2+1 &&
// 	  _pbWidth/2 -1 <= i && i <= _pbWidth/2 +1 ) {
// 	cerr << _texReaction[index] << "  ";
//       }

      index += 4;
    }
  }

//   cerr << endl;
    
//   fprintf( stdout, "Rate unnormalized  %f %f\n", _vmag_min, _vmag_max );
//   fprintf( stdout, "Rate normalized    %f %f\n\n", min_mag_norm, max_mag_norm );

  updateTexture( _gl_ConstsTexID[REACTION], _texReaction );
}


//------------------------------------------------------------------------
// function    : updateConstantsDT() 
// description : update the texture that stores the DT.
//               NOTE: anisotropic diffusion based on vector field data
//------------------------------------------------------------------------
void ReactDiffuse::updateConstantsDT()
{
  // generate the diffusion tensor with gradients
  generateDiffusionTensor( true, 0 );

  unsigned int index = 0;

  for ( unsigned int j=0; j<_pbHeight; j++ ) {
    for ( unsigned int i=0; i<_pbWidth; i++ ) {

      // diffusion tensor
      _texDiffusion[index++] = _dt_data[j][i][0];
      _texDiffusion[index++] = _dt_data[j][i][1];
      _texDiffusion[index++] = _dt_data[j][i][2];
      _texDiffusion[index++] = _dt_data[j][i][3];
    }
  }

  updateTexture( _gl_ConstsTexID[DIFFUSION], _texDiffusion );

  // generate the diffusion tensor with gradients
  generateDiffusionTensor( true, 2 );
  
  index = 0;

  for ( unsigned int j=0; j<_pbHeight; j++ ) {
    for ( unsigned int i=0; i<_pbWidth; i++ ) {

      // diffusion tensor
      _texDiffusion2[index++] = _dt_data[j][i][0];
      _texDiffusion2[index++] = _dt_data[j][i][1];
      _texDiffusion2[index++] = _dt_data[j][i][2];
      _texDiffusion2[index++] = _dt_data[j][i][3];
    }
  }

  updateTexture( _gl_ConstsTexID[DIFFUSION2], _texDiffusion2 );
}


//------------------------------------------------------------------------
// function    : generateConstantsTex() 
// description : generate the texture that stores the
//               reaction_const0, reaction_const1 and reaction_const2
//               values of the reaction diffusion equations, the reaction
//               rates, and finite difference kernel for the diffusion
//               NOTE: we do anisotropic diffusion based on vector field
//                     data, and also modulate the reaction rates based on
//                     the vec data
//------------------------------------------------------------------------
void ReactDiffuse::generateConstantsTex()
{
  // allocate memory for the data
  _texMorphigens = new float [_pbWidth*_pbHeight*4];
  _texReaction   = new float [_pbWidth*_pbHeight*4];
  _texDiffusion  = new float [_pbWidth*_pbHeight*4];
  _texDiffusion2 = new float [_pbWidth*_pbHeight*4];
  _texVariance   = new float [_pbWidth*_pbHeight*4];
  _texRHS        = new float [_pbWidth*_pbHeight*4];
  _texResiduals  = new float [_pbWidth*_pbHeight*4];
  _texReduction  = new float [_pbWidth*_pbHeight*4];

  _reaction_const0_data = new float * [_pbHeight];
  _reaction_const1_data = new float * [_pbHeight];
  _reaction_const2_data = new float * [_pbHeight];

  _rr_data = new float * [_pbHeight];
  _dt_data = new float **[_pbHeight];

  if( _texMorphigens == 0 ||
      _texReaction   == 0 ||
      _texDiffusion  == 0 ||
      _texDiffusion2 == 0 ||
      _texVariance   == 0 ||
      _texRHS        == 0 ||
      _texResiduals  == 0 ||
      _texReduction  == 0 ||

      _reaction_const0_data == 0 ||
      _reaction_const1_data == 0 ||
      _reaction_const2_data == 0 ||

      _rr_data == 0 ||
      _dt_data == 0 ) {

    fprintf( stderr, "No temporary height memory %d\n", _pbHeight );
    exit( -1 );
  }

  for ( unsigned int j = 0; j < _pbHeight; j++ ) {
    _reaction_const0_data[j] = new float [_pbWidth];
    _reaction_const1_data[j] = new float [_pbWidth];
    _reaction_const2_data[j] = new float [_pbWidth];

    _rr_data[j] = new float  [_pbWidth];
    _dt_data[j] = new float *[_pbWidth];

    if( _reaction_const0_data[j] == 0 ||
	_reaction_const1_data[j] == 0 ||
	_reaction_const2_data[j] == 0 ||

	_rr_data[j] == 0 ||
	_dt_data[j] == 0) {
      fprintf( stderr, "No temporary width memory %d\n", _pbWidth );
      exit( -1 );
    }

    for ( unsigned int i = 0; i < _pbWidth; i++ ) {
      _dt_data[j][i] = new float[4];
      
      if( _dt_data[j][i] == 0 ) {
	fprintf( stderr, "No temporary depth memory %d %d\n", j, i );
	exit( -1 );
      }
    }
  }

  // seed the random number generator.
  long seed = -clock();

//  seed = -100;

  register int index;
  
  index = 0;
  // variance texture
  for ( unsigned int j=0; j<_pbHeight; j++ ) {
    for ( unsigned int i=0; i<_pbWidth; i++ ) {

      if( _reaction == GRAY_SCOTT ) {

	unsigned int offw = 12;
	unsigned int offh = 12;

	if( _pbWidth /2 - offw <= i &&  i < _pbWidth /2 + offw && 
	    _pbHeight/2 - offh <= j &&  j < _pbHeight/2 + offh ) {
	  _texVariance[index++] = 1.0;
	  _texVariance[index++] = 1.0;
	} else {
	  _texVariance[index++] = 0.0;
	  _texVariance[index++] = 2.0;
	}
	
	_texVariance[index++] = 1.0;
	_texVariance[index++] = 1.0 + 
	  frand( -_reaction_const_variance, _reaction_const_variance, seed );

      } else {
	_texVariance[index++] =
	  1.0 + frand( -_ab_variance, _ab_variance, seed );
	_texVariance[index++] =
	  1.0 + frand( -_ab_variance, _ab_variance, seed );

	_texVariance[index++] = 1.0 + 
	  frand( -_reaction_const_variance, _reaction_const_variance, seed );
	_texVariance[index++] = 1.0 + 
          frand( -_reaction_const_variance, _reaction_const_variance, seed );
      }
    }
  }

  updateTexture( _gl_ConstsTexID[VARIANCE], _texVariance );

  // initialize the reaction
  float v_scale = 1.0 / (_vmag_max -_vmag_min);
  float rc0_2 = _pbWidth  / 2;
  float rc1_2 = _pbHeight / 2;

  // Reaction Constants
  float rc0_range = 0.00;      /* Maximum random range reaction. */
  float rc1_range = 0.00;      /* Maximum random range reaction. */

  float ts, u;

  if( _solution == EULER_EXPLICIT )
    ts = _time_step;
  else
    ts = 1.0;

  index = 0;
  for ( unsigned int j=0; j<_pbHeight; j++ ) {
    for ( unsigned int i=0; i<_pbWidth; i++ ) {

      float variance2 = _texVariance[index + 2];
      float variance3 = _texVariance[index + 3];

      float di = (float) (i-rc0_2) / rc0_2;
      float dj = (float) (j-rc1_2) / rc1_2;


      // Radial stripes to spots.
      if( MAP_TOBY_PUFFER || PAPUA_TOBY_PUFFER ) {
	rc0_range = 0.00;      // Maximum random range reaction.
	rc1_range = 0.375;

	_texReaction[index++] = _reaction_const0_data[j][i] =
	  _reaction_const0; // * variance2 + rc0_range * di;

	float dr = sqrt( di*di + dj*dj);
	float db = 0.125;

	float dt = 2.0 *
	  fabs( sin( 8.0 * atan2( dj, di ) ) ) *
	  fabs( sin( 8.0 * atan2( dj, di ) ) );

	// 	  float dt = atan2( dj, di ) / 3.1415; // Range 0->1, -1->0
	// 	  if ( dt < 0 ) dt += 2.0; // Range 0 -> +2
	// 	  dt *= 8.0;               // Range 0 -> +16
	// 	  dt += 0.25;              // Range 0.25 -> +16.25 // off axis
	// 	  dt -= ((int) dt);        // Range 0->1 (mod)
	// 	  dt *= 2.0;               // Range 0->2 (mod)


	if( PAPUA_TOBY_PUFFER ) { // PAPUA TOBY PUFFER
	  if( dj > -.3 ) {
	    if( dr < db )
	      _texReaction[index++] = _reaction_const1_data[j][i] =
		_reaction_const1 * variance3 +
		-rc1_range * dr/db;

	    else if( dr < 2.0*db )
	      _texReaction[index++] = _reaction_const1_data[j][i] =
		_reaction_const1 * variance3 +
		-rc1_range + (dr-db)/db * rc1_range;

	    else
	      _texReaction[index++] = _reaction_const1_data[j][i] =
		_reaction_const1 * variance3 +
		-rc1_range * dt;
	  } else {
	    _texReaction[index++] = _reaction_const1_data[j][i] =
	      _reaction_const1 * variance3;
	  }

	} else if( MAP_TOBY_PUFFER ) { // MAP TOBY PUFFER

	  if( dr < db )
	    _texReaction[index++] = _reaction_const1_data[j][i] =
	      _reaction_const1 * variance3 +
	      -rc1_range * dr/db;

	  else if( dr < 2.0*db )
	    _texReaction[index++] = _reaction_const1_data[j][i] =
	      _reaction_const1 * variance3 +
	      -rc1_range + (dr-db)/db * rc1_range;

	  else if( dr < 1.0 ) {
	    _texReaction[index++] = _reaction_const1_data[j][i] =
	      _reaction_const1 * variance3 +
	      -rc1_range * dt;
	  } else {
	    _texReaction[index++] = _reaction_const1_data[j][i] =
	      _reaction_const1 * variance3;
	  }
	}
      } else if( ZEBRA_GOBY ) {
	rc0_range = 0.00;      // Maximum random range reaction.
	rc1_range = 0.375;

	_texReaction[index++] = _reaction_const0_data[j][i] =
	  _reaction_const0 + rc0_range * di;

	if( 0 && ( -0.75 < di && di < -0.25 && 0.25 < dj && dj < 0.75 ) )
	  _texReaction[index++] = _reaction_const1_data[j][i] =
	    _reaction_const1 + rc1_range/2;
	else
	  _texReaction[index++] = _reaction_const1_data[j][i] =
	    _reaction_const1 + rc1_range * dj;

	  
	// Circular stripes to spots.
      } else if( SPOTTED_PUFFER ) {
	rc0_range = 0.00;      // Maximum random range reaction
	rc1_range = 0.50;      

	_texReaction[index++] = _reaction_const0_data[j][i] =
	  _reaction_const0; // * variance2 + rc0_range * di;

	float dr = sqrt( di*di + dj*dj);
	float db = 0.25;

	if( dr < db )
	  _texReaction[index++] = _reaction_const1_data[j][i] =
	    _reaction_const1 // * variance3 +
	    -rc1_range * dr/db;

	else if( dr < 2.0*db )
	  _texReaction[index++] = _reaction_const1_data[j][i] =
	    _reaction_const1 // * variance3 +
	    -rc1_range * (1.0 - (dr-db)/db);

	else
	  _texReaction[index++] = _reaction_const1_data[j][i] =
	    _reaction_const1 * variance3;

      } else {

	// reaction_const0 value  ->  +-variance
	_texReaction[index++] = _reaction_const0_data[j][i] =
	  _reaction_const0 * variance2 + rc0_range * di;

	// reaction_const1 value  ->  +-variance
	_texReaction[index++] = _reaction_const1_data[j][i] =
	  _reaction_const1 * variance3 + rc1_range * dj;
      }

      //reaction_const2 value  ->  +-variance
      _texReaction[index++] = _reaction_const2_data[j][i] =
	_reaction_const2 * (variance2+variance3)/2.0;


      // reaction rate
      if( _v_data[j][i][3] > 0 ) {

	u = (_v_data[j][i][3]-_vmag_min) * v_scale;

	if( _reaction == TURING )
	  _texReaction[index++] = _rr_data[j][i] =
	    ts / (_rr_coef_1 + _rr_coef_2 * u);
	else
	  _texReaction[index++] = _rr_data[j][i] =
	    ts * (_rr_coef_1 + _rr_coef_2 * u );
      } else {
	_texReaction[index++] = _rr_data[j][i] = 0;
      }
    }
  }

  updateTexture( _gl_ConstsTexID[REACTION], _texReaction  );

  // generate the diffusion tensor
  generateDiffusionTensor( false, 0 );

  index = 0;
  for ( unsigned int j=0; j<_pbHeight; j++ ) {
    for ( unsigned int i=0; i<_pbWidth; i++ ) {
      _texDiffusion [index  ] = _dt_data[j][i][0];
      _texDiffusion2[index++] = _dt_data[j][i][0];
      _texDiffusion [index  ] = _dt_data[j][i][1];
      _texDiffusion2[index++] = _dt_data[j][i][1];
      _texDiffusion [index  ] = _dt_data[j][i][2];
      _texDiffusion2[index++] = _dt_data[j][i][2];
      _texDiffusion [index  ] = _dt_data[j][i][3];
      _texDiffusion2[index++] = _dt_data[j][i][3];
    }
  }

  updateTexture( _gl_ConstsTexID[DIFFUSION],  _texDiffusion  );
  updateTexture( _gl_ConstsTexID[DIFFUSION2], _texDiffusion2 );

  updateTexture( _gl_MorphogenTexID[0], 0 ); // _texMorphogen );
  updateTexture( _gl_MorphogenTexID[1], 0 ); // _texMorphogen );
  updateTexture( _gl_RHSTexID,          0 ); // _texRHS );
  updateTexture( _gl_ResidualsTexID,    0 ); // _texResiduals );
  updateTexture( _gl_ReductionTexID[0], 0 ); // _texReduction );
  updateTexture( _gl_ReductionTexID[1], 0 ); // _texReduction );

  float min_mag_norm[4], max_mag_norm[4];

  for ( unsigned int t=0; t<4; t++ ) {
    min_mag_norm[t] =  MAX_FLOAT;
    max_mag_norm[t] = -MAX_FLOAT;
  }
}


//------------------------------------------------------------------------
// function    : generateDiffusionTensor()
// description : generate the finite difference diffusion kernel for 
//               anisotropic diffusion based on a vector field
//------------------------------------------------------------------------
void ReactDiffuse::generateDiffusionTensor( bool gradient, unsigned int var )
{ 
  float min_mag_norm[4], max_mag_norm[4];

  for( unsigned int t=0; t<4; t++ ) {
    min_mag_norm[t] =  MAX_FLOAT;
    max_mag_norm[t] = -MAX_FLOAT;
  }

  if( gradient ) {
    // get the texture values
    getTexValues();

    unsigned int width4 = _pbWidth*4;

    unsigned int height_1 = _pbHeight - 1;
    unsigned int width_1  = _pbWidth  - 1;

    float dg, grad[3], dotSign = 1.0;

    if( _reaction == TURING ||
	_reaction == BRUSSELATOR )
      dotSign = -1.0;

    // Do just the interior. The boundary does not matter.
    for ( unsigned int j=width4, jj=1; jj<height_1; j+=width4, jj++ ) {
      for ( unsigned int i=var+j+4, ii=1; ii<width_1; i+=4, ii++ ) {
	dg = 0;

	// Calculate the gradient using central differences
	// of the a morphigen.
	grad[0] = _texMorphigens[i+     4] -  _texMorphigens[i-     4];
	grad[1] = _texMorphigens[i+width4] -  _texMorphigens[i-width4];

	if( fabs(grad[0]) > MIN_FLOAT || fabs(grad[1]) > MIN_FLOAT ) {
	  grad[2] = sqrt(grad[0]*grad[0]+grad[1]*grad[1]);
	  
	  // Get the dot product of the morphigen gradient and
	  // the vector field. This is normalized (-1,1).
	  float dotProd = dotSign * (grad[0] * _v_data[jj][ii][0] +
				     grad[1] * _v_data[jj][ii][1]) / (grad[2]);

	  // Depending on the dot product change the diffusion.
	  dg = gammp_table[ INDEX(dotProd, MAX_GAMMP) ];
	}
    
	// setup the principal diffusivity matrix
	float pd00 = _diff_coef_1 + dg;
	float pd11 = _diff_coef_2 - dg;

	// square the principal diffusivity matrix so it is positive
	pd00 *= pd00;
	pd11 *= pd11;

	float cos_ang = _v_data[jj][ii][0];
	float sin_ang = _v_data[jj][ii][1];
      
	float cos2 = cos_ang*cos_ang;
	float sin2 = sin_ang*sin_ang;

	if( _reaction == TURING && _laplacian == UNIFORM ) {
	  // calculate the diffusion matrix for
	  // uniform anisotropic diffusion

	  // NOTE: premultiple the secondary diffusivity values by 1/4
	  // This is so that it does not need to be done when calculating
	  // the finite differences.
	  _dt_data[jj][ii][0] = (pd00 * cos2 + pd11 * sin2);
	  _dt_data[jj][ii][1] =
	  _dt_data[jj][ii][2] = (pd00 - pd11) * cos_ang * sin_ang * 0.25;
	  _dt_data[jj][ii][3] = (pd00 * sin2 + pd11 * cos2);
	} else {
	  // calculate the diffusion matrix for
	  // inhomogeneous anisotropic diffusion.

	  // NOTE: premultiple the principal diffusivity values by 1/2
	  // NOTE: premultiple the secondary diffusivity values by 1/4
	  // This is so that it does not need to be done when
	  // calculating the finite differences.
	  _dt_data[jj][ii][0] = (pd00 * cos2 + pd11 * sin2) * 0.5;
	  _dt_data[jj][ii][1] =
	  _dt_data[jj][ii][2] = (pd00 - pd11) * cos_ang * sin_ang * 0.25;
	  _dt_data[jj][ii][3] = (pd00 * sin2 + pd11 * cos2) * 0.5;
	}
      }
    }
  } else {

    for ( unsigned int jj=0; jj<_pbHeight; jj++ ) {
      for ( unsigned int ii=0; ii<_pbWidth; ii++ ) {
	
	// setup the principal diffusivity matrix for diffusion
	float pd00 = _diff_coef_1;
	float pd11 = _diff_coef_2;
    
	// square the principal diffusivity matrix so it is positive
	pd00 *= pd00;
	pd11 *= pd11;

	float cos_ang = _v_data[jj][ii][0];
	float sin_ang = _v_data[jj][ii][1];
      
	float cos2 = cos_ang*cos_ang;
	float sin2 = sin_ang*sin_ang;

	if( _reaction == TURING && _laplacian == UNIFORM ) {
	  // calculate the diffusion matrix for
	  // uniform anisotropic diffusion

	  // NOTE: premultiple the secondary diffusivity values by 1/4
	  // This is so that it does not need to be done when calculating
	  // the finite differences.
	  _dt_data[jj][ii][0] = (pd00 * cos2 + pd11 * sin2);
	  _dt_data[jj][ii][1] =
	  _dt_data[jj][ii][2] = (pd00 - pd11) * cos_ang * sin_ang * 0.25;
	  _dt_data[jj][ii][3] = (pd00 * sin2 + pd11 * cos2);
	} else {
	  // calculate the diffusion matrix for
	  // inhomogeneous anisotropic diffusion.

	  // NOTE: premultiple the principal diffusivity values by 1/2
	  // NOTE: premultiple the secondary diffusivity values by 1/4
	  // This is so that it does not need to be done when
	  // calculating the finite differences.
	  _dt_data[jj][ii][0] = (pd00 * cos2 + pd11 * sin2) * 0.5;
	  _dt_data[jj][ii][1] =
	  _dt_data[jj][ii][2] = (pd00 - pd11) * cos_ang * sin_ang * 0.25;
	  _dt_data[jj][ii][3] = (pd00 * sin2 + pd11 * cos2) * 0.5;
	}
      }
    }
  }
}


//------------------------------------------------------------------------
// function    : createVectorData() 
// description : create the vector data
//------------------------------------------------------------------------
void ReactDiffuse::createVectorData( )
{
  _vmag_min = +MAX_FLOAT;
  _vmag_max = -MAX_FLOAT;

  if( _v_data ) {
    for ( unsigned int j=0; j<_pbHeight; j++ ) {
      for ( unsigned int i=0; i<_pbWidth; i++ )
	delete[] _v_data[j][i];

      delete[] _v_data[j];
    }

    delete[] _v_data;
  }

  float mid = 64;

  unsigned int dims[3];

  dims[0] = 1;
  dims[1] = dims[2] = 2 * (unsigned int) mid;

  // set the fbuffer width and height based on the data
  _pbHeight = dims[1];
  _pbWidth  = dims[2];


  // allocate memory for the data
  _v_data = new float **[_mult*_pbHeight];

  // read in the data -- NOTE: this is 3D data!
  for ( unsigned int j = 0; j < _mult*_pbHeight; j++ ) {
    _v_data[j] = new float *[_mult*_pbWidth];
      
    for ( unsigned int i = 0; i < _mult*_pbWidth; i++ ) {
      _v_data[j][i] = new float [4];
	
      if( j < _pbHeight && i < _pbWidth ) {
 
	_v_data[j][i][0] =  ((float) j - mid) / mid;
	_v_data[j][i][1] = -((float) i - mid) / mid;
	_v_data[j][i][2] = 0;
	_v_data[j][i][3] =
	  sqrt( _v_data[j][i][0] * _v_data[j][i][0] +
		_v_data[j][i][1] * _v_data[j][i][1] +
		_v_data[j][i][2] * _v_data[j][i][2] );

	if( _vmag_min > _v_data[j][i][3] )  _vmag_min = _v_data[j][i][3];
	if( _vmag_max < _v_data[j][i][3] )  _vmag_max = _v_data[j][i][3];

	if( _v_data[j][i][3] < MIN_FLOAT ) {
	  _v_data[j][i][0] = 0;
	  _v_data[j][i][1] = 1;
	  _v_data[j][i][3] = MIN_FLOAT;
	} else {
	  _v_data[j][i][0] /= _v_data[j][i][3];
	  _v_data[j][i][1] /= _v_data[j][i][3];
	}
      }
    }
  }

  if( _mult > 1 ) {
    dims[1] *= _mult;
    dims[2] *= _mult;

    for (int jj=dims[1]-1; jj>=0; jj--) {
      
      unsigned int j0 = (int) (jj / _mult);
      unsigned int j1 = (j0+1) % _pbHeight;
      
      float v = (float) (jj % _mult) / (float) (_mult);
      
      for (int ii=dims[2]-1; ii>=0; ii--) {

	unsigned int i0 = (int) (ii / _mult);
	unsigned int i1 = (i0+1) % _pbWidth;
	  
	float u = (float) (ii % _mult) / (float) (_mult);

	for( unsigned int index=0; index<4; index++ )
	  _v_data[jj][ii][index] =
	    LERP( v,
		  LERP( u, _v_data[j0][i0][index], _v_data[j0][i1][index] ),
		  LERP( u, _v_data[j1][i0][index], _v_data[j1][i1][index] ) );
      }
    }

    _pbHeight *= _mult;
    _pbWidth  *= _mult;
  }

//  _pbHeight = dims[1] = 98;
//  _pbWidth  = dims[2] = 98;

  /* Min - max are the same. */
  if( _vmag_max-_vmag_min < MIN_FLOAT ) {
    _vmag_min -= 1.0;
    _vmag_max += 1.0;
  }

  /* Min - max have a small range. */
  else if( _vmag_max-_vmag_min < 1.0e-4 ) {
    float ave  = (_vmag_max+_vmag_min) / 2.0;
    float diff = (_vmag_max-_vmag_min);

    _vmag_min = ave - 1.0e3 * diff;
    _vmag_max = ave + 1.0e3 * diff;
  }
}

//------------------------------------------------------------------------
// function    : generateDisplayLists() 
// description : generate the display lists for the updateState()
//               rendering pass -- NOTE: the texture coordinates are 
//               used in the frag shader to access the constants texture
//
//               the number of texture coords passed in can vary depending
//               on the card used. the nv30 series allows for 4 texture
//               coords. passing them in elimates having to compute the 
//               coords every time the frag shader is used
//------------------------------------------------------------------------
void ReactDiffuse::generateDisplayLists()
{
  if ( glIsList( _gl_dlAll ) == GL_FALSE ) {
    // create the gl lists
    _gl_dlAll = glGenLists( 6 );

    if( _gl_dlAll == 0 ) {
      GLenum errorCode;
      if ((errorCode = glGetError()) != GL_NO_ERROR) 
	fprintf( stderr, "glGenLists(): ERROR: %s\n",
		 gluErrorString(errorCode) );
    }

    _gl_dlAllZeroFlux = _gl_dlAll + 1;

    _gl_dlInteriorZeroFlux = _gl_dlAllZeroFlux + 1;
    _gl_dlBoundaryZeroFlux = _gl_dlInteriorZeroFlux + 1;

    _gl_dlInteriorPeriodic = _gl_dlBoundaryZeroFlux + 1;
    _gl_dlBoundaryPeriodic = _gl_dlInteriorPeriodic + 1;
  }

  //----------------------------------------------------------------------
  // _dlAll rendering mode
  //
  //----------------------------------------------------------------------
  glNewList( _gl_dlAll, GL_COMPILE );

  // render 4 boundary edges 
  //  NOTE: need to render this as quads to ensure correct behavior!
  glBegin( GL_QUADS );
  glMultiTexCoord2fARB(GL_TEXTURE0_ARB,         0,        0);
  glVertex2f( 0, 0 );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB, _pbWidth,         0);
  glVertex2f( _pbWidth, 0 );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB, _pbWidth, _pbHeight);
  glVertex2f( _pbWidth, _pbHeight );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB,        0, _pbHeight);
  glVertex2f( 0, _pbHeight );
  glEnd();

  glEndList();

  //
  // this is setup to use 3 texture coordinates to store the
  // coordinates of a pixel's neighbors and the offset into
  // the other quads.
  //
  //        -------------
  //        | 0 |   |   |
  //        -------------
  //        |   | 1 |   |     NEIGHBORHOOD
  //        -------------
  //        |   |   | 2 |
  //        -------------
  //                     -------------
  //                     | 3 |   |   |
  //                     -------------
  //                     |   | 4 |   |     NEIGHBORHOOD
  //                     -------------
  //                     |   |   | 5 |
  //                     -------------
  //
  //  GL_TEXTURE0_ARB --> x,y for 0 
  //  GL_TEXTURE1_ARB --> x,y for 1
  //  GL_TEXTURE2_ARB --> x,y for 2
  //                GL_TEXTURE0_ARB --> z,w for 3 + _pbWidth, _pbHeight 
  //                GL_TEXTURE1_ARB --> z,w for 4 + _pbWidth, _pbHeight 
  //                GL_TEXTURE2_ARB --> z,w for 5 + _pbWidth, _pbHeight 
  //
  // these neighbor coordinates are used in the frag shader in the 
  //   diffusion calculation, and eliminates calculating the coords
  //   every time the frag shader is run
  // if working with a card that has more texture coordinates available
  //   coordinates could be added to precompute the coordinates of 
  //   the access to the consts tex (for now this is done in the frag
  //   shader)
  //

  //----------------------------------------------------------------------
  // interior rendering mode
  //----------------------------------------------------------------------
  glNewList( _gl_dlAllZeroFlux, GL_COMPILE );

  // render with an offset of one pixel border (for both position and texture)
  //  NOTE: need to render this as quads to ensure correct behavior!
  glBegin( GL_QUADS );
  glMultiTexCoord2fARB(GL_TEXTURE0_ARB,         -1,           1);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB,          0,           0);
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB,          1,          -1);
  glVertex2f( 0, 0 );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB, _pbWidth-1,           1);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB, _pbWidth  ,           0);
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB, _pbWidth+1,          -1);
  glVertex2f( _pbWidth, 0 );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB, _pbWidth-1, _pbHeight+1);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB, _pbWidth  , _pbHeight  );
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB, _pbWidth+1, _pbHeight-1);
  glVertex2f( _pbWidth, _pbHeight );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB,         -1, _pbHeight+1);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB,          0, _pbHeight  );
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB,          1, _pbHeight-1);
  glVertex2f( 0, _pbHeight );
  glEnd();

  glEndList();


  //----------------------------------------------------------------------
  // Periodic interior rendering mode
  //----------------------------------------------------------------------
  glNewList( _gl_dlInteriorPeriodic, GL_COMPILE );

  // render with an offset of one pixel border (for both position and texture)
  //  NOTE: need to render this as quads to ensure correct behavior!
  glBegin( GL_QUADS );
  glMultiTexCoord2fARB(GL_TEXTURE0_ARB,          0,           2);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB,          1,           1);
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB,          2,           0);
  glVertex2f( 1, 1 );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB, _pbWidth-2,           2);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB, _pbWidth-1,           1);
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB, _pbWidth  ,           0);
  glVertex2f( _pbWidth-1, 1 );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB, _pbWidth-2, _pbHeight  );
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB, _pbWidth-1, _pbHeight-1);
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB, _pbWidth,   _pbHeight-2);
  glVertex2f( _pbWidth-1, _pbHeight-1 );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB,          0, _pbHeight  );
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB,          1, _pbHeight-1);
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB,          2, _pbHeight-2);
  glVertex2f( 1, _pbHeight-1 );
  glEnd();

  glEndList();


  //----------------------------------------------------------------------
  // boundary rendering mode
  //
  // NOTE --> using periodic boundary conditions!!!!
  //----------------------------------------------------------------------
  glNewList( _gl_dlBoundaryPeriodic, GL_COMPILE );

  // render 4 boundary edges 
  //  NOTE: need to render this as quads to ensure correct behavior!
  glBegin( GL_QUADS );

  // top edge
  glMultiTexCoord2fARB(GL_TEXTURE0_ARB,          0,           0);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB,          1, _pbHeight-1);
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB,          2, _pbHeight-2);
  glVertex2f( 1,_pbHeight-1 );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB, _pbWidth-2,           0);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB, _pbWidth-1, _pbHeight-1);
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB, _pbWidth,   _pbHeight-2);
  glVertex2f( _pbWidth-1,_pbHeight-1 );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB, _pbWidth-2,           1);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB, _pbWidth-1, _pbHeight  );
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB, _pbWidth,   _pbHeight-1);
  glVertex2f( _pbWidth-1, _pbHeight );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB,          0,           1);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB,          1, _pbHeight  );
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB,          2, _pbHeight-1);
  glVertex2f( 1, _pbHeight );

  // bottom edge
  glMultiTexCoord2fARB(GL_TEXTURE0_ARB,          0,           1);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB,          1,           0);
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB,          2, _pbHeight-1);
  glVertex2f( 1, 0 );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB, _pbWidth-2,           1);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB, _pbWidth-1,           0);
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB, _pbWidth,   _pbHeight-1);
  glVertex2f( _pbWidth-1, 0 );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB, _pbWidth-2,           2);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB, _pbWidth-1,           1);
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB, _pbWidth,   _pbHeight  );
  glVertex2f( _pbWidth-1, 1 );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB,          0,           2);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB,          1,           1);
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB,          2, _pbHeight  );
  glVertex2f( 1, 1 );


  // left edge
  glMultiTexCoord2fARB(GL_TEXTURE0_ARB, _pbWidth-1,           2);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB,          0,           1);
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB,          1,           0);
  glVertex2f( 0, 1 );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB, _pbWidth,             2);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB,          1,           1);
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB,          2,           0);
  glVertex2f( 1, 1 );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB, _pbWidth,   _pbHeight  );
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB,          1, _pbHeight-1);
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB,          2, _pbHeight-2);
  glVertex2f( 1, _pbHeight-1 );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB, _pbWidth-1, _pbHeight  );
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB,          0, _pbHeight-1);
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB,          1, _pbHeight-2);
  glVertex2f( 0, _pbHeight-1 );


  // right edge
  glMultiTexCoord2fARB(GL_TEXTURE0_ARB, _pbWidth-2,           2);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB, _pbWidth-1,           1);
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB,          0,           0);
  glVertex2f( _pbWidth-1, 1 );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB, _pbWidth-1,           2);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB, _pbWidth,             1);
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB,          1,           0);
  glVertex2f( _pbWidth,   1 );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB, _pbWidth-1, _pbHeight  );
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB, _pbWidth,   _pbHeight-1);
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB,          1, _pbHeight-2);
  glVertex2f( _pbWidth,   _pbHeight-1 );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB, _pbWidth-2, _pbHeight  );
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB, _pbWidth-1, _pbHeight-1);
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB,          0, _pbHeight-2);
  glVertex2f( _pbWidth-1, _pbHeight-1 );


  // top left corner
  glMultiTexCoord2fARB(GL_TEXTURE0_ARB, _pbWidth-1,           0);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB,          0, _pbHeight-1);
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB,          1, _pbHeight-2);
  glVertex2f( 0, _pbHeight-1 );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB, _pbWidth,             0);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB,        1,   _pbHeight-1);
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB,        2,   _pbHeight-2);
  glVertex2f( 1, _pbHeight-1 );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB, _pbWidth,             1);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB,        1,   _pbHeight  );
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB,        2,   _pbHeight-1);
  glVertex2f( 1, _pbHeight );
 
  glMultiTexCoord2fARB(GL_TEXTURE0_ARB, _pbWidth-1,           1);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB,          0, _pbHeight  );
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB,          1, _pbHeight-1);
  glVertex2f( 0, _pbHeight );

  // top right corner
  glMultiTexCoord2fARB(GL_TEXTURE0_ARB, _pbWidth-2,           0);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB, _pbWidth-1, _pbHeight-1);
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB,          0, _pbHeight-2);
  glVertex2f( _pbWidth-1, _pbHeight-1 );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB, _pbWidth-1,           0);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB, _pbWidth,   _pbHeight-1);
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB,          1, _pbHeight-2);
  glVertex2f( _pbWidth, _pbHeight-1 );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB, _pbWidth-1,           1);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB, _pbWidth,   _pbHeight  );
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB,          1, _pbHeight-1);
  glVertex2f( _pbWidth, _pbHeight );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB, _pbWidth-2,           1);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB, _pbWidth-1, _pbHeight  );
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB,          0, _pbHeight-1);
  glVertex2f( _pbWidth-1, _pbHeight );

  // bottom left corner
  glMultiTexCoord2fARB(GL_TEXTURE0_ARB, _pbWidth-1,           1);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB,          0,           0);
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB,          1, _pbHeight-1);
  glVertex2f( 0, 0 );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB, _pbWidth,             1);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB,          1,           0);
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB,          2, _pbHeight-1);
  glVertex2f( 1, 0 );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB, _pbWidth,             2);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB,          1,           1);
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB,          2, _pbHeight  );
  glVertex2f( 1, 1 );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB, _pbWidth-1,           2);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB,          0,           1);
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB,          1, _pbHeight  );
  glVertex2f( 0, 1 );

  // bottom right corner
  glMultiTexCoord2fARB(GL_TEXTURE0_ARB, _pbWidth-2,           1);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB, _pbWidth-1,           0);
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB,          0, _pbHeight-1);
  glVertex2f( _pbWidth-1, 0 );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB, _pbWidth-1,           1);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB, _pbWidth,             0);
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB,          1, _pbHeight-1);
  glVertex2f( _pbWidth,   0 );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB, _pbWidth-1,           2);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB, _pbWidth,             1);
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB,          1, _pbHeight  );
  glVertex2f( _pbWidth,   1 );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB, _pbWidth-2,           2);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB, _pbWidth-1,           1);
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB,          0, _pbHeight  );
  glVertex2f( _pbWidth-1, 1 );

  glEnd(); 

  glEndList();


  //----------------------------------------------------------------------
  // Zero Flux interior rendering mode
  //----------------------------------------------------------------------
  glNewList( _gl_dlInteriorZeroFlux, GL_COMPILE );

  unsigned int n   = 1;
  unsigned int n_1 = n - 1;
  unsigned int n1  = n + 1;

  // render with an offset of one pixel border (for both position and texture)
  //  NOTE: need to render this as quads to ensure correct behavior!
  glBegin( GL_QUADS );
  glMultiTexCoord2fARB(GL_TEXTURE0_ARB,        n_1,          n1);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB,          n,           n);
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB,         n1,         n_1);
  glVertex2f( n, n );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB, _pbWidth-n1,         n1);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB, _pbWidth-n,           n);
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB, _pbWidth-n_1,       n_1);
  glVertex2f( _pbWidth-n, n );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB, _pbWidth-n1,  _pbHeight-n_1);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB, _pbWidth-n,   _pbHeight-n  );
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB, _pbWidth-n_1, _pbHeight-n1 );
  glVertex2f( _pbWidth-n, _pbHeight-n );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB,        n_1, _pbHeight-n_1);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB,          n, _pbHeight-n  );
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB,         n1, _pbHeight-n1 );
  glVertex2f( n, _pbHeight-n );
  glEnd();

  glEndList();


  //----------------------------------------------------------------------
  // boundary rendering mode
  //
  // NOTE --> using zero flux boundary conditions!!!!
  //----------------------------------------------------------------------
  glNewList( _gl_dlBoundaryZeroFlux, GL_COMPILE );

  // render 4 boundary edges 
  //  NOTE: need to render this as quads to ensure correct behavior!
  glBegin( GL_QUADS );

  // top edge
  glMultiTexCoord2fARB(GL_TEXTURE0_ARB,         -1, _pbHeight-n_1);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB,          0, _pbHeight-n  );
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB,          1, _pbHeight-n1 );
  glVertex2f(        0, _pbHeight-n );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB, _pbWidth-1, _pbHeight-n_1);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB, _pbWidth  , _pbHeight-n  );
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB, _pbWidth+1, _pbHeight-n1 );
  glVertex2f( _pbWidth, _pbHeight-n );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB, _pbWidth-1, _pbHeight+1);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB, _pbWidth  , _pbHeight  );
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB, _pbWidth+1, _pbHeight-1);
  glVertex2f( _pbWidth, _pbHeight );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB,         -1, _pbHeight+1);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB,          0, _pbHeight  );
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB,          1, _pbHeight-1);
  glVertex2f(        0, _pbHeight );


  // bottom edge
  glMultiTexCoord2fARB(GL_TEXTURE0_ARB,         -1,           1);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB,          0,           0);
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB,          1,          -1);
  glVertex2f(        0, 0 );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB, _pbWidth-1,           1);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB, _pbWidth  ,           0);
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB, _pbWidth+1,          -1);
  glVertex2f( _pbWidth, 0 );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB, _pbWidth-1,           n1 );
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB, _pbWidth ,            n  );
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB, _pbWidth+1,           n_1);
  glVertex2f( _pbWidth, n );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB,         -1,           n1 );
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB,          0,           n  );
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB,          1,           n_1);
  glVertex2f(        0, n );


  // left edge
  glMultiTexCoord2fARB(GL_TEXTURE0_ARB,         -1,           1);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB,          0,           0);
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB,          1,          -1);
  glVertex2f( 0, 0 );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB,        n_1,           1);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB,          n,           0);
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB,         n1,          -1);
  glVertex2f( n, 0 );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB,        n_1, _pbHeight+1);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB,          n, _pbHeight  );
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB,         n1, _pbHeight-1);
  glVertex2f( n, _pbHeight );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB,         -1, _pbHeight+1);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB,          0, _pbHeight  );
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB,          1, _pbHeight-1);
  glVertex2f( 0, _pbHeight );


  // right edge
  glMultiTexCoord2fARB(GL_TEXTURE0_ARB, _pbWidth-n1,          1);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB, _pbWidth-n,           0);
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB, _pbWidth-n_1,        -1);
  glVertex2f( _pbWidth-n, 0 );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB, _pbWidth-1,           1);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB, _pbWidth,             0);
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB, _pbWidth+1,          -1);
  glVertex2f( _pbWidth,   0 );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB, _pbWidth-1, _pbHeight+1);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB, _pbWidth,   _pbHeight  );
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB, _pbWidth+1, _pbHeight-1);
  glVertex2f( _pbWidth,   _pbHeight );

  glMultiTexCoord2fARB(GL_TEXTURE0_ARB, _pbWidth-n1,  _pbHeight+1);
  glMultiTexCoord2fARB(GL_TEXTURE1_ARB, _pbWidth-n,   _pbHeight  );
  glMultiTexCoord2fARB(GL_TEXTURE2_ARB, _pbWidth-n_1, _pbHeight-1);
  glVertex2f( _pbWidth-n, _pbHeight );
  glEnd();

  glEndList();

#ifdef OGL_ERROR_CHECKING
  GLenum errorCode;
  if ((errorCode = glGetError()) != GL_NO_ERROR) 
    fprintf( stderr, "generateDisplayLists(): ERROR: %s\n",
	     gluErrorString(errorCode) );
#endif
}


//------------------------------------------------------------------------
// function    : updateStateExplicit()
// description : iteration function that computes the reaction diffusion
//               equations for the next time step
//------------------------------------------------------------------------
void ReactDiffuse::updateStateExplicit( unsigned int num_passes )
{
  if( _last_solution != _solution ) {
    _last_solution = _solution;

    updateConstantsReact();

    if( _reaction == TURING )
      _cg_Program = _cg_Explicit[_reaction+_laplacian];
    else
      _cg_Program = _cg_Explicit[_reaction];
    
    // setup the link between this code and the CG program variables so 
    // that we can modify the inputs on the fly
    _cg_Morphigens = cgGetNamedParameter( _cg_Program, "morphigens" );
    
    for( unsigned int i=REACTION; i<=DIFFUSION2; i++ )
      _cg_ConstsTex[i] =
	cgGetNamedParameter( _cg_Program, constant_names[i].c_str() );
  
    _cg_DiffRates = cgGetNamedParameter( _cg_Program, "diffRates" );
    _cg_MixRates  = cgGetNamedParameter( _cg_Program, "mixingRate" );
  }

  // render a quad to update the new state --> here the front buffer is
  //  the old (current) timestep, and the back buffer is where the next
  //  (new) timestep will be computed --> the front buffer is bound as
  //  an input texture...
  _fbuffer->enable();

  cgGLEnableProfile( _cg_Profile );

  cgGLBindProgram( _cg_Program );
  
  cgGLSetParameter4f( _cg_DiffRates,
		      _time_step * _a_diff_rate,
		      _time_step * _b_diff_rate,
		      _time_step * _c_diff_rate,
		      _time_step * _d_diff_rate );

  cgGLSetParameter1f( _cg_MixRates, _mixing_rate );

  for( unsigned int i=REACTION; i<=DIFFUSION2; i++ )
    cgGLSetTextureParameter( _cg_ConstsTex[i], _gl_ConstsTexID[i] );  

  for( unsigned int i=REACTION; i<=DIFFUSION2; i++ )
    cgGLEnableTextureParameter( _cg_ConstsTex[i] );

 _fbuffer->attach( _gl_MorphogenTexID[0], 0 );
 _fbuffer->attach( _gl_MorphogenTexID[1], 1 );

  if( _boundary == ZERO_FLUX_ALL || _boundary == ZERO_FLUX_VAR ) {
    for ( unsigned int i=0; i<num_passes/2; i++ ) {
      cgGLSetTextureParameter( _cg_Morphigens, _gl_MorphogenTexID[0] );
      cgGLEnableTextureParameter( _cg_Morphigens );
      glDrawBuffer( GL_COLOR_ATTACHMENT1_EXT );
      glCallList( _gl_dlAllZeroFlux );

      cgGLSetTextureParameter( _cg_Morphigens, _gl_MorphogenTexID[1] );
      cgGLEnableTextureParameter( _cg_Morphigens );
      glDrawBuffer( GL_COLOR_ATTACHMENT0_EXT );
      glCallList( _gl_dlAllZeroFlux );
    }
  } else {
    for ( unsigned int i=0; i<num_passes/2; i++ ) {
      cgGLSetTextureParameter( _cg_Morphigens, _gl_MorphogenTexID[0] );
      cgGLEnableTextureParameter( _cg_Morphigens );
      glDrawBuffer( GL_COLOR_ATTACHMENT1_EXT );
      glCallList( _gl_dlInterior );
      glCallList( _gl_dlBoundary );

      cgGLSetTextureParameter( _cg_Morphigens, _gl_MorphogenTexID[1] );
      cgGLEnableTextureParameter( _cg_Morphigens );
      glDrawBuffer( GL_COLOR_ATTACHMENT0_EXT );
      glCallList( _gl_dlInterior );
      glCallList( _gl_dlBoundary );
    }
  }

  cgGLDisableTextureParameter( _cg_Morphigens );
  for( unsigned int i=REACTION; i<=DIFFUSION2; i++ )
    cgGLDisableTextureParameter( _cg_ConstsTex[i] );

  cgGLDisableProfile( _cg_Profile );

  _fbuffer->disable();

  // Update the diffusion tensor which will calculate the gradient.
  if( _gradient )
    updateConstantsDT();
}


//------------------------------------------------------------------------
// function    : getTexValues()
// description : Get the values from a particular buffer
//               
//------------------------------------------------------------------------
void ReactDiffuse::getTexValues()
{
  glEnable(GL_TEXTURE_RECTANGLE_NV);
  glBindTexture( GL_TEXTURE_RECTANGLE_NV, _gl_MorphogenTexID[0] );
  glGetTexImage( GL_TEXTURE_RECTANGLE_NV, 0, GL_RGBA, GL_FLOAT, _texMorphigens );
  glBindTexture(GL_TEXTURE_RECTANGLE_NV, 0);
  glDisable(GL_TEXTURE_RECTANGLE_NV);
}
