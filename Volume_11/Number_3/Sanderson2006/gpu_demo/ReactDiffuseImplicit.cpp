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

using namespace std;

//------------------------------------------------------------------------
// function    : updateStateImplicit()
// description : iteration function that computes the reaction diffusion
//               equations for the next time step
//------------------------------------------------------------------------
void ReactDiffuse::updateStateImplicit( unsigned int num_passes )
{
  if( _last_solution != _solution ) {
    _last_solution = _solution;

    updateConstantsReact();

    if( _last_solution == EULER_SEMI_IMPLICIT ) {
      _a_diff_rate_theta = _a_diff_rate;
      _b_diff_rate_theta = _b_diff_rate;
      _c_diff_rate_theta = _c_diff_rate;
      _d_diff_rate_theta = _d_diff_rate;

      _a_diff_rate_1_theta = 0;
      _b_diff_rate_1_theta = 0;
      _c_diff_rate_1_theta = 0;
      _d_diff_rate_1_theta = 0;

    } else {
      _a_diff_rate_theta = _cn_theta*_a_diff_rate;
      _b_diff_rate_theta = _cn_theta*_b_diff_rate;
      _c_diff_rate_theta = _cn_theta*_c_diff_rate;
      _d_diff_rate_theta = _cn_theta*_d_diff_rate;

      _a_diff_rate_1_theta = (1.0-_cn_theta)*_a_diff_rate;
      _b_diff_rate_1_theta = (1.0-_cn_theta)*_b_diff_rate;
      _c_diff_rate_1_theta = (1.0-_cn_theta)*_c_diff_rate;
      _d_diff_rate_1_theta = (1.0-_cn_theta)*_d_diff_rate;
    }
  }

  _fbuffer->enable();
  
  for( unsigned int j=0; j<5; j++ ) {
    for( unsigned int i=0; i<num_passes/5; i++ ) {

      for( unsigned int m=0; m<1; m++ ) {
 	updateStateImplicitRHS( m );

	unsigned int cc = 0, dd = 0, diff_iterations = 5, max_iterations = 100;
	float current_residual = 1.0e4, last_residual = 1.0e5;
	float max_difference = 1.0e-4, max_residual = 1.0e-4;

	// Initerate until a) the residual is below the maximum value,
	// b) the residual does not change for a set number of iterations,
	// or c) if the maximum number of iterations is reached.
	while( cc++ < max_iterations ) {

	  updateStateImplicitRelaxGPU( 24, m );

	  // Calculate the sum of squares residual.
 	  current_residual = updateStateImplicitResiduals( m );

	  // Stop if the current residual is below the maximum value.
	  if( current_residual < max_residual )
	    break;

	  // Stop if the current residual does not change significantly
	  // after 5 iterations.
	  if( fabs(last_residual-current_residual) > max_difference ) {
	    dd = 1;

	    last_residual = current_residual;

	  } else if( ++dd == 5 )
	    break;
	}

	updateStateImplicitClamp();
      }
    }

    // Update the diffusion tensor which will calculate the gradient.
    if( _gradient )
      updateConstantsDT();
  }

  _fbuffer->disable();
}



//------------------------------------------------------------------------
// function    : updateStateImplicitRHS()
// description : iteration function that computes the reaction diffusion
//               equations for the next time step
//------------------------------------------------------------------------
void ReactDiffuse::updateStateImplicitRHS( unsigned int morph )
{
  _cg_Program = _cg_Implicit_RHS[_last_solution];

  // setup the link between this code and the CG program variables so 
  // that we can modify the inputs on the fly  
  _cg_Morphigens = cgGetNamedParameter( _cg_Program, "morphigens" );    

  for( unsigned int i=REACTION; i<=DIFFUSION; i++ )
    _cg_ConstsTex[i] = cgGetNamedParameter( _cg_Program,
					    constant_names[i].c_str() );
  
  // render a quad to update the new state --> here the front buffer is
  //  the old (current) timestep, and the back buffer is where the next
  //  (new) timestep will be computed --> the front buffer is bound as
  //  an input texture...

  _fbuffer->attach( _gl_RHSTexID, 1 );

  glDrawBuffer( GL_COLOR_ATTACHMENT1_EXT );

  cgGLEnableProfile( _cg_Profile );
  
  cgGLBindProgram( _cg_Program );

  cgGLSetParameter4f( cgGetNamedParameter( _cg_Program, "diffRates" ),
		      _a_diff_rate_1_theta,
		      _b_diff_rate_1_theta,
		      _c_diff_rate_1_theta,
		      _d_diff_rate_1_theta );

  cgGLSetParameter1f( cgGetNamedParameter( _cg_Program, "timeStep_inv" ),
		      _time_step_inv );

  cgGLSetTextureParameter( _cg_Morphigens, getTextureID() );
  for( unsigned int i=REACTION; i<=DIFFUSION; i++)
    cgGLSetTextureParameter( _cg_ConstsTex[i], _gl_ConstsTexID[i] );  

  cgGLEnableTextureParameter( _cg_Morphigens );
  for( unsigned int i=REACTION; i<=DIFFUSION; i++)
    cgGLEnableTextureParameter( _cg_ConstsTex[i] );

  // Do the rendering.
  if( _boundary == ZERO_FLUX_ALL )
    glCallList( _gl_dlAllZeroFlux );
  else {
    glCallList( _gl_dlInterior );
    glCallList( _gl_dlBoundary );
  }

  cgGLDisableTextureParameter( _cg_Morphigens );
  for( unsigned int i=REACTION; i<=DIFFUSION; i++)
    cgGLDisableTextureParameter( _cg_ConstsTex[i] );

  cgGLDisableProfile( _cg_Profile );
}


//------------------------------------------------------------------------
// function    : updateStateImplicitRelaxGPU()
// description : iteration function that computes the reaction diffusion
//               equations for the next time step
//------------------------------------------------------------------------
bool ReactDiffuse::updateStateImplicitRelaxGPU( unsigned int nr_steps,
						unsigned int morph )
{
  bool converged = false;

  unsigned int nSamples = _pbWidth*_pbHeight;
  unsigned int step, check = 4;

  _fbuffer->attach( _gl_MorphogenTexID[0], 0 );
  _fbuffer->attach( _gl_MorphogenTexID[1], 1 );
  
  // Do the rendering.
  for( step=0; step<nr_steps; step+=check ) {

    _cg_Program = _cg_Implicit_Relax[_last_solution];

    _cg_Morphigens = cgGetNamedParameter( _cg_Program, "morphigens" );
    _cg_RHS = cgGetNamedParameter( _cg_Program, "rhs" );

    for( unsigned int i=REACTION; i<=DIFFUSION; i++ )
      _cg_ConstsTex[i] =
	cgGetNamedParameter( _cg_Program, constant_names[i].c_str() );
  
    // render a quad to update the new state --> here the front buffer is
    //  the old (current) timestep, and the back buffer is where the next
    //  (new) timestep will be computed --> the front buffer is bound as
    //  an input texture...

    glDrawBuffer( GL_COLOR_ATTACHMENT0_EXT );
    cgGLEnableProfile( _cg_Profile );
  
    cgGLBindProgram( _cg_Program );

    cgGLSetParameter4f( cgGetNamedParameter( _cg_Program, "diffRates" ),
			_a_diff_rate_theta,
			_b_diff_rate_theta,
			_c_diff_rate_theta,
			_d_diff_rate_theta );

    cgGLSetParameter1f( cgGetNamedParameter( _cg_Program, "timeStep_inv" ),
		      _time_step_inv );

    cgGLSetTextureParameter( _cg_Morphigens, getTextureID() );
    cgGLSetTextureParameter( _cg_RHS, _gl_RHSTexID );

    for( unsigned int i=REACTION; i<=DIFFUSION; i++)
      cgGLSetTextureParameter( _cg_ConstsTex[i], _gl_ConstsTexID[i] );  

    cgGLEnableTextureParameter( _cg_Morphigens );
    cgGLEnableTextureParameter( _cg_RHS );

    for( unsigned int i=REACTION; i<=DIFFUSION; i++)
      cgGLEnableTextureParameter( _cg_ConstsTex[i] );

    if( _boundary == ZERO_FLUX_ALL ) {
      for( unsigned int i=0; i<check/2; i++ ) {
	cgGLSetTextureParameter( _cg_Morphigens, _gl_MorphogenTexID[0] );
	cgGLEnableTextureParameter( _cg_Morphigens );
	glDrawBuffer( GL_COLOR_ATTACHMENT1_EXT );
	glCallList( _gl_dlAllZeroFlux );

	cgGLSetTextureParameter( _cg_Morphigens, _gl_MorphogenTexID[1] );
	cgGLEnableTextureParameter( _cg_Morphigens );
	glDrawBuffer( GL_COLOR_ATTACHMENT0_EXT );
	glCallList( _gl_dlAllZeroFlux );
      }

    } else if( _boundary == ZERO_FLUX_VAR ) {
      for( unsigned int i=0; i<check/2; i++ ) {
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

      // Extra relaxations around the boundary.
      for( unsigned int i=0; i<check/2; i++ ) {
	cgGLSetTextureParameter( _cg_Morphigens, _gl_MorphogenTexID[0] );
	cgGLEnableTextureParameter( _cg_Morphigens );
	glDrawBuffer( GL_COLOR_ATTACHMENT1_EXT );
	glCallList( _gl_dlBoundary );

	cgGLSetTextureParameter( _cg_Morphigens, _gl_MorphogenTexID[1] );
	cgGLEnableTextureParameter( _cg_Morphigens );
	glDrawBuffer( GL_COLOR_ATTACHMENT0_EXT );
	glCallList( _gl_dlBoundary );
      }

    } else /* if( _boundary == PERIODIC ) */ {
      for( unsigned int i=0; i<check/2; i++ ) {
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
    cgGLDisableTextureParameter( _cg_RHS );

    for( unsigned int i=REACTION; i<=DIFFUSION; i++)
      cgGLDisableTextureParameter( _cg_ConstsTex[i] );
    
    cgGLDisableProfile( _cg_Profile );

    // Test for convergence using the reduction.
    if( converged = (reduction( getTextureID() ) == 1.0) )
      break;
  }

  return converged;
}


//------------------------------------------------------------------------
// function    : updateStateImplicitResiduals()
// description : iteration function that computes the reaction diffusion
//               equations for the next time step
//------------------------------------------------------------------------
float ReactDiffuse::updateStateImplicitResiduals( unsigned int morph )
{
  float sum = 0;

  _fbuffer->attach( _gl_ResidualsTexID, 1 );

  glDrawBuffer( GL_COLOR_ATTACHMENT1_EXT );

  cgGLEnableProfile( _cg_Profile );

  _cg_Program = _cg_Implicit_Residuals[_last_solution];

  cgGLBindProgram( _cg_Program );
  
  // setup the link between this code and the CG program variables so 
  // that we can modify the inputs on the fly
  _cg_Morphigens = cgGetNamedParameter( _cg_Program, "morphigens" );
  _cg_RHS = cgGetNamedParameter( _cg_Program, "rhs" );

  _cg_ConstsTex[DIFFUSION] =
    cgGetNamedParameter( _cg_Program, constant_names[DIFFUSION].c_str() );
  
  cgGLSetParameter4f( cgGetNamedParameter( _cg_Program, "diffRates" ),
		      _a_diff_rate_theta,
		      _b_diff_rate_theta,
		      _c_diff_rate_theta,
		      _d_diff_rate_theta );

  cgGLSetParameter1f( cgGetNamedParameter( _cg_Program, "timeStep_inv" ),
		      _time_step_inv );

  cgGLSetTextureParameter( _cg_Morphigens, getTextureID() );
  cgGLSetTextureParameter( _cg_RHS, _gl_RHSTexID );
  cgGLSetTextureParameter( _cg_ConstsTex[DIFFUSION], _gl_ConstsTexID[DIFFUSION] );  

  cgGLEnableTextureParameter( _cg_Morphigens );
  cgGLEnableTextureParameter( _cg_RHS );
  cgGLEnableTextureParameter( _cg_ConstsTex[DIFFUSION] );

  if( _boundary == ZERO_FLUX_ALL )
    glCallList( _gl_dlAllZeroFlux );
  else {
    glCallList( _gl_dlInterior );
    glCallList( _gl_dlBoundary );
  }
  
  cgGLDisableTextureParameter( _cg_Morphigens );
  cgGLDisableTextureParameter( _cg_RHS );
  cgGLDisableTextureParameter( _cg_ConstsTex[DIFFUSION] );
      
  cgGLDisableProfile( _cg_Profile );

  sum = reduction( _gl_ResidualsTexID );

  return sqrt( sum );
}


//------------------------------------------------------------------------
// function    : updateStateImplicitClamp()
// description : iteration function that computes the reaction diffusion
//               equations for the next time step
//------------------------------------------------------------------------
void ReactDiffuse::updateStateImplicitClamp()
{
  _cg_Program = _cg_Implicit_Clamp;

  _cg_Morphigens = cgGetNamedParameter( _cg_Program, "morphigens" );

  // render a quad to update the new state --> here the front buffer is
  //  the old (current) timestep, and the back buffer is where the next
  //  (new) timestep will be computed --> the front buffer is bound as
  //  an input texture...

  _fbuffer->attach( _gl_MorphogenTexID[1], 1 );

  glDrawBuffer( GL_COLOR_ATTACHMENT1_EXT );

  cgGLEnableProfile( _cg_Profile );
  
  cgGLBindProgram( _cg_Program );

  cgGLSetTextureParameter( _cg_Morphigens, _gl_MorphogenTexID[0] );
  cgGLEnableTextureParameter( _cg_Morphigens );

  glCallList( _gl_dlAll );

  cgGLDisableTextureParameter( _cg_Morphigens );
      
  cgGLDisableProfile( _cg_Profile );

  glReadBuffer( GL_COLOR_ATTACHMENT1_EXT );
  glEnable(GL_TEXTURE_RECTANGLE_NV);
  glBindTexture( GL_TEXTURE_RECTANGLE_NV, _gl_MorphogenTexID[0] );
  glCopyTexSubImage2D(GL_TEXTURE_RECTANGLE_NV, 0,
		      0, 0, 0, 0, _pbWidth, _pbHeight);
  glBindTexture( GL_TEXTURE_RECTANGLE_NV, 0 );
  glDisable(GL_TEXTURE_RECTANGLE_NV);
}


//------------------------------------------------------------------------
// function    : reduction()
// description : 
//------------------------------------------------------------------------
float ReactDiffuse::reduction( GLuint gl_InputTexID )
{
  // render a quad to update the new state --> here the front buffer is
  //  the old (current) timestep, and the back buffer is where the next
  //  (new) timestep will be computed --> the front buffer is bound as
  //  an input texture...

  unsigned int o_height = _pbHeight;
  unsigned int o_width  = _pbWidth;

  unsigned int n_height = _pbHeight;
  unsigned int n_width  = _pbWidth;

  unsigned int i = 0;

  glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,
			     GL_TEXTURE_RECTANGLE_NV, _gl_ReductionTexID[0], 0 );

  glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT1_EXT,
			     GL_TEXTURE_RECTANGLE_NV, _gl_ReductionTexID[1], 0 );
  while( 1 ) {

    n_height = o_height / 2;
    n_width  = o_width  / 2;

    if( o_height != n_height * 2 &&
	o_width  != n_width  * 2 ) {

      n_height = o_height;
      n_width  = o_width;
      break;
    }

    float step = -0.5;

    _cg_Program = _cg_Summation;

    cgGLEnableProfile( _cg_Profile );
  
    cgGLBindProgram( _cg_Program );

    glEnable(GL_TEXTURE_RECTANGLE_NV);
    
    // setup the link between this code and the CG program variables so 
    // that we can modify the inputs on the fly  
    _cg_Input = cgGetNamedParameter( _cg_Program, "input" );    

    glDrawBuffer( GL_COLOR_ATTACHMENT0_EXT+i%2);

    if( i == 0 )
      cgGLSetTextureParameter( _cg_Input, gl_InputTexID );
    else
      cgGLSetTextureParameter( _cg_Input, _gl_ReductionTexID[(i+1)%2] );

    cgGLEnableTextureParameter( _cg_Input );
  
    // Do the rendering.
    glBegin( GL_QUADS );
    for( unsigned int k=0; k<2; k++ )
      glMultiTexCoord2fARB(GL_TEXTURE0_ARB+k,         k+step,          k+step);
    glVertex2f( 0, 0 );

    for( unsigned int k=0; k<2; k++ )
      glMultiTexCoord2fARB(GL_TEXTURE0_ARB+k, o_width+k+step,          k+step);
    glVertex2f( n_width, 0 );
      
    for( unsigned int k=0; k<2; k++ )
      glMultiTexCoord2fARB(GL_TEXTURE0_ARB+k, o_width+k+step, o_height+k+step);
    glVertex2f( n_width, n_height );

    for( unsigned int k=0; k<2; k++ )
      glMultiTexCoord2fARB(GL_TEXTURE0_ARB+k,         k+step, o_height+k+step);
    glVertex2f( 0, n_height);
    glEnd();

    cgGLDisableTextureParameter( _cg_Input );

    cgGLDisableProfile( _cg_Profile );

    o_height = n_height;
    o_width  = n_width;

    i++;
  }

  // Read the total number of samples.
  glReadBuffer( GL_COLOR_ATTACHMENT0_EXT+(i+1)%2 );

  glReadPixels( 0, 0, n_width, n_height,
		GL_RGBA, GL_FLOAT, _texReduction );

  float nsamples = n_width * n_height * 4; 
  float sumN = 0, sumD = _pbHeight * _pbWidth;


  for( unsigned int cc=3; cc<nsamples; cc+=4 )
    sumN += _texReduction[cc];

  return sumN / sumD;
}
