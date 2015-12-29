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

#include <iostream>

#include <math.h>

#include "rd_turing.h"

using namespace std;

/******************************************************************************
Turing reaction-diffusion system.
******************************************************************************/

rd_turing::rd_turing( ) :
  alpha( NULL ), beta( NULL )
{
  nMorphs = 2;

  a_ = 4.0;
  b_ = 4.0;
  ab_variance_ = 0.000;
  reaction_const0_ = 12.0;
  reaction_const1_ = 16.0;
  reaction_const_variance_ = 0.001;
  time_step_ = 0.50;
  time_mult_ = 25.0;
  time_step_inv_ = 1.0 / (time_step_ * time_mult_);
  theta_ = 0.50;

  rr_coef_1_ = 100.0;
  rr_coef_2_ = 250.0;

  diff_coef_1_ = 1.5;
  diff_coef_2_ = 0.5;

  a_diff_rate_ = 1.0 / 16.0;
  b_diff_rate_ = 1.0 /  4.0;

  gradient_  = 1;
}

/******************************************************************************
Allocate the data space
******************************************************************************/

void rd_turing::alloc( unsigned int *dims, int mult )
{
  rd_base::alloc( dims, mult );

  alpha = new float*[ height_ ];
  beta  = new float*[ height_ ];
  
  for (int j = 0; j < height_; j++) {
    alpha[j] = new float[ width_ ];
    beta[j]  = new float[ width_ ];
  }
}


/******************************************************************************
Run Turing reaction-diffusion system.
******************************************************************************/
void rd_turing::initialize( float ***vector, bool reset )
{
//   float min_alpha_norm=MAX_FLOAT, max_alpha_norm=-MAX_FLOAT;
//   float  min_beta_norm=MAX_FLOAT,  max_beta_norm=-MAX_FLOAT;

  // seed the random number generator.
  long seed = -clock_t();
  seed = -100;

  rd_base::initialize( vector );

  float morph_steady[2];

  morph_steady[0] = a_;               /* Inital a steady state values. */
  morph_steady[1] = b_;               /* Inital b steady state values. */

  float alpha_init = reaction_const0_;
  float beta_init  = reaction_const1_;

  /* Maximum random amount in substrate. */
  morph_variance = ab_variance_;
  react_variance = reaction_const_variance_;

  /* calculate semistable equilibria */
  for (int j = 0; j < height_; j++) {
    for (int i = 0; i < width_; i++) {
      for( unsigned int n=0; reset && n<nMorphs; n++ )
	morphigen[n][j][i] = morph_steady[n];
	
      frand(-react_variance, react_variance, seed);
      frand(-react_variance, react_variance, seed);
      
      alpha[j][i] = alpha_init;
      beta [j][i] =  beta_init;

      alpha[j][i] *= (1.0 + frand(-react_variance, react_variance, seed));
      beta [j][i] *= (1.0 + frand(-react_variance, react_variance, seed));
    }
  }
}


/******************************************************************************
Turing's reaction equations.
******************************************************************************/
inline float rd_turing::reaction( unsigned int n,
				  unsigned int i,
				  unsigned int j )
{
  float aVal = morphigen[0][j][i];
  float bVal = morphigen[1][j][i];

  float alphaVal = alpha[j][i];
  float betaVal  =  beta[j][i];

  switch ( n ) {
  case 0:
    return (-alphaVal + aVal * bVal - aVal);
  case 1:
    return (  betaVal - aVal * bVal);
  }
}
