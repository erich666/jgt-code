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

#include <stdio.h>


#include "rd_gray_scott.h"

using namespace std;

/******************************************************************************
Run Gray-Scott reaction-diffusion system.
******************************************************************************/

rd_gray_scott::rd_gray_scott() :
  conv( NULL ), feed( NULL ), perturb( 12 )
{
  nMorphs = 2;

  a_ = 0.25;
  b_ = 0.50;
  ab_variance_ = 0.000;
  reaction_const0_ = 0.0925;
  reaction_const1_ = 0.0300;
  reaction_const_variance_ = 0.001;
  time_step_ = 0.10;
  time_mult_ = 25.0;
  time_step_inv_ = 1.0 / (time_step_ * time_mult_);
  theta_ = 0.50;

  rr_coef_1_ = 4.50;
  rr_coef_2_ = 4.00;

  diff_coef_1_ = 1.5;
  diff_coef_2_ = 0.5;

  a_diff_rate_ = 4.0e-5;
  b_diff_rate_ = 8.0e-5;

  gradient_  = 1;
}

/******************************************************************************
Allocate the data space
******************************************************************************/

void rd_gray_scott::alloc( unsigned int *dims, int mult )
{
  rd_base::alloc( dims, mult );

  conv = new float*[ height_ ];
  feed = new float*[ height_ ];
  
  for (int j = 0; j < height_; j++) {
    conv[j] = new float[ width_ ];
    feed[j] = new float[ width_ ];
  }
}


/******************************************************************************
Run Gray-Scott reaction-diffusion system.
******************************************************************************/

void rd_gray_scott::initialize( float ***vector, bool reset )
{
  // seed the random number generator.
  long seed = -clock_t();
  seed = -100;

  rd_base::initialize( vector );

  float morph_steady[2];

  morph_steady[0] = 0.0;        // Inital a steady state values.
  morph_steady[1] = 1.0;        // Inital b steady state values.

  conv_init = reaction_const0_;  // Const
  feed_init = reaction_const1_;  // Feed

  morph_variance = ab_variance_;
  react_variance = reaction_const_variance_;

  c_range =  0.030;          /* Maximum range in substrate. */
  f_range =  0.020;          /* Maximum range in substrate. */

  /* calculate semistable equilibria */

  float d1_2 = height_/2;
  float d2_2 = width_/2;

  for (int j=0; j<height_; j++) {
    for (int i=0; i<width_; i++) {

      for( unsigned int n=0; reset && n<nMorphs; n++ ) {
	morphigen[n][j][i] = morph_steady[n];
	morphigen[n][j][i] *=
	  (1.0 + frand(-morph_variance, morph_variance, seed));
      }

      conv[j][i] = conv_init;
      feed[j][i] = feed_init;

      conv[j][i] *= (1.0 + frand(-react_variance, react_variance, seed));
      feed[j][i] *= (1.0 + frand(-react_variance, react_variance, seed));
    }
  }

  morph_steady[0] = 0.250;   /* Inital b perturbed state values. */
  morph_steady[1] = 0.500;   /* Inital a perturbed state values. */

  /* Perturb the central area. */
  int pj, pi;

  if( height_/4 > perturb ) pj = perturb;
  else                      pj = 0;

  if( width_/4 > perturb ) pi = perturb;
  else                      pi = 0;

  if( pj && pi ) {
    for (int j =  height_/2-pj; j < height_/2+pj; j++) {
      for (int i = width_/2-pi; i < width_/2+pi; i++) {
	for( unsigned int n=0; reset && n<nMorphs; n++ ) {
	  morphigen[n][j][i] = morph_steady[n];
	  morphigen[n][j][i] *=
	    (1.0 + frand(-morph_variance, morph_variance, seed));
	}
      }
    }
  } else if(  pj && pi ) {

    for (int j =  height_/2-pj; j < height_/2+pj; j++) {
      for (int i = width_/2-pi; i < width_/2+pi; i++) {
	for( unsigned int n=0; n<nMorphs; n++ ) {
	  morphigen[n][j][i] = morph_steady[n];
	  morphigen[n][j][i] *=
	    (1.0 + frand(-morph_variance, morph_variance, seed));
	}
      }
    }
  } else if( pi ) {
    int j = 0;

    for (int i = width_/2-pi; i < width_/2+pi; i++) {
      for( unsigned int n=0; n<nMorphs; n++ ) {
	morphigen[n][j][i] = morph_steady[n];
	morphigen[n][j][i] *=
	  (1.0 + frand(-morph_variance, morph_variance, seed));
      }
    }
  }
}


/******************************************************************************
Gray-Scott's reaction equations.
******************************************************************************/

inline float rd_gray_scott::reaction( unsigned int n,
				      unsigned int i,
				      unsigned int j )
{
  float aVal = morphigen[0][j][i];
  float bVal = morphigen[1][j][i];

  float K = conv[j][i];
  float F = feed[j][i];

  switch ( n ) {
  case 0:
    return (    aVal * aVal * bVal -  K * aVal);
  case 1:
    return (F - aVal * aVal * bVal -  F * bVal);
  }
}
