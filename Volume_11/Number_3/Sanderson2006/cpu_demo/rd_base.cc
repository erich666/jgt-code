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
#include <math.h>

#include "rand.h"
#include "gammp.h"

#include "rd_base.h"

using namespace std;

#define LERP( t, x0, x1 ) \
( (float) (x0) + (t) * ( (float) (x1) - (float) (x0)) )


#define SGN( x ) \
	( (x) < 0 ? -1.0 : 1.0 )


#define INDEX( x, range ) \
	( (int) (x*range) + range )


#define MAX_GAMMP 500

/******************************************************************************
rd_base
******************************************************************************/
rd_base::rd_base( ) :
  cell_mult_(1),
  gradient_( 0 ),
  theta_( 0.5 ),
  nMorphs(0),

  relaxation_(RED_BLACK),
  reaction_(TURING),
  laplacian_(INHOMOGENEOUS),
  diffusion_(ANISOTROPIC),
  solution_(EXPLICIT),

  vector_data(0),
  vector_diverge(0),

  morphigen(0),
  dmorphigen(0),

  diff_rate(0),

  react_rate(0),
  
  d_tensor(0),
  d_template(0),

  u_h(0),
  rhs_func(0),
  residuals(0),
  errors(0),
  tmp(0),

  neighborhood(8),

  min_error_(1.0e-4)
{
  gammp_table = new float[2*MAX_GAMMP+1];

  gammp_table[MAX_GAMMP] = 0;

  for( unsigned int i=1; i<=MAX_GAMMP; i++ ) {
    float val = gammp( 3.0, 10.0 * (float) i / (float) MAX_GAMMP );

    gammp_table[MAX_GAMMP-i] = -0.25 * val;
    gammp_table[MAX_GAMMP+i] = +0.25 * val;
  }
}

/******************************************************************************
frand - Pick a random number between min and max.
******************************************************************************/
float rd_base::frand( float min, float max, long &seed ) {
  return (min + gasdev( &seed ) * (max - min));
}


/******************************************************************************
next_step_euler - next_step - explicit euler solution
******************************************************************************/
bool rd_base::next_step_explicit_euler( )
{
  react_and_diffuse( );

  /* Add the change in the morphigen to each cell. */
  for( unsigned int n=0; n<nMorphs; n++ ) {
    for (int j=0; j<height_; j++) {
      for (int i=0; i<width_; i++) {
	
	double mval = morphigen[n][j][i];
	
	morphigen[n][j][i] += dmorphigen[n][j][i];
	
	if( morphigen[n][j][i] < 0.0 )
	  morphigen[n][j][i] = 0.0;
      }
    }
  }

  return false;
}


/******************************************************************************
next_step - semi implicit euler solution
******************************************************************************/

#define LOCK_STEP 1
//#define STAGGERED 1

bool rd_base::next_step_implicit_euler( )
{
#ifdef LOCK_STEP
  // Get the inital RHS for the morphigen which is the
  // current value plus the its current reaction.
  for( unsigned int m=0; m<nMorphs; m++ )
    implicit_euler_rhs( rhs_func[m], (morphigen_type) m );
  
  // Solve for the next value.
  for( unsigned int m=0; m<nMorphs; m++ )
    implicit_solve( morphigen[m], rhs_func[m], theta_*diff_rate[m] );

#else //STAGGERED
  for( unsigned int m=0; m<nMorphs; m++ ) {

    // Get the inital RHS for the morphigen which is the
    // current value plus the its current reaction.
    implicit_euler_rhs( rhs_func[m], (morphigen_type) m );
    
    // Solve for the next value.
    implicit_solve( morphigen[m], rhs_func[m], theta_*diff_rate[m] );
  }
#endif
}


/******************************************************************************
implicit_solve - base solution
******************************************************************************/
void rd_base::implicit_solve( float **v, float **rhs, float diffusion ) {

  int nr_steps = 20; // Number of smoothing steps for the exact solution.

  int cc = 0, dd = 0, max_iterations = 100;
  float current_residual = 1.0e4, last_residual = 1.0e5;
  float max_difference = min_error_, max_residual = min_error_;

  // Initerate until a) the residual is below the maximum value,
  // b) the residual does not change for a set number of iterations,
  // or c) if the maximum number of iterations is reached.
  while( cc++ < max_iterations ) {

    // Set the current error to zero.
    for (unsigned int j=0; j<height_; j++)
      for (unsigned int i=0; i<width_; i++)
	errors[j][i] = 0;

    // Get the next solution using relaxation.
    bool converged = false;

    for(unsigned int i=0; i<nr_steps && !converged; i++)
      converged = relax( v, rhs, diffusion );

    // Calculate the sum of squares residual.
    current_residual = implicit_euler_residual(residuals, v, rhs, diffusion );

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

  // Clamp any negative results to zero
  for (unsigned int j=0; j<height_; j++)
    for (unsigned int i=0; i<width_; i++)
      if( v[j][i] < 0 )
	v[j][i] = 0;
}


/******************************************************************************
difference - mulitgrid difference
******************************************************************************/
float rd_base::difference( float **u_new, float **u_old  )
{
  register float sum = 0.0;

  for (int j=0; j<height_; j+=1) {
    for (int i=0; i<width_; i+=1) {
      register float diff = u_new[j][i] - u_old[j][i];

      sum += diff * diff;
    }
  }

  return sqrt( sum / ((width_/1)*(height_/1)) );
}


/******************************************************************************
relax - relaxation using XXX relaxation
******************************************************************************/
inline bool rd_base::relax( float **u, float **rhs, float diff )
{
  if( relaxation_ == RED_BLACK )
    return relax_gs_rb( u, rhs, diff );

  else if( relaxation_ == GAUSS_SEIDEL )
    return relax_gs( u, rhs, diff );

  else if( relaxation_ == JACOBI )
    return relax_jacobi( u, rhs, diff );
}


/******************************************************************************
relax_gs - relaxation using Gauss-Seidel relaxation
******************************************************************************/
bool rd_base::relax_gs( float **u, float **rhs, float diff )
{
  bool convergence = true;

  for (int j=0; j<height_; j+=1) {
    int j_1 = jminus1[j];
    int j1  = jplus1 [j];
      
    for (int i=0; i<width_; i+=1) {
      int i_1 = iminus1[i];
      int i1  = iplus1 [i];

      float old_u = u[j][i];

      u[j][i] = implicit_euler_relax( u, rhs, diff,
				      i_1,  j_1,
				      i,    j,
				      i1,   j1 );

      if( convergence && fabs( old_u - u[j][i] ) > min_error_ )
	convergence = false;
    }
  }
  
  return convergence;
}


/******************************************************************************
relax_gs_rb - relaxation using Gauss-Seidel red-black relaxation
******************************************************************************/
bool rd_base::relax_gs_rb( float **u, float **rhs, float diff )
{
  bool convergence = true;

  for( int pass=0; pass<2; pass++ ) {

    for (int j=0; j<height_; j+=1) {
      int j_1 = jminus1[j];
      int j1  = jplus1 [j];
      
//    j/1 gives the actual row,
//    the remainder gives even - old,
      int offset = (j/1) % 2;

//    If on the second pass flip;
      if( pass )
	offset = !offset;

      for (int i=offset*1; i<width_; i+=(1*2)) {

	float old_u = u[j][i];

	int i_1 = iminus1[i];
	int i1  = iplus1 [i];

	u[j][i] = implicit_euler_relax( u, rhs, diff,
					i_1,  j_1,
					i,    j,
					i1,   j1 );

	if( convergence && fabs( old_u - u[j][i] ) > min_error_ )
	  convergence = false;
      }
    }
  }

  return convergence;
}


/******************************************************************************
relax_jacobi - relaxation using Jacobi relaxation
******************************************************************************/
bool rd_base::relax_jacobi( float **u, float **rhs, float diff )
{
  bool convergence = true;

  for (int j=0; j<height_; j+=1) {
    int j_1 = jminus1[j];
    int j1  = jplus1 [j];
      
    for (int i=0; i<width_; i+=1) {
      int i_1 = iminus1[i];
      int i1  = iplus1 [i];

      tmp[j][i] = implicit_euler_relax( u, rhs, diff,
					i_1,  j_1,
					i,    j,
					i1,   j1 );

      if( convergence && fabs( u[j][i] - tmp[j][i] ) > min_error_ )
	convergence = false;
    }
  }
  
  for (int j=0; j<height_; j+=1)
    for (int i=0; i<width_; i+=1)
      u[j][i] = tmp[j][i];
  
  return convergence;
}


/******************************************************************************
relax - relaxation for a single grid location
******************************************************************************/
inline float rd_base::implicit_euler_relax( float **u,
					    float **rhs, float diff,
 					    int i_1, int j_1,
 					    int i,   int j,
 					    int i1,  int j1 )
{
  float h = 1.0 / 1;
  float h2 = h * h;

  // Solve for the grid next value current values of the neighbors.
  // This is just the diffusion with u[j][i] on the LHS.
  float f3 =
    ( (d_tensor[j][i1 ][0][1] * (u[j1][i1 ] - u[j_1][i1 ]) -
       d_tensor[j][i_1][0][1] * (u[j1][i_1] - u[j_1][i_1]) )  +
      
      (d_tensor[j1 ][i][1][0] * (u[j1 ][i1] - u[j1 ][i_1]) -
       d_tensor[j_1][i][1][0] * (u[j_1][i1] - u[j_1][i_1]) ) ) / h2;
      
  float d_t0 = d_tensor[j ][i1][0][0] + d_tensor[j  ][i  ][0][0];
  float d_t1 = d_tensor[j ][i ][0][0] + d_tensor[j  ][i_1][0][0];
  float d_t2 = d_tensor[j1][i ][1][1] + d_tensor[j  ][i  ][1][1];	  
  float d_t3 = d_tensor[j ][i ][1][1] + d_tensor[j_1][i  ][1][1];

  float f1a = (d_t0 * u[j ][i1] + d_t1 * u[j  ][i_1] +
	       d_t2 * u[j1][i ] + d_t3 * u[j_1][i  ]) / h2;

  float f1b_mult = -(d_t0 + d_t1 + d_t2 + d_t3) / h2;

  float val = (rhs[j][i] + diff * (f1a + f3)) /
    (time_step_inv_ - diff * f1b_mult);

  return val;
}


/******************************************************************************
Implicit RHS Calculation.
******************************************************************************/
void rd_base::implicit_euler_rhs( float **rhs, morphigen_type mt )
{
  int n = (int) mt;

  if( theta_ == 1.0 ) {
    for (int j=0; j <height_; j++)
      for (int i=0; i<width_; i++)
	rhs[j][i] = morphigen[n][j][i] * time_step_inv_ +
	  react_rate[j][i] * reaction( n, i, j );
  } else {
    float diff = (1.0-theta_) * diff_rate[n];
    
    for (int j=0; j<height_; j++)
      for (int i=0; i<width_; i++)
	rhs[j][i] = morphigen[n][j][i] * time_step_inv_ +
	  react_rate[j][i] * reaction( n, i, j ) +
	  diff * diffusion( morphigen[n], i, j ) ;
  }
}



/******************************************************************************
 The residual is the difference between the RHS
 (current u + current u reaction) and  the LHS (next u - next u diffusion).
******************************************************************************/
float rd_base::implicit_euler_residual( float **resid,
					float **u,
					float **rhs,
					float diff )
{
  float sum = 0.0;

  for (int j=0; j<height_; j+=1) {
    for (int i=0; i<width_; i+=1) {
      resid[j][i] = rhs[j][i] -
	(u[j][i] * time_step_inv_ - diff * diffusion( u, i, j ) );

      sum += resid[j][i] * resid[j][i];
    }
  }

  return sqrt( sum / ((height_/1)*(width_/1)) );
}


/******************************************************************************
react_and_diffuse.
******************************************************************************/
void rd_base::react_and_diffuse( )
{
  // Compute the change in the morphigen in each cell.
  for (unsigned int j=0; j<height_; j++) {
    for (unsigned int i=0; i<width_; i++) {
	
      for( unsigned int n=0; n<nMorphs; n++ ) {
	dmorphigen[n][j][i] =

	  react_rate[j][i] * reaction( n, i, j ) +

	  diff_rate[n] * diffusion( morphigen[n], i, j );
      }
    }
  }
}


/******************************************************************************
Diffusion equations.
******************************************************************************/
float rd_base::diffusion( float **morph, unsigned int i, unsigned int j )
{
  float diff = 0;

  int j_1 = jminus1[j];
  int j1  = jplus1 [j];
      
  int i_1 = iminus1[i];
  int i1  = iplus1 [i];

  if( laplacian_ == INHOMOGENEOUS ) {

//    float e1 = 0.5, e2 = 0.5, e3 = 0.5, e4 = 0.5;

    float f1 =
      ( (d_tensor[j  ][i1 ][0][0] + d_tensor[j][i][0][0]) * (morph[j  ][i1 ] - morph[j][i]) +
	(d_tensor[j  ][i_1][0][0] + d_tensor[j][i][0][0]) * (morph[j  ][i_1] - morph[j][i]) +
	(d_tensor[j1 ][i  ][1][1] + d_tensor[j][i][1][1]) * (morph[j1 ][i  ] - morph[j][i]) +
	(d_tensor[j_1][i  ][1][1] + d_tensor[j][i][1][1]) * (morph[j_1][i  ] - morph[j][i]) );

    // / 2.0; - the divide by 2 is not needed as it was already done for the principal
    // difusivity values.  This only holds when solely using f1 and f3

    /*
    float f2 =
      ( (d_tensor[j1 ][i1][0][0] + d_tensor[j1 ][i  ][0][0]) * (morph[j1 ][i1] - morph[j1 ][i  ]) - 
	(d_tensor[j1 ][i ][0][0] + d_tensor[j1 ][i_1][0][0]) * (morph[j1 ][i ] - morph[j1 ][i_1]) ) / 8.0 +

      ( (d_tensor[j  ][i1][0][0] + d_tensor[j  ][i  ][0][0]) * (morph[j  ][i1] - morph[j  ][i  ]) - 
	(d_tensor[j  ][i ][0][0] + d_tensor[j  ][i_1][0][0]) * (morph[j  ][i ] - morph[j  ][i_1]) ) / 4.0 +

      ( (d_tensor[j_1][i1][0][0] + d_tensor[j_1][i  ][0][0]) * (morph[j_1][i1] - morph[j_1][i  ]) - 
	(d_tensor[j_1][i ][0][0] + d_tensor[j_1][i_1][0][0]) * (morph[j_1][i ] - morph[j_1][i_1]) ) / 8.0 +


      ( (d_tensor[j1][i1 ][1][1] + d_tensor[j  ][i1 ][1][1]) * (morph[j1][i1 ] - morph[j  ][i1 ]) - 
	(d_tensor[j ][i1 ][1][1] + d_tensor[j_1][i1 ][1][1]) * (morph[j ][i1 ] - morph[j_1][i1 ]) ) / 8.0 +

      ( (d_tensor[j1][i  ][1][1] + d_tensor[j  ][i  ][1][1]) * (morph[j1][i  ] - morph[j  ][i  ]) - 
	(d_tensor[j ][i  ][1][1] + d_tensor[j_1][i  ][1][1]) * (morph[j ][i  ] - morph[j_1][i  ]) ) / 4.0 +

      ( (d_tensor[j1][i_1][1][1] + d_tensor[j  ][i_1][1][1]) * (morph[j1][i_1] - morph[j  ][i_1]) - 
	(d_tensor[j ][i_1][1][1] + d_tensor[j_1][i_1][1][1]) * (morph[j ][i_1] - morph[j_1][i_1]) ) / 8.0;
    */

    float f3 =
      ( (d_tensor[j][i1 ][0][1] * (morph[j1][i1 ] - morph[j_1][i1 ]) -
	 d_tensor[j][i_1][0][1] * (morph[j1][i_1] - morph[j_1][i_1]) ) +
	
	(d_tensor[j1 ][i][1][0] * (morph[j1 ][i1] - morph[j1 ][i_1]) -
	 d_tensor[j_1][i][1][0] * (morph[j_1][i1] - morph[j_1][i_1]) ) );

    // / 4.0; - the divide by 4 is not needed as it was already done
    // for the secondary difusivity values.  This only holds when
    // solely using f1 and f3
 
    /*
    float f4 =
      (d_tensor[j][i1][0][1] * (morph[j1][i1] - morph[j][i1]) -
       d_tensor[j][i ][0][1] * (morph[j1][i ] - morph[j][i ]) +
      
       d_tensor[j][i  ][0][1] * (morph[j][i  ] - morph[j_1][i]) -
       d_tensor[j][i_1][0][1] * (morph[j][i_1] - morph[j_1][i_1]) ) / 2.0 +

      (d_tensor[j1][i][1][0] * (morph[j1][i1] - morph[j1][i]) -
       d_tensor[j ][i][1][0] * (morph[j ][i1] - morph[j ][i]) +
      
       d_tensor[j  ][i][1][0] * (morph[j  ][i] - morph[j  ][i_1]) -
       d_tensor[j_1][i][1][0] * (morph[j_1][i] - morph[j_1][i_1]) ) / 2.0;
    */


    /*    diff = e1 * f1 + e2 * f2 + e3 * f3 + e4 * f4; */

    diff = f1 + f3;
  } else {

      /* 4 Neighborhood - isotropic. */
    if( neighborhood == 4 ) {

      diff = (morph[j_1][i] + morph[j1][i] +
	      -4.0 * morph[j][i] +
	      morph[j][i_1] + morph[j][i1]);

      /* 8 Neighborhood - anisotropic. */
    } else if( neighborhood == 8 ) {
      diff = (d_template[j][i][0][0] * morph[j_1][i_1] +
	      d_template[j][i][0][1] * morph[j_1][i  ] +
	      d_template[j][i][0][2] * morph[j_1][i1 ] +
	      
	      d_template[j][i][1][0] * morph[j][i_1] +
	      d_template[j][i][1][1] * morph[j][i  ] +
	      d_template[j][i][1][2] * morph[j][i1 ] +
	      
	      d_template[j][i][2][0] * morph[j1][i_1] +
	      d_template[j][i][2][1] * morph[j1][i  ] +
	      d_template[j][i][2][2] * morph[j1][i1 ]);
    }
  }
 
  return diff;
}


/******************************************************************************
Contiguous memory allocation 1D
******************************************************************************/
char *rd_base::alloc( unsigned int i,
		      unsigned int bytes )
{
  char *p1 = 0;

  if( p1 = (char *) malloc( i * bytes ) ) {
  }

  return p1;
}


/******************************************************************************
Contiguous memory allocation 2D
******************************************************************************/
char **rd_base::alloc( unsigned int j, unsigned int i,
		       unsigned int bytes )
{
  char **p2 = 0, *p1 = 0;

  unsigned int ic = i * bytes;

  if( p2 = (char **) malloc( j     * sizeof( char * ) +
			     j * i * bytes ) ) {
    p1 = (char *) (p2 + j);
      
    for( unsigned int jc=0; jc<j; jc++ ) {
      p2[jc] = p1;
	
      p1 += ic;
    }
  }

  return p2;
}


/******************************************************************************
Contiguous memory allocation 3D
******************************************************************************/
char ***rd_base::alloc( unsigned int k, unsigned int j, unsigned int i,
			unsigned int bytes )
{
  char ***p3 = 0, **p2 = 0, *p1 = 0;

  unsigned int ic = i * bytes;

  if( p3 = (char ***) malloc( k *         sizeof( char ** ) +
			      k * j *     sizeof( char *  ) +
			      k * j * i * bytes ) ) {
    p2 = (char **) (p3 + k);
    p1 = (char  *) (p3 + k + k * j);

    for( unsigned int kc=0; kc<k; kc++ ) {
      p3[kc] = p2;

      p2 += j;

      for( unsigned int jc=0; jc<j; jc++ ) {
	p3[kc][jc] = p1;

	p1 += ic;
      }
    }
  }

  return p3;
}


/******************************************************************************
Contiguous memory allocation 4D
******************************************************************************/
char ****rd_base::alloc( unsigned int l, unsigned int k,
			 unsigned int j, unsigned int i,
			 unsigned int bytes )
{
  char ****p4 = 0, ***p3 = 0, **p2 = 0, *p1 = 0;

  unsigned int ic = i * bytes;

  if( p4 = (char ****) malloc( l *             sizeof( char *** ) +
			       l * k *         sizeof( char **  ) +
			       l * k * j *     sizeof( char *   ) +
			       l * k * j * i * bytes ) ) {
    p3 = (char ***) (p4 + l);
    p2 = (char  **) (p4 + l + l * k);
    p1 = (char   *) (p4 + l + l * k + l * k * j);

    for( unsigned int lc=0; lc<l; lc++ ) {
      p4[lc] = p3;

      p3 += k;

      for( unsigned int kc=0; kc<k; kc++ ) {
	p4[lc][kc] = p2;

	p2 += j;

	for( unsigned int jc=0; jc<j; jc++ ) {
	  p4[lc][kc][jc] = p1;
	  
	  p1 += ic;
	}
      }
    }
  }

  return p4;
}


/******************************************************************************
Initalize the base system
******************************************************************************/
void rd_base::alloc(unsigned int *dims, int mult)
{
  cell_mult_ = mult;

  height_ = dims[0] * cell_mult_;
  width_  = dims[1] * cell_mult_;

  if( vector_data ) {
    free( vector_data );
    free( vector_diverge );

    free( morphigen );
    free( dmorphigen );

    free( diff_rate );

    free( react_rate );
  
    free( d_tensor );
    free( d_template );

    free( u_h );
    free( rhs_func );
    free( residuals );
    free( errors );
    free( tmp );
  }

  vector_data    = (float ***) alloc( height_, width_, 4, sizeof( float ) );
  vector_diverge = (float ***) alloc( height_, width_, 4, sizeof( float ) );

  morphigen  = (float ***) alloc( nMorphs, height_, width_, sizeof( float ) );
  dmorphigen = (float ***) alloc( nMorphs, height_, width_, sizeof( float ) );

  diff_rate  = (float   *) alloc( nMorphs,                   sizeof( float ) );

  react_rate = (float **) alloc( height_, width_, sizeof( float ) );
  
  d_tensor   = (float ****) alloc( height_, width_, 2, 2, sizeof( float ) );
  d_template = (float ****) alloc( height_, width_, 3, 3, sizeof( float ) );

  rhs_func  = (float ***) alloc( 2, height_, width_, sizeof( float ) );
  u_h       = (float **)  alloc( height_, width_, sizeof( float ) );
  residuals = (float **)  alloc( height_, width_, sizeof( float ) );
  errors    = (float **)  alloc( height_, width_, sizeof( float ) );
  tmp       = (float **)  alloc( height_, width_, sizeof( float ) );

  iminus1ZeroFlux = (int*) alloc( width_, sizeof( int ) );
  iplus1ZeroFlux  = (int*) alloc( width_, sizeof( int ) );

  jminus1ZeroFlux = (int*) alloc( height_, sizeof( int ) );
  jplus1ZeroFlux  = (int*) alloc( height_, sizeof( int ) );
  
  iminus1Periodic = (int*) alloc( width_, sizeof( int ) );
  iplus1Periodic  = (int*) alloc( width_, sizeof( int ) );

  jminus1Periodic = (int*) alloc( height_, sizeof( int ) );
  jplus1Periodic  = (int*) alloc( height_, sizeof( int ) );
  
  unsigned int i = 0;

  iminus1ZeroFlux[i] = i;
  iplus1ZeroFlux [i] = i + 1;

  iminus1Periodic[i] = (i + width_ - 1) % width_;
  iplus1Periodic [i] = (i + 1) % width_;

  for( i=1; i<width_-1; i+=1) {
    iminus1ZeroFlux[i] = iminus1Periodic[i] = (i - 1);
    iplus1ZeroFlux [i] = iplus1Periodic [i] = (i + 1);
  }

  iminus1ZeroFlux[i] = i - 1;
  iplus1ZeroFlux [i] = i;

  iminus1Periodic[i] = (i + width_ - 1) % width_;
  iplus1Periodic [i] = (i + 1) % width_;


  unsigned int j = 0;

  jminus1ZeroFlux[j] = j;
  jplus1ZeroFlux [j] = j + 1;

  jminus1Periodic[j] = (j + height_ - 1) % height_;
  jplus1Periodic [j] = (j + 1) % height_;      

  for( j=1; j<height_-1; j+=1) {
    jminus1ZeroFlux[j] = jminus1Periodic[j] = (j - 1);
    jplus1ZeroFlux [j] = jplus1Periodic [j] = (j + 1);      
  }

  jminus1ZeroFlux[j] = j - 1;
  jplus1ZeroFlux [j] = j;

  jminus1Periodic[j] = (j + height_ - 1) % height_;
  jplus1Periodic [j] = (j + 1) % height_;      
}


/******************************************************************************
Initalize the base vars.
******************************************************************************/
void rd_base::initialize( float ***vector, bool reset )
{
  register int k0, k1, k_1, j0, j1, j_1, i0, i1, i_1;
  register float u, v;

  vector_min=MAX_FLOAT, vector_max=-MAX_FLOAT;

  for (int jj=0; jj<height_; jj++) {

    if( height_ > 1 ) {
      j0 = (int) (jj / cell_mult_);
      j1 = (j0+1) % (height_ / cell_mult_);

      v = (float) (jj % cell_mult_) / (float) (cell_mult_);

    } else {
      j0 = 0;
      j1 = 0;
      v = 0;
    }

    for (int ii=0; ii<width_; ii++) {

      if( width_ > 1 ) {
	i0 = (int) (ii / cell_mult_);
	i1 = (i0+1) % (width_ / cell_mult_);
	  
	u = (float) (ii % cell_mult_) / (float) (cell_mult_);
      } else {
	i0 = 0;
	i1 = 0;
	u = 0;
      }

      for( int index=0; index<4; index++ )
	vector_data[jj][ii][index] =
	  LERP( v,
		LERP( u,
		      vector[j0][i0][index],
		      vector[j0][i1][index] ),
		LERP( u,
		      vector[j1][i0][index],
		      vector[j1][i1][index] ) );
    }
  }

  for (int j=0; j<height_; j++) {

    int j_1 = jminus1[j];
    int j1  = jplus1 [j];

    for (int i=0; i<width_; i++) {

      int i_1 = iminus1[i];
      int i1  = iplus1 [i];
    
      vector_diverge[j][i][0] =
	(vector_data[j][i][0] - vector_data[j][i_1][0]);
	
      vector_diverge[j][i][1] =
	(vector_data[j][i][1] - vector_data[j_1][i][1]);
	
      vector_diverge[j][i][2] = 0;
	
      vector_diverge[j][i][3] =
	vector_diverge[j][i][0] +
	vector_diverge[j][i][1] +
	vector_diverge[j][i][2];

      if( vector_min > vector_data[j][i][3] )
	vector_min = vector_data[j][i][3];
      if( vector_max < vector_data[j][i][3] )
	vector_max = vector_data[j][i][3];
    }
  }
  
  /* Min - max are the same. */
  if( vector_max-vector_min < 1.0e-8 ) {
    vector_min -= 1.0;
    vector_max += 1.0;
  }

  /* Min - max have a small range. */
  else if( vector_max-vector_min < 1.0e-4 ) {
    float ave  = (vector_max+vector_min) / 2.0;
    float diff = (vector_max-vector_min);

    vector_min = ave - 1.0e3 * diff;
    vector_max = ave + 1.0e3 * diff;
  }
}


/******************************************************************************
Set the reaction and diffusion rates.
******************************************************************************/

void rd_base::set_rates( bool rate_type )
{
  fprintf( stdout, "Rate unnormalized  %f %f\n", vector_min, vector_max );

  // Variable Reaction rates.
  set_reaction_rates( react_rate, rate_type );

  // Normally the diffusion rates are divided by the cell size.
  // However, this is done in the interface.
  // float cell_size = 1.00;
  // float cell_area = cell_size * cell_size;

  float ts;

  if( solution_ == EXPLICIT )
    ts = time_step_;
  else
    ts = 1.0;

  // Constant Diffusion rates.
  diff_rate[0] = ts * a_diff_rate_;
  diff_rate[1] = ts * b_diff_rate_;

  cerr << "Diffusion Rates " << diff_rate[0] << "  " << diff_rate[1] << endl;
  cerr << "Diffusion Rates " << a_diff_rate_ << "  " << b_diff_rate_ << endl;
}


/******************************************************************************
Set the reaction rates.
******************************************************************************/
void rd_base::set_reaction_rates( float **rate,
				  bool rate_type )
{
  float scale = 1.0 / (vector_max-vector_min);

  float min_mag_norm=MAX_FLOAT, max_mag_norm=-MAX_FLOAT;

  float ts, u;

  if( solution_ == EXPLICIT )
    ts = time_step_;
  else // if( solution_ == IMPLICIT )
    ts = 1.0;

  /* Normalize the values at the posts. */
  for (int k=0, kk=0; k<height_; k++) {
    for (int j=0, jj=0; j<height_; j++) {
      for (int i=0, ii=0; i<width_; i++) {

	u = (vector_data[j][i][3]-vector_min) * scale;

	if( reaction_ == TURING )
	  rate[j][i] = ts / (rr_coef_1_ + rr_coef_2_ * u);
	else
	  rate[j][i] = ts * (rr_coef_1_ + rr_coef_2_ * u);

	if( min_mag_norm > rate[j][i] ) min_mag_norm = rate[j][i];
	if( max_mag_norm < rate[j][i] ) max_mag_norm = rate[j][i];
      }
    }
  }

  fprintf( stdout,
	   "Rate normalized    %f %f\n\n", min_mag_norm, max_mag_norm );
}


/******************************************************************************
Set the Diffusion Matrix.
******************************************************************************/
void rd_base::set_diffusion( unsigned int diffusion )
{
  if( laplacian_ == UNIFORM )
    set_diffusion_uniform( );
  else if( laplacian_ == INHOMOGENEOUS   )
    set_diffusion_inhomogeneous( );
}


/******************************************************************************
Set the Diffusion Matrix.
******************************************************************************/
#define BOX 0

void rd_base::set_diffusion_uniform()
{
  // First set up the diffusion tensors - this is exactly the same as for the
  // inhomogeneoust method.
  set_diffusion_inhomogeneous( );

  // By assuming the diffusion does not change with cell location
  // a mask for convolving can be created.
  // This is not correct but is what Witkin and Kass assumed
  // and it gives descent results.
  for (int jj=0; jj<height_; jj++) {
    for (int ii=0; ii<width_; ii++) {
      
      // Remove the predivsion done for the invarient method.
      d_tensor[jj][ii][0][0] *= 2.0;
      d_tensor[jj][ii][0][1] *= 4.0;
      d_tensor[jj][ii][1][0] *= 4.0;
      d_tensor[jj][ii][1][1] *= 2.0;

      // This for the + plus (cross) template of the Laplacian.
      float f = (1./4.); // Const for the cross derivatives.
	  
      d_template[jj][ii][0][0] =  f  * d_tensor[jj][ii][0][1];
      d_template[jj][ii][0][1] =  1. * d_tensor[jj][ii][1][1];
      d_template[jj][ii][0][2] = -f  * d_tensor[jj][ii][0][1];

      d_template[jj][ii][1][0] =  1. * d_tensor[jj][ii][0][0];
      d_template[jj][ii][1][1] = -2. *
	(d_tensor[jj][ii][0][0] + d_tensor[jj][ii][1][1]);
      d_template[jj][ii][1][2] =  1. * d_tensor[jj][ii][0][0];

      d_template[jj][ii][2][0] = -f  * d_tensor[jj][ii][1][0];
      d_template[jj][ii][2][1] =  1. * d_tensor[jj][ii][1][1];
      d_template[jj][ii][2][2] =  f  * d_tensor[jj][ii][1][0];


      //  For adding in the x template with the + template for a box template.
      if( BOX ) {

	float b = (1./2.) * (2./3.); // (1./2.) is a const do not change.

	d_template[jj][ii][0][0] += b *  1. * d_tensor[jj][ii][0][1];
	d_template[jj][ii][0][1] += b * -2. * d_tensor[jj][ii][1][1];
	d_template[jj][ii][0][2] += b *  1. * d_tensor[jj][ii][0][1];

	d_template[jj][ii][1][0] += b * -2. * d_tensor[jj][ii][0][0];
	d_template[jj][ii][1][1] += b *
	  ( 4.0 * (d_tensor[jj][ii][0][0] + d_tensor[jj][ii][1][1]) -
	    2.0 * (d_tensor[jj][ii][0][1] + d_tensor[jj][ii][1][0]) );
	d_template[jj][ii][1][2] += b * -2. * d_tensor[jj][ii][0][0];

	d_template[jj][ii][2][0] += b *  1. * d_tensor[jj][ii][1][0];
	d_template[jj][ii][2][1] += b * -2. * d_tensor[jj][ii][1][1];
	d_template[jj][ii][2][2] += b *  1. * d_tensor[jj][ii][1][0];
      }
    }
  }

  // For debugging - make sure the sum of the template is zero.
  int jj = height_ / 4;
  int ii = width_  / 4;

  float sum = 0;

  fprintf( stdout, " %d  %d\n", ii, jj );

  fprintf( stdout, " %f  %f\n %f  %f\n\n",
	   d_tensor[jj][ii][0][0],
	   d_tensor[jj][ii][0][1],
	   d_tensor[jj][ii][1][0],
	   d_tensor[jj][ii][1][1] );


  for( int mj=0; mj<3; mj++ ) {
    for( int mi=0; mi<3; mi++ ) {
      sum += d_template[jj][ii][mj][mi];

      fprintf( stdout, " %f  ", d_template[jj][ii][mj][mi] );
    }
    fprintf( stdout, "\n" );
  }
  fprintf( stdout, "\n" );

  fprintf( stdout, "\nsum %f \n\n", sum );}


/******************************************************************************
Set the Diffusion Matrix.
******************************************************************************/
void rd_base::set_diffusion_inhomogeneous()
{
  float min_mag_norm[4], max_mag_norm[4];

  for( unsigned int t=0; t<4; t++ ) {
    min_mag_norm[t] =  MAX_FLOAT;
    max_mag_norm[t] = -MAX_FLOAT;
  }

  /* Calculate the diffusion tensor between the post values. */

  if( gradient_ ) {
    int i0, i1, i_1, j0, j1, j_1, k0, k1, k_1;
    float u, v, w;

    float dg, grad[3], dotSign = 1.0;

    if( reaction_ == rd_base::TURING )
      dotSign = -1.0;

    for (int j=1; j<height_-1; j++) {
      for (int i=1; i<width_-1; i++) {
	dg = 0;

	// Calculate the gradient using central differences
	// of the a morphigen.
	grad[0] = morphigen[0][j  ][i+1] - morphigen[0][j  ][i-1];
	grad[1] = morphigen[0][j+1][i  ] - morphigen[0][j-1][i  ];

	if( fabs(grad[0]) > MIN_FLOAT || fabs(grad[1]) > MIN_FLOAT ) {
	  grad[2] = sqrt(grad[0]*grad[0] + grad[1]*grad[1]);
	      
	  // Get the dot product of the morphigen gradient and
	  // the vector field.
	  float dotProd = dotSign * (grad[0] * vector_data[j][i][0] +
				     grad[1] * vector_data[j][i][1]) / grad[2];
	    
	  // Depending on the dot product change the diffusion.
	  dg = gammp_table[ INDEX(dotProd, MAX_GAMMP) ];
	}

	// setup the principal diffusivity matrix
	float pd00 = diff_coef_1_ + dg;
	float pd11 = diff_coef_2_ - dg;

	// Square the difusion matrix so that it is positive.
	pd00 *= pd00;
	pd11 *= pd11;

	float cos_ang = vector_data[j][i][0];
	float sin_ang = vector_data[j][i][1];
	
	float cos2 = cos_ang*cos_ang;
	float sin2 = sin_ang*sin_ang;

	// Calculate the tensor for this particular vector.
	// NOTE: premultiple the principal difisivity valuies by 1/2
	// NOTE: premultiple the secondary difisivity valuies by 1/4
	// This is so that it does not need to be done when calculating
	// the finite differences.

	d_tensor[j][i][0][0] = (pd00 * cos2 + pd11 * sin2) * 0.5;
	d_tensor[j][i][0][1] = 
	d_tensor[j][i][1][0] = (pd00 - pd11) * cos_ang * sin_ang * 0.25;
	d_tensor[j][i][1][1] = (pd00 * sin2 + pd11 * cos2) * 0.5;

	for( unsigned int t=0; t<4; t++ ) {
	  if( min_mag_norm[t] > d_tensor[j][i][t/2][t%2] )
	    min_mag_norm[t] = d_tensor[j][i][t/2][t%2];
	  if( max_mag_norm[t] < d_tensor[j][i][t/2][t%2] )
	    max_mag_norm[t] = d_tensor[j][i][t/2][t%2];
	}
      }
    }

  } else {

    for (int j=0; j<height_; j++) {
      for (int i=0; i<width_; i++) {

	// setup the principal diffusivity matrix
	float pd00 = diff_coef_1_;
	float pd11 = diff_coef_2_;

	// Square the difusion matrix so that it is positive.
	pd00 *= pd00;
	pd11 *= pd11;

	float cos_ang = vector_data[j][i][0];
	float sin_ang = vector_data[j][i][1];
	
	float cos2 = cos_ang*cos_ang;
	float sin2 = sin_ang*sin_ang;
	/*
	  if( j == height_ / 4 && i == width_ / 4 ) {
	  cerr << aval << endl;
	  cerr << pd00 << endl;
	  cerr << pd11 << endl;
	  cerr << cos_ang << endl;
	  cerr << sin_ang << endl;
	  }
	*/
	
	// Calculate the tensor for this particular vector.
	// NOTE: premultiple the principal diffusivity valuies by 1/2
	// NOTE: premultiple the secondary diffusivity valuies by 1/4
	// This is so that it does not need to be done when calculating
	// the finite differences.
	d_tensor[j][i][0][0] = (pd00 * cos2 + pd11 * sin2) * 0.5;
	d_tensor[j][i][0][1] = 
	  d_tensor[j][i][1][0] = (pd00 - pd11) * cos_ang * sin_ang * 0.25;
	d_tensor[j][i][1][1] = (pd00 * sin2 + pd11 * cos2) * 0.5;

	for( unsigned int t=0; t<4; t++ ) {
	  if( min_mag_norm[t] > d_tensor[j][i][t/2][t%2] )
	    min_mag_norm[t] = d_tensor[j][i][t/2][t%2];
	  if( max_mag_norm[t] < d_tensor[j][i][t/2][t%2] )
	    max_mag_norm[t] = d_tensor[j][i][t/2][t%2];
	}
      }
    }

    // For debugging print out what the mask would look like.
//     int j = height_ / 4;
//     int i = width_  / 4;

//     int i1 = i+1;
//     int j1 = j+1;
	    
//     int i_1 = i-1;
//     int j_1 = j-1;

//     fprintf( stdout, " %d  %d\n", i, j );

//     fprintf( stdout, " %f  %f\n %f  %f\n\n",
// 	     d_tensor[j][i][0][0] * 2.0,
// 	     d_tensor[j][i][0][1] * 4.0,
// 	     d_tensor[j][i][1][0] * 4.0,
// 	     d_tensor[j][i][1][1] * 2.0 );

//     fprintf( stdout, " %f  %f  %f\n %f  %f  %f\n\n",
// 	     d_tensor[j ][i1][0][0], d_tensor[j][i  ][0][0], d_tensor[j  ][i_1][0][0],
// 	     d_tensor[j1][i ][1][1], d_tensor[j][i  ][1][1], d_tensor[j_1][i  ][1][1] );

//     fprintf( stdout, "Nonuniforn Anisotropic\n  %f  %f  %f\n %f  %f  %f\n %f  %f  %f\n\n\n",

// 	     -d_tensor[j ][i1 ][0][1],
// 	     (d_tensor[j1][i  ][1][1] +     d_tensor[j][i][1][1]),
// 	     d_tensor[j ][i_1][0][1],

// 	     ( d_tensor[j ][i ][0][0] +     d_tensor[j][i_1][0][0]),
// 	     -( d_tensor[j ][i1][0][0] + 2.0*d_tensor[j][i  ][0][0] + d_tensor[j  ][i_1][0][0]) -
// 	     ( d_tensor[j1][i ][1][1] + 2.0*d_tensor[j][i  ][1][1] + d_tensor[j_1][i  ][1][1]),
// 	     ( d_tensor[j ][i1][0][0] +     d_tensor[j][i  ][0][0]),
		     
		     
// 	     d_tensor[j1 ][i][1][0],
// 	     (d_tensor[j  ][i][1][1] +      d_tensor[j_1][i][1][1]),
// 	     -d_tensor[j_1][i][1][0] );

//     fprintf( stdout, "Uniforn Anisotropic - Approximation\n %f  %f  %f\n %f  %f  %f\n %f  %f  %f\n\n\n",

// 	     -d_tensor[j][i][0][1],
// 	     (d_tensor[j][i][1][1] +     d_tensor[j][i][1][1]),
// 	     d_tensor[j][i][0][1],

// 	     ( d_tensor[j][i][0][0] +     d_tensor[j][i][0][0]),
// 	     -( d_tensor[j][i][0][0] + 2.0*d_tensor[j][i][0][0] + d_tensor[j][i][0][0]) -
// 	     ( d_tensor[j][i][1][1] + 2.0*d_tensor[j][i][1][1] + d_tensor[j][i][1][1]),
// 	     ( d_tensor[j][i][0][0] +     d_tensor[j][i][0][0]),
		     
		     
// 	     d_tensor[j][i][1][0],
// 	     (d_tensor[j][i][1][1] +      d_tensor[j][i][1][1]),
// 	     -d_tensor[j][i][1][0] );
  }    

//   fprintf( stdout, "Tensors" );
  
//   for( unsigned int t=0; t<4; t++ )
//     fprintf( stdout, "   %f %f", min_mag_norm[t], max_mag_norm[t] );

//   fprintf( stdout, "\n" );
}
