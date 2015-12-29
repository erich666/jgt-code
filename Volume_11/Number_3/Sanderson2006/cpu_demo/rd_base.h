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

#ifndef DEFINE_RD_BASE_H
#define DEFINE_RD_BASE_H 1

#include <vector>

#define MIN_FLOAT 1.0e-8
#define MAX_FLOAT 1.0e12

/******************************************************************************
rd_base class
******************************************************************************/
class rd_base
{
public:

  enum { NORMALIZED=0, VARIABLE=1 };
  enum diffusion_type  { ISOTROPIC=0, ANISOTROPIC=1 };
  enum relaxation_type { JACOBI=0, GAUSS_SEIDEL=1, RED_BLACK=2 };
  enum reaction_type   { TURING=0, GRAY_SCOTT=1, MEINHARDT=2, OREGONATOR=3 };
  enum laplacian_type  { INHOMOGENEOUS=0, UNIFORM=1, ARS=2 };
  enum solution_type   { EXPLICIT=0, IMPLICIT=1, THETA=2, };
  enum boundary_type   { ZERO_FLUX=0, PERIODIC=1 };
  enum morphigen_type  { MORPHIGEN_A=0, MORPHIGEN_B=1 };

protected:

public:
  //--------------------------------------------------------------//
  // getter and setter functions                                  //
  void setSolution (int solution  )
  { solution_   = (solution_type) solution; };
  void setBoundary(int boundary) {
    boundary_ = (boundary_type) boundary;

    if( boundary_ == ZERO_FLUX ) {
      iplus1  = iplus1ZeroFlux;
      iminus1 = iminus1ZeroFlux;
      jplus1  = jplus1ZeroFlux;
      jminus1 = jminus1ZeroFlux;

    } else {
      iplus1  = iplus1Periodic;
      iminus1 = iminus1Periodic;
      jplus1  = jplus1Periodic;
      jminus1 = jminus1Periodic;
    }
  };

  void setReaction (int reaction  )
  { reaction_   = (reaction_type) reaction; };
  void setDiffusion (int diffusion  )
  { diffusion_   = (diffusion_type) diffusion; };
  void setLaplacian (int laplacian  )
  { laplacian_   = (laplacian_type) laplacian; };
  void setRelaxation(int relaxation)
  { relaxation_ = (relaxation_type) relaxation; };


  void setABVariance(float variance) { ab_variance_ = variance; };
  void setReactionConstVariance(float var) { reaction_const_variance_ = var; };

  void setA(float a) { a_ = a; };
  void setB(float b) { b_ = b; };

  void setReaction_Const0(float val) { reaction_const0_ = val; };
  void setReaction_Const1(float val) { reaction_const1_ = val; };

  void setTimeStep(float ts) {
    time_step_ = ts;
    time_step_inv_ = 1.0 / (time_step_ * time_mult_);
  };
  void setTimeMult(float tm) {
    time_mult_ = tm;
    time_step_inv_ = 1.0 / (time_step_ * time_mult_);

  };

  void setTheta(float theta) { theta_ = theta; };
 
  void setRRCoef1(float c1) { rr_coef_1_ = c1; };
  void setRRCoef2(float c2) { rr_coef_2_ = c2; };

  void setDiffCoef1(float dc) { diff_coef_1_ = dc; };
  void setDiffCoef2(float dc) { diff_coef_2_ = dc; };

  void setADiffRate(float dr) { a_diff_rate_ = dr; };
  void setBDiffRate(float dr) { b_diff_rate_ = dr; };

  void setGradient(int gradient) { gradient_ = gradient; };
  void setMult( int mult ) { cell_mult_ = mult; };

  float getABVariance() { return ab_variance_; };
  float getReactionConstVariance() { return reaction_const_variance_; };

  float getA() { return a_; };
  float getB() { return b_; };

  float getReaction_Const0() { return reaction_const0_; };
  float getReaction_Const1() { return reaction_const1_; };

  float getTimeStep() { return time_step_; };
  float getTimeMult() { return time_mult_; };

  float getTheta() { return theta_; };

  float getRRCoef1() { return rr_coef_1_; };
  float getRRCoef2() { return rr_coef_2_; };

  float getDiffCoef1() { return diff_coef_1_; };
  float getDiffCoef2() { return diff_coef_2_; };

  float getADiffRate() { return a_diff_rate_; };
  float getBDiffRate() { return b_diff_rate_; };

  int   getGradient()  { return gradient_; };
  int   getMult()      { return cell_mult_; };

  rd_base();

  char    *alloc( unsigned int i,
		  unsigned int bytes );
  char   **alloc( unsigned int j, unsigned int i,
		  unsigned int bytes );
  char  ***alloc( unsigned int k,
		  unsigned int j, unsigned int i,
		  unsigned int bytes );
  char ****alloc( unsigned int l, unsigned int k,
		  unsigned int j, unsigned int i,
		  unsigned int bytes );

  virtual void alloc( unsigned int *dims, int mult = 1);

  virtual void initialize( float ***vector, bool reset = 1 );

  virtual void set_rates( bool rate_type );
 
  virtual void set_reaction_rates( float **rate,
				   bool rate_type );

  void set_diffusion( unsigned int diffusion );
  void set_diffusion_uniform( );
  void set_diffusion_inhomogeneous( );

  virtual float diffusion( float **, unsigned int i, unsigned int j );

  static float reaction_rate( float u, bool rate_type );
  static float diffusion_rate_A( float u, bool rate_type );
  static float diffusion_rate_B( float u, bool rate_type );

  virtual float reaction( unsigned n,
			  unsigned int i, unsigned int j ) = 0;

  virtual bool next_step_explicit_euler( );
  virtual bool next_step_implicit_euler( );
  virtual void implicit_euler_rhs( float **rhs, morphigen_type morphigen );
  virtual float implicit_euler_residual( float **resid,
					 float **u, float **rhs, float diff );
  
  virtual float implicit_euler_relax( float **u, float **rhs, float diff,
				      int i_1, int j_1,
				      int i,   int j,
				      int i1,  int j1 );

  virtual void react_and_diffuse( );

  virtual void implicit_solve( float **v, float **rhs, float diffusion );

  virtual bool relax       ( float **u, float **rhs, float diff );
  virtual bool relax_gs    ( float **u, float **rhs, float diff );
  virtual bool relax_gs_rb ( float **u, float **rhs, float diff );
  virtual bool relax_jacobi( float **u, float **rhs, float diff );
  virtual float difference( float **u_new, float **u_old  );


  relaxation_type relaxation_;
  reaction_type reaction_;
  laplacian_type laplacian_;
  diffusion_type diffusion_;
  solution_type solution_;
  boundary_type boundary_;

  float time_step_;              //  Euler integration step.
  float time_mult_;              //  Inverse Euler integration mult step.
  float time_step_inv_;          //  Inverse Euler integration step.

  float theta_;

  int   gradient_;

  float a_, b_, reaction_const0_, reaction_const1_;
  float ab_variance_, reaction_const_variance_;

  float rr_coef_1_, rr_coef_2_;

  float a_diff_rate_, b_diff_rate_;

  float diff_coef_1_, diff_coef_2_;


  float _a_diff_rate_theta, _b_diff_rate_theta;
  float _a_diff_rate_1_theta, _b_diff_rate_1_theta;




  unsigned int height_, width_;

  int cell_mult_;

  int *iplus1, *iminus1;
  int *jplus1, *jminus1;

  int *iplus1ZeroFlux, *iminus1ZeroFlux;
  int *jplus1ZeroFlux, *jminus1ZeroFlux;

  int *iplus1Periodic, *iminus1Periodic;
  int *jplus1Periodic, *jminus1Periodic;

  float ***vector_data;
  float ***vector_diverge;

  float vector_max, vector_min;

  int nMorphs;          //  Number of morphigens.
  int mIndex;           //  Morphigen index.

  float ***morphigen;   //  Morphogen concentration.
  float ***dmorphigen;  //  Change in concentration of the morphogen.


  float **react_rate;   //  Reaction rate   - one for each morpgiden cell..
  float *diff_rate;     //  Diffusion rates - one for each morphigen.


  float ****d_tensor;   //  Diffusion tensor.
  float ****d_template; //  Diffusion template.

  float morph_variance; //  Morphigen varaince.
  float react_variance; //  Reaction varaince.

  int neighborhood;

  float *gammp_table;

  float ***rhs_func;    //  RHS function.
  float **u_h;          //  U.
  float **residuals;    //  Residuals.
  float **errors;       //  Errors.

  float **tmp;          //  Temporary Storage.

  float min_error_;

protected:

  // Pick a random number between min and max.
  float frand( float min, float max, long &seed );

private:
  unsigned int unsteady;

  float last_min_da, last_min_db;
  float last_max_da, last_max_db;

  float min_da, min_db;
  float max_da, max_db;

  float min_a, min_b;
  float max_a, max_b;

  float min_diff;
};

#endif  // DEFINE_RD_BASE_H
