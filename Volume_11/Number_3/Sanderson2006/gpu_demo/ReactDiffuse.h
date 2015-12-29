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

#ifndef __REACT_DIFFUSE_H__
#define __REACT_DIFFUSE_H__

#include <GL/glew.h>

#include "Fbuffer.h"

#include <stdlib.h>

#include <Cg/cgGL.h>
#include <iostream>

#define MAX_TEXTURES 5
#define MAX_RATES    10

#define NORMAL 1
#define SPOTTED_PUFFER 0
#define MAP_TOBY_PUFFER 0
#define PAPUA_TOBY_PUFFER 0
#define ZEBRA_GOBY 0

void cgErrorCallback();

class ReactDiffuse  
{
public:

  ReactDiffuse();
  ~ReactDiffuse();

  void oglErrorCallback( char* func );

  //---------------------------------------------------------------//
  // initialize cg                                                 //
  void initCG();

  //---------------------------------------------------------------//
  // initialize the frmebuffer, view, and display lists               //
  void initFbuffer();

  void deleteGrids();


  //---------------------------------------------------------------//
  // iterate through time over the domain using the reaction       //
  //  diffusion equations                                          //  
  void setInitalState();

  void updateStateExplicit( unsigned int num_passes = 1);
  void updateStateImplicit( unsigned int num_passes = 1);
  void updateStateImplicitRHS( unsigned int morph );
  bool updateStateImplicitRelaxGPU( unsigned int nr_steps, unsigned int morph);
  float updateStateImplicitResiduals( unsigned int morph );
  void updateStateImplicitClamp();

  float reduction( GLuint gl_InputTexID );

  //--------------------------------------------------------------//
  // functions for rendering the reaction diffusion pbuffer       //
  void getTextureCoords(unsigned int &b, unsigned int &t,
			unsigned int &l, unsigned int &r);
  GLuint getTextureID() { return _gl_MorphogenTexID[0]; };

  unsigned int getTextureWidth()       { return _pbWidth; };
  unsigned int getTextureHeight()      { return _pbHeight; };

  //--------------------------------------------------------------//
  // getter and setter functions                                  //
  void setSolution(unsigned int solution) {
    _solution = (solution_type) solution;
  };
  void setLaplacian(unsigned int laplacian) {
    _laplacian = (laplacian_type) laplacian;
  };
  void setBoundary(unsigned int boundary) {
    _boundary = (boundary_type) boundary;
    if( _boundary == ZERO_FLUX_VAR ) {
      _gl_dlInterior = _gl_dlInteriorZeroFlux;
      _gl_dlBoundary = _gl_dlBoundaryZeroFlux;
    } else if( _boundary == PERIODIC ) {
      _gl_dlInterior = _gl_dlInteriorPeriodic;
      _gl_dlBoundary = _gl_dlBoundaryPeriodic;
    }
  };

  void setReaction(unsigned int reaction) {
    if( reaction )
      _reaction = (reaction_type) (reaction + 1);
    else
      _reaction = (reaction_type) reaction;
  };

  void setABVariance(float variance) { _ab_variance = variance; };
  void setReactionConstVariance(float var) { _reaction_const_variance = var; };

  void setA(float a) { _a = a; };
  void setB(float b) { _b = b; };

  void setReaction_Const0(float val) { _reaction_const0 = val; };
  void setReaction_Const1(float val) { _reaction_const1 = val; };
  void setReaction_Const2(float val) { _reaction_const2 = val; };

  void setTimeStep(float ts) {
    _time_step = ts;
    _time_step_inv = (float) 1.0 / (_time_step * _time_mult);
  };
  void setTimeMult(float tm) {
    _time_mult = tm;
    _time_step_inv = (float) 1.0 / (_time_step * _time_mult);
  };

  void setTheta(float theta) { _cn_theta = theta; };
 
  void setRRCoef1(float c1) { _rr_coef_1 = c1; };
  void setRRCoef2(float c2) { _rr_coef_2 = c2; };

  void setMixingRate(float mr) { _mixing_rate = mr; };

  void setDiffCoef1(float dc) { _diff_coef_1 = dc; };
  void setDiffCoef2(float dc) { _diff_coef_2 = dc; };

  void setADiffRate(float dr) { _a_diff_rate = dr; };
  void setBDiffRate(float dr) { _b_diff_rate = dr; };
  void setCDiffRate(float dr) { _c_diff_rate = dr; };
  void setDDiffRate(float dr) { _d_diff_rate = dr; };

  void setGradient(unsigned int gradient) { _gradient = gradient; };
  void setMult(unsigned int mult ) { _mult = mult; };

  void setDataFile(std::string fname, unsigned int t) {
    _data_file.filename = fname; 
    _data_file.type = t;
  };

  float getABVariance() { return _ab_variance; };
  float getReactionConstVariance() { return _reaction_const_variance; };

  float getA() { return _a; };
  float getB() { return _b; };

  float getReaction_Const0() { return _reaction_const0; };
  float getReaction_Const1() { return _reaction_const1; };
  float getReaction_Const2() { return _reaction_const2; };

  float getTimeStep() { return _time_step; };
  float getTimeMult() { return _time_mult; };

  float getTheta() { return _cn_theta; };

  float getMixingRate() { return _mixing_rate; };

  float getRRCoef1() { return _rr_coef_1; };
  float getRRCoef2() { return _rr_coef_2; };

  float getDiffCoef1() { return _diff_coef_1; };
  float getDiffCoef2() { return _diff_coef_2; };

  float getADiffRate() { return _a_diff_rate; };
  float getBDiffRate() { return _b_diff_rate; };
  float getCDiffRate() { return _c_diff_rate; };
  float getDDiffRate() { return _d_diff_rate; };

  unsigned int getGradient()  { return _gradient; };
  unsigned int getMult()      { return _mult; };

  std::string getDataFileName() { return _data_file.filename; };


  enum reaction_type { TURING=0,
		       GRAY_SCOTT=2,
		       BRUSSELATOR=3,
		       MAX_REACTIONS=4 };

  enum solution_type { EULER_EXPLICIT=0,
		       EULER_SEMI_IMPLICIT=1, EULER_THETA_IMPLICIT=2,
		       MAX_SOLUTIONS=3 };

  enum boundary_type { ZERO_FLUX_ALL=0, ZERO_FLUX_VAR=1, PERIODIC=2 };

  enum laplacian_type { INHOMOGENEOUS=0, UNIFORM=1 };

  enum constant_type { REACTION=0, DIFFUSION=1, DIFFUSION2=2, VARIANCE=3 };

  std::string constant_names[MAX_TEXTURES];

  typedef struct {
    std::string filename;
    unsigned int type;
  } datafile;
  
  float frand( float min, float max, long &seed );

private:
  Fbuffer *_fbuffer;       // the fbuffer

  datafile _data_file;

  unsigned int _mult, _pbWidth, _pbHeight;  // h and w of fbuffer in pixels
  
  reaction_type  _reaction;
  laplacian_type _laplacian;
  solution_type  _solution;
  solution_type  _last_solution;
  boundary_type  _boundary;

  float _time_step, _time_mult, _time_step_inv;

  float _cn_theta; 

  unsigned int _gradient;

  float _mixing_rate;

  float _a, _b, _reaction_const0, _reaction_const1, _reaction_const2;
  float _ab_variance, _reaction_const_variance;

  float _rr_coef_1, _rr_coef_2;

  float _a_diff_rate, _b_diff_rate, _c_diff_rate, _d_diff_rate;

  float _diff_coef_1, _diff_coef_2;


  float _a_diff_rate_theta, _b_diff_rate_theta,
    _c_diff_rate_theta, _d_diff_rate_theta;
  float _a_diff_rate_1_theta, _b_diff_rate_1_theta,
    _c_diff_rate_1_theta, _d_diff_rate_1_theta;

  float *gammp_table;

  float ***_v_data;
  float **_reaction_const0_data, **_reaction_const1_data, **_reaction_const2_data;
  float **_rr_data, ***_dt_data;

  float _vmag_min, _vmag_max;

public:
  float *_texMorphigens;
  float *_texReaction, *_texDiffusion, *_texDiffusion2, *_texVariance;
  float *_texRHS, *_texResiduals, *_texReduction;

private:

  CGcontext   _cg_Context;

  // initialization of frag programs
  CGprofile _cg_Profile;
  CGprogram _cg_InitProgram;
  CGprogram _cg_Program;

  // reaction diffusion frag programs
  CGprogram _cg_Explicit[MAX_REACTIONS];

  CGprogram _cg_Implicit_RHS[MAX_SOLUTIONS];
  CGprogram _cg_Implicit_Relax[MAX_SOLUTIONS];
  CGprogram _cg_Implicit_Residuals[MAX_SOLUTIONS];
  CGprogram _cg_Implicit_Clamp;
  CGprogram _cg_Summation;

  CGparameter _cg_Morphigens, _cg_ConstsTex[MAX_TEXTURES];
  CGparameter _cg_DiffRates,  _cg_MixRates;
  CGparameter _cg_RHS;
  CGparameter _cg_Input;

  GLuint _gl_MorphogenTexID[2];         // the texture ID for the Morphogens
  GLuint _gl_RHSTexID;                  // the texture ID for the RHS
  GLuint _gl_ResidualsTexID;            // the texture ID for the Residuals
  GLuint _gl_ConstsTexID[MAX_TEXTURES]; // the texture ID for the Constants
  GLuint _gl_ReductionTexID[2];

  // display lists
  GLuint _gl_dlAll;
  GLuint _gl_dlAllZeroFlux;
  GLuint _gl_dlInteriorZeroFlux, _gl_dlBoundaryZeroFlux;
  GLuint _gl_dlInteriorPeriodic, _gl_dlBoundaryPeriodic;

  GLuint _gl_dlInterior, _gl_dlBoundary;

  GLenum errorCode;

public:
  void updateTexture(GLuint &texID, float *texData);
  void saveConstTex(constant_type type);
  void saveVarianceTex();
  void generateConstantsTex();
  void generateDiffusionTensor( bool gradient, unsigned int var );

  void createVectorData();

  void updateConstantsDT();
  void updateConstantsReact();

  void generateDisplayLists();

  void getTexValues( );
};

#endif // __REACT_DIFFUSE_H__
