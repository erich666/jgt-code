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

#include "rand.h"

using namespace std;

//------------------------------------------------------------------------
// Global Functions 
//
// function    : cgErrorCallback() 
// description : print out captured CG errors
//------------------------------------------------------------------------
void cgErrorCallback(void)
{
  CGerror LastError = cgGetError();

  if(LastError) {
    cerr << "---------------------------------------------------" << endl;
    cerr << cgGetErrorString(LastError) << endl;
    cerr << "---------------------------------------------------" << endl;
    cerr << endl;
  }
}


//------------------------------------------------------------------------
// function    : oglErrorCallback() 
// description : print out captured OGL errors
//               
//------------------------------------------------------------------------
void ReactDiffuse::oglErrorCallback( char* func ) {
  GLenum errorCode;
  if ((errorCode = glGetError()) != GL_NO_ERROR) 
    fprintf( stderr, "%s(): ERROR: %s\n",
             func, gluErrorString(errorCode) );
}


//------------------------------------------------------------------------
// function    : getTextureCoords() 
// description : return the pbuffer dimensions for use when rendering
//               the pbuffer
//------------------------------------------------------------------------
void ReactDiffuse::getTextureCoords( unsigned int &b, unsigned int &t,
				     unsigned int &l, unsigned int &r )
{
  b = 0; t = _pbHeight; l = 0; r = _pbWidth;
}


//------------------------------------------------------------------------
// function    : frand() 
// description : return a randon value with a guassian deviation
//------------------------------------------------------------------------
float ReactDiffuse::frand( float min, float max, long &seed ) {
  return (min + gasdev( &seed ) * (max-min) );
}
