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

#ifndef DEFINE_RD_TURING_H
#define DEFINE_RD_TURING_H 1

#include "rd_base.h"

/******************************************************************************
turing_rd class
******************************************************************************/
class rd_turing: public rd_base
{
public:
  rd_turing();

  virtual void initialize( float ***vector, bool reset = 1 );

  virtual void alloc( unsigned int *dims, int mult = 1);

  virtual float reaction( unsigned int n, unsigned int i, unsigned int j);

  float **alpha;     // Growth parameter.
  float **beta;      // Decay parameter.
};

#endif  // DEFINE_TURING_RD_H
