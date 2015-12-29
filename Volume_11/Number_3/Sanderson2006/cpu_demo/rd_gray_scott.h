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

#ifndef DEFINE_RD_GRAY_SCOTT_H
#define DEFINE_RD_GRAY_SCOTT_H 1

#include "rd_base.h"

/******************************************************************************
rd_gray_scott class
******************************************************************************/
class rd_gray_scott: public rd_base
{
public:
  rd_gray_scott();

  virtual void initialize( float ***vector, bool reset );

  virtual void alloc( unsigned int *dims, int mult = 1);

  virtual float reaction( unsigned int n,
			  unsigned int i, unsigned int j);

  float **feed;   /* Feed rate parameter. */
  float **conv;  /*  Conversion rate parameter. */

  float feed_init;    /* Feed rate parameter. */
  float conv_init;    /* Conversion rate parameter. */

  float f_varriance;  // Maximum varriance in substrate base value.
  float c_varriance;  // Maximum varriance in substrate base value.

  float f_range;      // The varriance affect regularity of pattern.
  float c_range;      // The varriance affect regularity of pattern.

  int perturb;         /* Amount of the perturbed central area. */
};

#endif // DEFINE_RD_GRAY_SCOTT_H
