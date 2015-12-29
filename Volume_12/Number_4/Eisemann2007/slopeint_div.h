/******************************************************************************

  This source code accompanies the Journal of Graphics Tools paper:

  "Fast Ray / Axis-Aligned Bounding Box Overlap Tests using Ray Slopes" 
  by Martin Eisemann, Thorsten Grosch, Stefan Müller and Marcus Magnor
  Computer Graphics Lab, TU Braunschweig, Germany and
  University of Koblenz-Landau, Germany

  This source code is public domain, but please mention us if you use it.

******************************************************************************/
#ifndef _SLOPEINT_DIV_H
#define _SLOPEINT_DIV_H

#include "ray.h"
#include "aabox.h"

bool slopeint_div(ray *r, aabox *b, float *t);

#endif
