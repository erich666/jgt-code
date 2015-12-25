/******************************************************************************

  This source code accompanies the Journal of Graphics Tools paper:

  "Fast Ray-Axis Aligned Bounding Box Overlap Tests With Pluecker Coordinates" by
  Jeffrey Mahovsky and Brian Wyvill
  Department of Computer Science, University of Calgary

  This source code is public domain, but please mention us if you use it.

 ******************************************************************************/

#ifndef _PLUECKER_CLS_H
#define _PLUECKER_CLS_H

#include "ray.h"
#include "aabox.h"

bool pluecker_cls(ray *r, aabox *b);

#endif
