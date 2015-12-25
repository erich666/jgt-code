/******************************************************************************

  This source code accompanies the Journal of Graphics Tools paper:

  "Fast Ray-Axis Aligned Bounding Box Overlap Tests With Pluecker Coordinates" by
  Jeffrey Mahovsky and Brian Wyvill
  Department of Computer Science, University of Calgary

  This source code is public domain, but please mention us if you use it.

 ******************************************************************************/

#ifndef _AABOX_H
#define _AABOX_H

// axis-aligned bounding box structure

struct aabox
{
	double x0, y0, z0, x1, y1, z1;
};

void make_aabox(double x0, double y0, double z0, double x1, double y1, double z1, aabox *a);

#endif
