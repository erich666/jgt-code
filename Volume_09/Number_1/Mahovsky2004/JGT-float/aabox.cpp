/******************************************************************************

  This source code accompanies the Journal of Graphics Tools paper:

  "Fast Ray-Axis Aligned Bounding Box Overlap Tests With Pluecker Coordinates" by
  Jeffrey Mahovsky and Brian Wyvill
  Department of Computer Science, University of Calgary

  This source code is public domain, but please mention us if you use it.

 ******************************************************************************/

#include "aabox.h"

void make_aabox(float x0, float y0, float z0, float x1, float y1, float z1, aabox *a)
{
	if(x0 > x1)
	{
		a->x0 = x1;
		a->x1 = x0;
	}
	else
	{
		a->x0 = x0;
		a->x1 = x1;
	}
	if(y0 > y1)
	{
		a->y0 = y1;
		a->y1 = y0;
	}
	else
	{
		a->y0 = y0;
		a->y1 = y1;
	}
	if(z0 > z1)
	{
		a->z0 = z1;
		a->z1 = z0;
	}
	else
	{
		a->z0 = z0;
		a->z1 = z1;
	}
}
