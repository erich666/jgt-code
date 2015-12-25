/******************************************************************************

  This source code accompanies the Journal of Graphics Tools paper:

  "Fast Ray-Axis Aligned Bounding Box Overlap Tests With Pluecker Coordinates" by
  Jeffrey Mahovsky and Brian Wyvill
  Department of Computer Science, University of Calgary

  This source code is public domain, but please mention us if you use it.

 ******************************************************************************/

#include "smits_div.h"


bool smits_div(ray *r, aabox *b, double *t)
{
	double tnear = -1e6;
	double tfar = 1e6;

	{	
		double t1 = (b->x0 - r->x) / r->i;
		double t2 = (b->x1 - r->x) / r->i;

		if(t1 > t2)
		{
			double temp = t1;
			t1 = t2;
			t2 = temp;
		}
		if(t1 > tnear)
			tnear = t1;
		if(t2 < tfar)
			tfar = t2;

		if(tnear > tfar)
			return false;
		if(tfar < 0.0)
			return false;
	}
	{
		double t1 = (b->y0 - r->y) / r->j;
		double t2 = (b->y1 - r->y) / r->j;

		if(t1 > t2)
		{
			double temp = t1;
			t1 = t2;
			t2 = temp;
		}
		if(t1 > tnear)
			tnear = t1;
		if(t2 < tfar)
			tfar = t2;

		if(tnear > tfar)
			return false;
		if(tfar < 0.0)
			return false;
	}
	{
		double t1 = (b->z0 - r->z) / r->k;
		double t2 = (b->z1 - r->z) / r->k;

		if(t1 > t2)
		{
			double temp = t1;
			t1 = t2;
			t2 = temp;
		}
		if(t1 > tnear)
			tnear = t1;
		if(t2 < tfar)
			tfar = t2;

		if(tnear > tfar)
			return false;
		if(tfar < 0.0)
			return false;
	}

	*t = tnear;
	return true;
}
