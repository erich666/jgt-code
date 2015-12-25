/******************************************************************************

  This source code accompanies the Journal of Graphics Tools paper:

  "Fast Ray-Axis Aligned Bounding Box Overlap Tests With Pluecker Coordinates" by
  Jeffrey Mahovsky and Brian Wyvill
  Department of Computer Science, University of Calgary

  This source code is public domain, but please mention us if you use it.

 ******************************************************************************/

#include "ray.h"

void make_ray(double x, double y, double z, double i, double j, double k, ray *r)
{
	r->x = x;
	r->y = y;
	r->z = z;
	r->i = i;
	r->j = j;
	r->k = k;
	r->ii = 1.0 / i;
	r->ij = 1.0 / j;
	r->ik = 1.0 / k;
	r->R0 = x * j - i * y;
	r->R1 = x * k - i * z;
	r->R3 = y * k - j * z;

	if(i < 0)
	{
		if(j < 0)
		{
			if(k < 0)
				r->classification = MMM;
			else
				r->classification = MMP;
		}
		else
		{
			if(k < 0)
				r->classification = MPM;
			else
				r->classification = MPP;
		}
	}
	else
	{
		if(j < 0)
		{
			if(k < 0)
				r->classification = PMM;
			else
				r->classification = PMP;
		}
		else
		{
			if(k < 0)
				r->classification = PPM;
			else
				r->classification = PPP;
		}
	}
}
