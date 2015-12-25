/******************************************************************************

  This source code accompanies the Journal of Graphics Tools paper:

  "Fast Ray-Axis Aligned Bounding Box Overlap Tests With Pluecker Coordinates" by
  Jeffrey Mahovsky and Brian Wyvill
  Department of Computer Science, University of Calgary

  This source code is public domain, but please mention us if you use it.

 ******************************************************************************/

#include "smits_div_cls.h"


bool smits_div_cls(ray *r, aabox *b, float *t)
{
	float tnear = -1e6;
	float tfar = 1e6;

	switch(r->classification)
	{
	case MMM:
		{	
			float t1 = (b->x1 - r->x) / r->i;
			float t2 = (b->x0 - r->x) / r->i;

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
			float t1 = (b->y1 - r->y) / r->j;
			float t2 = (b->y0 - r->y) / r->j;

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
			float t1 = (b->z1 - r->z) / r->k;
			float t2 = (b->z0 - r->z) / r->k;

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

		break;

	case MMP:
		{	
			float t1 = (b->x1 - r->x) / r->i;
			float t2 = (b->x0 - r->x) / r->i;

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
			float t1 = (b->y1 - r->y) / r->j;
			float t2 = (b->y0 - r->y) / r->j;

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
			float t1 = (b->z0 - r->z) / r->k;
			float t2 = (b->z1 - r->z) / r->k;

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

		break;

	case MPM:
		{	
			float t1 = (b->x1 - r->x) / r->i;
			float t2 = (b->x0 - r->x) / r->i;

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
			float t1 = (b->y0 - r->y) / r->j;
			float t2 = (b->y1 - r->y) / r->j;

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
			float t1 = (b->z1 - r->z) / r->k;
			float t2 = (b->z0 - r->z) / r->k;

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

		break;

	case MPP:
		{	
			float t1 = (b->x1 - r->x) / r->i;
			float t2 = (b->x0 - r->x) / r->i;

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
			float t1 = (b->y0 - r->y) / r->j;
			float t2 = (b->y1 - r->y) / r->j;

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
			float t1 = (b->z0 - r->z) / r->k;
			float t2 = (b->z1 - r->z) / r->k;

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

		break;

	case PMM:
		{	
			float t1 = (b->x0 - r->x) / r->i;
			float t2 = (b->x1 - r->x) / r->i;

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
			float t1 = (b->y1 - r->y) / r->j;
			float t2 = (b->y0 - r->y) / r->j;

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
			float t1 = (b->z1 - r->z) / r->k;
			float t2 = (b->z0 - r->z) / r->k;

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

		break;

	case PMP:
		{	
			float t1 = (b->x0 - r->x) / r->i;
			float t2 = (b->x1 - r->x) / r->i;

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
			float t1 = (b->y1 - r->y) / r->j;
			float t2 = (b->y0 - r->y) / r->j;

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
			float t1 = (b->z0 - r->z) / r->k;
			float t2 = (b->z1 - r->z) / r->k;

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

		break;

	case PPM:
		{	
			float t1 = (b->x0 - r->x) / r->i;
			float t2 = (b->x1 - r->x) / r->i;

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
			float t1 = (b->y0 - r->y) / r->j;
			float t2 = (b->y1 - r->y) / r->j;

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
			float t1 = (b->z1 - r->z) / r->k;
			float t2 = (b->z0 - r->z) / r->k;

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

		break;

	case PPP:
		{	
			float t1 = (b->x0 - r->x) / r->i;
			float t2 = (b->x1 - r->x) / r->i;

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
			float t1 = (b->y0 - r->y) / r->j;
			float t2 = (b->y1 - r->y) / r->j;

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
			float t1 = (b->z0 - r->z) / r->k;
			float t2 = (b->z1 - r->z) / r->k;

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

		break;
	}
	return false;
}
