/******************************************************************************

  This source code accompanies the Journal of Graphics Tools paper:

  "Fast Ray-Axis Aligned Bounding Box Overlap Tests With Pluecker Coordinates" by
  Jeffrey Mahovsky and Brian Wyvill
  Department of Computer Science, University of Calgary

  This source code is public domain, but please mention us if you use it.

 ******************************************************************************/

#include "plueckerint_mul_cls_cff.h"


bool plueckerint_mul_cls_cff(ray *r, aabox *b, double *t)
{
	switch (r->classification)
	{
	case MMM:
		{
			// side(R,HD) < 0 or side(R,FB) > 0 or side(R,EF) > 0 or side(R,DC) < 0 or side(R,CB) < 0 or side(R,HE) > 0 to miss

			if ((r->x < b->x0) || (r->y < b->y0) || (r->z < b->z0) ||
				(r->R0 + r->i * b->y0 - r->j * b->x1 < 0) ||
				(r->R0 + r->i * b->y1 - r->j * b->x0 > 0) ||
				(r->R1 + r->i * b->z1 - r->k * b->x0 > 0) ||
				(r->R1 + r->i * b->z0 - r->k * b->x1 < 0) ||
				(r->R3 - r->k * b->y1 + r->j * b->z0 < 0) ||
				(r->R3 - r->k * b->y0 + r->j * b->z1 > 0))
				return false;

			// compute the intersection distance

			*t = (b->x1 - r->x) * r->ii;
			double t1 = (b->y1 - r->y) * r->ij;
			if(t1 > *t)
				*t = t1;
			double t2 = (b->z1 - r->z) * r->ik;
			if(t2 > *t)
				*t = t2;

			return true;
		}

	case MMP:
		{
			// side(R,HD) < 0 or side(R,FB) > 0 or side(R,HG) > 0 or side(R,AB) < 0 or side(R,DA) < 0 or side(R,GF) > 0 to miss

			if ((r->x < b->x0) || (r->y < b->y0) || (r->z > b->z1) ||
				(r->R0 + r->i * b->y0 - r->j * b->x1 < 0) ||
				(r->R0 + r->i * b->y1 - r->j * b->x0 > 0) ||
				(r->R1 + r->i * b->z1 - r->k * b->x1 > 0) ||
				(r->R1 + r->i * b->z0 - r->k * b->x0 < 0) ||
				(r->R3 - r->k * b->y0 + r->j * b->z0 < 0) ||
				(r->R3 - r->k * b->y1 + r->j * b->z1 > 0))
				return false;

			*t = (b->x1 - r->x) * r->ii;
			double t1 = (b->y1 - r->y) * r->ij;
			if(t1 > *t)
				*t = t1;
			double t2 = (b->z0 - r->z) * r->ik;
			if(t2 > *t)
				*t = t2;

			return true;
		}

	case MPM:
		{
			// side(R,EA) < 0 or side(R,GC) > 0 or side(R,EF) > 0 or side(R,DC) < 0 or side(R,GF) < 0 or side(R,DA) > 0 to miss

			if ((r->x < b->x0) || (r->y > b->y1) || (r->z < b->z0) ||
				(r->R0 + r->i * b->y0 - r->j * b->x0 < 0) ||
				(r->R0 + r->i * b->y1 - r->j * b->x1 > 0) ||
				(r->R1 + r->i * b->z1 - r->k * b->x0 > 0) ||
				(r->R1 + r->i * b->z0 - r->k * b->x1 < 0) ||
				(r->R3 - r->k * b->y1 + r->j * b->z1 < 0) ||
				(r->R3 - r->k * b->y0 + r->j * b->z0 > 0))
				return false;

			*t = (b->x1 - r->x) * r->ii;
			double t1 = (b->y0 - r->y) * r->ij;
			if(t1 > *t)
				*t = t1;
			double t2 = (b->z1 - r->z) * r->ik;
			if(t2 > *t)
				*t = t2;

			return true;
		}

	case MPP:
		{
			// side(R,EA) < 0 or side(R,GC) > 0 or side(R,HG) > 0 or side(R,AB) < 0 or side(R,HE) < 0 or side(R,CB) > 0 to miss

			if ((r->x < b->x0) || (r->y > b->y1) || (r->z > b->z1) ||
				(r->R0 + r->i * b->y0 - r->j * b->x0 < 0) ||
				(r->R0 + r->i * b->y1 - r->j * b->x1 > 0) ||
				(r->R1 + r->i * b->z1 - r->k * b->x1 > 0) ||
				(r->R1 + r->i * b->z0 - r->k * b->x0 < 0) ||
				(r->R3 - r->k * b->y0 + r->j * b->z1 < 0) ||
				(r->R3 - r->k * b->y1 + r->j * b->z0 > 0))
				return false;

			*t = (b->x1 - r->x) * r->ii;
			double t1 = (b->y0 - r->y) * r->ij;
			if(t1 > *t)
				*t = t1;
			double t2 = (b->z0 - r->z) * r->ik;
			if(t2 > *t)
				*t = t2;

			return true;
		}

	case PMM:
		{
			// side(R,GC) < 0 or side(R,EA) > 0 or side(R,AB) > 0 or side(R,HG) < 0 or side(R,CB) < 0 or side(R,HE) > 0 to miss

			if ((r->x > b->x1) || (r->y < b->y0) || (r->z < b->z0) ||
				(r->R0 + r->i * b->y1 - r->j * b->x1 < 0) ||
				(r->R0 + r->i * b->y0 - r->j * b->x0 > 0) ||
				(r->R1 + r->i * b->z0 - r->k * b->x0 > 0) ||
				(r->R1 + r->i * b->z1 - r->k * b->x1 < 0) ||
				(r->R3 - r->k * b->y1 + r->j * b->z0 < 0) ||
				(r->R3 - r->k * b->y0 + r->j * b->z1 > 0))
				return false;

			*t = (b->x0 - r->x) * r->ii;
			double t1 = (b->y1 - r->y) * r->ij;
			if(t1 > *t)
				*t = t1;
			double t2 = (b->z1 - r->z) * r->ik;
			if(t2 > *t)
				*t = t2;

			return true;
		}

	case PMP:
		{
			// side(R,GC) < 0 or side(R,EA) > 0 or side(R,DC) > 0 or side(R,EF) < 0 or side(R,DA) < 0 or side(R,GF) > 0 to miss

			if ((r->x > b->x1) || (r->y < b->y0) || (r->z > b->z1) ||
				(r->R0 + r->i * b->y1 - r->j * b->x1 < 0) ||
				(r->R0 + r->i * b->y0 - r->j * b->x0 > 0) ||
				(r->R1 + r->i * b->z0 - r->k * b->x1 > 0) ||
				(r->R1 + r->i * b->z1 - r->k * b->x0 < 0) ||
				(r->R3 - r->k * b->y0 + r->j * b->z0 < 0) ||
				(r->R3 - r->k * b->y1 + r->j * b->z1 > 0))
				return false;

			*t = (b->x0 - r->x) * r->ii;
			double t1 = (b->y1 - r->y) * r->ij;
			if(t1 > *t)
				*t = t1;
			double t2 = (b->z0 - r->z) * r->ik;
			if(t2 > *t)
				*t = t2;

			return true;
		}

	case PPM:
		{
			// side(R,FB) < 0 or side(R,HD) > 0 or side(R,AB) > 0 or side(R,HG) < 0 or side(R,GF) < 0 or side(R,DA) > 0 to miss

			if ((r->x > b->x1) || (r->y > b->y1) || (r->z < b->z0) ||
				(r->R0 + r->i * b->y1 - r->j * b->x0 < 0) ||
				(r->R0 + r->i * b->y0 - r->j * b->x1 > 0) ||
				(r->R1 + r->i * b->z0 - r->k * b->x0 > 0) ||
				(r->R1 + r->i * b->z1 - r->k * b->x1 < 0) ||
				(r->R3 - r->k * b->y1 + r->j * b->z1 < 0) ||
				(r->R3 - r->k * b->y0 + r->j * b->z0 > 0))
				return false;

			*t = (b->x0 - r->x) * r->ii;
			double t1 = (b->y0 - r->y) * r->ij;
			if(t1 > *t)
				*t = t1;
			double t2 = (b->z1 - r->z) * r->ik;
			if(t2 > *t)
				*t = t2;

			return true;
		}

	case PPP:
		{
			// side(R,FB) < 0 or side(R,HD) > 0 or side(R,DC) > 0 or side(R,EF) < 0 or side(R,HE) < 0 or side(R,CB) > 0 to miss

			if ((r->x > b->x1) || (r->y > b->y1) || (r->z > b->z1) ||
				(r->R0 + r->i * b->y1 - r->j * b->x0 < 0) ||
				(r->R0 + r->i * b->y0 - r->j * b->x1 > 0) ||
				(r->R1 + r->i * b->z0 - r->k * b->x1 > 0) ||
				(r->R1 + r->i * b->z1 - r->k * b->x0 < 0) ||
				(r->R3 - r->k * b->y0 + r->j * b->z1 < 0) ||
				(r->R3 - r->k * b->y1 + r->j * b->z0 > 0))
				return false;

			*t = (b->x0 - r->x) * r->ii;
			double t1 = (b->y0 - r->y) * r->ij;
			if(t1 > *t)
				*t = t1;
			double t2 = (b->z0 - r->z) * r->ik;
			if(t2 > *t)
				*t = t2;

			return true;
		}
	}

	return false;
}
