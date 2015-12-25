/******************************************************************************

This source code accompanies the Journal of Graphics Tools paper:

"Fast Ray-Axis Aligned Bounding Box Overlap Tests With Pluecker Coordinates" by
Jeffrey Mahovsky and Brian Wyvill
Department of Computer Science, University of Calgary

This source code is public domain, but please mention us if you use it.

******************************************************************************/

#include "plueckerint_mul_cls.h"


bool plueckerint_mul_cls(ray *r, aabox *b, double *t)
{
	switch (r->classification)
	{
	case MMM:
		{
			// side(R,HD) < 0 or side(R,FB) > 0 or side(R,EF) > 0 or side(R,DC) < 0 or side(R,CB) < 0 or side(R,HE) > 0 to miss

			if ((r->x < b->x0) || (r->y < b->y0) || (r->z < b->z0))
				return false;

			double xa = b->x0 - r->x; 
			double ya = b->y0 - r->y; 
			double za = b->z0 - r->z; 
			double xb = b->x1 - r->x;
			double yb = b->y1 - r->y;
			double zb = b->z1 - r->z;

			if(	(r->i * ya - r->j * xb < 0) ||
				(r->i * yb - r->j * xa > 0) ||
				(r->i * zb - r->k * xa > 0) ||
				(r->i * za - r->k * xb < 0) ||
				(r->j * za - r->k * yb < 0) ||
				(r->j * zb - r->k * ya > 0))
				return false;

			// compute the intersection distance

			*t = xb * r->ii;
			double t1 = yb * r->ij;
			if(t1 > *t)
				*t = t1;
			double t2 = zb * r->ik;
			if(t2 > *t)
				*t = t2;

			return true;
		}

	case MMP:
		{
			// side(R,HD) < 0 or side(R,FB) > 0 or side(R,HG) > 0 or side(R,AB) < 0 or side(R,DA) < 0 or side(R,GF) > 0 to miss

			if ((r->x < b->x0) || (r->y < b->y0) || (r->z > b->z1))
				return false;

			double xa = b->x0 - r->x; 
			double ya = b->y0 - r->y; 
			double za = b->z0 - r->z; 
			double xb = b->x1 - r->x;
			double yb = b->y1 - r->y;
			double zb = b->z1 - r->z;

			if(	(r->i * ya - r->j * xb < 0) ||
				(r->i * yb - r->j * xa > 0) ||
				(r->i * zb - r->k * xb > 0) ||
				(r->i * za - r->k * xa < 0) ||
				(r->j * za - r->k * ya < 0) ||
				(r->j * zb - r->k * yb > 0))
				return false;

			*t = xb * r->ii;
			double t1 = yb * r->ij;
			if(t1 > *t)
				*t = t1;
			double t2 = za * r->ik;
			if(t2 > *t)
				*t = t2;

			return true;
		}
	case MPM:
		{
			// side(R,EA) < 0 or side(R,GC) > 0 or side(R,EF) > 0 or side(R,DC) < 0 or side(R,GF) < 0 or side(R,DA) > 0 to miss

			if ((r->x < b->x0) || (r->y > b->y1) || (r->z < b->z0))
				return false;

			double xa = b->x0 - r->x; 
			double ya = b->y0 - r->y; 
			double za = b->z0 - r->z; 
			double xb = b->x1 - r->x;
			double yb = b->y1 - r->y;
			double zb = b->z1 - r->z;

			if(	(r->i * ya - r->j * xa < 0) ||
				(r->i * yb - r->j * xb > 0) ||
				(r->i * zb - r->k * xa > 0) ||
				(r->i * za - r->k * xb < 0) ||
				(r->j * zb - r->k * yb < 0) ||
				(r->j * za - r->k * ya > 0))
				return false;

			*t = xb * r->ii;
			double t1 = ya * r->ij;
			if(t1 > *t)
				*t = t1;
			double t2 = zb * r->ik;
			if(t2 > *t)
				*t = t2;

			return true;
		}
	case MPP:
		{
			// side(R,EA) < 0 or side(R,GC) > 0 or side(R,HG) > 0 or side(R,AB) < 0 or side(R,HE) < 0 or side(R,CB) > 0 to miss

			if ((r->x < b->x0) || (r->y > b->y1) || (r->z > b->z1))
				return false;

			double xa = b->x0 - r->x; 
			double ya = b->y0 - r->y; 
			double za = b->z0 - r->z; 
			double xb = b->x1 - r->x;
			double yb = b->y1 - r->y;
			double zb = b->z1 - r->z;

			if(	(r->i * ya - r->j * xa < 0) ||
				(r->i * yb - r->j * xb > 0) ||
				(r->i * zb - r->k * xb > 0) ||
				(r->i * za - r->k * xa < 0) ||
				(r->j * zb - r->k * ya < 0) ||
				(r->j * za - r->k * yb > 0))
				return false;

			*t = xb * r->ii;
			double t1 = ya * r->ij;
			if(t1 > *t)
				*t = t1;
			double t2 = za * r->ik;
			if(t2 > *t)
				*t = t2;

			return true;
		}
	case PMM:
		{
			// side(R,GC) < 0 or side(R,EA) > 0 or side(R,AB) > 0 or side(R,HG) < 0 or side(R,CB) < 0 or side(R,HE) > 0 to miss

			if ((r->x > b->x1) || (r->y < b->y0) || (r->z < b->z0))
				return false;

			double xa = b->x0 - r->x; 
			double ya = b->y0 - r->y; 
			double za = b->z0 - r->z; 
			double xb = b->x1 - r->x;
			double yb = b->y1 - r->y;
			double zb = b->z1 - r->z;

			if(	(r->i * yb - r->j * xb < 0) ||
				(r->i * ya - r->j * xa > 0) ||
				(r->i * za - r->k * xa > 0) ||
				(r->i * zb - r->k * xb < 0) ||
				(r->j * za - r->k * yb < 0) ||
				(r->j * zb - r->k * ya > 0))
				return false;

			*t = xa * r->ii;
			double t1 = yb * r->ij;
			if(t1 > *t)
				*t = t1;
			double t2 = zb * r->ik;
			if(t2 > *t)
				*t = t2;

			return true;
		}
	case PMP:
		{
			// side(R,GC) < 0 or side(R,EA) > 0 or side(R,DC) > 0 or side(R,EF) < 0 or side(R,DA) < 0 or side(R,GF) > 0 to miss

			if ((r->x > b->x1) || (r->y < b->y0) || (r->z > b->z1))
				return false;

			double xa = b->x0 - r->x; 
			double ya = b->y0 - r->y; 
			double za = b->z0 - r->z; 
			double xb = b->x1 - r->x;
			double yb = b->y1 - r->y;
			double zb = b->z1 - r->z;

			if(	(r->i * yb - r->j * xb < 0) ||
				(r->i * ya - r->j * xa > 0) ||
				(r->i * za - r->k * xb > 0) ||
				(r->i * zb - r->k * xa < 0) ||
				(r->j * za - r->k * ya < 0) ||
				(r->j * zb - r->k * yb > 0))
				return false;

			*t = xa * r->ii;
			double t1 = yb * r->ij;
			if(t1 > *t)
				*t = t1;
			double t2 = za * r->ik;
			if(t2 > *t)
				*t = t2;

			return true;
		}
	case PPM:
		{
			// side(R,FB) < 0 or side(R,HD) > 0 or side(R,AB) > 0 or side(R,HG) < 0 or side(R,GF) < 0 or side(R,DA) > 0 to miss

			if ((r->x > b->x1) || (r->y > b->y1) || (r->z < b->z0))
				return false;

			double xa = b->x0 - r->x; 
			double ya = b->y0 - r->y; 
			double za = b->z0 - r->z; 
			double xb = b->x1 - r->x;
			double yb = b->y1 - r->y;
			double zb = b->z1 - r->z;

			if(	(r->i * yb - r->j * xa < 0) ||
				(r->i * ya - r->j * xb > 0) ||
				(r->i * za - r->k * xa > 0) ||
				(r->i * zb - r->k * xb < 0) ||
				(r->j * zb - r->k * yb < 0) ||
				(r->j * za - r->k * ya > 0))
				return false;

			*t = xa * r->ii;
			double t1 = ya * r->ij;
			if(t1 > *t)
				*t = t1;
			double t2 = zb * r->ik;
			if(t2 > *t)
				*t = t2;

			return true;
		}
	case PPP:
		{
			// side(R,FB) < 0 or side(R,HD) > 0 or side(R,DC) > 0 or side(R,EF) < 0 or side(R,HE) < 0 or side(R,CB) > 0 to miss

			if ((r->x > b->x1) || (r->y > b->y1) || (r->z > b->z1))
				return false;

			double xa = b->x0 - r->x; 
			double ya = b->y0 - r->y; 
			double za = b->z0 - r->z; 
			double xb = b->x1 - r->x;
			double yb = b->y1 - r->y;
			double zb = b->z1 - r->z;

			if(	(r->i * yb - r->j * xa < 0) ||
				(r->i * ya - r->j * xb > 0) ||
				(r->i * za - r->k * xb > 0) ||
				(r->i * zb - r->k * xa < 0) ||
				(r->j * zb - r->k * ya < 0) ||
				(r->j * za - r->k * yb > 0))
				return false;

			*t = xa * r->ii;
			double t1 = ya * r->ij;
			if(t1 > *t)
				*t = t1;
			double t2 = za * r->ik;
			if(t2 > *t)
				*t = t2;

			return true;
		}
	}

	return false;
}
