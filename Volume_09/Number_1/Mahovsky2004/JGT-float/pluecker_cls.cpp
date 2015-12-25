/******************************************************************************

This source code accompanies the Journal of Graphics Tools paper:

"Fast Ray-Axis Aligned Bounding Box Overlap Tests With Pluecker Coordinates" by
Jeffrey Mahovsky and Brian Wyvill
Department of Computer Science, University of Calgary

This source code is public domain, but please mention us if you use it.

******************************************************************************/

#include "pluecker_cls.h"


bool pluecker_cls(ray *r, aabox *b)
{
	switch (r->classification)
	{
	case MMM:
		{
			// side(R,HD) < 0 or side(R,FB) > 0 or side(R,EF) > 0 or side(R,DC) < 0 or side(R,CB) < 0 or side(R,HE) > 0 to miss

			if ((r->x < b->x0) || (r->y < b->y0) || (r->z < b->z0))
				return false;

			float xa = b->x0 - r->x; 
			float ya = b->y0 - r->y; 
			float za = b->z0 - r->z; 
			float xb = b->x1 - r->x;
			float yb = b->y1 - r->y;
			float zb = b->z1 - r->z;

			if(	(r->i * ya - r->j * xb < 0) ||
				(r->i * yb - r->j * xa > 0) ||
				(r->i * zb - r->k * xa > 0) ||
				(r->i * za - r->k * xb < 0) ||
				(r->j * za - r->k * yb < 0) ||
				(r->j * zb - r->k * ya > 0))
				return false;

			return true;
		}

	case MMP:
		{
			// side(R,HD) < 0 or side(R,FB) > 0 or side(R,HG) > 0 or side(R,AB) < 0 or side(R,DA) < 0 or side(R,GF) > 0 to miss

			if ((r->x < b->x0) || (r->y < b->y0) || (r->z > b->z1))
				return false;

			float xa = b->x0 - r->x; 
			float ya = b->y0 - r->y; 
			float za = b->z0 - r->z; 
			float xb = b->x1 - r->x;
			float yb = b->y1 - r->y;
			float zb = b->z1 - r->z;

			if(	(r->i * ya - r->j * xb < 0) ||
				(r->i * yb - r->j * xa > 0) ||
				(r->i * zb - r->k * xb > 0) ||
				(r->i * za - r->k * xa < 0) ||
				(r->j * za - r->k * ya < 0) ||
				(r->j * zb - r->k * yb > 0))
				return false;

			return true;
		}
	case MPM:
		{
			// side(R,EA) < 0 or side(R,GC) > 0 or side(R,EF) > 0 or side(R,DC) < 0 or side(R,GF) < 0 or side(R,DA) > 0 to miss

			if ((r->x < b->x0) || (r->y > b->y1) || (r->z < b->z0))
				return false;

			float xa = b->x0 - r->x; 
			float ya = b->y0 - r->y; 
			float za = b->z0 - r->z; 
			float xb = b->x1 - r->x;
			float yb = b->y1 - r->y;
			float zb = b->z1 - r->z;

			if(	(r->i * ya - r->j * xa < 0) ||
				(r->i * yb - r->j * xb > 0) ||
				(r->i * zb - r->k * xa > 0) ||
				(r->i * za - r->k * xb < 0) ||
				(r->j * zb - r->k * yb < 0) ||
				(r->j * za - r->k * ya > 0))
				return false;

			return true;
		}
	case MPP:
		{
			// side(R,EA) < 0 or side(R,GC) > 0 or side(R,HG) > 0 or side(R,AB) < 0 or side(R,HE) < 0 or side(R,CB) > 0 to miss

			if ((r->x < b->x0) || (r->y > b->y1) || (r->z > b->z1))
				return false;

			float xa = b->x0 - r->x; 
			float ya = b->y0 - r->y; 
			float za = b->z0 - r->z; 
			float xb = b->x1 - r->x;
			float yb = b->y1 - r->y;
			float zb = b->z1 - r->z;

			if(	(r->i * ya - r->j * xa < 0) ||
				(r->i * yb - r->j * xb > 0) ||
				(r->i * zb - r->k * xb > 0) ||
				(r->i * za - r->k * xa < 0) ||
				(r->j * zb - r->k * ya < 0) ||
				(r->j * za - r->k * yb > 0))
				return false;

			return true;
		}
	case PMM:
		{
			// side(R,GC) < 0 or side(R,EA) > 0 or side(R,AB) > 0 or side(R,HG) < 0 or side(R,CB) < 0 or side(R,HE) > 0 to miss

			if ((r->x > b->x1) || (r->y < b->y0) || (r->z < b->z0))
				return false;

			float xa = b->x0 - r->x; 
			float ya = b->y0 - r->y; 
			float za = b->z0 - r->z; 
			float xb = b->x1 - r->x;
			float yb = b->y1 - r->y;
			float zb = b->z1 - r->z;

			if(	(r->i * yb - r->j * xb < 0) ||
				(r->i * ya - r->j * xa > 0) ||
				(r->i * za - r->k * xa > 0) ||
				(r->i * zb - r->k * xb < 0) ||
				(r->j * za - r->k * yb < 0) ||
				(r->j * zb - r->k * ya > 0))
				return false;

			return true;
		}
	case PMP:
		{
			// side(R,GC) < 0 or side(R,EA) > 0 or side(R,DC) > 0 or side(R,EF) < 0 or side(R,DA) < 0 or side(R,GF) > 0 to miss

			if ((r->x > b->x1) || (r->y < b->y0) || (r->z > b->z1))
				return false;

			float xa = b->x0 - r->x; 
			float ya = b->y0 - r->y; 
			float za = b->z0 - r->z; 
			float xb = b->x1 - r->x;
			float yb = b->y1 - r->y;
			float zb = b->z1 - r->z;

			if(	(r->i * yb - r->j * xb < 0) ||
				(r->i * ya - r->j * xa > 0) ||
				(r->i * za - r->k * xb > 0) ||
				(r->i * zb - r->k * xa < 0) ||
				(r->j * za - r->k * ya < 0) ||
				(r->j * zb - r->k * yb > 0))
				return false;

			return true;
		}
	case PPM:
		{
			// side(R,FB) < 0 or side(R,HD) > 0 or side(R,AB) > 0 or side(R,HG) < 0 or side(R,GF) < 0 or side(R,DA) > 0 to miss

			if ((r->x > b->x1) || (r->y > b->y1) || (r->z < b->z0))
				return false;

			float xa = b->x0 - r->x; 
			float ya = b->y0 - r->y; 
			float za = b->z0 - r->z; 
			float xb = b->x1 - r->x;
			float yb = b->y1 - r->y;
			float zb = b->z1 - r->z;

			if(	(r->i * yb - r->j * xa < 0) ||
				(r->i * ya - r->j * xb > 0) ||
				(r->i * za - r->k * xa > 0) ||
				(r->i * zb - r->k * xb < 0) ||
				(r->j * zb - r->k * yb < 0) ||
				(r->j * za - r->k * ya > 0))
				return false;

			return true;
		}
	case PPP:
		{
			// side(R,FB) < 0 or side(R,HD) > 0 or side(R,DC) > 0 or side(R,EF) < 0 or side(R,HE) < 0 or side(R,CB) > 0 to miss

			if ((r->x > b->x1) || (r->y > b->y1) || (r->z > b->z1))
				return false;

			float xa = b->x0 - r->x; 
			float ya = b->y0 - r->y; 
			float za = b->z0 - r->z; 
			float xb = b->x1 - r->x;
			float yb = b->y1 - r->y;
			float zb = b->z1 - r->z;

			if(	(r->i * yb - r->j * xa < 0) ||
				(r->i * ya - r->j * xb > 0) ||
				(r->i * za - r->k * xb > 0) ||
				(r->i * zb - r->k * xa < 0) ||
				(r->j * zb - r->k * ya < 0) ||
				(r->j * za - r->k * yb > 0))
				return false;

			return true;
		}
	}

	return false;
}
