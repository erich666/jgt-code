/******************************************************************************

  This source code accompanies the Journal of Graphics Tools paper:

  "Fast Ray-Axis Aligned Bounding Box Overlap Tests With Pluecker Coordinates" by
  Jeffrey Mahovsky and Brian Wyvill
  Department of Computer Science, University of Calgary

  This source code is public domain, but please mention us if you use it.

 ******************************************************************************/

#include "pluecker_cls_cff.h"


bool pluecker_cls_cff(ray *r, aabox *b)
{
	switch (r->classification)
	{
	case MMM:
		// side(R,HD) < 0 or side(R,FB) > 0 or side(R,EF) > 0 or side(R,DC) < 0 or side(R,CB) < 0 or side(R,HE) > 0 to miss

		if ((r->x < b->x0) || (r->y < b->y0) || (r->z < b->z0) ||
			(r->R0 + r->i * b->y0 - r->j * b->x1 < 0) ||
			(r->R0 + r->i * b->y1 - r->j * b->x0 > 0) ||
			(r->R1 + r->i * b->z1 - r->k * b->x0 > 0) ||
			(r->R1 + r->i * b->z0 - r->k * b->x1 < 0) ||
			(r->R3 - r->k * b->y1 + r->j * b->z0 < 0) ||
			(r->R3 - r->k * b->y0 + r->j * b->z1 > 0))
			return false;
		
		return true;

	case MMP:
		// side(R,HD) < 0 or side(R,FB) > 0 or side(R,HG) > 0 or side(R,AB) < 0 or side(R,DA) < 0 or side(R,GF) > 0 to miss

		if ((r->x < b->x0) || (r->y < b->y0) || (r->z > b->z1) ||
			(r->R0 + r->i * b->y0 - r->j * b->x1 < 0) ||
			(r->R0 + r->i * b->y1 - r->j * b->x0 > 0) ||
			(r->R1 + r->i * b->z1 - r->k * b->x1 > 0) ||
			(r->R1 + r->i * b->z0 - r->k * b->x0 < 0) ||
			(r->R3 - r->k * b->y0 + r->j * b->z0 < 0) ||
			(r->R3 - r->k * b->y1 + r->j * b->z1 > 0))
			return false;
		
		return true;

	case MPM:
		// side(R,EA) < 0 or side(R,GC) > 0 or side(R,EF) > 0 or side(R,DC) < 0 or side(R,GF) < 0 or side(R,DA) > 0 to miss

		if ((r->x < b->x0) || (r->y > b->y1) || (r->z < b->z0) ||
			(r->R0 + r->i * b->y0 - r->j * b->x0 < 0) ||
			(r->R0 + r->i * b->y1 - r->j * b->x1 > 0) ||
			(r->R1 + r->i * b->z1 - r->k * b->x0 > 0) ||
			(r->R1 + r->i * b->z0 - r->k * b->x1 < 0) ||
			(r->R3 - r->k * b->y1 + r->j * b->z1 < 0) ||
			(r->R3 - r->k * b->y0 + r->j * b->z0 > 0))
			return false;
		
		return true;

	case MPP:
		// side(R,EA) < 0 or side(R,GC) > 0 or side(R,HG) > 0 or side(R,AB) < 0 or side(R,HE) < 0 or side(R,CB) > 0 to miss

		if ((r->x < b->x0) || (r->y > b->y1) || (r->z > b->z1) ||
			(r->R0 + r->i * b->y0 - r->j * b->x0 < 0) ||
			(r->R0 + r->i * b->y1 - r->j * b->x1 > 0) ||
			(r->R1 + r->i * b->z1 - r->k * b->x1 > 0) ||
			(r->R1 + r->i * b->z0 - r->k * b->x0 < 0) ||
			(r->R3 - r->k * b->y0 + r->j * b->z1 < 0) ||
			(r->R3 - r->k * b->y1 + r->j * b->z0 > 0))
			return false;
		
		return true;

	case PMM:
		// side(R,GC) < 0 or side(R,EA) > 0 or side(R,AB) > 0 or side(R,HG) < 0 or side(R,CB) < 0 or side(R,HE) > 0 to miss

		if ((r->x > b->x1) || (r->y < b->y0) || (r->z < b->z0) ||
			(r->R0 + r->i * b->y1 - r->j * b->x1 < 0) ||
			(r->R0 + r->i * b->y0 - r->j * b->x0 > 0) ||
			(r->R1 + r->i * b->z0 - r->k * b->x0 > 0) ||
			(r->R1 + r->i * b->z1 - r->k * b->x1 < 0) ||
			(r->R3 - r->k * b->y1 + r->j * b->z0 < 0) ||
			(r->R3 - r->k * b->y0 + r->j * b->z1 > 0))
			return false;
		
		return true;

	case PMP:
		// side(R,GC) < 0 or side(R,EA) > 0 or side(R,DC) > 0 or side(R,EF) < 0 or side(R,DA) < 0 or side(R,GF) > 0 to miss

		if ((r->x > b->x1) || (r->y < b->y0) || (r->z > b->z1) ||
			(r->R0 + r->i * b->y1 - r->j * b->x1 < 0) ||
			(r->R0 + r->i * b->y0 - r->j * b->x0 > 0) ||
			(r->R1 + r->i * b->z0 - r->k * b->x1 > 0) ||
			(r->R1 + r->i * b->z1 - r->k * b->x0 < 0) ||
			(r->R3 - r->k * b->y0 + r->j * b->z0 < 0) ||
			(r->R3 - r->k * b->y1 + r->j * b->z1 > 0))
			return false;
		
		return true;

	case PPM:
		// side(R,FB) < 0 or side(R,HD) > 0 or side(R,AB) > 0 or side(R,HG) < 0 or side(R,GF) < 0 or side(R,DA) > 0 to miss

		if ((r->x > b->x1) || (r->y > b->y1) || (r->z < b->z0) ||
			(r->R0 + r->i * b->y1 - r->j * b->x0 < 0) ||
			(r->R0 + r->i * b->y0 - r->j * b->x1 > 0) ||
			(r->R1 + r->i * b->z0 - r->k * b->x0 > 0) ||
			(r->R1 + r->i * b->z1 - r->k * b->x1 < 0) ||
			(r->R3 - r->k * b->y1 + r->j * b->z1 < 0) ||
			(r->R3 - r->k * b->y0 + r->j * b->z0 > 0))
			return false;
		
		return true;

	case PPP:
		// side(R,FB) < 0 or side(R,HD) > 0 or side(R,DC) > 0 or side(R,EF) < 0 or side(R,HE) < 0 or side(R,CB) > 0 to miss

		if ((r->x > b->x1) || (r->y > b->y1) || (r->z > b->z1) ||
			(r->R0 + r->i * b->y1 - r->j * b->x0 < 0) ||
			(r->R0 + r->i * b->y0 - r->j * b->x1 > 0) ||
			(r->R1 + r->i * b->z0 - r->k * b->x1 > 0) ||
			(r->R1 + r->i * b->z1 - r->k * b->x0 < 0) ||
			(r->R3 - r->k * b->y0 + r->j * b->z1 < 0) ||
			(r->R3 - r->k * b->y1 + r->j * b->z0 > 0))
			return false;
		
		return true;
	}

	return false;
}
