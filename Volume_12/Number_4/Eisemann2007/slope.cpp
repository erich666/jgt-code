/******************************************************************************

  This source code accompanies the Journal of Graphics Tools paper:

  "Fast Ray / Axis-Aligned Bounding Box Overlap Tests using Ray Slopes" 
  by Martin Eisemann, Thorsten Grosch, Stefan Müller and Marcus Magnor
  Computer Graphics Lab, TU Braunschweig, Germany and
  University of Koblenz-Landau, Germany
  
  This source code is public domain, but please mention us if you use it.

******************************************************************************/
#include "slope.h"

bool slope(ray *r, aabox *b){

    switch (r->classification)
    {
	case MMM:
		
		if ((r->x < b->x0) || (r->y < b->y0) || (r->z < b->z0)
			|| (r->jbyi * b->x0 - b->y1 + r->c_xy > 0)
			|| (r->ibyj * b->y0 - b->x1 + r->c_yx > 0)
			|| (r->jbyk * b->z0 - b->y1 + r->c_zy > 0)
			|| (r->kbyj * b->y0 - b->z1 + r->c_yz > 0)
			|| (r->kbyi * b->x0 - b->z1 + r->c_xz > 0)
			|| (r->ibyk * b->z0 - b->x1 + r->c_zx > 0)
			)
			return false;
		
		return true;

	case MMP:
		
		if ((r->x < b->x0) || (r->y < b->y0) || (r->z > b->z1)
			|| (r->jbyi * b->x0 - b->y1 + r->c_xy > 0)
			|| (r->ibyj * b->y0 - b->x1 + r->c_yx > 0)
			|| (r->jbyk * b->z1 - b->y1 + r->c_zy > 0)
			|| (r->kbyj * b->y0 - b->z0 + r->c_yz < 0)
			|| (r->kbyi * b->x0 - b->z0 + r->c_xz < 0)
			|| (r->ibyk * b->z1 - b->x1 + r->c_zx > 0)
			)
			return false;
		
		return true;

	case MPM:
		
		if ((r->x < b->x0) || (r->y > b->y1) || (r->z < b->z0)
			|| (r->jbyi * b->x0 - b->y0 + r->c_xy < 0) 
			|| (r->ibyj * b->y1 - b->x1 + r->c_yx > 0)
			|| (r->jbyk * b->z0 - b->y0 + r->c_zy < 0) 
			|| (r->kbyj * b->y1 - b->z1 + r->c_yz > 0)
			|| (r->kbyi * b->x0 - b->z1 + r->c_xz > 0)
			|| (r->ibyk * b->z0 - b->x1 + r->c_zx > 0)
			)
			return false;
		
		return true;

	case MPP:
	
		if ((r->x < b->x0) || (r->y > b->y1) || (r->z > b->z1)
			|| (r->jbyi * b->x0 - b->y0 + r->c_xy < 0) 
			|| (r->ibyj * b->y1 - b->x1 + r->c_yx > 0)
			|| (r->jbyk * b->z1 - b->y0 + r->c_zy < 0)
			|| (r->kbyj * b->y1 - b->z0 + r->c_yz < 0)
			|| (r->kbyi * b->x0 - b->z0 + r->c_xz < 0)
			|| (r->ibyk * b->z1 - b->x1 + r->c_zx > 0)
			)
			return false;
		
		return true;

	case PMM:

		if ((r->x > b->x1) || (r->y < b->y0) || (r->z < b->z0)
			|| (r->jbyi * b->x1 - b->y1 + r->c_xy > 0)
			|| (r->ibyj * b->y0 - b->x0 + r->c_yx < 0)
			|| (r->jbyk * b->z0 - b->y1 + r->c_zy > 0)
			|| (r->kbyj * b->y0 - b->z1 + r->c_yz > 0)
			|| (r->kbyi * b->x1 - b->z1 + r->c_xz > 0)
			|| (r->ibyk * b->z0 - b->x0 + r->c_zx < 0)
			)
			return false;

		return true;

	case PMP:

		if ((r->x > b->x1) || (r->y < b->y0) || (r->z > b->z1)
			|| (r->jbyi * b->x1 - b->y1 + r->c_xy > 0)
			|| (r->ibyj * b->y0 - b->x0 + r->c_yx < 0)
			|| (r->jbyk * b->z1 - b->y1 + r->c_zy > 0)
			|| (r->kbyj * b->y0 - b->z0 + r->c_yz < 0)
			|| (r->kbyi * b->x1 - b->z0 + r->c_xz < 0)
			|| (r->ibyk * b->z1 - b->x0 + r->c_zx < 0)
			)
			return false;

		return true;

	case PPM:

		if ((r->x > b->x1) || (r->y > b->y1) || (r->z < b->z0)
			|| (r->jbyi * b->x1 - b->y0 + r->c_xy < 0)
			|| (r->ibyj * b->y1 - b->x0 + r->c_yx < 0)
			|| (r->jbyk * b->z0 - b->y0 + r->c_zy < 0) 
			|| (r->kbyj * b->y1 - b->z1 + r->c_yz > 0)
			|| (r->kbyi * b->x1 - b->z1 + r->c_xz > 0)
			|| (r->ibyk * b->z0 - b->x0 + r->c_zx < 0)
			)
			return false;
		
		return true;

	case PPP:

		if ((r->x > b->x1) || (r->y > b->y1) || (r->z > b->z1)
			|| (r->jbyi * b->x1 - b->y0 + r->c_xy < 0)
			|| (r->ibyj * b->y1 - b->x0 + r->c_yx < 0)
			|| (r->jbyk * b->z1 - b->y0 + r->c_zy < 0)
			|| (r->kbyj * b->y1 - b->z0 + r->c_yz < 0)
			|| (r->kbyi * b->x1 - b->z0 + r->c_xz < 0)
			|| (r->ibyk * b->z1 - b->x0 + r->c_zx < 0)
			)
			return false;
		
		return true;

	case OMM:

		if((r->x < b->x0) || (r->x > b->x1)
			|| (r->y < b->y0) || (r->z < b->z0)
			|| (r->jbyk * b->z0 - b->y1 + r->c_zy > 0)
			|| (r->kbyj * b->y0 - b->z1 + r->c_yz > 0)
			)
			return false;

		return true;

	case OMP:

		if((r->x < b->x0) || (r->x > b->x1)
			|| (r->y < b->y0) || (r->z > b->z1)
			|| (r->jbyk * b->z1 - b->y1 + r->c_zy > 0)
			|| (r->kbyj * b->y0 - b->z0 + r->c_yz < 0)
			)
			return false;

		return true;

	case OPM:

		if((r->x < b->x0) || (r->x > b->x1)
			|| (r->y > b->y1) || (r->z < b->z0)
			|| (r->jbyk * b->z0 - b->y0 + r->c_zy < 0) 
			|| (r->kbyj * b->y1 - b->z1 + r->c_yz > 0)
			)
			return false;

		return true;

	case OPP:

		if((r->x < b->x0) || (r->x > b->x1)
			|| (r->y > b->y1) || (r->z > b->z1)
			|| (r->jbyk * b->z1 - b->y0 + r->c_zy < 0)
			|| (r->kbyj * b->y1 - b->z0 + r->c_yz < 0)
			)
			return false;

		return true;

	case MOM:

		if((r->y < b->y0) || (r->y > b->y1)
			|| (r->x < b->x0) || (r->z < b->z0) 
			|| (r->kbyi * b->x0 - b->z1 + r->c_xz > 0)
			|| (r->ibyk * b->z0 - b->x1 + r->c_zx > 0)
			)
			return false;

		return true;

	case MOP:

		if((r->y < b->y0) || (r->y > b->y1)
			|| (r->x < b->x0) || (r->z > b->z1) 
			|| (r->kbyi * b->x0 - b->z0 + r->c_xz < 0)
			|| (r->ibyk * b->z1 - b->x1 + r->c_zx > 0)
			)
			return false;

		return true;

	case POM:

		if((r->y < b->y0) || (r->y > b->y1)
			|| (r->x > b->x1) || (r->z < b->z0)
			|| (r->kbyi * b->x1 - b->z1 + r->c_xz > 0)
			|| (r->ibyk * b->z0 - b->x0 + r->c_zx < 0)
			)
			return false;

		return true;

	case POP:

		if((r->y < b->y0) || (r->y > b->y1)
			|| (r->x > b->x1) || (r->z > b->z1)
			|| (r->kbyi * b->x1 - b->z0 + r->c_xz < 0)
			|| (r->ibyk * b->z1 - b->x0 + r->c_zx < 0)
			)
			return false;

		return true;

	case MMO:

		if((r->z < b->z0) || (r->z > b->z1)
			|| (r->x < b->x0) || (r->y < b->y0) 
			|| (r->jbyi * b->x0 - b->y1 + r->c_xy > 0)
			|| (r->ibyj * b->y0 - b->x1 + r->c_yx > 0)
			)
			return false;

		return true;

	case MPO:

		if((r->z < b->z0) || (r->z > b->z1)
			|| (r->x < b->x0) || (r->y > b->y1) 
			|| (r->jbyi * b->x0 - b->y0 + r->c_xy < 0) 
			|| (r->ibyj * b->y1 - b->x1 + r->c_yx > 0)
			)
			return false;

		return true;

	case PMO:

		if((r->z < b->z0) || (r->z > b->z1)
			|| (r->x > b->x1) || (r->y < b->y0) 
			|| (r->jbyi * b->x1 - b->y1 + r->c_xy > 0)
			|| (r->ibyj * b->y0 - b->x0 + r->c_yx < 0)  
			)
			return false;

		return true;

	case PPO:

		if((r->z < b->z0) || (r->z > b->z1)
			|| (r->x > b->x1) || (r->y > b->y1)
			|| (r->jbyi * b->x1 - b->y0 + r->c_xy < 0)
			|| (r->ibyj * b->y1 - b->x0 + r->c_yx < 0)
			)
			return false;

		return true;

	case MOO:

		if((r->x < b->x0)
			|| (r->y < b->y0) || (r->y > b->y1)
			|| (r->z < b->z0) || (r->z > b->z1)
			)
			return false;

		return true;

	case POO:

		if((r->x > b->x1)
			|| (r->y < b->y0) || (r->y > b->y1)
			|| (r->z < b->z0) || (r->z > b->z1)
			)
			return false;

		return true;

	case OMO:

		if((r->y < b->y0)
			|| (r->x < b->x0) || (r->x > b->x1)
			|| (r->z < b->z0) || (r->z > b->z1)
			)
			return false;

	case OPO:

		if((r->y > b->y1)
			|| (r->x < b->x0) || (r->x > b->x1)
			|| (r->z < b->z0) || (r->z > b->z1)
			)
			return false;

	case OOM:

		if((r->z < b->z0)
			|| (r->x < b->x0) || (r->x > b->x1)
			|| (r->y < b->y0) || (r->y > b->y1)
			)
			return false;

	case OOP:

		if((r->z > b->z1)
			|| (r->x < b->x0) || (r->x > b->x1)
			|| (r->y < b->y0) || (r->y > b->y1)
			)
			return false;

		return true;
	
	}

	return false;
}
