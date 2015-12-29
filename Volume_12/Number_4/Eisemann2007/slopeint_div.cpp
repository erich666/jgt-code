/******************************************************************************

  This source code accompanies the Journal of Graphics Tools paper:

  "Fast Ray / Axis-Aligned Bounding Box Overlap Tests using Ray Slopes" 
  by Martin Eisemann, Thorsten Grosch, Stefan Müller and Marcus Magnor
  Computer Graphics Lab, TU Braunschweig, Germany and
  University of Koblenz-Landau, Germany


  This source code is public domain, but please mention us if you use it.

******************************************************************************/
#include "slopeint_div.h"

bool slopeint_div(ray *r, aabox *b, float *t){

	switch (r->classification)
	{
	case MMM:
		{
		if ((r->x < b->x0) || (r->y < b->y0) || (r->z < b->z0)
			|| (r->jbyi * b->x0 - b->y1 + r->c_xy > 0)
			|| (r->ibyj * b->y0 - b->x1 + r->c_yx > 0)
			|| (r->jbyk * b->z0 - b->y1 + r->c_zy > 0)
			|| (r->kbyj * b->y0 - b->z1 + r->c_yz > 0)
			|| (r->kbyi * b->x0 - b->z1 + r->c_xz > 0)
			|| (r->ibyk * b->z0 - b->x1 + r->c_zx > 0)
			)
			return false;
		
		*t = (b->x1 - r->x) / r->i;
		float t1 = (b->y1 - r->y) / r->j;
		if(t1 > *t)
			*t = t1;
		float t2 = (b->z1 - r->z) / r->k;
		if(t2 > *t)
			*t = t2;

		return true;
		}


	case MMP:
		{		
		if ((r->x < b->x0) || (r->y < b->y0) || (r->z > b->z1)
			|| (r->jbyi * b->x0 - b->y1 + r->c_xy > 0)
			|| (r->ibyj * b->y0 - b->x1 + r->c_yx > 0)
			|| (r->jbyk * b->z1 - b->y1 + r->c_zy > 0)
			|| (r->kbyj * b->y0 - b->z0 + r->c_yz < 0)
			|| (r->kbyi * b->x0 - b->z0 + r->c_xz < 0)
			|| (r->ibyk * b->z1 - b->x1 + r->c_zx > 0)
			)
			return false;
		*t = (b->x1 - r->x) / r->i;
		float t1 = (b->y1 - r->y) / r->j;
		if(t1 > *t)
			*t = t1;
		float t2 = (b->z0 - r->z) / r->k;
		if(t2 > *t)
			*t = t2;

		return true;
		}


	case MPM:
		{		
		if ((r->x < b->x0) || (r->y > b->y1) || (r->z < b->z0)
			|| (r->jbyi * b->x0 - b->y0 + r->c_xy < 0) 
			|| (r->ibyj * b->y1 - b->x1 + r->c_yx > 0)
			|| (r->jbyk * b->z0 - b->y0 + r->c_zy < 0) 
			|| (r->kbyj * b->y1 - b->z1 + r->c_yz > 0)
			|| (r->kbyi * b->x0 - b->z1 + r->c_xz > 0)
			|| (r->ibyk * b->z0 - b->x1 + r->c_zx > 0)
			)
			return false;
		
		*t = (b->x1 - r->x) / r->i;
		float t1 = (b->y0 - r->y) / r->j;
		if(t1 > *t)
			*t = t1;
		float t2 = (b->z1 - r->z) / r->k;
		if(t2 > *t)
			*t = t2;

		return true;
		}

	case MPP:
		{
		if ((r->x < b->x0) || (r->y > b->y1) || (r->z > b->z1)
			|| (r->jbyi * b->x0 - b->y0 + r->c_xy < 0) 
			|| (r->ibyj * b->y1 - b->x1 + r->c_yx > 0)
			|| (r->jbyk * b->z1 - b->y0 + r->c_zy < 0)
			|| (r->kbyj * b->y1 - b->z0 + r->c_yz < 0)
			|| (r->kbyi * b->x0 - b->z0 + r->c_xz < 0)
			|| (r->ibyk * b->z1 - b->x1 + r->c_zx > 0)
			)
			return false;
		
		*t = (b->x1 - r->x) / r->i;
		float t1 = (b->y0 - r->y) / r->j;
		if(t1 > *t)
			*t = t1;
		float t2 = (b->z0 - r->z) / r->k;
		if(t2 > *t)
			*t = t2;

		return true;
		}	

	case PMM:
		{
		if ((r->x > b->x1) || (r->y < b->y0) || (r->z < b->z0)
			|| (r->jbyi * b->x1 - b->y1 + r->c_xy > 0)
			|| (r->ibyj * b->y0 - b->x0 + r->c_yx < 0)
			|| (r->jbyk * b->z0 - b->y1 + r->c_zy > 0)
			|| (r->kbyj * b->y0 - b->z1 + r->c_yz > 0)
			|| (r->kbyi * b->x1 - b->z1 + r->c_xz > 0)
			|| (r->ibyk * b->z0 - b->x0 + r->c_zx < 0)
			)
			return false;

		*t = (b->x0 - r->x) / r->i;
		float t1 = (b->y1 - r->y) / r->j;
		if(t1 > *t)
			*t = t1;
		float t2 = (b->z1 - r->z) / r->k;
		if(t2 > *t)
			*t = t2;

		return true;
		}

	case PMP:
		{
		if ((r->x > b->x1) || (r->y < b->y0) || (r->z > b->z1)
			|| (r->jbyi * b->x1 - b->y1 + r->c_xy > 0)
			|| (r->ibyj * b->y0 - b->x0 + r->c_yx < 0)
			|| (r->jbyk * b->z1 - b->y1 + r->c_zy > 0)
			|| (r->kbyj * b->y0 - b->z0 + r->c_yz < 0)
			|| (r->kbyi * b->x1 - b->z0 + r->c_xz < 0)
			|| (r->ibyk * b->z1 - b->x0 + r->c_zx < 0)
			)
			return false;

		*t = (b->x0 - r->x) / r->i;
		float t1 = (b->y1 - r->y) / r->j;
		if(t1 > *t)
			*t = t1;
		float t2 = (b->z0 - r->z) / r->k;
		if(t2 > *t)
			*t = t2;

		return true;
		}

	case PPM:
		{
		if ((r->x > b->x1) || (r->y > b->y1) || (r->z < b->z0)
			|| (r->jbyi * b->x1 - b->y0 + r->c_xy < 0)
			|| (r->ibyj * b->y1 - b->x0 + r->c_yx < 0)
			|| (r->jbyk * b->z0 - b->y0 + r->c_zy < 0) 
			|| (r->kbyj * b->y1 - b->z1 + r->c_yz > 0)
			|| (r->kbyi * b->x1 - b->z1 + r->c_xz > 0)
			|| (r->ibyk * b->z0 - b->x0 + r->c_zx < 0)
			)
			return false;
		
		*t = (b->x0 - r->x) / r->i;
		float t1 = (b->y0 - r->y) / r->j;
		if(t1 > *t)
			*t = t1;
		float t2 = (b->z1 - r->z) / r->k;
		if(t2 > *t)
			*t = t2;

		return true;
		}

	case PPP:
		{
		if ((r->x > b->x1) || (r->y > b->y1) || (r->z > b->z1)
			|| (r->jbyi * b->x1 - b->y0 + r->c_xy < 0)
			|| (r->ibyj * b->y1 - b->x0 + r->c_yx < 0)
			|| (r->jbyk * b->z1 - b->y0 + r->c_zy < 0)
			|| (r->kbyj * b->y1 - b->z0 + r->c_yz < 0)
			|| (r->kbyi * b->x1 - b->z0 + r->c_xz < 0)
			|| (r->ibyk * b->z1 - b->x0 + r->c_zx < 0)
			)
			return false;
		
		*t = (b->x0 - r->x) / r->i;
		float t1 = (b->y0 - r->y) / r->j;
		if(t1 > *t)
			*t = t1;
		float t2 = (b->z0 - r->z) / r->k;
		if(t2 > *t)
			*t = t2;

		return true;
		}

	case OMM:
		{
		if((r->x < b->x0) || (r->x > b->x1)
			|| (r->y < b->y0) || (r->z < b->z0)
			|| (r->jbyk * b->z0 - b->y1 + r->c_zy > 0)
			|| (r->kbyj * b->y0 - b->z1 + r->c_yz > 0)
			)
			return false;

		*t = (b->y1 - r->y) / r->j;
		float t2 = (b->z1 - r->z) / r->k;
		if(t2 > *t)
			*t = t2;

		return true;
		}

	case OMP:
		{
		if((r->x < b->x0) || (r->x > b->x1)
			|| (r->y < b->y0) || (r->z > b->z1)
			|| (r->jbyk * b->z1 - b->y1 + r->c_zy > 0)
			|| (r->kbyj * b->y0 - b->z0 + r->c_yz < 0)
			)
			return false;

		*t = (b->y1 - r->y) / r->j;
		float t2 = (b->z0 - r->z) / r->k;
		if(t2 > *t)
			*t = t2;

		return true;
		}

	case OPM:
		{
		if((r->x < b->x0) || (r->x > b->x1)
			|| (r->y > b->y1) || (r->z < b->z0)
			|| (r->jbyk * b->z0 - b->y0 + r->c_zy < 0) 
			|| (r->kbyj * b->y1 - b->z1 + r->c_yz > 0)
			)
			return false;

		*t = (b->y0 - r->y) / r->j;
		float t2 = (b->z1 - r->z) / r->k;
		if(t2 > *t)
			*t = t2;

		return true;
		}

	case OPP:
		{
		if((r->x < b->x0) || (r->x > b->x1)
			|| (r->y > b->y1) || (r->z > b->z1)
			|| (r->jbyk * b->z1 - b->y0 + r->c_zy < 0)
			|| (r->kbyj * b->y1 - b->z0 + r->c_yz < 0)
			)
			return false;

		*t = (b->y0 - r->y) / r->j;
		float t2 = (b->z0 - r->z) / r->k;
		if(t2 > *t)
			*t = t2;

		return true;
		}

	case MOM:
		{
		if((r->y < b->y0) || (r->y > b->y1)
			|| (r->x < b->x0) || (r->z < b->z0) 
			|| (r->kbyi * b->x0 - b->z1 + r->c_xz > 0)
			|| (r->ibyk * b->z0 - b->x1 + r->c_zx > 0)
			)
			return false;

		*t = (b->x1 - r->x) / r->i;
		float t2 = (b->z1 - r->z) / r->k;
		if(t2 > *t)
			*t = t2;

		return true;
		}

	case MOP:
		{
		if((r->y < b->y0) || (r->y > b->y1)
			|| (r->x < b->x0) || (r->z > b->z1)
			|| (r->kbyi * b->x0 - b->z0 + r->c_xz < 0)
			|| (r->ibyk * b->z1 - b->x1 + r->c_zx > 0)
			)
			return false;

		*t = (b->x1 - r->x) / r->i;
		float t2 = (b->z0 - r->z) / r->k;
		if(t2 > *t)
			*t = t2;

		return true;
		}

	case POM:
		{
		if((r->y < b->y0) || (r->y > b->y1)
			|| (r->x > b->x1) || (r->z < b->z0)
			|| (r->kbyi * b->x1 - b->z1 + r->c_xz > 0)
			|| (r->ibyk * b->z0 - b->x0 + r->c_zx < 0)
			)
			return false;

		*t = (b->x0 - r->x) / r->i;
		float t2 = (b->z1 - r->z) / r->k;
		if(t2 > *t)
			*t = t2;

		return true;
		}	

	case POP:
		{
		if((r->y < b->y0) || (r->y > b->y1)
			|| (r->x > b->x1) || (r->z > b->z1)
			|| (r->kbyi * b->x1 - b->z0 + r->c_xz < 0)
			|| (r->ibyk * b->z1 - b->x0 + r->c_zx < 0)
			)
			return false;

		*t = (b->x0 - r->x) / r->i;
		float t2 = (b->z0 - r->z) / r->k;
		if(t2 > *t)
			*t = t2;

		return true;
		}

	case MMO:
		{
		if((r->z < b->z0) || (r->z > b->z1)
			|| (r->x < b->x0) || (r->y < b->y0)
			|| (r->jbyi * b->x0 - b->y1 + r->c_xy > 0)
			|| (r->ibyj * b->y0 - b->x1 + r->c_yx > 0)
			)
			return false;

		*t = (b->x1 - r->x) / r->i;
		float t1 = (b->y1 - r->y) / r->j;
		if(t1 > *t)
			*t = t1;

		return true;
		}

	case MPO:
		{
		if((r->z < b->z0) || (r->z > b->z1)
			|| (r->x < b->x0) || (r->y > b->y1) 
			|| (r->jbyi * b->x0 - b->y0 + r->c_xy < 0) 
			|| (r->ibyj * b->y1 - b->x1 + r->c_yx > 0)
			)
			return false;

		*t = (b->x1 - r->x) / r->i;
		float t1 = (b->y0 - r->y) / r->j;
		if(t1 > *t)
			*t = t1;

		return true;
		}

	case PMO:
		{
		if((r->z < b->z0) || (r->z > b->z1)
			|| (r->x > b->x1) || (r->y < b->y0) 
			|| (r->jbyi * b->x1 - b->y1 + r->c_xy > 0)
			|| (r->ibyj * b->y0 - b->x0 + r->c_yx < 0) 
			)
			return false;

		*t = (b->x0 - r->x) / r->i;
		float t1 = (b->y1 - r->y) / r->j;
		if(t1 > *t)
			*t = t1;

		return true;
		}

	case PPO:
		{
		if((r->z < b->z0) || (r->z > b->z1)
			|| (r->x > b->x1) || (r->y > b->y1)  
			|| (r->jbyi * b->x1 - b->y0 + r->c_xy < 0)
			|| (r->ibyj * b->y1 - b->x0 + r->c_yx < 0)
			)
			return false;

		*t = (b->x0 - r->x) / r->i;
		float t1 = (b->y0 - r->y) / r->j;
		if(t1 > *t)
			*t = t1;

		return true;
		}
	case MOO:
		{
		if((r->x < b->x0)
			|| (r->y < b->y0) || (r->y > b->y1)
			|| (r->z < b->z0) || (r->z > b->z1)
			)
			return false;

		*t = (b->x1 - r->x) / r->i;
		return true;
		}

	case POO:
		{
		if((r->x > b->x1)
			|| (r->y < b->y0) || (r->y > b->y1)
			|| (r->z < b->z0) || (r->z > b->z1)
			)
			return false;

		*t = (b->x0 - r->x) / r->i;
		return true;
		}

	case OMO:
		{
		if((r->y < b->y0)
			|| (r->x < b->x0) || (r->x > b->x1)
			|| (r->z < b->z0) || (r->z > b->z1)
			)
			return false;
		
		*t = (b->y1 - r->y) / r->j;
		return true;
		}

	case OPO:
		{
		if((r->y > b->y1)
			|| (r->x < b->x0) || (r->x > b->x1)
			|| (r->z < b->z0) || (r->z > b->z1)
			)
			return false;

		*t = (b->y0 - r->y) / r->j;
		return true;
	}


	case OOM:
		{
		if((r->z < b->z0)
			|| (r->x < b->x0) || (r->x > b->x1)
			|| (r->y < b->y0) || (r->y > b->y1)
			)
			return false;

		*t = (b->z1 - r->z) / r->k;
		return true;
		}

	case OOP:
		{
		if((r->z > b->z1)
			|| (r->x < b->x0) || (r->x > b->x1)
			|| (r->y < b->y0) || (r->y > b->y1)
			)
			return false;

		*t = (b->z0 - r->z) / r->k;
		return true;
		}
	
	}

	return false;
}
