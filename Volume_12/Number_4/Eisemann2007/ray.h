/******************************************************************************

  This source code accompanies the Journal of Graphics Tools paper:

  "Fast Ray / Axis-Aligned Bounding Box Overlap Tests using Ray Slopes" 
  by Martin Eisemann, Thorsten Grosch, Stefan Müller and Marcus Magnor
  Computer Graphics Lab, TU Braunschweig, Germany and
  University of Koblenz-Landau, Germany

  Parts of this code are taken from
  "Fast Ray-Axis Aligned Bounding Box Overlap Tests With Pluecker Coordinates" 
  by Jeffrey Mahovsky and Brian Wyvill
  Department of Computer Science, University of Calgary

  This source code is public domain, but please mention us if you use it.

******************************************************************************/
#ifndef _RAY_H
#define _RAY_H

enum CLASSIFICATION
{ MMM, MMP, MPM, MPP, PMM, PMP, PPM, PPP, POO, MOO, OPO, OMO, OOP, OOM,
	OMM,OMP,OPM,OPP,MOM,MOP,POM,POP,MMO,MPO,PMO,PPO};

struct ray
{	
	//common variables
	float x, y, z;		// ray origin	
	float i, j, k;		// ray direction	
	float ii, ij, ik;	// inverses of direction components
	
	// ray slope
	int classification;
	float ibyj, jbyi, kbyj, jbyk, ibyk, kbyi; //slope
	float c_xy, c_xz, c_yx, c_yz, c_zx, c_zy;	
};

void make_ray(float x, float y, float z, float i, float j, float k, ray *r);

#endif
