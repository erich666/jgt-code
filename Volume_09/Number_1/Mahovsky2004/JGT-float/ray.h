/******************************************************************************

  This source code accompanies the Journal of Graphics Tools paper:

  "Fast Ray-Axis Aligned Bounding Box Overlap Tests With Pluecker Coordinates" by
  Jeffrey Mahovsky and Brian Wyvill
  Department of Computer Science, University of Calgary

  This source code is public domain, but please mention us if you use it.

 ******************************************************************************/

#ifndef _RAY_H
#define _RAY_H

enum CLASSIFICATION
{ MMM, MMP, MPM, MPP, PMM, PMP, PPM, PPP };

struct ray
{
	int classification;	// MMM, MMP, etc.
	float x, y, z;		// ray origin
	float R0;			// Pluecker coefficient R0
	float R1;			// Pluecker coefficient R1
	float R3;			// Pluecker coefficient R3
	float i;			// -R2 or i direction component
	float j;			// R5 or j direction component
	float k;			// -R4 or k direction component
	float ii, ij, ik;	// inverses of direction components
};

void make_ray(float x, float y, float z, float i, float j, float k, ray *r);

#endif
