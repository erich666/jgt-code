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
	double x, y, z;		// ray origin
	double R0;			// Pluecker coefficient R0
	double R1;			// Pluecker coefficient R1
	double R3;			// Pluecker coefficient R3
	double i;			// -R2 or i direction component
	double j;			// R5 or j direction component
	double k;			// -R4 or k direction component
	double ii, ij, ik;	// inverses of direction components
};

void make_ray(double x, double y, double z, double i, double j, double k, ray *r);

#endif
