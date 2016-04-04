#ifndef __UDPOINT_H
#define __UDPOINT_H


/*
 * This function cannot generate "uniformly" distributed random 
 * vector.
 */
void OnCube(float *result, int maxno);
void PlaneRandom(float *result, int maxno);

/*
 * Generate a uniformly distributed random ray 
 * See paper for detail description of the technique
 */
void SphereStripReg(float *result, int maxno);
void SphereStripRnd(float *result, int maxno);

/* 
 * The following function generate non-uniform distributed pointset.
 * The point will be dense at the two poles and sparse at the equator.
 */
void TwoPoleReg(float *result, int maxno);
void TwoPoleRnd(float *result, int maxno);

/*
 * Harmmersley point sets
 */
void PlaneHammersley(float *result, int n);
void PlaneHammersley2(float *result, int n, int p1);
void SphereHammersley(float *result, int n);
void SphereHammersley2(float *result, int n, int p1);


/*
 * Halton sequences
 */ 
void PlaneHalton(float *result, int n, int p2=3);
void PlaneHalton2(float *result, int n, int p1=2, int p2=3);
void SphereHalton(float *result, int n, int p2=3);
void SphereHalton2(float *result, int n, int p1=2, int p2=3);

#endif
