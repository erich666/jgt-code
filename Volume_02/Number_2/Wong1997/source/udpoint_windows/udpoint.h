#ifndef __UDPOINT_H
#define __UDPOINT_H

/*
 * Generate a uniformly distributed random point on sphere and plane.
 */
void PlaneRandom(float *result, int maxno);
void SphereStripRnd(float *result, int maxno);

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
