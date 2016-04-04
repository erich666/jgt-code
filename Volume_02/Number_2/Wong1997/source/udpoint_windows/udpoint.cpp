/*
 * udpoint.cpp
 *
 * An archive of functions to generate uniformly distributed points on
 * sphere and plane.
 *
 * (c) Copyright  Tien-Tsin Wong, 1996.
 * All Rights Reserved.
 * 
 * 19 Oct 1996: First release
 * 19 Aug 2000: Modified to support Window platform
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#ifndef WIN32
#include <values.h>
#endif
#include "vecmath.h"
#include "udpoint.h"

#define X   0
#define Y   1
#define Z   2

#ifdef WIN32  // for windows only
  #define M_PI	   3.14159265358979323846   // There is no M_PI defined in window
  #define RANDOM   ((float)rand()/(float)RAND_MAX)
#else  // for UNIX only
  #define RANDOM   ((float)random()/(float)MAXLONG)
#endif



/*
 * A function to uniformly distribute points on the spherical surface
 * randomly.
 */
void SphereStripRnd(float *result, int maxno)
{
  float theta, angle2;
  int i, pos;
  /*
   * Generate random point on the surface of a sphere
   */
  for (i=0, pos=0 ; i<maxno ; i++, pos+=3)
  {
    theta = acos(1.0 - 2.0*RANDOM);
    angle2 = 2.0*M_PI*RANDOM;
    result[pos]   = sin(theta)*cos(angle2);
    result[pos+1] = sin(theta)*sin(angle2);
    result[pos+2] = cos(theta);
  }
}


/*
 * Hammersley point sets. Deterministic and look random.
 * Base p = 2, which is especially fast for computation.
 */
void SphereHammersley(float *result, int n)
{
  float p, t, st, phi, phirad;
  int k, kk, pos;
  
  for (k=0, pos=0 ; k<n ; k++)
  {
    t = 0;
    for (p=0.5, kk=k ; kk ; p*=0.5, kk>>=1)
      if (kk & 1)                           // kk mod 2 == 1
	t += p;
    t = 2.0 * t  - 1.0;                     // map from [0,1] to [-1,1]

    phi = (k + 0.5) / n;                    // a slight shift
    phirad =  phi * 2.0 * M_PI;             // map to [0, 2 pi)

    st = sqrt(1.0-t*t);
    result[pos++] = st * cos(phirad);
    result[pos++] = st * sin(phirad);
    result[pos++] = t;
  }
}


/*
 * Hammersley point sets for any base p1. Deterministic and look random.
 */
void SphereHammersley2(float *result, int n, int p1)
{
  float a, p, ip, t, st, phi, phirad;
  int k, kk, pos;
  
  for (k=0, pos=0 ; k<n ; k++)
  {
    t = 0;
    ip = 1.0/p1;                           // recipical of p1
    for (p=ip, kk=k ; kk ; p*=ip, kk/=p1)  // kk = (int)(kk/p1)
      if ((a = kk % p1))
	t += a * p;
    t = 2.0 * t  - 1.0;                    // map from [0,1] to [-1,1]

    phi = (k + 0.5) / n;
    phirad =  phi * 2.0 * M_PI;            // map to [0, 2 pi)

    st = sqrt(1.0-t*t);
    result[pos++] = st * cos(phirad);
    result[pos++] = st * sin(phirad);
    result[pos++] = t;
  }
}


/*
 * Halton point set generation
 * two p-adic Van der Corport sequences
 * Useful for incremental approach.
 * p1 = 2, p2 = 3(default)
 */
void SphereHalton(float *result, int n, int p2)
{
  float p, t, st, phi, phirad, ip;
  int k, kk, pos, a;
  
  for (k=0, pos=0 ; k<n ; k++)
  {
    t = 0;
    for (p=0.5, kk=k ; kk ; p*=0.5, kk>>=1)
      if (kk & 1)                          // kk mod 2 == 1
	t += p;
    t = 2.0 * t - 1.0;                     // map from [0,1] to [-1,1]
    st = sqrt(1.0-t*t);

    phi = 0;
    ip = 1.0/p2;                           // recipical of p2
    for (p=ip, kk=k ; kk ; p*=ip, kk/=p2)  // kk = (int)(kk/p2)
      if ((a = kk % p2))
	phi += a * p;
    phirad =  phi * 4.0 * M_PI;            // map from [0,0.5] to [0, 2 pi)

    result[pos++] = st * cos(phirad);
    result[pos++] = st * sin(phirad);
    result[pos++] = t;
  }
}



/*
 * Halton point set generation
 * two p-adic Van der Corport sequences
 * Useful for incremental approach.
 * p1 = 2(default), p2 = 3(default)
 */
void SphereHalton2(float *result, int n, int p1, int p2)
{
  float p, t, st, phi, phirad, ip;
  int k, kk, pos, a;
  
  for (k=0, pos=0 ; k<n ; k++)
  {
    t = 0;
    ip = 1.0/p1;                           // recipical of p1
    for (p=ip, kk=k ; kk ; p*=ip, kk/=p1)  // kk = (int)(kk/p1)
      if ((a = kk % p1))
	t += a * p;
    t = 2.0 * t  - 1.0;                    // map from [0,1] to [-1,1]
    st = sqrt(1.0-t*t);

    phi = 0;
    ip = 1.0/p2;                           // recipical of p2
    for (p=ip, kk=k ; kk ; p*=ip, kk/=p2)  // kk = (int)(kk/p2)
      if ((a = kk % p2))
	phi += a * p;
    phirad =  phi * 4.0 * M_PI;            // map from [0,0.5] to [0, 2 pi)

    result[pos++] = st * cos(phirad);
    result[pos++] = st * sin(phirad);
    result[pos++] = t;
  }
}


///////////////////////////////// Plane ///////////////////////////////////

/*
 * Hammersley point sets. Deterministic and look random.
 * Base p = 2, which is especially fast for computation.
 */
void PlaneHammersley(float *result, int n)
{
  float p, u, v;
  int k, kk, pos;
  
  for (k=0, pos=0 ; k<n ; k++)
  {
    u = 0;
    for (p=0.5, kk=k ; kk ; p*=0.5, kk>>=1)
      if (kk & 1)                           // kk mod 2 == 1
	u += p;

    v = (k + 0.5) / n;

    result[pos++] = u;
    result[pos++] = v;
  }
}



/*
 * Hammersley point sets for any base p1. Deterministic and look random.
 */
void PlaneHammersley2(float *result, int n, int p1)
{
  float a, p, ip, u, v;
  int k, kk, pos;
  
  for (k=0, pos=0 ; k<n ; k++)
  {
    u = 0;
    ip = 1.0/p1;                           // recipical of p1
    for (p=ip, kk=k ; kk ; p*=ip, kk/=p1)  // kk = (int)(kk/p1)
      if ((a = kk % p1))
	u += a * p;

    v = (k + 0.5) / n;

    result[pos++] = u;
    result[pos++] = v;
  }
}


/*
 * Halton point set generation
 * two p-adic Van der Corport sequences
 * Useful for incremental approach.
 * p1 = 2, p2 = 3(default)
 */
void PlaneHalton(float *result, int n, int p2)
{
  float p, u, v, ip;
  int k, kk, pos, a;
  
  for (k=0, pos=0 ; k<n ; k++)
  {
    u = 0;
    for (p=0.5, kk=k ; kk ; p*=0.5, kk>>=1)
      if (kk & 1)                          // kk mod 2 == 1
	u += p;

    v = 0;
    ip = 1.0/p2;                           // recipical of p2
    for (p=ip, kk=k ; kk ; p*=ip, kk/=p2)  // kk = (int)(kk/p2)
      if ((a = kk % p2))
	v += a * p;

    result[pos++] = u;
    result[pos++] = v;
  }
}


/*
 * Halton point set generation
 * two p-adic Van der Corport sequences
 * Useful for incremental approach.
 * p1 = 2(default), p2 = 3(default)
 */
void PlaneHalton2(float *result, int n, int p1, int p2)
{
  float p, u, v, ip;
  int k, kk, pos, a;
  
  for (k=0, pos=0 ; k<n ; k++)
  {
    u = 0;
    ip = 1.0/p1;                           // recipical of p1
    for (p=ip, kk=k ; kk ; p*=ip, kk/=p1)  // kk = (int)(kk/p1)
      if ((a = kk % p1))
	u += a * p;

    v = 0;
    ip = 1.0/p2;                           // recipical of p2
    for (p=ip, kk=k ; kk ; p*=ip, kk/=p2)  // kk = (int)(kk/p2)
      if ((a = kk % p2))
	v += a * p;

    result[pos++] = u;
    result[pos++] = v;
  }
}


/*
 * Distribute random points on the 2D plane.
 */
void PlaneRandom(float *result, int maxno)
{
  int i, pos=0;
  for (i=0, pos=0 ; i<maxno ; i++)
  {
    result[pos++] = RANDOM;
    result[pos++] = RANDOM;
  }
}
