/*
 * udpoint.cc
 *
 * An archive of methods to generate uniformly distributed points on
 * sphere. 
 *
 * Copyrighted by Tien-Tsin Wong, 1996.
 * 
 * 19 October 1996.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <values.h>
#include "vecmath.h"
#include "udpoint.h"

#define X   0
#define Y   1
#define Z   2


#ifndef UNIX
float random(){return 0;} // to fool the dos version
#endif

/*
 * This function cannot generate "uniformly" distributed random
 * vector.
 */
void OnCube(float *result, int maxno)
{
  float randomdir[3];
  int i;
  /*
   * Notice that the following way of generation of random direction
   * cannot generate random points statistically evenly distribute on
   * a sphere. Instead it generates random points statistically evenly
   * distribute on a cube. A more suitable generation method sure be
   * used later.
   */
  for (i=0 ; i<maxno ; i++)
  {
    randomdir[X] = (float)random()/(float)MAXLONG - 0.5;
    randomdir[Y] = (float)random()/(float)MAXLONG - 0.5;
    randomdir[Z] = (float)random()/(float)MAXLONG - 0.5;
    vnormal(randomdir);
    vcopy(randomdir, &(result[i*3]));
  }
}



/*
 * Generate a uniformly distributed random ray 
 * See paper for detail description of the technique
 * This the regular version.
 */
void SphereStripReg(float *result, int maxno)
{
  int counter=0;
  int i, j, paramax;
  float randomdir[3];
  float theta, angle2;
  /*
   * Generate random point on the surface of a sphere
   * See diary how I generate random point evenly on a sphere
   */
  paramax = (int)ceil(sqrt(maxno));
  printf("paramax=%d %d\n", paramax, maxno);
  for (i=0 ; i<paramax ; i++)
  {
    angle2 = 2*M_PI*((float)i/(float)paramax);
    for (j=0 ; j<paramax ; j++)
    {
      theta = acos(1 - 2*(float)j/(float)paramax); 
      randomdir[X] = sin(theta)*cos(angle2);
      randomdir[Y] = sin(theta)*sin(angle2);
      randomdir[Z] = cos(theta);
      vcopy(randomdir,&(result[counter*3]));
      counter++;
      if (counter >= maxno)
        return;
    }
  }
}


/*
 * This is the random version
 */
void SphereStripRnd(float *result, int maxno)
{
  float theta, angle2;
  int i, pos;
  /*
   * Generate random point on the surface of a sphere
   * See diary how I generate random point evenly on a sphere
   */
  for (i=0, pos=0 ; i<maxno ; i++, pos+=3)
  {
    theta = acos(1.0 - 2.0*(float)random()/(float)MAXLONG);
    angle2 = 2.0*M_PI*(float)random()/(float)MAXLONG;
    result[pos]   = sin(theta)*cos(angle2);
    result[pos+1] = sin(theta)*sin(angle2);
    result[pos+2] = cos(theta);
  }
}



/* 
 * The following function generate uniformly distributed pointset.
 * It is same as SphereStripReg(). Although the code look different
 * They can be reduced to the same equation.
 * This the regular version.
 */
void TwoPoleReg(float *result, int maxno)
{
  float randomdir[3];
  float t, phi, tmp;
  int i, j, paramax, counter=0;
  
  paramax = (int)ceil(sqrt(maxno));
  printf("paramax=%d\n", paramax);
  for (i=0 ; i<paramax ; i++)
  {
    phi = 2.0*M_PI*(float)i/(float)paramax;
    for (j=0 ; j<paramax ; j++)
    {
      t = 2.0*(float)j/(float)paramax - 1;
      tmp = sqrt(1.0-t*t);
  
      randomdir[X] = tmp*cos(phi);
      randomdir[Y] = tmp*sin(phi);
      randomdir[Z] = t;
      vcopy(randomdir, &(result[counter*3]));
      counter++;
      if (counter >= maxno)
        return;
    }
  }
}


/*
 * This is the randomized version
 */
void TwoPoleRnd(float *result, int maxno)
{
  float t, phi, tmp;
  int i, pos;

  for (i=0, pos=0 ; i<maxno ; i++, pos+=3)
  {
    t = 2.0*(float)random()/(float)MAXLONG - 1.0;
    phi = 2.0*M_PI*(float)random()/(float)MAXLONG;
    tmp = sqrt(1.0-t*t);

    result[pos]   = tmp*cos(phi);
    result[pos+1] = tmp*sin(phi);
    result[pos+2] = t;
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



void PlaneRandom(float *result, int maxno)
{
  int i, pos=0;
  for (i=0, pos=0 ; i<maxno ; i++)
  {
    result[pos++] = (float)random()/(float)MAXLONG;
    result[pos++] = (float)random()/(float)MAXLONG;
  }
}
