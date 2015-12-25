/******************************************************************************

  This source code accompanies the Journal of Graphics Tools paper:

  "Fast Ray-Axis Aligned Bounding Box Overlap Tests With Pluecker Coordinates" by
  Jeffrey Mahovsky and Brian Wyvill
  Department of Computer Science, University of Calgary

  This source code is public domain, but please mention us if you use it.

 ******************************************************************************/

/*
  This is the double-precision version of the test code for the paper.  The 
  individual test cases are separated into individual .cpp/.h files that
  correspond to the tests in the paper.

  The code was tested under both Microsoft Visual C++ .NET 2003 and gcc 2.96
  under Linux.  A solution (.sln) file is included to facilitate compiling 
  under VC++.  To compile under Linux, the command "g++ -O2 *.cpp" will
  produce an executable file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "ray.h"
#include "aabox.h"

#include "pluecker.h"
#include "plueckerint_div.h"
#include "plueckerint_mul.h"
#include "pluecker_cls.h"
#include "pluecker_cls_cff.h"
#include "plueckerint_div_cls.h"
#include "plueckerint_div_cls_cff.h"
#include "plueckerint_mul_cls.h"
#include "plueckerint_mul_cls_cff.h"
#include "smits_div.h"
#include "smits_div_cls.h"
#include "smits_mul.h"
#include "smits_mul_cls.h"
#include "standard_div.h"
#include "standard_mul.h"

double drand()
{
	// returns value in range -1.0 to 1.0
	return (double)(rand() - RAND_MAX / 2) / (double)(RAND_MAX / 2);
}

#define LOOPS 100
#define EPSILON 1e-10


int main(int argc, char *argv[])
{
	if(argc < 3)
	{
		printf("Usage: %s <cases> <hitcases>\n", argv[0]);
		exit(0);
	}

	int cases = atoi(argv[1]);
	int hitcases = atoi(argv[2]);

	printf("AABox double precision benchmark: cases = %d, hitcases = %d, looping %d times\n\n", 
		cases, hitcases, LOOPS);

	ray *rays = new ray[cases];
	aabox *aabbs = new aabox[cases];


	// generate the ray-box combinations that intersect

	int numcases = 0;

	while(numcases < hitcases)
	{
		ray r;
		aabox b;
		make_ray(drand(), drand(), drand(), drand(), drand(), drand(), &r);
		make_aabox(drand(), drand(), drand(), drand(), drand(), drand(), &b);

		double t;
		if(standard_div(&r, &b, &t))
		{
			rays[numcases] = r;
			aabbs[numcases] = b;
			numcases++;
		}
	}

	// generate the ray-box combinations that don't intersect

	while(numcases < cases)
	{
		ray r;
		aabox b;
		make_ray(drand(), drand(), drand(), drand(), drand(), drand(), &r);
		make_aabox(drand(), drand(), drand(), drand(), drand(), drand(), &b);

		double t;
		if(!standard_div(&r, &b, &t))
		{
			rays[numcases] = r;
			aabbs[numcases] = b;
			numcases++;
		}
	}

	// verify that all the different algorithms produce similar results

	for(int i = 0; i < cases; i++)
	{
		double hit_t = 0;
		bool hit = standard_div(&rays[i], &aabbs[i], &hit_t);

		double t = 0;

		if(hit != standard_mul(&rays[i], &aabbs[i], &t))
			printf("error: standard_mul()\n");
		if(hit && (fabs(hit_t - t) > EPSILON))
			printf("error: standard_mul() - t\n");

		if(hit != pluecker(&rays[i], &aabbs[i]))
			printf("error: pluecker()\n");

		if(hit != plueckerint_div(&rays[i], &aabbs[i], &t))
			printf("error: plueckerint_div()\n");
		if(hit && (fabs(hit_t - t) > EPSILON))
			printf("error: plueckerint_div() - t\n");

		if(hit != plueckerint_mul(&rays[i], &aabbs[i], &t))
			printf("error: plueckerint_mul()\n");
		if(hit && (fabs(hit_t - t) > EPSILON))
			printf("error: plueckerint_mul() - t\n");

		if(hit != pluecker_cls(&rays[i], &aabbs[i]))
			printf("error: pluecker_cls()\n");

		if(hit != pluecker_cls_cff(&rays[i], &aabbs[i]))
			printf("error: pluecker_cls_cff()\n");

		if(hit != plueckerint_div_cls(&rays[i], &aabbs[i], &t))
			printf("error: plueckerint_div_cls()\n");
		if(hit && (fabs(hit_t - t) > EPSILON))
			printf("error: plueckerint_div_cls() - t\n");

		if(hit != plueckerint_div_cls_cff(&rays[i], &aabbs[i], &t))
			printf("error: plueckerint_div_cls_cff()\n");
		if(hit && (fabs(hit_t - t) > EPSILON))
			printf("error: plueckerint_div_cls_cff() - t\n");

		if(hit != plueckerint_mul_cls(&rays[i], &aabbs[i], &t))
			printf("error: plueckerint_mul_cls()\n");
		if(hit && (fabs(hit_t - t) > EPSILON))
			printf("error: plueckerint_mul_cls() - t\n");

		if(hit != plueckerint_mul_cls_cff(&rays[i], &aabbs[i], &t))
			printf("error: plueckerint_mul_cls_cff()\n");
		if(hit && (fabs(hit_t - t) > EPSILON))
			printf("error: plueckerint_mul_cls_cff() - t\n");

		if(hit != smits_div(&rays[i], &aabbs[i], &t))
			printf("error: smits_div()\n");
		if(hit && (fabs(hit_t - t) > EPSILON))
			printf("error: smits_div() - t\n");

		if(hit != smits_div_cls(&rays[i], &aabbs[i], &t))
			printf("error: smits_div_cls()\n");
		if(hit && (fabs(hit_t - t) > EPSILON))
			printf("error: smits_div_cls() - t\n");

		if(hit != smits_mul(&rays[i], &aabbs[i], &t))
			printf("error: smits_mul()\n");
		if(hit && (fabs(hit_t - t) > EPSILON))
			printf("error: smits_mul() - t\n");

		if(hit != smits_mul_cls(&rays[i], &aabbs[i], &t))
			printf("error: smits_mul_cls()\n");
		if(hit && (fabs(hit_t - t) > EPSILON))
			printf("error: smits_mul_cls() - t\n");
	}


	// benchmark the individual algorithms

	{
		clock_t starttime = clock();

		int hits = 0;

		for(int l = 0; l < LOOPS; l++)
			for(int i = 0; i < cases; i++)
			{
				if(pluecker(&rays[i], &aabbs[i]))
					hits++;
			}

			clock_t endtime = clock();

			if(hits != hitcases * LOOPS)
				printf("error\n");

			printf("pluecker:\t\t time = %fs\n", (double)(endtime - starttime) / CLOCKS_PER_SEC);
	}

	{
		clock_t starttime = clock();

		int hits = 0;
		
		for(int l = 0; l < LOOPS; l++)
			for(int i = 0; i < cases; i++)
			{
				if(pluecker_cls(&rays[i], &aabbs[i]))
					hits++;
			}

			clock_t endtime = clock();

			if(hits != hitcases * LOOPS)
				printf("error\n");

			printf("pluecker_cls:\t\t time = %fs\n", (double)(endtime - starttime) / CLOCKS_PER_SEC);
	}

	{
		clock_t starttime = clock();

		int hits = 0;
		
		for(int l = 0; l < LOOPS; l++)
			for(int i = 0; i < cases; i++)
			{
				if(pluecker_cls_cff(&rays[i], &aabbs[i]))
					hits++;
			}

			clock_t endtime = clock();

			if(hits != hitcases * LOOPS)
				printf("error\n");

			printf("pluecker_cls_cff:\t time = %fs\n", (double)(endtime - starttime) / CLOCKS_PER_SEC);
	}

	printf("\n");

	{
		clock_t starttime = clock();

		int hits = 0;
		double t = 0;

		for(int l = 0; l < LOOPS; l++)
			for(int i = 0; i < cases; i++)
			{
				if(plueckerint_div(&rays[i], &aabbs[i], &t))
					hits++;
			}

			clock_t endtime = clock();

			if(hits != hitcases * LOOPS)
				printf("error\n");

			printf("plueckerint_div:\t time = %fs\n", (double)(endtime - starttime) / CLOCKS_PER_SEC);
	}

	{
		clock_t starttime = clock();

		int hits = 0;
		double t = 0;

		for(int l = 0; l < LOOPS; l++)
			for(int i = 0; i < cases; i++)
			{
				if(plueckerint_div_cls(&rays[i], &aabbs[i], &t))
					hits++;
			}

			clock_t endtime = clock();

			if(hits != hitcases * LOOPS)
				printf("error\n");

			printf("plueckerint_div_cls:\t time = %fs\n", (double)(endtime - starttime) / CLOCKS_PER_SEC);
	}

	{
		clock_t starttime = clock();

		int hits = 0;
		double t = 0;

		for(int l = 0; l < LOOPS; l++)
			for(int i = 0; i < cases; i++)
			{
				if(plueckerint_div_cls_cff(&rays[i], &aabbs[i], &t))
					hits++;
			}

			clock_t endtime = clock();

			if(hits != hitcases * LOOPS)
				printf("error\n");
					
			printf("plueckerint_div_cls_cff: time = %fs\n", (double)(endtime - starttime) / CLOCKS_PER_SEC);
	}

	{
		clock_t starttime = clock();

		int hits = 0;
		double t = 0;

		for(int l = 0; l < LOOPS; l++)
			for(int i = 0; i < cases; i++)
			{
				if(plueckerint_mul(&rays[i], &aabbs[i], &t))
					hits++;
			}

			clock_t endtime = clock();

			if(hits != hitcases * LOOPS)
				printf("error\n");

			printf("plueckerint_mul:\t time = %fs\n", (double)(endtime - starttime) / CLOCKS_PER_SEC);
	}

	{
		clock_t starttime = clock();

		int hits = 0;
		double t = 0;

		for(int l = 0; l < LOOPS; l++)
			for(int i = 0; i < cases; i++)
			{
				if(plueckerint_mul_cls(&rays[i], &aabbs[i], &t))
					hits++;
			}

			clock_t endtime = clock();

			if(hits != hitcases * LOOPS)
				printf("error\n");

			printf("plueckerint_mul_cls:\t time = %fs\n", (double)(endtime - starttime) / CLOCKS_PER_SEC);
	}

	{
		clock_t starttime = clock();

		int hits = 0;
		double t = 0;

		for(int l = 0; l < LOOPS; l++)
			for(int i = 0; i < cases; i++)
			{
				if(plueckerint_mul_cls_cff(&rays[i], &aabbs[i], &t))
					hits++;
			}

			clock_t endtime = clock();

			if(hits != hitcases * LOOPS)
				printf("error\n");

			printf("plueckerint_mul_cls_cff: time = %fs\n", (double)(endtime - starttime) / CLOCKS_PER_SEC);
	}

	printf("\n");

	{
		clock_t starttime = clock();

		int hits = 0;
		double t = 0;

		for(int l = 0; l < LOOPS; l++)
			for(int i = 0; i < cases; i++)
			{
				if(standard_div(&rays[i], &aabbs[i], &t))
					hits++;
			}

			clock_t endtime = clock();

			if(hits != hitcases * LOOPS)
				printf("error\n");

			printf("standard_div:\t\t time = %fs\n", (double)(endtime - starttime) / CLOCKS_PER_SEC);
	}

	{
		clock_t starttime = clock();

		int hits = 0;
		double t = 0;

		for(int l = 0; l < LOOPS; l++)
			for(int i = 0; i < cases; i++)
			{
				if(standard_mul(&rays[i], &aabbs[i], &t))
					hits++;
			}

			clock_t endtime = clock();

			if(hits != hitcases * LOOPS)
				printf("error\n");

			printf("standard_mul:\t\t time = %fs\n", (double)(endtime - starttime) / CLOCKS_PER_SEC);
	}

	printf("\n");

	{
		clock_t starttime = clock();

		int hits = 0;
		double t = 0;

		for(int l = 0; l < LOOPS; l++)
			for(int i = 0; i < cases; i++)
			{
				if(smits_div(&rays[i], &aabbs[i], &t))
					hits++;
			}

			clock_t endtime = clock();

			if(hits != hitcases * LOOPS)
				printf("error\n");

			printf("smits_div:\t\t time = %fs\n", (double)(endtime - starttime) / CLOCKS_PER_SEC);
	}

	{
		clock_t starttime = clock();

		int hits = 0;
		double t = 0;

		for(int l = 0; l < LOOPS; l++)
			for(int i = 0; i < cases; i++)
			{
				if(smits_div_cls(&rays[i], &aabbs[i], &t))
					hits++;
			}

			clock_t endtime = clock();

			if(hits != hitcases * LOOPS)
				printf("error\n");

			printf("smits_div_cls:\t\t time = %fs\n", (double)(endtime - starttime) / CLOCKS_PER_SEC);
	}

	{
		clock_t starttime = clock();

		int hits = 0;
		double t = 0;

		for(int l = 0; l < LOOPS; l++)
			for(int i = 0; i < cases; i++)
			{
				if(smits_mul(&rays[i], &aabbs[i], &t))
					hits++;
			}

			clock_t endtime = clock();

			if(hits != hitcases * LOOPS)
				printf("error\n");

			printf("smits_mul:\t\t time = %fs\n", (double)(endtime - starttime) / CLOCKS_PER_SEC);
	}

	{
		clock_t starttime = clock();

		int hits = 0;
		double t = 0;

		for(int l = 0; l < LOOPS; l++)
			for(int i = 0; i < cases; i++)
			{
				if(smits_mul_cls(&rays[i], &aabbs[i], &t))
					hits++;
			}

			clock_t endtime = clock();

			if(hits != hitcases * LOOPS)
				printf("error\n");

			printf("smits_mul_cls:\t\t time = %fs\n", (double)(endtime - starttime) / CLOCKS_PER_SEC);
	}

	return 0;
}

