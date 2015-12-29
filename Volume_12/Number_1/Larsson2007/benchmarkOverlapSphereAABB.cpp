/****************************************************************/
/* Benchmark code for Sphere-AABB overlap tests                 */
/*                                                              */
/* This file contains the code used to get the benchmarking     */
/* results for the Sphere-AABB overlap tests presented in       */
/* the paper "On Faster Sphere-Box Overlap Testing" by          */
/* Thomas Larsson, Tomas Akenine-Moller and Eric Lengyel.       */
/*                                                              */
/* History:                                                     */
/*   2005-12-05: First version of source code created           */
/*   2006-05-08: Updated code to include SSE overlap test       */
/*                                                              */
/****************************************************************/

#define WIN_LEAN_AND_MEAN
#include <windows.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "timing.h"

#include "overlapSphereBox.h"

#define  NUMBER_OF_METHODS    5
#define  NUMBER_OF_BV_PAIRS 2000000
#define  REPETITIONS          10

Sphere3D sphereArray[NUMBER_OF_BV_PAIRS];
Box3D    boxArray[NUMBER_OF_BV_PAIRS];

char algorithmName[NUMBER_OF_METHODS][255] = {
	"overlapSphereAABB_Arvo",
	"overlapSphereAABB_QRI", 
	"overlapSphereAABB_QRF", 
	"overlapSphereAABB_Cons",
	"overlapSphereAABB_SSE"
};

double time[NUMBER_OF_METHODS];
int noOverlaps[NUMBER_OF_METHODS];

float getRandomScalar(float min, float max) 
{
	float frand = ((float)rand()) / ((float)RAND_MAX);
	return min + frand * (max-min);
}

void printResult(char * s, int overlaps, double time)
{	
	printf("Algorithm: %s\n", s);
	printf("Number of overlap tests: %d\n", NUMBER_OF_BV_PAIRS);
	printf("Number of overlaps found: %d\n", overlaps);
	printf("Overlap percentage: %f\n", 100.0f * (float)overlaps / (float) NUMBER_OF_BV_PAIRS);	
	printf("Total running time: %f\n", time);
	printf("Average time per overlap test: %.10lf\n", (double)time / NUMBER_OF_BV_PAIRS);
	printf("\n");
}

inline float MaxZero(const float& f)
{
	long i = *(long *) &f;
	i &= ~(i >> 31);
	return (*(float *) &i);
}

void main(void)
{
	int i, k;
	Timing watch;
	Point3D mid, ext;

	float cubeHalfSide = 100.0f; 
	float minSphereRadius; 
	float minHalfBoxExtent; 
	float maxSphereRadius;  
	float maxHalfBoxExtent; 

	// Note: Change the value of 'testCase' in [0, 4] to vary 
	// the number of overlaps in the generated test data.
	int testCase = 0;

	switch (testCase) {
		case 0:
			minSphereRadius = 1.0f;
			minHalfBoxExtent = 1.0f;
			maxSphereRadius = 32.5f;
			maxHalfBoxExtent = 32.5f;
			break;
		case 1:
			minSphereRadius = 1.0f;
			minHalfBoxExtent = 1.0f;			
			maxSphereRadius = 55.0f;
			maxHalfBoxExtent = 55.0f;
			break;
		case 2:
			minSphereRadius = 1.0f;
			minHalfBoxExtent = 1.0f;
			maxSphereRadius = 77.5f;
			maxHalfBoxExtent = 77.5f;
			break;
		case 3:
			minSphereRadius = 13.5f;
			minHalfBoxExtent = 13.5f;
			maxSphereRadius = 85.0f;
			maxHalfBoxExtent = 85.0f;
			break;
		case 4:
			minSphereRadius = 28.0f;
			minHalfBoxExtent = 28.0f;
			maxSphereRadius = 99.0f;
			maxHalfBoxExtent = 99.0f;
			break;
		default:
			minSphereRadius = 1.0f;
			minHalfBoxExtent = 1.0f;
			maxSphereRadius = 10.0f;
			maxHalfBoxExtent = 10.0f;
	}

	for (i = 0; i < NUMBER_OF_BV_PAIRS; i++) {
		sphereArray[i].r = getRandomScalar(minSphereRadius, maxSphereRadius);
		
		sphereArray[i].c.x = getRandomScalar(-cubeHalfSide + sphereArray[i].r, cubeHalfSide - sphereArray[i].r);
		sphereArray[i].c.y = getRandomScalar(-cubeHalfSide + sphereArray[i].r, cubeHalfSide - sphereArray[i].r);
		sphereArray[i].c.z = getRandomScalar(-cubeHalfSide + sphereArray[i].r, cubeHalfSide - sphereArray[i].r);
		
		ext.x = getRandomScalar(minHalfBoxExtent, maxHalfBoxExtent);
		ext.y = getRandomScalar(minHalfBoxExtent, maxHalfBoxExtent);
		ext.z = getRandomScalar(minHalfBoxExtent, maxHalfBoxExtent);
		
		mid.x = getRandomScalar(-cubeHalfSide + ext.x, cubeHalfSide - ext.x);
		mid.y = getRandomScalar(-cubeHalfSide + ext.y, cubeHalfSide - ext.y);
		mid.z = getRandomScalar(-cubeHalfSide + ext.z, cubeHalfSide - ext.z);

		boxArray[i].min.x = mid.x - ext.x;
		boxArray[i].min.y = mid.y - ext.y;
		boxArray[i].min.z = mid.z - ext.z;
		boxArray[i].max.x = mid.x + ext.x;
		boxArray[i].max.y = mid.y + ext.y;
		boxArray[i].max.z = mid.z + ext.z;
	}

	/* ==== Method 1 ==== */
	noOverlaps[0] = 0;
	watch.start();
	for (k=0; k < REPETITIONS; k++) {		
		for (i = 0; i < NUMBER_OF_BV_PAIRS; i++) {
			noOverlaps[0] += overlapSphereAABB_Arvo(sphereArray[i], boxArray[i]);
		}
	}
	time[0] = watch.stop() / REPETITIONS;
	noOverlaps[0] /= REPETITIONS;
	printResult(algorithmName[0], noOverlaps[0], time[0]);

	/* ==== Method 2 ==== */
	noOverlaps[1] = 0;
	watch.start();
	for (k=0; k < REPETITIONS; k++) {		
		for (i = 0; i < NUMBER_OF_BV_PAIRS; i++) {
			noOverlaps[1] += overlapSphereAABB_QRI(sphereArray[i], boxArray[i]);
		}
	}
	time[1] = watch.stop() / REPETITIONS;
	noOverlaps[1] /= REPETITIONS;
	printResult(algorithmName[1], noOverlaps[1], time[1]);

	/* ==== Method 3 ==== */
	noOverlaps[2] = 0;
	watch.start();
	for (k=0; k < REPETITIONS; k++) {		
		for (i = 0; i < NUMBER_OF_BV_PAIRS; i++) {
			noOverlaps[2] += overlapSphereAABB_QRF(sphereArray[i], boxArray[i]);
		}
	}
	time[2] = watch.stop() / REPETITIONS;
	noOverlaps[2] /= REPETITIONS;
	printResult(algorithmName[2], noOverlaps[2], time[2]);

	/* ==== Method 4 ==== */
	noOverlaps[3] = 0;
	watch.start();
	for (k=0; k < REPETITIONS; k++) {		
		for (i = 0; i < NUMBER_OF_BV_PAIRS; i++) {
			noOverlaps[3] += overlapSphereAABB_Cons(sphereArray[i], boxArray[i]);
		}
	}
	time[3] = watch.stop() / REPETITIONS;
	noOverlaps[3] /= REPETITIONS;
	printResult(algorithmName[3], noOverlaps[3], time[3]);

	/* ==== Method 5 ==== */
	noOverlaps[4] = 0;
	watch.start();
	for (k=0; k < REPETITIONS; k++) {		
		for (i = 0; i < NUMBER_OF_BV_PAIRS; i++) {
			noOverlaps[4] += overlapSphereAABB_SSE(sphereArray[i], boxArray[i]);
		}
	}
	time[4] = watch.stop() / REPETITIONS;
	noOverlaps[4] /= REPETITIONS;
	printResult(algorithmName[4], noOverlaps[4], time[4]);

	/* ==== Result Summary ==== */
	printf("=== Result summary ===\n\n");
	
	printf("Number of overlap tests: %d\n\n", NUMBER_OF_BV_PAIRS);

	printf("Algorithm\t\tTime\t(Speedup)\n");
	printf("%s\t%.4f\n", algorithmName[0], time[0]);
	for (i = 1; i < NUMBER_OF_METHODS; i++) {
		printf("%s\t%.4f\t(%.3lf)\n", algorithmName[i], time[i], time[0] / time[i]);
	}
	printf("\n");

	printf("Overlaps: %.2f percent\n", 100.0f * (float)noOverlaps[0] / NUMBER_OF_BV_PAIRS);	
	int noOverlapsReal = noOverlaps[0];
	int noOverlapsConservative = noOverlaps[3];
	printf("False positives reported in conservative test: %.3f percent\n", 100.0f * (noOverlapsConservative - noOverlapsReal) / (float)noOverlapsReal);
	getchar();
}