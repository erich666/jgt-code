/*
 * Simple test program for evaluating the precision and performance of the
 * equal-area mapping, using all three versions of the transform.
 *
 * Written by Petrik Clarberg <petrik@cs.lth.se>, Lund University, 2007-2008.
 * This code is released as public domain for use free of charge for any
 * purpose, but without any kind of warranty.
 */
#include <iostream>
#include <iomanip>
#include <sstream>

#include "mapping.h"
#include "mappingfast.h"
#include "mappingsse.h"

using namespace std;
using namespace mapping;	


/// Returns a random number in [0,1).
float uniform() { return (float)std::rand() / (1.f+RAND_MAX); }

// Timer functions (tic/toc, returns time in seconds)
// Don't forget to defined MACOSX if you compile on Mac
#ifdef MACOSX
#include <CoreServices/CoreServices.h>
AbsoluteTime _start;
void tic() { _start = UpTime(); }
double toc() {
	AbsoluteTime now = UpTime();
	AbsoluteTime absTime = SubAbsoluteFromAbsolute(now, _start);
	Nanoseconds ns = AbsoluteToNanoseconds(absTime);
	return (double)UnsignedWideToUInt64(ns) / 1.0e9;
}
#else // Fallback for non-Mac systems (poor precision)
#include <ctime>
std::clock_t _start;
void tic() { _start = std::clock(); }
double toc() {
	std::clock_t now = std::clock();
	return (double)(now-_start) / CLOCKS_PER_SEC;
}
#endif


//  ----------------------------------------------------------------------------
/// Computes an estimates of the precision by transforming a large number of
/// points/vectors using both the exact and approximate mappings. The error
/// is measured as Euclidian distance in 3D (approximately the same as arc
/// length on the unit sphere since the errors are very small).
//  ----------------------------------------------------------------------------
void precision()
{
	const int N = 30000;			// NxN points are used
	
	double maxDist1=0.0, maxDist2=0.0;
	double avgDist1=0.0, avgDist2=0.0;
	int count = 0;
	
	for(int y=0; y<N; y++)
	{
		for(int x=0; x<N; x++)
		{
			// Compute original point in 2D and exact vector in 3D.
			// The unit square is sampled using NxN stratified random points.
			vec2f p = vec2f( ((float)x+uniform())/N, ((float)y+uniform())/N );
			vec3f d = sqr2sph(p);
			vec2f_4 p4(p);						// p4 = original point as SSE vector
			vec3f_4 d4(d);						// d4 = original vector as SSE vector

			// Test precision of SSE-optimized version.
			vec2f_4 p4q = sph2sqr_sse(d4);		// p4q = approx pos in 2D (SSE)
			vec3f_4 d4q = sqr2sph_sse(p4);		// d4q = approx vector in 3D (SSE)
			vec3f dq1 = sqr2sph(p4q.at(0));		// dq1 = 3D vector corresponding to approx pos
			vec3f dq2 = d4q.at(0);				// dq2 = 3D vector corresponding to approx vector
			
			// Test precision of optimized scalar version (should be the same as the SSE version).
//			vec2f pq = sph2sqr_fast(d);		// pq = approx pos in 2D (scalar)
//			vec3f dq = sqr2sph_fast(p);		// dq = approx vector in 3D (scalar)
//			vec3f dq1 = sqr2sph(pq);		// dq1 = 3D vector corresponding to approx pos
//			vec3f dq2 = dq;					// dq2 = 3D vector corresponding to approx vector			
			
			// Compute errors as max and average Euclidian distance.
			double d1 = (d-dq1).len();
			double d2 = (d-dq2).len();
			if(d1 > maxDist1) maxDist1 = d1;			
			if(d2 > maxDist2) maxDist2 = d2;
			avgDist1 += d1;
			avgDist2 += d2;
			count++;
		}
	}
	avgDist1 /= (double)count;
	avgDist2 /= (double)count;
	
	cout << "Precision of sphere->square transform" << endl;
	cout << "  max distance: " << maxDist1 << endl;
	cout << "  avg distance: " << avgDist1 << endl;
	cout << endl;
	cout << "Precision of square->sphere transform" << endl;
	cout << "  max distance: " << maxDist2 << endl;
	cout << "  avg distance: " << avgDist2 << endl;
	cout << endl;
}



//  ----------------------------------------------------------------------------
/// Performs benchmarking of the different implementations using a specified
/// number of points/vectors. The tests are repeted M times to get better
/// accuracy. The timings are presented as clock cycles / transform, hence
/// we need to current CPU frequency (in Hz)
//  ----------------------------------------------------------------------------
void benchmark(int N, int numIter, double cpuFreq)
{
	// Allocate temporary storage
	vec2f* pos = new vec2f[N];
	vec3f* dir = new vec3f[N];
	vec2f_4* pos4 = new vec2f_4[N/4];
	vec3f_4* dir4 = new vec3f_4[N/4];

	double total = (double)N * numIter;
	double t1,t2,t3,c1,c2,c3;

	//  -------------------------------
	// Test square -> sphere transform
	//  -------------------------------
	
	cout << "Testing the square to sphere transform ..." << endl;
	cout << "  #points     : " << N << endl;
	cout << "  #iterations : " << numIter << endl;
	
	// Setup array of N random 2D points in [0,1].
	for(int i=0; i<N; i++)
	{
		vec2f p(uniform(),uniform());
		pos[i] = p;
		pos4[i/4].fx[i&3] = p.x;
		pos4[i/4].fy[i&3] = p.y;
	}

	// Perform two passes of the benchmarking to make sure the cache
	// is initialized before making the final run.
	for(int j=0; j<2; j++)
	{	
		// Timing of scalar version
		tic();
		for(int i=0; i<numIter; i++)
		{
			for(int j=0; j<N; j++)
				dir[j] = sqr2sph( pos[j] );
		}
		t1 = toc();

		// Timing of optimized scalar version
		tic();
		for(int i=0; i<numIter; i++)
		{
			for(int j=0; j<N; j++)
				dir[j] = sqr2sph_fast( pos[j] );
		}
		t2 = toc();
		
		// Timing of SSE version
		tic();
		for(int i=0; i<numIter; i++)
		{
			for(int j=0; j<N/4; j++)			
				dir4[j] = sqr2sph_sse( pos4[j] );
		}
		t3 = toc();
	}

	// Compute number of clock cycles / transform.
	total = (double)N * numIter;
	c1 = cpuFreq*t1 / total;
	c2 = cpuFreq*t2 / total;
	c3 = cpuFreq*t3 / total;

	// Print results.
	cout << "  time normal scalar   : " << t1 << " s" << endl;
	cout << "  time fast scalar     : " << t2 << " s" << endl;
	cout << "  time fast SSE        : " << t3 << " s" << endl;
	cout << "  cycles normal scalar : " << c1 << endl;
	cout << "  cycles fast scalar   : " << c2 << endl;
	cout << "  cycles fast SSE      : " << c3 << endl;	
	cout << endl;


	//  -------------------------------
	// Test sphere -> square transform
	//  -------------------------------
	
	cout << "Testing the sphere to square transform ..." << endl;
	cout << "  #vectors    : " << N << endl;
	cout << "  #iterations : " << numIter << endl;
	
	// Setup array of N random 3D vectors on the sphere.
	for(int i=0; i<N; i++)
	{
		vec2f p(uniform(),uniform());
		vec3f d = sqr2sph(p);
		d.normalize();	// just in case
		dir[i] = d;
		dir4[i/4].setAt(i&3,d);
	}
	
	// Perform two passes of the benchmarking to make sure the cache
	// is initialized before making the final run.
	for(int j=0; j<2; j++)
	{			
		// Timing of scalar version
		tic();
		for(int i=0; i<numIter; i++)
		{
			for(int j=0; j<N; j++)
				pos[j] = sph2sqr( dir[j] );
		}
		t1 = toc();
		
		// Timing of optimized scalar version
		tic();
		for(int i=0; i<numIter; i++)
		{
			for(int j=0; j<N; j++)
				pos[j] = sph2sqr_fast( dir[j] );
		}
		t2 = toc();
		
		// Timing of SSE version
		tic();
		for(int i=0; i<numIter; i++)
		{
			for(int j=0; j<N/4; j++)			
				pos4[j] = sph2sqr_sse( dir4[j] );
		}
		t3 = toc();
	}
	
	// Compute number of clock cycles / transform.
	c1 = cpuFreq*t1 / total;
	c2 = cpuFreq*t2 / total;
	c3 = cpuFreq*t3 / total;
	
	// Print results.
	cout << "  time normal scalar   : " << t1 << " s" << endl;
	cout << "  time fast scalar     : " << t2 << " s" << endl;
	cout << "  time fast SSE        : " << t3 << " s" << endl;
	cout << "  cycles normal scalar : " << c1 << endl;
	cout << "  cycles fast scalar   : " << c2 << endl;
	cout << "  cycles fast SSE      : " << c3 << endl;	
	cout << endl;
	
	// Delete buffers.
	delete[] pos;
	delete[] dir;
	delete[] pos4;
	delete[] dir4;
}	



//  ----------------------------------------------------------------------------
int main (int argc, char * const argv[])
{
	cout << "Test program for SIMD mapping." << endl;
	cout << "Written by Petrik Clarberg <petrik@cs.lth.se>, Lund University, 2007-2008." << endl;
	cout << "Released as public domain, without warranty, free of charge for any use." << endl;
	cout << endl;
	
	// Estimate the max and avergae approximation errors.
	// This function can take a while (couple of minutes).
	cout << "Estimating the approximation errors" << endl;
	precision();
	
	// Benchmark the implementations using different dataset sizes.
	// We ask the user about the CPU speed first.
	double cpuSpeed;
	cout << "What is your CPU frequency in GHz? " << flush;
	cin >> cpuSpeed;
	cpuSpeed *= 1.0e9;

	benchmark(1<<8,  1<<22, cpuSpeed);		// 256
	benchmark(1<<12, 1<<18, cpuSpeed);		// 4 k
	benchmark(1<<16, 1<<14, cpuSpeed);		// 64 k
	benchmark(1<<20, 1<<10, cpuSpeed);		// 1 M
	benchmark(1<<24, 1<<6, cpuSpeed);		// 16 M

	// Test bilinear interpolation
//	setup_bilerp(8);		// set map size to 2^8=256 pixels
//	vec2f pos(0.1f,0.7f);	// 2D coordinates of the point to be interpolated
//	int32_4 idx;
//	f32_4 w;
//	bilerp(pos, idx, w);	// idx & w holds the pixel indices and weights
//	cout << "indices : " << idx << endl;
//	cout << "weights : " << w << endl;

	return 0;
}
