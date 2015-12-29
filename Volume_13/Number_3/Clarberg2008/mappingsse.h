/* 
 * Contains implementations of the forward and inverse transform using
 * SSE/SSE2 instructions. Four points/vectors are transformed at the time.
 *
 * The following functions are defined:
 *
 *     sph2sqr_sse  - maps 4x 3D vector to 2D points in the unit square
 *     sqr2sph_sse  - maps 4x 2D point in the unit square to 3D vectors
 *
 * and also:
 *
 * setup_bilerp - initializes the bilinear interpolation
 * bilerp	      - computes bilinear interpolation weights and pixel indices
 *
 * Written by Petrik Clarberg <petrik@cs.lth.se>, Lund University, 2007-2008.
 * This code is released as public domain for use free of charge for any
 * purpose, but without any kind of warranty.
 */
#ifndef __MAPPINGSSE_H__
#define __MAPPINGSSE_H__

#include <cmath>
#include <iostream>
#include <iomanip>

#include "vecsse.h"

namespace mapping
{
#ifdef __SSE__

	// 4-wide math constants
	const __m128 mfZERO	= _mm_set_ps1(0.f);			// 0
	const __m128 mfHALF	= _mm_set_ps1(0.5f);		// 0.5
	const __m128 mfONE	= _mm_set_ps1(1.f);			// 1
	const __m128 mfTWO	= _mm_set_ps1(2.f);			// 2
	const __m128 mfSIGN = f32_4(0x80000000U);		// sign bits set to 1

	// Coefficients for minimax approximation of sin(x*pi/4), x=[0,2].
	const __m128 _s1 = _mm_set_ps1( 0.7853975892066955566406250000000000f);
	const __m128 _s2 = _mm_set_ps1(-0.0807407423853874206542968750000000f);
	const __m128 _s3 = _mm_set_ps1( 0.0024843954015523195266723632812500f);
	const __m128 _s4 = _mm_set_ps1(-0.0000341485538228880614042282104492f);

	// Coefficients for minimax approximation of cos(x*pi/4), x=[0,2].
	const __m128 _c1 = _mm_set_ps1( 0.9999932952821962577665326692990000f);
	const __m128 _c2 = _mm_set_ps1(-0.3083711259464511647371969120320000f);
	const __m128 _c3 = _mm_set_ps1( 0.0157862649459062213825197189573000f);
	const __m128 _c4 = _mm_set_ps1(-0.0002983708648233575495551227373110f);

	// Coefficients for 6th degree minimax approximation of atan(x)*2/pi, x=[0,1].
	const __m128 _t1 = _mm_set_ps1( 0.406758566246788489601959989e-5f);
	const __m128 _t2 = _mm_set_ps1( 0.636226545274016134946890922156f);
	const __m128 _t3 = _mm_set_ps1( 0.61572017898280213493197203466e-2f);
	const __m128 _t4 = _mm_set_ps1(-0.247333733281268944196501420480f);
	const __m128 _t5 = _mm_set_ps1( 0.881770664775316294736387951347e-1f);
	const __m128 _t6 = _mm_set_ps1( 0.419038818029165735901852432784e-1f);
	const __m128 _t7 = _mm_set_ps1(-0.251390972343483509333252996350e-1f);


	//  ------------------------------------------------------------------------
	/// SSE-optimized transform of 4 points in the unit square to the unit
	/// sphere. Uses approximations of the trig operations and avoids branching.
	///
	/// Note that we introduce a bit too many registers to make the code more
	/// readable. Only x and y are used as general purpose registers, but a good
	/// compiler should be able to optimize the register allocation (re-using
	/// registers instead of allocating new ones). It is also possible the
	/// order of the operations can be optimized by moving the code blocks
	/// around. We have not experimented much with this.
	//  ------------------------------------------------------------------------
	inline static vec3f_4 sqr2sph_sse(const vec2f_4& p)
	{
		__m128 u,v,x,y;
		
		// Transform p from unit square to (u,v) in [-1,1].
		u = _mm_add_ps(p.x, p.x);
		v = _mm_add_ps(p.y, p.y);
		u = _mm_sub_ps(u, mfONE);				// u = 2*p.x-1	(in [-1,1])
		v = _mm_sub_ps(v, mfONE);				// v = 2*p.y-1	(in [-1,1])

		// Compute (x,y) as absolute of (u,v)
		x = _mm_andnot_ps(mfSIGN, u);			// x = |u|
		y = _mm_andnot_ps(mfSIGN, v);			// y = |v|

		// Compute signed distance along diagonal 'd' and 'radius'.
		__m128 sd, r;
		sd = _mm_add_ps(y, x);					// sd = y+x
		sd = _mm_sub_ps(mfONE, sd);				// sd = 1-(x+y)	(1->0->-1)
		r = _mm_andnot_ps(mfSIGN, sd);			// r = |sd|
		r = _mm_sub_ps(mfONE, r);				// r = 1-|sd|	(0->1->0)
		
		__m128 phi;
		phi = _mm_sub_ps(y, x);					// phi = y-x
		phi = _mm_div_ps(phi, r);				// phi = (y-x)/r	(in [-1,1])
		// Note: replacing this div with rcp+iteration was slower on Intel Core 2 Duo
		
		x = _mm_cmpneq_ps(r, _mm_setzero_ps());	// x=1..1 if r!=0, x=0..0 if r==0
		phi = _mm_and_ps(phi, x);				// set phi=0 if r==0 (avoid NaN from div-by-zero)
		phi = _mm_add_ps(mfONE, phi);			// phi = (y-x)/r+1	(in [0,2])

		// Compute minimax approx of cos(phi) & sin(phi) in first quadrant
		__m128 sp, cp;
		x = _mm_mul_ps(phi, phi);				// x = phi^2
		sp = _mm_madd_ps(_s3, _s4, x);
		sp = _mm_madd_ps(_s2, sp, x);
		sp = _mm_madd_ps(_s1 ,sp, x);
		sp = _mm_mul_ps(sp, phi);				// sp = s1*phi + s2*phi^3 + s3*phi^5 + s4*phi^7

		cp = _mm_madd_ps(_c3, _c4, x);
		cp = _mm_madd_ps(_c2, cp, x);
		cp = _mm_madd_ps(_c1, cp, x);			// cp = c1 + c2*phi^2 + c3*phi^4 + c4*phi^6
		
		// Flip signs of cos & sin based on the signs of u and v
		u = _mm_and_ps(mfSIGN, u);
		v = _mm_and_ps(mfSIGN, v);
		cp = _mm_xor_ps(cp, u);
		sp = _mm_xor_ps(sp, v);

		// Compute sin(theta). Note: replacing sqrt with rsqrt+iteration was slower on Intel Core 2 Duo.
		__m128 st;
		x = _mm_mul_ps(r, r);					// x = r^2
		st = _mm_sub_ps(mfTWO, x);				// st = 2-r^2
		st = _mm_sqrt_ps(st);					// st = sqrt(2-r^2)		
		st = _mm_mul_ps(st, r);					// st = r*sqrt(2-r^2)
		
		// Compute z = cos(theta)
		__m128 z;
		z = _mm_sub_ps(mfONE, x);				// z = 1-r^2
		x = _mm_and_ps(mfSIGN, sd);				// MSB 0 -> inner tri, 1 -> outer tri
		z = _mm_xor_ps(z, x);					// ct = -ct if outer triangle
		
		// Compute the output vector
		x = _mm_mul_ps(st, cp);					// x = sin(theta) * cos(phi)
		y = _mm_mul_ps(st, sp);					// y = sin(theta) * sin(phi)
												// z = cos(theta)		
		return vec3f_4(x,y,z);
	}



	//  ------------------------------------------------------------------------
	/// SSE-optimized transform of 4 unit vectors to the unit square,
	/// using minimax approximation of arctan and avoids branching.
	//  ------------------------------------------------------------------------
	inline static vec2f_4 sph2sqr_sse(const vec3f_4& d)
	{
		// Take the absolute of x,y to move the problem to the 1st quadrant
		__m128 u,v;
		u = _mm_andnot_ps(mfSIGN, d.x);			// u = |x|
		v = _mm_andnot_ps(mfSIGN, d.y);			// v = |y|
		
		// Compute atan(y/x). We reduce the problem to the angle [0,pi/4],
		// and use a minimax polynomial approximation of atan.
		__m128 a,b;
		a = _mm_max_ps(u, v);					// a = max(u,v)
		b = _mm_min_ps(u, v);					// b = min(u,v)
		b = _mm_div_ps(b,a);					// b = min(u,v) / max(u,v)
		a = _mm_cmpneq_ps(a, mfZERO);			// a=0..0 if max(u,v)==0
		b = _mm_and_ps(b, a);					// set b=0 if max(u,v)=0 to avoid NaN from div-by-zero
		
		// Use 6th degree minimax approximation of atan(x)*2/pi.
		// For clarity, we use a custom madd (mul+add) instruction.
		__m128 phi,r;
		phi = _mm_madd_ps(_t6, _t7, b);
		phi = _mm_madd_ps(_t5, phi, b);
		phi = _mm_madd_ps(_t4, phi, b);
		phi = _mm_madd_ps(_t3, phi, b);
		phi = _mm_madd_ps(_t2, phi, b);
		phi = _mm_madd_ps(_t1, phi, b);			// phi = atan(b)*2/pi, in [0,0.5]
		
		b = _mm_cmplt_ps(u,v);					// b=1..1 if u<v  (phi in [0.5,1])
		a = _mm_and_ps(mfONE, b);				// a = u<v ? 1 : 0
		b = _mm_and_ps(mfSIGN, b);				// sign bit of b
		phi = _mm_xor_ps(phi, b);				// flip sign of phi
		phi = _mm_add_ps(a, phi);				// phi = u<v ? 1-phi : phi;
		
		// Compute radius r.
		a = _mm_andnot_ps(mfSIGN,d.z);			// a = |z|
		r = _mm_sub_ps(mfONE, a);				// r = 1-|z|
		r = _mm_sqrt_ps(r);						// r = sqrt(1-|z|)
		
		// Compute position (u,v) in the inner triangle of the first quadrant
		v = _mm_mul_ps(phi, r);					// v = phi*r
		u = _mm_sub_ps(r, v);					// u = r-v
		
		// Mirror (u,v) to the outer triangle if z<0  (we reuse a,b,r)
		r = _mm_cmplt_ps(d.z, mfZERO);			// r=1..1 if z<0
		a = _mm_sub_ps(mfONE, v);				// a=1-v
		b = _mm_sub_ps(mfONE, u);				// b=1-u
		u = _mm_andnot_ps(r, u);				// u = z<0 ? 0 : u
		v = _mm_andnot_ps(r, v);				// v = z<0 ? 0 : v
		a = _mm_and_ps(r, a);					// a = z<0 ? 1-v : 0
		b = _mm_and_ps(r, b);					// b = z<0 ? 1-u : 0
		u = _mm_or_ps(u, a);					// u = z<0 ? 1-v : u
		v = _mm_or_ps(v, b);					// v = z<0 ? 1-u : v
		
		// Flip signs of (u,v) based on signs of (x,y).
		a = _mm_and_ps(mfSIGN, d.x);
		b = _mm_and_ps(mfSIGN, d.y);
		u = _mm_xor_ps(u, a);					// u = x<0 ? -u : u
		v = _mm_xor_ps(v, b);					// v = y<0 ? -v : v
		
		// Transform (u,v) from [-1,1] to [0,1].
		u = _mm_add_ps(u, mfONE);
		v = _mm_add_ps(v, mfONE);
		u = _mm_mul_ps(u, mfHALF);
		v = _mm_mul_ps(v, mfHALF);
		
		return vec2f_4(u,v);
	}

#endif // __SSE__

	
#ifdef __SSE2__
	
	// Constants needed for the SSE-optimized bilinear interpolation.
	// Initialize with the setup_bilerp() function.
	static int MAP_LEVEL;			///< The map is N=2^level pixels on each side.
	static __m128 MAP4_SIZE;		///< 4 packed floats set to N
	static __m128i MAP8_SIZE;		///< 8 packed 16-bit ints set to N
	static __m128i MAP8_SIZE_1;		///< 8 packed 16-bit ints set to N-1
	
	// Fixed constants for the bilerp function
	const __m128i MAP_DELTAS = int16_8(0,0,1,1,0,1,0,1);

	//  ------------------------------------------------------------------------
	/// Setup the bilinear interpolation function to use a map size of NxN
	/// pixels. N is computed as 2^level, as it has to be a power-of-two.
	/// If level=8 then the map size is N = 2^8, i.e., 256x256 pixels.
	//  ------------------------------------------------------------------------
	void setup_bilerp(int level)
	{
		int n = 1<<level;
		MAP_LEVEL = level;
		MAP4_SIZE = f32_4((float)n);
		MAP8_SIZE = int16_8(n);
		MAP8_SIZE_1 = int16_8(n-1);
	}
	
	//  ------------------------------------------------------------------------
	/// Computes the indices and weights needed for bilinear interpolation.
	/// The input is a 2D point pos in [0,1] (points outside this range are 
	/// wrapped using "mirrored repeat tiling" to their correct places).
	/// The output is a four packed ints (idx) representing the pixel indicies, 
	/// i.e., y*N+x, where N is the map size, and four packed floats (weight) 
	/// storing the interpolation weights (sum to 1).
	/// We use SSE2 to process 8 pixel coordinates (4 x and y) in parallel as 
	/// 16-bit integers. This limits the map size to 64k x 64k = 4G pixels!
	//  ------------------------------------------------------------------------
	void bilerp(const vec2f& pos, int32_4& idx, f32_4& weight)
	{
		// Input: pos = (s,t) = coordinates in [0,1].

		// Scale input coordinates by N and round to integer to find the offset 
		// for the bottom-right pixel (x+1,y+1). This assumes the SSE rounding
		// mode is set to round-to-nearest, which should be the default.
		// By using rounding, we implicitly take the pixel center (0.5,0.5) into
		// account. However, in the computation of the fractional part, we have
		// to add 0.5 to get the correct weights.
		
		// We use 6 XMM registers:
		__m128 p, q, w;
		__m128i c, d, m;
		
		p = _mm_set_ps(pos.y, pos.y, pos.x, pos.x);		// p = [t t s s]
		p = _mm_mul_ps(p, MAP4_SIZE);			// p = (s,t)*N
		c = _mm_cvtps_epi32(p);					// c = [y+1 y+1 x+1 x+1] as 32-bit ints
		q = _mm_cvtepi32_ps(c);					// q = round(p) as float
		
		// Compute the interpolation weights.		
		p = _mm_sub_ps(p,q);					// p = p-round(p)
		p = _mm_add_ps(p, mfHALF);				// p = p-round(p)+0.5 = [wy wy wx wx] fractions in [0,1)
		q = _mm_sub_ps(mfONE, p);				// q = [1-wy 1-wy 1-wx 1-wx]	
		w  = _mm_unpacklo_ps(q,p);				// w  = [wx 1-wx wx 1-wx]
		p = _mm_shuffle_ps(q,p, _MM_SHUFFLE(2,2,2,2));	// p = [wy wy 1-wy 1-wy]
		w = _mm_mul_ps(w,p);					// w = [wx*wy (1-wx)*wy wx*(1-wy) (1-wx)*(1-wy)]
		
		// Compute the integer indices (x,y) of the four pixels.
		// We use 16-bit integers and pack 8 coordinates into each XMM register.
		// Logic operations (xor, and) are used to detect the pixels (if any) 
		// that lie in an odd square and need to be mirrored.
		
		c = _mm_shufflelo_epi16(c, _MM_SHUFFLE(0,0,0,0));	// set lo 4 words to the value of bits  0..15 = x truncated to 16 bits
		c = _mm_shufflehi_epi16(c, _MM_SHUFFLE(2,2,2,2));	// set hi 4 words to the value of bits 32..47 = y truncated to 16 bits
															// c = [y y y y  x x x x]+1 as 16-bit ints
		c = _mm_sub_epi16(c, MAP_DELTAS);					// c = [y+1 y+1 y y  x+1 x x+1 x]
		m = _mm_shuffle_epi32(c, _MM_SHUFFLE(1,0,3,2));		// m = [x+1 x x+1 x  y+1 y+1 y y]
		
		m = _mm_xor_si128(c,m);					// m = x^y
		m = _mm_and_si128(m, MAP8_SIZE);		// m = (x^y) & N
		m = _mm_cmpeq_epi16(m, MAP8_SIZE);		// m = bit mask 1..1 => mirror, 0..0 otherwise
		
		c = _mm_and_si128(c, MAP8_SIZE_1);		// c = (x,y) mod N
		d = _mm_sub_epi16(MAP8_SIZE_1, c);		// d = N-1 - (x,y)  (mirrored coordinates)
		d = _mm_and_si128(m, d);				// d = mirror ? N-1-c : 0
		c = _mm_andnot_si128(m, c);				// c = mirror ? 0 : c
		c = _mm_or_si128(c, d);					// c = mirror ? N-1-c : c
		
		// Now, c holds the (x,y) coordinates of each pixel in the range {0,..,N-1}.
		// Perform address computation as: y*N+x using bit shifts and add.
		// We unpack the 16-bit ints to 32 bits as otherwise the max map size
		// would be 256x256 = 64k pixels.
		
		d = _mm_setzero_si128();				// d = 0..0
		m = _mm_unpackhi_epi16(c, d);			// m = [y3 y2 y1 y0] as 32-bit ints (upper halves set to 0)	
		d = _mm_unpacklo_epi16(c, d);			// d = [x3 x2 x1 x0] as 32-bit ints (upper halves set to 0)
		m = _mm_slli_epi32(m, MAP_LEVEL);		// m = y<<k, k=log2(N) <=> y*N
		m = _mm_add_epi32(m,d);					// m = y*N + x
		
		// Output:
		// m = integer indices (y*N+x) of each pixel in the bilinear lookup.
		// w = their respective interpolation weights, sum to 1.

		idx = int32_4(m);
		weight = f32_4(w);
	}

#endif // __SSE2__


} // namespace mapping
#endif // __MAPPINGSSE_H__
