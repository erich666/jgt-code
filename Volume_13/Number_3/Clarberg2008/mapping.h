/* 
 * Contains implementations of the forward and inverse transform using
 * straightforward scalar code (built-in trig operations and branching).
 *
 * The following functions are defined:
 *
 *     sph2sqr - maps a 3D vector to a 2D point in the unit square
 *     sqr2sph - maps a 2D point in the unit square to a 3D vector
 *
 * Written by Petrik Clarberg <petrik@cs.lth.se>, Lund University, 2007-2008.
 * This code is released as public domain for use free of charge for any
 * purpose, but without any kind of warranty.
 */
#ifndef __MAPPING_H__
#define __MAPPING_H__

#include <cmath>
#include <iostream>
#include <iomanip>

#include "vec.h"


namespace mapping
{
	
	//  ------------------------------------------------------------------------
	/// Transform a 2D position p=(u,v) in the unit square to a normalized 3D 
	/// vector on the unit sphere. Straightforward scalar implementation.
	//  ------------------------------------------------------------------------
	inline static vec3f sqr2sph(const vec2f& p)
	{
		// Transform point from [0,1] to [-1,1]
		float u = 2.f*p.x - 1.f;
		float v = 2.f*p.y - 1.f;
		
		// Compute lengths (a,b) for rotated square (-45deg) scaled by sqrt(2).
		float a = v + u;
		float b = v - u;
		
		// Compute (r,phi) differently based on which quadrant we are in.
		// There are 8 different cases (3 levels of nestled if-statements).
		// We set z to -1 or 1 based on which hemisphere we are in.
		
		float r, phi, z;
		
		if(v >= 0.f)
		{
			if(u >= 0.f)	// quadrant 1
			{
				if(a <= 1.f)	// north
				{
					r = a;
					z = 1.f;
					phi = v/r;
				}
				else			// south
				{
					r = 2.f - a;
					z = -1.f;
					phi = (1.f - u) / r;
				}
			}
			else			// quadrant 2
			{
				if(b<=1.f)		// north
				{
					r = b;
					z = 1.f;
					phi = 1.f - u / r;
				}
				else
				{
					r = 2.f - b;
					z = -1.f;
					phi = 1.f + (1.f - v) / r;
				}
			}
		}
		else
		{
			if(u<0.0f)		// quadrant 3
			{
				if(a>=-1.f)		// north
				{
					r = -a;
					z = 1.f;
					phi = 2.f - v / r;
				}
				else			// south
				{
					r = 2.f + a;
					z = -1.f;
					phi = 2.f + (1.f + u) / r;
				}
			}
			else			// quadrant 4
			{
				if(b>=-1.f)		// north
				{
					r = -b;
					z = 1.f;
					phi = 3.f + u / r;
				}
				else			// south
				{
					r = 2.f + b;
					z = -1.f;
					phi = 3.f + (1.f + v) / r;
				}
			}
		}
		
		// Fix division-by-zero problem (r=0).
		if(r==0.f) phi = 0.f;
		
		// Compute 3D coordinate (x,y,z)
		float r2 = r*r;
		phi *= fPI_2;
		float sin_t = r * std::sqrt(2.f - r2);			// sin(theta)
		
		float x = sin_t * std::cos(phi);
		float y = sin_t * std::sin(phi);
		z *= 1.f - r2;
		
		return vec3f(x,y,z);
	}
	

	
	//  ------------------------------------------------------------------------
	/// Transforms a normalized 3D vector to a 2D position in the unit square.
	/// Straightforward scalar implementation using built-in trigonometric
	/// operations and branching.
	//  ------------------------------------------------------------------------
	inline static vec2f sph2sqr(const vec3f& d)
	{		
		float phi = std::atan2(d.y,d.x) * f2_PI;		// phi in [-2,2]
		float u, v;
		
		// There are 8 different cases we need to test (3 levels of nestled
		// if-statements to compute the (u,v) coordiantes in the square.
		
		if(d.z < 0.0f)		// southern hemisphere
		{
			float r = std::sqrt(1.f + d.z);
			
			if(phi >= 0.f)
			{
				if(phi <= 1.f)
				{
					u = 1.f - r*phi;
					v = 2.f - r - u;
				}
				else
				{
					u = r * (2.f - phi) - 1.f;
					v = 2.f - r + u;
				}
			}
			else
			{
				if(phi >= -1.f)
				{
					u = r*phi + 1.f;
					v = r - 2.f + u;
				}
				else
				{
					u = r * (2.f + phi) - 1.f;
					v = r - 2.f - u;
				}
			}		
		}
		else				// northern hemisphere
		{
			float r = std::sqrt(1.f - d.z);
			
			if(phi >= 0.f)
			{
				if(phi < 1.f)
				{
					v = r * phi;
					u = r - v;
				}
				else
				{
					v = r * (2.f - phi);
					u = v - r;
				}
			}
			else
			{
				if(phi > -1.f)
				{
					v = r * phi;
					u = r + v;
				}
				else
				{
					v = -r * (2.f + phi);
					u = -(r + v);
				}
			}
		}
		
		// Transform (u,v) from [-1,1] to [0,1]
		u = 0.5f * (u + 1.f);
		v = 0.5f * (v + 1.f);
		
		return vec2f(u,v);
	}


} // namespace mapping
#endif // __MAPPING_H__
