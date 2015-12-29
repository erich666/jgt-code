/* 
 * Contains simple vector classes for 2D and 3D vectors stored as floats.
 * Replace with your own or extend these as desired.
 *
 * Written by Petrik Clarberg <petrik@cs.lth.se>, Lund University, 2007-2008.
 * This code is released as public domain for use free of charge for any
 * purpose, but without any kind of warranty.
 */
#ifndef __VEC_H__
#define __VEC_H__

#include <cmath>
#include <iostream>
#include <iomanip>

namespace mapping 
{
	// Useful math constants
	const float fPI_2	= 1.57079632679489661923132169163975144f;	// pi/2
	const float fPI_4	= 0.785398163397448309615660845819875721f;	// pi/4
	const float f2_PI	= 0.636619772367581343075535053490057448f;	// 2/pi
	const float f4_PI	= 1.27323954473516f;						// 4/pi
	
	
	//  ------------------------------------------------------------------------
	/// Represents a 2D vector (x,y) with the elements stored as floats.	
	//  ------------------------------------------------------------------------
	class vec2f
	{
	public:
		float x;
		float y;
		
	public:
		vec2f() {}
		~vec2f() {}
		
		/// Initialize all elements to the value s.
		inline explicit vec2f(float s) : x(s), y(s) {}
		
		/// Initializing the vector to (x,y).
		inline vec2f(float _x, float _y) : x(_x), y(_y) {}

		/// Write elements to an output stream nicely formatted.
		friend std::ostream& operator<< (std::ostream& os, const vec2f& rhs)
		{
			using namespace std;
			os << setprecision(6);
			os << setw(14) << rhs.x << setw(14) << rhs.y;
			return os;		
		}
	}; // class vec2f



	//  ------------------------------------------------------------------------
	/// Represents a 3D vector (x,y,z) with the elements stored as floats.	
	//  ------------------------------------------------------------------------
	class vec3f
	{
	public:
		float x;
		float y;
		float z;
		
	public:
		vec3f() {}
		~vec3f() {}
		
		/// Initialize all elements to the value s.
		inline explicit vec3f(float s) : x(s), y(s), z(s) {}
		
		/// Initializing the vector to (x,y,z).
		inline vec3f(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}

		/// Addition of two vectors.
		inline vec3f operator+ (const vec3f& rhs) const { return vec3f(x+rhs.x, y+rhs.y, z+rhs.z); }

		/// Subtraction of two vectors.
		inline vec3f operator- (const vec3f& rhs) const { return vec3f(x-rhs.x, y-rhs.y, z-rhs.z); }

		/// Returns the length of the vector.
		inline float len() const { return std::sqrt(x*x + y*y + z*z); }

		/// Normalize vector to unit length.
		inline vec3f& normalize()
		{ 
			float s = 1.f / len();
			x*=s; y*=s; z*=s;
			return *this;
		}

		/// Write elements to an output stream nicely formatted.
		friend std::ostream& operator<< (std::ostream& os, const vec3f& rhs)
		{
			using namespace std;
			os << setprecision(6);
			os << setw(14) << rhs.x << setw(14) << rhs.y << setw(14) << rhs.z;
			return os;		
		}	
	}; // class vec3f

} // namespace mapping
#endif // __VEC_H__
