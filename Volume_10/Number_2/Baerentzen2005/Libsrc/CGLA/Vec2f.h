#ifndef __VEC2F_H
#define __VEC2F_H

#include "ArithVec.h"
#include "Vec2i.h"


namespace CGLA {

	/** 2D floating point vector */

	class Vec2f: public ArithVec<float,Vec2f,2>
	{
	public:

		Vec2f() {}

		Vec2f(float _a,float _b): ArithVec<float,Vec2f,2>(_a,_b) {}

		explicit Vec2f(const Vec2i& v): ArithVec<float,Vec2f,2>(v[0],v[1]) {}
		explicit Vec2f(float a): ArithVec<float,Vec2f,2>(a,a) {}
  
		/// Return Euclidean length
		float length() const 
    {
      return sqrt(sqr_length(*this));
		}

		/// Normalize vector
		void normalize() {*this/=length();}

	};


	/// Returns normalized vector
	inline Vec2f normalize(const Vec2f& v) 
	{
		return v/v.length();
	}

	/// Rotates vector 90 degrees to obtain orthogonal vector
	inline Vec2f orthogonal(const Vec2f& v) 
	{
		return Vec2f(-v[1],v[0]);
	}

	// Computes (scalar) cross product from two vectors
	inline float cross(const Vec2f& a, const Vec2f& b)
	{
		return a[0]*b[1]-a[1]*b[0];
	}

	/** The two last (scalar) arguments are the linear combination of 
			the two first arguments (vectors) which produces the third argument.
			*/
	bool linear_combine(const Vec2f&,const Vec2f&,const Vec2f&,float&,float&);



}
#endif
