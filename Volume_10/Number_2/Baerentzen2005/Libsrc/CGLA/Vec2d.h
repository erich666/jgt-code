#ifndef __VEC2D_H
#define __VEC2D_H

#include "ArithVec.h"
#include "Vec2i.h"


namespace CGLA {

	/** 2D double floating point vector */

	class Vec2d: public ArithVec<double,Vec2d,2>
	{
	public:

		Vec2d() {}

		Vec2d(double _a,double _b): ArithVec<double,Vec2d,2>(_a,_b) {}

		explicit Vec2d(const Vec2i& v): ArithVec<double,Vec2d,2>(v[0],v[1]) {}
		explicit Vec2d(double a): ArithVec<double,Vec2d,2>(a,a) {}
  
		/// Return Euclidean length
		double length() const 
    {
      return sqrt(data[0]*data[0]+
                  data[1]*data[1]);
		}

		/// Normalize vector
		void normalize() {*this/=length();}

	};


	/// Returns normalized vector
	inline Vec2d normalize(const Vec2d& v) 
	{
		return v/v.length();
	}

	/// Rotates vector 90 degrees to obtain orthogonal vector
	inline Vec2d orthogonal(const Vec2d& v) 
	{
		return Vec2d(-v[1],v[0]);
	}

	// Computes (scalar) cross product from two vectors
	inline double cross(const Vec2d& a, const Vec2d& b)
	{
		return a[0]*b[1]-a[1]*b[0];
	}

}
#endif
