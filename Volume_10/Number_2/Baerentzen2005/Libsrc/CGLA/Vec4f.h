#ifndef __VEC4F_H
#define __VEC4F_H

#include "Vec3f.h"

namespace CGLA {

	/** A four dimensional floating point vector. 
			This class is also used (via typedef) for
			homogeneous vectors.
	*/

	class Vec4f: public ArithVec<float,Vec4f,4>
	{
	public:
  
		/// Construct a (0,0,0,0) homogenous Vector
		Vec4f(): ArithVec<float,Vec4f,4>(0,0,0,0.0) {}

		/// Construct a (0,0,0,0) homogenous Vector
		explicit Vec4f(float _a): ArithVec<float,Vec4f,4>(_a,_a,_a,_a) {}

		/// Construct a 4D vector
		Vec4f(float _a, float _b, float _c, float _d): 
			ArithVec<float,Vec4f,4>(_a,_b,_c,_d) {}

		/// Construct a homogenous vector (a,b,c,1)
		Vec4f(float _a, float _b, float _c): 
			ArithVec<float,Vec4f,4>(_a,_b,_c,1.0) {}

		/// Construct a homogenous vector from a non-homogenous.
		explicit Vec4f(const Vec3f& v): 
			ArithVec<float,Vec4f,4>(v[0],v[1],v[2],1.0) {}

		/// Construct a homogenous vector from a non-homogenous.
		explicit Vec4f(const Vec3f& v,float _d): 
			ArithVec<float,Vec4f,4>(v[0],v[1],v[2],_d) {}

		/// Divide all coordinates by the fourth coordinate
		void de_homogenize();  

		operator Vec3f() const
		{
			float k = 1.0f/data[3];
			return  Vec3f(data[0]*k,data[1]*k,data[2]*k);
		}

		/// Compute Euclidean length.
		float length() const 
    {
      return sqrt(sqr_length(*this));
    }
	};

	/** This function divides a vector (x,y,z,w) by w 
			to obtain a new 4D vector where w=1. */
	inline void Vec4f::de_homogenize() 
	{
		float k = 1.0f/data[3];
		data[0] = data[0]*k;
		data[1] = data[1]*k;
		data[2] = data[2]*k;
		data[3] = 1.0;
	}

}
#endif

