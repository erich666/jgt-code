#ifndef __VEC3I_H
#define __VEC3I_H

#include "ArithVec.h"

namespace CGLA 
{
	class Vec3f;
	class Vec3uc;
	class Vec3usi;

	/** 3D integer vector. This class does not really extend the template
			and hence provides only the basic facilities of an ArithVec. 
			The class is typically used for indices to 3D voxel grids. */
	class Vec3i: public ArithVec<int,Vec3i,3>
	{
	public:
  
		/// Construct 0 vector.
		Vec3i() {}

		/// Construct a 3D integer vector.
		Vec3i(int _a,int _b,int _c): ArithVec<int,Vec3i,3>(_a,_b,_c) {}

		/// Construct a 3D integer vector with 3 identical coordinates.
		explicit Vec3i(int a): ArithVec<int,Vec3i,3>(a,a,a) {}
	
		/// Construct from a Vec3f.
		explicit Vec3i(const Vec3f& v);

		/// Construct from a Vec3uc.
		explicit Vec3i(const Vec3uc& v);

		/// Construct from a Vec3usi.
		explicit Vec3i(const Vec3usi& v);

	};

	/// Returns cross product of arguments
	inline Vec3i cross( const Vec3i& x, const Vec3i& y ) 
	{
		return Vec3i( x[1] * y[2] - x[2] * y[1], 
									x[2] * y[0] - x[0] * y[2], 
									x[0] * y[1] - x[1] * y[0] );
	}

}
#endif
