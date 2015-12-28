#ifndef __VEC3USI_H
#define __VEC3USI_H

#include "Vec3i.h"

namespace CGLA {
	typedef unsigned short int USInt;

	/** Unsigned short int 3D vector class. 
			This class is mainly useful if we need a 3D int vector that takes up
			less room than a Vec3i but holds larger numbers than a Vec3c. */
	class Vec3usi: public ArithVec<int,Vec3usi,3>
	{

	public:

		/// Construct 0 vector.
		Vec3usi() {}

		/// Construct a Vec3usi
		Vec3usi(USInt _a, USInt _b, USInt _c): ArithVec<int,Vec3usi,3>(_a,_b,_c) {}

		/// Construct a Vec3usi from a Vec3i. 
		explicit Vec3usi(const Vec3i& v): 
			ArithVec<int,Vec3usi,3>(v[0]&0xffff, v[1]&0xffff, v[2]&0xffff) {}
	};


}
#endif

