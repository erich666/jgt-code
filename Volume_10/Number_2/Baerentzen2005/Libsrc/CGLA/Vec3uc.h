#ifndef __VEC3UC_H
#define __VEC3UC_H

#include "Vec3i.h"

namespace CGLA {
	typedef unsigned char UChar;

	/** 3D unsigned char vector. */
	class Vec3uc: public ArithVec<UChar,Vec3uc,3>
	{

	public:
		
		/// Construct 0 vector
		Vec3uc() {}

		/// Construct 3D uchar vector
		Vec3uc(UChar _a, UChar _b, UChar _c): ArithVec<UChar,Vec3uc,3>(_a,_b,_c) {}

		/// Convert from int vector. 
		explicit Vec3uc(const Vec3i& v): 
			ArithVec<UChar,Vec3uc,3>(v[0]&0xff, v[1]&0xff, v[2]&0xff) {}
	};


}
#endif

