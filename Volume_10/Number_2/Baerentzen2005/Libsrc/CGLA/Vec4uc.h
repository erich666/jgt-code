#ifndef __VEC4UC_H
#define __VEC4UC_H

#include "Vec4f.h"

namespace CGLA {
	typedef unsigned char UChar;

	/** 4D unsigned char vector. */
	class Vec4uc: public ArithVec<UChar,Vec4uc,4>
	{

	public:
		
		/// Construct 0 vector
		Vec4uc() {}

		/// Construct 0 vector
		Vec4uc(unsigned char a): ArithVec<UChar,Vec4uc,4>(a,a,a,a) {}

		/// Construct 4D uchar vector
		Vec4uc(UChar _a, UChar _b, UChar _c,UChar _d): 
			ArithVec<UChar,Vec4uc,4>(_a,_b,_c,_d) {}

		/// Convert from float vector. 
		explicit Vec4uc(const Vec4f& v): 
			ArithVec<UChar,Vec4uc,4>(UChar(v[0]), UChar(v[1]), 
															 UChar(v[2]), UChar(v[3])) {}

		operator Vec4f() const
		{
			return  Vec4f(data[0],data[1],data[2],data[3]);
		}

	};


}
#endif

