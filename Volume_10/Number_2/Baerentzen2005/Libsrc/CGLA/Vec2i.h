#ifndef __VEC2I_H
#define __VEC2I_H

#include "ArithVec.h"

namespace CGLA {
	class Vec2f;

	/** 2D Integer vector. */
	
	class Vec2i: public ArithVec<int,Vec2i,2>
	{
	public:
		
		/// Construct 0 vector
		Vec2i() {}

		/// Construct 2D int vector
		Vec2i(int _a,int _b): ArithVec<int,Vec2i,2>(_a,_b) {}

		/// Convert from 2D float vector
		explicit Vec2i(const Vec2f& v);
  
	};

}
#endif
