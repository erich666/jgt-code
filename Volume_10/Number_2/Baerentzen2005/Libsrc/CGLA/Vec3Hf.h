#ifndef __VEC3HF_H
#define __VEC3HF_H

/** The H in Vec3Hf stands for homogenous. 

*/

#include "Vec4f.h"

namespace CGLA {

	/** A 3D homogeneous vector is simply a four D vector.
			I find this simpler than a special class for homogeneous
			vectors. */
	typedef Vec4f Vec3Hf;

}
#endif

