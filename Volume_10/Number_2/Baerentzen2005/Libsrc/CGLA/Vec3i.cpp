#include "Vec3i.h"
#include "Vec3f.h"
#include "Vec3uc.h"

namespace CGLA {

	Vec3i::Vec3i(const Vec3f& v):
		ArithVec<int,Vec3i,3>(int(floor(v[0])), 
													int(floor(v[1])), 
													int(floor(v[2]))) {}

	Vec3i::Vec3i(const Vec3uc& v): ArithVec<int,Vec3i,3>(v[0],v[1],v[2]) {}

	Vec3i::Vec3i(const Vec3usi& v): ArithVec<int,Vec3i,3>(v[0],v[1],v[2]) {}

}
