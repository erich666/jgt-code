#include "Vec2i.h"
#include "Vec2f.h"

namespace CGLA {

	Vec2i::Vec2i(const Vec2f& v):
		ArithVec<int,Vec2i,2>(int(floor(v[0])), int(floor(v[1]))) {}

}
