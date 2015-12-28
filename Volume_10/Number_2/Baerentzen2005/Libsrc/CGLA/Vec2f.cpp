#include <iostream>
#include <algorithm>
#include "Vec2f.h"
#include "Mat2x2f.h"
#include "CGLA.h"

namespace CGLA {
	using namespace std;

	bool linear_combine(const Vec2f& a, const Vec2f& b, const Vec2f& c,
											float& x, float& y)
	{
		Mat2x2f inv_mat, mat(a[0],b[0],a[1],b[1]);
		if(invert(mat,inv_mat))
			{
				const Mat2x2f blob(inv_mat);
				Vec2f xy = blob * c;
				x = xy[0];
				y = xy[1];
				return true;
			}
		return false;
	}


}
