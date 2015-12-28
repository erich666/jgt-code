#include <algorithm>

#include "Vec3f.h"
#include "Vec3d.h"
#include "Vec3Hf.h"
#include "Quaternion.h"

using namespace std;
using namespace CMN;

namespace CGLA {

	Vec3f::Vec3f(const Quaternion& q):
		ArithVec<float,Vec3f,3>(q.qv) {}

	Vec3f::Vec3f(const Vec3d& v):
		ArithVec<float,Vec3f,3>(v[0], v[1], v[2]) {}

	void Vec3f::get_spherical( float &theta, float &phi, float &rlen ) const
	{  
		rlen = length();
		theta = acos(data[2]/rlen);    
		if (data[0]>0)
			phi = atan(data[1]/data[0]);
		else 
			if (data[0]<0)
				phi = atan(data[1]/data[0]) + M_PI;
			else 
				phi = (data[1]>0) ? M_PI_2 : -1 * M_PI_2;
	}


	void Vec3f::set_spherical( float theta, float phi, float rlen )
	{
		data[0] = rlen * sin(theta) * cos(phi);
		data[1] = rlen * sin(theta) * sin(phi);
		data[2] = rlen * cos(theta);
	}


	void orthogonal(const Vec3f& _a, Vec3f& b, Vec3f& c)
	{
		Vec3f a = normalize(_a);
		float max_sqval=sqr(a[0]);
		int mi=0;
		for(int i=1;i<3;i++)
			{
				float sqval = sqr(a[i]);
				if(max_sqval<sqval)
					{
						max_sqval = sqval;
						mi = i;
					}
			}
		b[mi] = 0;
		b[(mi+1)%3] = 1;
		b[(mi+2)%3] = 0;

		b = normalize(b-a*dot(b,a));
		c = normalize(cross(a,b));

		if(dot(cross(b,c), a) < 0) swap(b,c);
	}

}
