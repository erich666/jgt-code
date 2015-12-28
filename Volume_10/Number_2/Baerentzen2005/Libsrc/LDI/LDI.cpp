#include "LDI.h"

using namespace CGLA;
using namespace std;

namespace LDI
{

	bool LDILayer::convert_to_points(const Mat4x4f& itransf, 
																	 vector<PointRecord>& points)
	{
		int points_num = 0;
		for(int j=0;j<normal_buffer.get_ydim();++j)
			for(int i=0;i<normal_buffer.get_xdim();++i)
				{
					float z = depth_buffer(i,j);
					if(z > 0.0f && z < 1.0f)
						{
							Vec3f n = 2.0f*(normal_buffer(i,j) - Vec3f(0.5));
							float x = (i+0.5f)/float(normal_buffer.get_xdim());
							float y = (j+0.5f)/float(normal_buffer.get_ydim());
							Vec3f p = itransf.mul_3D_point(Vec3f(x,y,1-z));
							Vec4f c = colour_buffer(i,j);
							PointRecord r = {c,n,p};
							points.push_back(r);
							++points_num;
						}
				}
		return points_num>0;
	}


}
