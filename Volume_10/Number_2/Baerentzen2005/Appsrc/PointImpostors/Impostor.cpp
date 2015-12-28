#include <GLEW/glew.h>
#include <GL/glu.h>
#include <cstring>
#include <string>

#include "Common/CommonDefs.h"
#include "Impostor.h"

using namespace std;
using namespace LDI;
using namespace CMN;
using namespace CGLA;

namespace
{
	class DistAtten
	{
		float x0, x1, a, b, k1, k2;
	public:
		DistAtten(float _x0, float _a, float _x1, float _b):
			x0(_x0), x1(_x1), a(_a), b(_b) 
		{
			k1 = (b-a)*(sqr(x0)*sqr(x1))/(sqr(x0)-sqr(x1));
			k2 = b - k1/sqr(x1);
		}
	
		float operator()(float x) const
		{
			return k1/sqr(x) + k2;
		}

	};

}

Impostor::Impostor(const std::string& file,
					 const std::string& point_file,
					 float _pre_scale, 
					 const Vec4f& _pre_rot,
					 const Vec3f& _pre_trans):
	pre_scale(_pre_scale), 
	pre_rot(_pre_rot),
	pre_trans(_pre_trans)
{
	vector<PointRecord> _points;
	load_points(point_file, _points, unit_scale);

	list = 0;
	Vec3f mip = _points[0].p;
	Vec3f map = _points[0].p;

	for(unsigned int i=1;i<_points.size();++i)
		{
			mip = v_min(mip, _points[i].p);
			map = v_max(map, _points[i].p);

			//cout << _points[i].p << " " ;
			//cout << _points[i].n << endl;
		}

	bsphere_centre = mip + 0.5 * (map-mip);
	bsphere_rad = 0.5 * (map-mip).length() * pre_scale;
	
	// Perform the random shuffling of points
	srand(0);
	random_shuffle(_points.begin(), _points.end());
	no_points = static_cast<int>(_points.size());


	cout << " No points = " << no_points << endl;

	glGenBuffersARB(1,&point_buffer);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, point_buffer);
	glBufferDataARB(GL_ARRAY_BUFFER_ARB, no_points*40, &_points[0],
									GL_STATIC_DRAW_ARB);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);	
}


int Impostor::draw_points(float point_frac) const
{
	glTranslatef(pre_trans[0],pre_trans[1],pre_trans[2]);
	glRotatef(pre_rot[3],pre_rot[0],pre_rot[1],pre_rot[2]);
	glScalef(pre_scale,pre_scale,pre_scale);
	int points_to_draw = static_cast<int>(point_frac*no_points);

	glBindBufferARB(GL_ARRAY_BUFFER_ARB, point_buffer);
	
	int max_pt = 1000;
	for(int i=0;i<points_to_draw; i+=max_pt)
		{
			glInterleavedArrays(GL_C4F_N3F_V3F, 40, (char*) NULL + 40*i);
			glDrawArrays(GL_POINTS, 0, min(max_pt,points_to_draw-i));
		}
	return points_to_draw;

}

