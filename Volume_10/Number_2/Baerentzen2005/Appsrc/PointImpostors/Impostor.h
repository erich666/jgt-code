#ifndef IMPOSTOR_H
#define IMPOSTOR_H

#include <string>
#include <vector>
#include "CGLA/Vec3f.h"
#include "CGLA/Vec4f.h"
#include "LDI/PointRecord.h"
//#include "LDI/point_impostor.h"

class Impostor
{
	int list;
	unsigned int point_buffer;
	float unit_scale;
	
	const float pre_scale;
	const CGLA::Vec4f pre_rot;
	const CGLA::Vec3f pre_trans;

	LDI::PointRecord* points;
	int no_points;
	std::vector<int> bins;

	CGLA::Vec3f bsphere_centre;
	float bsphere_rad;

public:

	Impostor(const std::string& _file,
					 const std::string& _point_file,
					 float _pre_scale, 
					 const CGLA::Vec4f& _pre_rot,
					 const CGLA::Vec3f& _pre_trans);

	int draw_points(float point_frac) const;
	int draw_points(float Z0, float dist) const
		{
			float epsilon = (Z0/dist) * unit_scale * pre_scale;
			float point_frac = CMN::sqr(epsilon);
			return draw_points(1.0f<point_frac ? 1.0f : point_frac);
		}

	void get_bsphere(CGLA::Vec3f& centre, float& rad)
	{
		centre = bsphere_centre;
		rad = bsphere_rad;
	}

};


#endif
