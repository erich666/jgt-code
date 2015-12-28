#ifndef __POINTRECORD_H
#define __POINTRECORD_H

#include <string>
#include <vector>
#include "CGLA/Vec3f.h"
#include "CGLA/Vec4f.h"

namespace LDI
{
	struct PointRecord
	{
		CGLA::Vec4f c;
		CGLA::Vec3f n;
		CGLA::Vec3f p;
	};


	bool load_points(const std::string&, 
									 std::vector<PointRecord>&,
									 float& unit_scale);
	bool save_points(const std::string&, 
									 const std::vector<PointRecord>&,
									 float unit_scale);
}

#endif
