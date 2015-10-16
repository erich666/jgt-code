#pragma once

struct RAYTRI
{
	float org[3];
	float end[3];
	float dir[3];
	float v0[3],v1[3],v2[3];

	struct PLANE
	{
		float x, y, z, d;
		enum MAIN_AXIS { X, Y, Z };
		MAIN_AXIS type;
	};
	PLANE plane;
};
