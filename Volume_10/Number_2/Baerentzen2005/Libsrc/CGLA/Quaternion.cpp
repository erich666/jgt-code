#include "Vec3f.h"
#include "Quaternion.h"

using namespace CMN;

namespace CGLA {
	void Quaternion::make_rot(float angle, const Vec3f& v)
	{
		angle = angle/2;
		qv = CGLA::normalize(v);
		qv *= sin(angle);
		qw  = cos(angle);
	};

	void Quaternion::make_rot(const Vec3f& _v0, const Vec3f& _v1)
	{
		Vec3f v0 = CGLA::normalize(_v0);
		Vec3f v1 = CGLA::normalize(_v1);
		qv = cross(v0, v1);
		float l = qv.length();
		if(l<TINY)
			qv = Vec3f(1,0,0);
		else
			qv.normalize();
		float a = acos(dot(v0,v1))/2;
		qw  = cos(a);
		qv *= sin(a);	
	};

	void Quaternion::get_rot(float& angle, Vec3f& v) 
	{
		angle=2*acos(qw);

		if (angle<TINY) 
			v = Vec3f(1,0,0);
		else 
			v = qv / sin(angle);

		if (angle>M_PI)
			v = -v;

		v.normalize();
	}

	Mat3x3f Quaternion::get_mat3x3f() const
	{
		float s=2/norm();
		float m[9] = {1 - s*(qv[1]*qv[1]+qv[2]*qv[2]), 
									s*(qv[0]*qv[1]-qw*qv[2]), 
									s*(qv[0]*qv[2]+qw*qv[1]), 
									s*(qv[0]*qv[1]+qw*qv[2]), 
									1 - s*(qv[0]*qv[0]+qv[2]*qv[2]), 
         s*(qv[1]*qv[2]-qw*qv[0]), 
									s*(qv[0]*qv[2]-qw*qv[1]), 
									s*(qv[1]*qv[2]+qw*qv[0]), 
									1 - s*(qv[0]*qv[0]+qv[1]*qv[1])};
		Mat3x3f mat;
		mat.set(m);
		return mat;
	}


	//This function just need to call the right initialiser

	Mat4x4f Quaternion::get_mat4x4f() const
	{
		float s=2/norm();
		float m[16] = {1 - s*(qv[1]*qv[1]+qv[2]*qv[2]), 
									 s*(qv[0]*qv[1]-qw*qv[2]), 
									 s*(qv[0]*qv[2]+qw*qv[1]), 
									 float(0),
									 s*(qv[0]*qv[1]+qw*qv[2]), 
									 1 - s*(qv[0]*qv[0]+qv[2]*qv[2]), 
									 s*(qv[1]*qv[2]-qw*qv[0]), 
									 float(0),
									 s*(qv[0]*qv[2]-qw*qv[1]), 
									 s*(qv[1]*qv[2]+qw*qv[0]), 
									 1 - s*(qv[0]*qv[0]+qv[1]*qv[1]), 
									 float(0),
									 float(0),             
									 float(0),                   
									 float(0),                
									 float(1)};
		Mat4x4f mat;
		mat.set(m);
		return mat;
	}


	Vec3f Quaternion::apply(const Vec3f& vec) const
	{
		return Vec3f((*this)*Quaternion(vec)*inverse());
	}


}
