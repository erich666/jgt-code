#ifndef __QUATERNION_H
#define __QUATERNION_H

#include "CGLA/Vec3f.h"
#include "CGLA/Vec4f.h"
#include "CGLA/Mat3x3f.h"
#include "CGLA/Mat4x4f.h"

namespace CGLA {

#ifndef M_PI
#define M_PI 3.1415926
#endif

	/** A Quaterinion class. Quaternions are algebraic entities 
			useful for rotation. */

	class Quaternion
	{
	public:

		/// Vector part of quaternion
		Vec3f qv;

		/// Scalar part of quaternion
		float qw;

		/// Construct 0 quaternion
		Quaternion(): qw(1) {}

		/// Construct quaternion from vector and scalar
		Quaternion(const Vec3f _qv, float _qw=1) : 
			qv(_qv) , qw(_qw) {}

		/// Construct quaternion from four scalars
		Quaternion(float x, float y, float z, float _qw) : 
			qv(x,y,z), qw(_qw) {}

		/// Assign values to a quaternion
		void set(float x, float y, float z, float _qw) 
		{
			qv.set(x,y,z);
			qw = _qw;
		}

		/// Get values from a quaternion
		void get(float& x, float& y, float& z, float& _qw) const
		{
			x  = qv[0];
			y  = qv[1];
			z  = qv[2];
			_qw = qw;
		}

		/// Get a 3x3 rotation matrix from a quaternion
		Mat3x3f get_mat3x3f() const;

		/// Get a 4x4 rotation matrix from a quaternion
		Mat4x4f get_mat4x4f() const;

		/// Construct a Quaternion from an angle and axis of rotation.
		void make_rot(float angle, const Vec3f&);

		/** Construct a Quaternion rotating from the direction given
				by the first argument to the direction given by the second.*/
		void make_rot(const Vec3f&,const Vec3f&);

		/// Obtain angle of rotation and axis
		void get_rot(float& angle, Vec3f&);

		/// Multiply two quaternions. (Combine their rotation)
		Quaternion operator *(Quaternion quat) const;

		/// Multiply scalar onto quaternion.
		Quaternion operator *(float scalar) const;

		/// Add two quaternions.
		Quaternion operator +(Quaternion quat) const;
		
		/// Invert quaternion
		Quaternion inverse() const;

		/// Return conjugate quaternion
		Quaternion conjugate() const;

		/// Compute norm of quaternion
		float norm() const;

		/// Normalize quaternion.
		Quaternion normalize();

		/// Rotate vector according to quaternion
		Vec3f apply(const Vec3f& vec) const;
	};

	/// Compare for equality.
	inline bool operator==(const Quaternion& q0, const Quaternion& q1)
	{
		return (q0.qw == q1.qw && q0.qv == q1.qv);
	}

	/// Print quaternion to stream.
	inline std::ostream& operator<<(std::ostream&os, const Quaternion v)
	{
		os << "[ ";
		for(int i=0;i<3;i++) os << v.qv[i] << " ";
		os << " ~ " << v.qw << " ";
		os << "]";

		return os;
	}

	inline Quaternion Quaternion::operator *(Quaternion quat) const
	{
		return Quaternion(cross(qv,quat.qv) + quat.qw*qv + qw*quat.qv, 
											qw*quat.qw - dot(qv,quat.qv));
	}

	inline Quaternion Quaternion::operator *(float scalar) const
	{
		return Quaternion(scalar*qv,scalar*qw);
	}

	/// Multiply scalar onto quaternion
	inline Quaternion operator *(float scalar, Quaternion quat) 
	{
		return Quaternion(scalar*quat.qv,scalar*quat.qw);
	}

	
	inline Quaternion Quaternion::operator +(Quaternion quat) const
	{
		return Quaternion(qv + quat.qv,qw + quat.qw);
	}

	inline float Quaternion::norm() const
	{
		return qv[0]*qv[0] + qv[1]*qv[1] + qv[2]*qv[2] + qw*qw;
	}

	inline Quaternion Quaternion::normalize() 
	{
		return Quaternion(1/norm()*(*this));
	}

	inline Quaternion Quaternion::conjugate() const
	{
		return Quaternion(-qv,qw);
	}

	inline Quaternion Quaternion::inverse() const
	{
		return Quaternion(1/norm()*conjugate());
	}


	/** Perform linear interpolation of two quaternions. 
			The last argument is the parameter used to interpolate
			between the two first. SLERP - invented by Shoemake -
			is a good way to interpolate because the interpolation
			is performed on the unit sphere. 	
	*/
	inline Quaternion slerp(Quaternion q0, Quaternion q1, float t) 
	{
		float angle=acos(q0.qv[0]*q1.qv[0]+q0.qv[1]*q1.qv[1]+q0.qv[2]*q1.qv[2]+q0.qw*q1.qw);
		return (q0*sin((1-t)*angle)+q1*sin(t*angle))*(1/sin(angle));
	}

}
#endif //__QUATERNION_H
