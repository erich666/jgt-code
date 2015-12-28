#ifndef __MAT4X4_H
#define __MAT4X4_H

#include "Common/ExceptionStandard.h"
#include "CGLA.h"
#include "Vec3f.h"
#include "Vec3Hf.h"
#include "Vec4f.h"
#include "ArithSqMat.h"


namespace CGLA {

	DERIVEEXCEPTION(Mat4x4fException, CMN::MotherException);
	DERIVEEXCEPTION(Mat4x4fNotAffine, Mat4x4fException);
	DERIVEEXCEPTION(Mat4x4fSingular,  Mat4x4fException);

	/** Four by four float matrix.
			This class is useful for transformations such as perspective projections 
			or translation where 3x3 matrices do not suffice. */
	class Mat4x4f: public ArithSqMat<Vec4f, Mat4x4f, 4>
	{
	public:
  
		/// Construct a Mat4x4f from four Vec4f vectors
		Mat4x4f(Vec4f _a, Vec4f _b, Vec4f _c, Vec4f _d): 
			ArithSqMat<Vec4f, Mat4x4f, 4> (_a,_b,_c,_d) {}
  
		/// Construct the 0 matrix
		Mat4x4f() {}

		/// Construct from a pointed to array of 16 floats.
		Mat4x4f(const float* sa): ArithSqMat<Vec4f, Mat4x4f, 4> (sa) {}

		/** Multiply vector onto matrix. Here the fourth coordinate 
				is se to 0. This removes any translation from the matrix.
				Useful if one wants to transform a vector which does not
				represent a point but a direction. Note that this is not
				correct for transforming normal vectors if the matric 
				contains anisotropic scaling. */
		const Vec3f mul_3D_vector(const Vec3f& v) const
		{
			Vec4f v4  = (*this) * Vec4f(v[0],v[1],v[2],0);
			return Vec3f(v4[0],v4[1],v4[2]);
		}

		/** Multiply 3D point onto matrix. Here the fourth coordinate 
				becomes 1 to ensure that the point is translated. Note that
				the vector is converted back into a Vec3f without any 
				division by w. This is deliberate: Typically, w=1 except
				for projections. If we are doing projection, we can use
				project_3D_point instead */
		const Vec3f mul_3D_point(const Vec3f& v) const
		{
			Vec4f v4  = (*this) * Vec4f(v);
			return Vec3f(v4[0],v4[1],v4[2]);
		}

		/** Multiply 3D point onto matrix. We set w=1 before 
				multiplication and divide by w after multiplication. */
		const Vec3f project_3D_point(const Vec3f& v) const
		{
			return Vec3f((*this)*Vec3Hf(v));
		}

	};

	/// Create a rotation _matrix. Rotates about one of the major axes.
	Mat4x4f rotation_Mat4x4f(CGLA::Axis axis, float angle);

	/// Create a translation matrix
	Mat4x4f translation_Mat4x4f(const Vec3f&);

	/// Create a scaling matrix.
	Mat4x4f scaling_Mat4x4f(const Vec3f&);

	/// Create an identity matrix.
	inline Mat4x4f identity_Mat4x4f()
	{
		return Mat4x4f(Vec4f(1,0,0,0), 
									 Vec4f(0,1,0,0), 
									 Vec4f(0,0,1,0), 
									 Vec4f(0,0,0,1));
	}

	/** Compute the adjoint of a matrix. This is the matrix where each
			entry is the subdeterminant of 'in' where the row and column of 
			the element is removed. Use mostly to compute the inverse */
	Mat4x4f adjoint(const Mat4x4f& in);
		
	/** Compute the determinant of a 4x4f matrix. */ 
	float determinant(const Mat4x4f&);

	/// Compute the inverse matrix of a Mat4x4f.
	Mat4x4f invert(const Mat4x4f&);

	/// Compute the inverse matrix of a Mat4x4f that is affine
	Mat4x4f invert_affine(const Mat4x4f&);




	/** Create a perspective matrix. Assumes the eye is at the origin and
			that we are looking down the negative z axis.
    
			ACTUALLY THE EYE IS NOT AT THE ORIGIN BUT BEHIND IT. CHECK UP ON
			THIS ONE */
	Mat4x4f perspective_Mat4x4f(float d);


}
#endif







