#ifndef __MAT3X3_H
#define __MAT3X3_H

#include "Common/ExceptionStandard.h"
#include "CGLA.h"
#include "Vec3f.h"
#include "ArithSqMat.h"

namespace CGLA {

	DERIVEEXCEPTION(Mat3x3fException, CMN::MotherException);
	DERIVEEXCEPTION(Mat3x3fSingular,  Mat3x3fException);

	/** 3 by 3 float matrix.
			This class will typically be used for rotation or
			scaling matrices for 3D vectors. */
	class Mat3x3f: public ArithSqMat<Vec3f, Mat3x3f, 3>
	{
	public:

		/// Construct matrix from 3 Vec3f vectors.
		Mat3x3f(Vec3f _a, Vec3f _b, Vec3f _c): 
			ArithSqMat<Vec3f, Mat3x3f, 3> (_a,_b,_c) {}
  
		/// Construct the 0 matrix
		Mat3x3f() {}

		/// Construct a matrix from a single scalar value.
		explicit Mat3x3f(float a): ArithSqMat<Vec3f, Mat3x3f, 3>(a) {}

	};

	/// Invert 3x3 matrix
	Mat3x3f invert(const Mat3x3f&);

	/// Create a rotation _matrix. Rotates about one of the major axes.
	Mat3x3f rotation_Mat3x3f(CGLA::Axis axis, float angle);

	/// Create a scaling matrix.
	Mat3x3f scaling_Mat3x3f(const Vec3f&);

	/// Create an identity matrix.
	inline Mat3x3f identity_Mat3x3f()
	{
		return Mat3x3f(Vec3f(1,0,0), Vec3f(0,1,0), Vec3f(0,0,1));
	}

	/** Compute determinant. There is a more generic function for
			computing determinants of square matrices (ArithSqMat). This one
			is faster but works only on Mat3x3f */
	inline float determinant(const Mat3x3f& m)
	{
		return 
			m[0][0]*m[1][1]*m[2][2] +
			m[0][1]*m[1][2]*m[2][0] +
			m[0][2]*m[1][0]*m[2][1] -
			m[0][2]*m[1][1]*m[2][0] -
			m[0][0]*m[1][2]*m[2][1] -
			m[0][1]*m[1][0]*m[2][2] ;
	}


}
#endif







