#ifndef __MAT3X3D_H
#define __MAT3X3D_H

#include "Common/ExceptionStandard.h"
#include "CGLA.h"
#include "Vec3d.h"
#include "ArithSqMat.h"

namespace CGLA {

	DERIVEEXCEPTION(Mat3x3dException, CMN::MotherException);
	DERIVEEXCEPTION(Mat3x3dSingular,  Mat3x3dException);

	/** 3 by 3 float matrix.
			This class will typically be used for rotation or
			scaling matrices for 3D vectors. */
	class Mat3x3d: public ArithSqMat<Vec3d, Mat3x3d, 3>
	{
	public:
			
		/// Construct matrix from 3 Vec3d vectors.
		Mat3x3d(Vec3d _a, Vec3d _b, Vec3d _c): 
			ArithSqMat<Vec3d, Mat3x3d, 3> (_a,_b,_c) {}
			
		/// Construct the 0 matrix
		Mat3x3d() {}
				
		/// Construct a matrix from a single scalar value.
		explicit Mat3x3d(float a): ArithSqMat<Vec3d, Mat3x3d, 3>(a) {}
					
	};

	/// Create an identity matrix.
	inline Mat3x3d identity_Mat3x3d()
	{
		return Mat3x3d(Vec3d(1,0,0), Vec3d(0,1,0), Vec3d(0,0,1));
	}

	/** Compute determinant. There is a more generic function for
			computing determinants of square matrices (ArithSqMat). This one
			is faster but works only on Mat3x3d */
	inline float determinant(const Mat3x3d& m)
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







