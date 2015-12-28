#ifndef __MAT2X2F_H
#define __MAT2X2F_H

#include "Common/ExceptionStandard.h"
#include "Vec2f.h"
#include "ArithSqMat.h"


namespace CGLA {

	DERIVEEXCEPTION(Mat2x2fException, CMN::MotherException);

	/** Two by two float matrix. This class is useful for various 
			vector transformations in the plane. */
	class Mat2x2f: public ArithSqMat<Vec2f, Mat2x2f, 2>
	{
	public:

		/// Construct a Mat2x2f from two Vec2f vectors.
		Mat2x2f(Vec2f _a, Vec2f _b): ArithSqMat<Vec2f, Mat2x2f, 2> (_a,_b) {}

		/// Construct a Mat2x2f from four scalars.
		Mat2x2f(float _a, float _b, float _c, float _d): 
			ArithSqMat<Vec2f, Mat2x2f, 2>(Vec2f(_a,_b),Vec2f(_c,_d)) {}
  
		/// Construct the 0 matrix
		Mat2x2f() {}

	};

	/** Compute the determinant of a Mat2x2f. This function is faster than
			the generic determinant function for ArithSqMat */
	inline float determinant(const Mat2x2f& m)
	{
		return m[0][0]*m[1][1]-m[0][1]*m[1][0];
	}

	/** Invert a two by two matrix. ((NOTE: Perhaps this function should be
			changed to return the inverse)). */
	bool invert(const Mat2x2f& m, Mat2x2f&);

}
#endif
