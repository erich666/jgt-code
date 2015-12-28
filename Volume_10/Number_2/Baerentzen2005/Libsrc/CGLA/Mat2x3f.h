#ifndef __MAT2X3F_H
#define __MAT2X3F_H
// Author: J. Andreas Bærentzen,
// Created: Mon Sep 25 11:55:4

#include "Vec2f.h"
#include "Vec3f.h"
#include "ArithMat.h"

namespace CGLA
{

	/**  2x3 float matrix class.
			 This class is useful for projecting a vector from 3D space to 2D.
	*/
	class Mat2x3f: public ArithMat<Vec2f, Vec3f, Mat2x3f, 2>
	{

	public:
		/// Construct Mat2x3f from two Vec3f vectors (vectors become rows)
		Mat2x3f(const Vec3f& _a, const Vec3f& _b): 
			ArithMat<Vec2f, Vec3f, Mat2x3f, 2> (_a,_b) {}

		/// Construct 0 matrix.
		Mat2x3f() {}

		/// Construct matrix from array of values.
		Mat2x3f(const float* sa): ArithMat<Vec2f, Vec3f, Mat2x3f, 2> (sa) {}
	};

	/**  3x2 float matrix class.
			 This class is useful for going from plane to 3D coordinates.
	*/
	class Mat3x2f: public ArithMat<Vec3f, Vec2f, Mat3x2f, 3>
	{

	public:

		/** Construct matrix from three Vec2f vectors which become the 
				rows of the matrix. */
		Mat3x2f(const Vec2f& _a, const Vec2f& _b, const Vec2f& _c): 
			ArithMat<Vec3f, Vec2f, Mat3x2f, 3> (_a,_b,_c) {}

		/// Construct 0 matrix.
		Mat3x2f() {}

		/// Construct matrix from array of values.
		Mat3x2f(const float* sa): ArithMat<Vec3f, Vec2f, Mat3x2f, 3> (sa) {}

	};


}
#endif
