#include "Common/CommonDefs.h"
#include "Mat4x4f.h"
#include "Mat3x3f.h"

namespace CGLA {

	namespace
	{

		/* Aux func. computes 3x3 determinants. */
		inline float d3x3f( float a1, float a2, float a3, 
												float b1, float b2, float b3, 
												float c1, float c2, float c3 )
		{
			return determinant(Mat3x3f(Vec3f(a1,a2,a3),
																 Vec3f(b1,b2,b3),
																 Vec3f(c1,c2,c3)));
		}
	}

	float determinant(const Mat4x4f& m )
	{
    float ans;
    float a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4, d1, d2, d3, d4;
    
    /* assign to individual variable names to aid selecting */
		/*  correct elements */
		
		a1 = m[0][0]; b1 = m[0][1]; 
		c1 = m[0][2]; d1 = m[0][3];
		
		a2 = m[1][0]; b2 = m[1][1]; 
		c2 = m[1][2]; d2 = m[1][3];
		
		a3 = m[2][0]; b3 = m[2][1]; 
		c3 = m[2][2]; d3 = m[2][3];
		
		a4 = m[3][0]; b4 = m[3][1]; 
		c4 = m[3][2]; d4 = m[3][3];

		ans = 
			a1 * d3x3f( b2, b3, b4, c2, c3, c4, d2, d3, d4)
			- b1 * d3x3f( a2, a3, a4, c2, c3, c4, d2, d3, d4)
			+ c1 * d3x3f( a2, a3, a4, b2, b3, b4, d2, d3, d4)
			- d1 * d3x3f( a2, a3, a4, b2, b3, b4, c2, c3, c4);
		return ans;
	}


	Mat4x4f invert_affine(const Mat4x4f& this_mat) 
	{
		//   The following code has been copied from a gem in Graphics Gems II by
		//   Kevin Wu.                      
		//   The function is very fast, but it can only invert affine matrices. An
		//   exception NotAffine is thrown if the matrix is not affine, and another
		//   exception Singular is thrown if the matrix is singular.

		Mat4x4f new_mat;
		float    det_1;
		float    pos, neg, temp;
      

		if (!(CMN::is_tiny(this_mat[3][0]) && 
					CMN::is_tiny(this_mat[3][1]) && 
					CMN::is_tiny(this_mat[3][2]) && 
					CMN::is_tiny(this_mat[3][3]-1.0f)))
			throw(Mat4x4fNotAffine("Can only invert affine matrices"));
    
#define ACCUMULATE if (temp >= 0.0) pos += temp; else neg += temp
  
		/*
		 * Calculate the determinant of submatrix A and determine if the
		 * the matrix is singular as limited by the float precision
		 * floating-point this_mat representation.
		 */
  
		pos = neg = 0.0;
		temp =  this_mat[0][0] * this_mat[1][1] * this_mat[2][2];
		ACCUMULATE;
		temp =  this_mat[1][0] * this_mat[2][1] * this_mat[0][2];
		ACCUMULATE;
		temp =  this_mat[2][0] * this_mat[0][1] * this_mat[1][2];
		ACCUMULATE;
		temp = -this_mat[2][0] * this_mat[1][1] * this_mat[0][2];
		ACCUMULATE;
		temp = -this_mat[1][0] * this_mat[0][1] * this_mat[2][2];
		ACCUMULATE;
		temp = -this_mat[0][0] * this_mat[2][1] * this_mat[1][2];
		ACCUMULATE;
		det_1 = pos + neg;
  
		/* Is the submatrix A singular? */
		if ((det_1 == 0.0) || (fabs(det_1 / (pos - neg)) < CMN::MINUTE)) 
			{
				/* Mat4x4f M has no inverse */
				throw(Mat4x4fSingular("Tried to invert Singular matrix"));
			}

		else {

			/* Calculate inverse(A) = adj(A) / det(A) */
			det_1 = 1.0 / det_1;
			new_mat[0][0] =   ( this_mat[1][1] * this_mat[2][2] -
													this_mat[2][1] * this_mat[1][2] )
				* det_1;
			new_mat[0][1] = - ( this_mat[0][1] * this_mat[2][2] -
													this_mat[2][1] * this_mat[0][2] )
				* det_1;
			new_mat[0][2] =   ( this_mat[0][1] * this_mat[1][2] -
													this_mat[1][1] * this_mat[0][2] )
				* det_1;
			new_mat[1][0] = - ( this_mat[1][0] * this_mat[2][2] -
													this_mat[2][0] * this_mat[1][2] )
				* det_1;
			new_mat[1][1] =   ( this_mat[0][0] * this_mat[2][2] -
													this_mat[2][0] * this_mat[0][2] )
				* det_1;
			new_mat[1][2] = - ( this_mat[0][0] * this_mat[1][2] -
													this_mat[1][0] * this_mat[0][2] )
				* det_1;
			new_mat[2][0] =   ( this_mat[1][0] * this_mat[2][1] -
													this_mat[2][0] * this_mat[1][1] )
				* det_1;
			new_mat[2][1] = - ( this_mat[0][0] * this_mat[2][1] -
													this_mat[2][0] * this_mat[0][1] )
				* det_1;
			new_mat[2][2] =   ( this_mat[0][0] * this_mat[1][1] -
													this_mat[1][0] * this_mat[0][1] )
				* det_1;

			/* Calculate -C * inverse(A) */
			new_mat[0][3] = - ( this_mat[0][3] * new_mat[0][0] +
													this_mat[1][3] * new_mat[0][1] +
													this_mat[2][3] * new_mat[0][2] );
			new_mat[1][3] = - ( this_mat[0][3] * new_mat[1][0] +
													this_mat[1][3] * new_mat[1][1] +
													this_mat[2][3] * new_mat[1][2] );
			new_mat[2][3] = - ( this_mat[0][3] * new_mat[2][0] +
													this_mat[1][3] * new_mat[2][1] +
													this_mat[2][3] * new_mat[2][2] );
    
			/* Fill in last column */
			new_mat[3][0] = new_mat[3][1] = new_mat[3][2] = 0.0;
			new_mat[3][3] = 1.0;

			return new_mat;
	
		}

#undef ACCUMULATE
	}




	Mat4x4f rotation_Mat4x4f(Axis axis, float angle)
	{
		Mat4x4f m;

		switch(axis)
			{
			case XAXIS:
				m[0][0] = 1.0;
				m[1][1] = cos(angle);
				m[1][2] = sin(angle);
				m[2][1] = -sin(angle);
				m[2][2] = cos(angle);
				m[3][3] = 1.0;
				break;
			case YAXIS:
				m[0][0] = cos(angle);
				m[0][2] = -sin(angle);
				m[2][0] = sin(angle);
				m[2][2] = cos(angle);
				m[1][1] = 1.0;
				m[3][3] = 1.0;
				break;
			case ZAXIS:
				m[0][0] = cos(angle);
				m[0][1] = sin(angle);
				m[1][0] = -sin(angle);
				m[1][1] = cos(angle);
				m[2][2] = 1.0;
				m[3][3] = 1.0;
				break;
			}

		return m;
	}

	Mat4x4f translation_Mat4x4f(const Vec3f& v)
	{
		Mat4x4f m;

		m[0][0] = 1.0;
		m[1][1] = 1.0;
		m[2][2] = 1.0;
		m[3][3] = 1.0;
  
		m[0][3] = v[0];
		m[1][3] = v[1];
		m[2][3] = v[2];
  
		return m;
	}

	Mat4x4f scaling_Mat4x4f(const Vec3f& v)
	{
		Mat4x4f m;

		m[0][0] = v[0];
		m[1][1] = v[1];
		m[2][2] = v[2];
		m[3][3] = 1.0;
   
		return m;
	}

	Mat4x4f perspective_Mat4x4f(float d)
	{
		Mat4x4f m;
  
		/* Eye at the origin, looking down the negative z axis */

		m[0][0] = 1.0;
		m[1][1] = 1.0;
		m[2][2] = 1.0;
		m[3][2] = -1.0/d;
   
		return m;
	}


	Mat4x4f adjoint(const Mat4x4f& in)
	{
		float a1, a2, a3, a4, b1, b2, b3, b4;
		float c1, c2, c3, c4, d1, d2, d3, d4;

		/* assign to individual variable names to aid  */
		/* selecting correct values  */
	
		a1 = in[0][0]; b1 = in[0][1]; 
		c1 = in[0][2]; d1 = in[0][3];

		a2 = in[1][0]; b2 = in[1][1]; 
		c2 = in[1][2]; d2 = in[1][3];

		a3 = in[2][0]; b3 = in[2][1];
		c3 = in[2][2]; d3 = in[2][3];

		a4 = in[3][0]; b4 = in[3][1]; 
		c4 = in[3][2]; d4 = in[3][3];


		/* row column labeling reversed since we transpose rows & columns */
	
		Mat4x4f out;
		out[0][0]  =   d3x3f( b2, b3, b4, c2, c3, c4, d2, d3, d4);
		out[1][0]  = - d3x3f( a2, a3, a4, c2, c3, c4, d2, d3, d4);
		out[2][0]  =   d3x3f( a2, a3, a4, b2, b3, b4, d2, d3, d4);
		out[3][0]  = - d3x3f( a2, a3, a4, b2, b3, b4, c2, c3, c4);
	
		out[0][1]  = - d3x3f( b1, b3, b4, c1, c3, c4, d1, d3, d4);
		out[1][1]  =   d3x3f( a1, a3, a4, c1, c3, c4, d1, d3, d4);
		out[2][1]  = - d3x3f( a1, a3, a4, b1, b3, b4, d1, d3, d4);
		out[3][1]  =   d3x3f( a1, a3, a4, b1, b3, b4, c1, c3, c4);
        
		out[0][2]  =   d3x3f( b1, b2, b4, c1, c2, c4, d1, d2, d4);
		out[1][2]  = - d3x3f( a1, a2, a4, c1, c2, c4, d1, d2, d4);
		out[2][2]  =   d3x3f( a1, a2, a4, b1, b2, b4, d1, d2, d4);
		out[3][2]  = - d3x3f( a1, a2, a4, b1, b2, b4, c1, c2, c4);
	
		out[0][3]  = - d3x3f( b1, b2, b3, c1, c2, c3, d1, d2, d3);
		out[1][3]  =   d3x3f( a1, a2, a3, c1, c2, c3, d1, d2, d3);
		out[2][3]  = - d3x3f( a1, a2, a3, b1, b2, b3, d1, d2, d3);
		out[3][3]  =   d3x3f( a1, a2, a3, b1, b2, b3, c1, c2, c3);

		return out;
	}


	Mat4x4f invert( const Mat4x4f& in)
	{
		float det = determinant( in );
		if (CMN::is_tiny(det)) 
			throw(Mat4x4fSingular("Tried to invert Singular matrix"));
		Mat4x4f out = adjoint(in);
		out/=det;
		return out;
	}


}
