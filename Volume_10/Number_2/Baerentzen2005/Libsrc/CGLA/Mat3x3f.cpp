#include "Mat3x3f.h"

namespace CGLA {

	Mat3x3f invert(const Mat3x3f& _a)
	{
		Mat3x3f a = _a;
		Mat3x3f	b = identity_Mat3x3f();

		int  i, j, i1;

		for (j=0; j<3; j++) 
			{   
				i1 = j;                 // Row with largest pivot candidate
				for (i=j+1; i<3; i++)
					if (fabs(a[i][j]) > fabs(a[i1][j]))
						i1 = i;
			
				// Swap rows i1 and j in a and b to put pivot on diagonal
				Vec3f a_tmp = a[i1];
				a[i1] = a[j];
				a[j]  = a_tmp;
			
				Vec3f b_tmp = b[i1];
				b[i1] = b[j];
				b[j]  = b_tmp;
		
				// Scale row j to have a unit diagonal
				if (a[j][j] == 0.0f)
					throw(Mat3x3fSingular("Tried to invert Singular matrix"));
			
				b[j] /= a[j][j];
				a[j] /= a[j][j];
			
				// Eliminate off-diagonal elems in col j of a, doing identical ops to b
				for (i=0; i<3; i++)
					if (i!=j) 
						{
							b[i] -= a[i][j] * b[j];
							a[i] -= a[i][j] * a[j];
						}
			}
		return b;
	}                                                                               
	Mat3x3f rotation_Mat3x3f(Axis axis, float angle)
	{
		Mat3x3f m;

		switch(axis)
			{
			case XAXIS:
				m[0][0] = 1.0;
				m[1][1] = cos(angle);
				m[1][2] = sin(angle);
				m[2][1] = -sin(angle);
				m[2][2] = cos(angle);
				break;
			case YAXIS:
				m[0][0] = cos(angle);
				m[0][2] = -sin(angle);
				m[2][0] = sin(angle);
				m[2][2] = cos(angle);
				m[1][1] = 1.0;
				break;
			case ZAXIS:
				m[0][0] = cos(angle);
				m[0][1] = sin(angle);
				m[1][0] = -sin(angle);
				m[1][1] = cos(angle);
				m[2][2] = 1.0;
				break;
			}

		return m;
	}

	Mat3x3f scaling_Mat3x3f(const Vec3f& v)
	{
		Mat3x3f m;
		m[0][0] = v[0];
		m[1][1] = v[1];
		m[2][2] = v[2];
		return m;
	}


}
