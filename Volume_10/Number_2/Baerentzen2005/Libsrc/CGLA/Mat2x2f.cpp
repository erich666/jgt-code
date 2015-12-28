// Author: J. Andreas Bærentzen,
// Created: Mon Jun  5 16:58:3

//#include <iostream>
#include "Common/CommonDefs.h"
#include "Mat2x2f.h"

using namespace std;

namespace CGLA {

	bool invert(const Mat2x2f& m, Mat2x2f& m_ret)
	{
		float det = determinant(m);
		if( fabs(det) > CMN::TINY)
			{
				m_ret =  Mat2x2f(m[1][1]/det, -m[0][1]/det,-m[1][0]/det, m[0][0]/det);
				return true;
			}
		return false;
	}

}
