/*
 *
 * RayTrace Software Package, prerelease 0.1.0.   June 2003
 *
 * Mathematics Subpackage (VrMath)
 *
 * Author: Samuel R. Buss
 *
 * Software accompanying the book
 *		3D Computer Graphics: A Mathematical Introduction with OpenGL,
 *		by S. Buss, Cambridge University Press, 2003.
 *
 * Software is "as-is" and carries no warranty.  It may be used without
 *   restriction, but if you modify it, please change the filenames to
 *   prevent confusion between different versions.  Please acknowledge
 *   all use of the software in any publications or products based on it.
 *
 * Bug reports: Sam Buss, sbuss@ucsd.edu.
 * Web page: http://math.ucsd.edu/~sbuss/MathCG
 *
 */

//
// VectorRn:  Vector over Rn  (Variable length vector)
//

#include "VectorRn.h"

VectorRn VectorRn::WorkVector;

double VectorRn::MaxAbs () const
{
	double result = 0.0;
	double* t = x;
	for ( long i = length; i>0; i-- ) {
		if ( (*t) > result ) {
			result = *t;
		}
		else if ( -(*t) > result ) {
			result = -(*t);
		}
		t++;
	}
	return result;
}