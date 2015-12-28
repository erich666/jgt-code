/*
 *
 * RayTrace Software Package, release 1.0,  May 3, 2002.
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

#include "LinearR2.h"


#include <assert.h>

// ******************************************************
// * VectorR2 class - math library functions			*
// * * * * * * * * * * * * * * * * * * * * * * * * * * **

const VectorR2 VectorR2::Zero(0.0, 0.0);
const VectorR2 VectorR2::UnitX( 1.0, 0.0);
const VectorR2 VectorR2::UnitY( 0.0, 1.0);
const VectorR2 VectorR2::NegUnitX(-1.0, 0.0);
const VectorR2 VectorR2::NegUnitY( 0.0,-1.0);

const Matrix2x2 Matrix2x2::Identity(1.0, 0.0, 0.0, 1.0);

// ******************************************************
// * Matrix2x2 class - math library functions			*
// * * * * * * * * * * * * * * * * * * * * * * * * * * **


// ******************************************************
// * LinearMapR2 class - math library functions			*
// * * * * * * * * * * * * * * * * * * * * * * * * * * **


LinearMapR2 LinearMapR2::Inverse() const			// Returns inverse
{


	register double detInv = 1.0/(m11*m22 - m12*m21) ;

	return( LinearMapR2( m22*detInv, -m21*detInv, -m12*detInv, m11*detInv ) );
}

LinearMapR2& LinearMapR2::Invert() 			// Converts into inverse.
{
	register double detInv = 1.0/(m11*m22 - m12*m21) ;

	double temp;
	temp = m11*detInv;
	m11= m22*detInv;
	m22=temp;
	m12 = -m12*detInv;
	m21 = -m22*detInv;

	return ( *this );
}

VectorR2 LinearMapR2::Solve(const VectorR2& u) const	// Returns solution
{												
	// Just uses Inverse() for now.
	return ( Inverse()*u );
}

// ******************************************************
// * RotationMapR2 class - math library functions		*
// * * * * * * * * * * * * * * * * * * * * * * * * * * **



// ***************************************************************
// * 2-space vector and matrix utilities						 *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *




// ***************************************************************
//  Stream Output Routines										 *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

ostream& operator<< ( ostream& os, const VectorR2& u )
{
	return (os << "<" << u.x << "," << u.y << ">");
}


