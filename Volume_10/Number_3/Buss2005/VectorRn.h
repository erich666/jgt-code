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
//    Not very sophisticated yet.  Needs more functionality
//		To do: better handling of resizing.
//

#ifndef VECTOR_RN_H
#define VECTOR_RN_H

#include <math.h>
#include <assert.h>
#include "LinearR3.h"

class VectorRn {
	friend class MatrixRmn;

public:
	VectorRn();					// Null constructor
	VectorRn( long length );	// Constructor with length
	~VectorRn();				// Destructor

	void SetLength( long newLength );
	long GetLength() const { return length; } 

	void SetZero(); 
	void Fill( double d );
	void Load( const double* d );
	void LoadScaled( const double* d, double scaleFactor );
	void Set ( const VectorRn& src );

	// Two access methods identical in functionality
	// Subscripts are ZERO-BASED!!
	const double& operator[]( long i ) const { assert ( 0<=i && i<length );	return *(x+i); }
	double& operator[]( long i ) { assert ( 0<=i && i<length );	return *(x+i); }
	double Get( long i ) const { assert ( 0<=i && i<length );	return *(x+i); }

	// Use GetPtr to get pointer into the array (efficient)
	// Is friendly in that anyone can change the array contents (be careful!)
	const double* GetPtr( long i ) const { assert(0<=i && i<length);  return (x+i); }
	double* GetPtr( long i ) { assert(0<=i && i<length);  return (x+i); }
	const double* GetPtr() const { return x; }
	double* GetPtr() { return x; }

	void Set( long i, double val ) { assert(0<=i && i<length), *(x+i) = val; }
	void SetTriple( long i, const VectorR3& u );

	VectorRn& operator+=( const VectorRn& src );
	VectorRn& operator-=( const VectorRn& src );
	void AddScaled (const VectorRn& src, double scaleFactor );

	VectorRn& operator*=( double f );
	double NormSq() const;
	double Norm() const { return sqrt(NormSq()); }

	double MaxAbs() const;
    
private:
	long length;				// Logical or actual length 
	long AllocLength;			// Allocated length
	double *x;					// Array of vector entries

	static VectorRn WorkVector;								// Serves as a temporary vector
	static VectorRn& GetWorkVector() { return WorkVector; }
	static VectorRn& GetWorkVector( long len ) { WorkVector.SetLength(len); return WorkVector; }
};

inline VectorRn::VectorRn() 
{
	length = 0;
	AllocLength = 0;
	x = 0;
}

inline VectorRn::VectorRn( long initLength ) 
{
	length = 0;
	AllocLength = 0;
	x = 0;
	SetLength( initLength );
}

inline VectorRn::~VectorRn() 
{
	delete x;
}

// Resize.  
// If the array is shortened, the information about the allocated length is lost.
inline void VectorRn::SetLength( long newLength )
{
	assert ( newLength>0 );
	if ( newLength>AllocLength ) {
		delete x;
		AllocLength = Max( newLength, AllocLength<<1 );
		x = new double[AllocLength];
	}
	length = newLength;
} 

// Zero out the entire vector
inline void VectorRn::SetZero()
{
	double* target = x;
	for ( long i=length; i>0; i-- ) {
		*(target++) = 0.0;
	}
}

// Set the value of the i-th triple of entries in the vector
inline void VectorRn::SetTriple( long i, const VectorR3& u ) 
{
	long j = 3*i;
	assert ( 0<=j && j+2 < length );
	u.Dump( x+j );
}

inline void VectorRn::Fill( double d ) 
{
	double *to = x;
	for ( long i=length; i>0; i-- ) {
		*(to++) = d;
	}
}

inline void VectorRn::Load( const double* d )
{
	double *to = x;
	for ( long i=length; i>0; i-- ) {
		*(to++) = *(d++);
	}
}

inline void VectorRn::LoadScaled( const double* d, double scaleFactor )
{
	double *to = x;
	for ( long i=length; i>0; i-- ) {
		*(to++) = (*(d++))*scaleFactor;
	}
}

inline void VectorRn::Set( const VectorRn& src )
{
	assert ( src.length == this->length );
	double* to = x;
	double* from = src.x;
	for ( long i=length; i>0; i-- ) {
		*(to++) = *(from++);
	}
}

inline VectorRn& VectorRn::operator+=( const VectorRn& src )
{
	assert ( src.length == this->length );
	double* to = x;
	double* from = src.x;
	for ( long i=length; i>0; i-- ) {
		*(to++) += *(from++);
	}
	return *this;
}

inline VectorRn& VectorRn::operator-=( const VectorRn& src )
{
	assert ( src.length == this->length );
	double* to = x;
	double* from = src.x;
	for ( long i=length; i>0; i-- ) {
		*(to++) -= *(from++);
	}
	return *this;
}

inline void VectorRn::AddScaled (const VectorRn& src, double scaleFactor )
{
	assert ( src.length == this->length );
	double* to = x;
	double* from = src.x;
	for ( long i=length; i>0; i-- ) {
		*(to++) += (*(from++))*scaleFactor;
	}
}

inline VectorRn& VectorRn::operator*=( double f ) 
{
	double* target = x;
	for ( long i=length; i>0; i-- ) {
		*(target++) *= f;
	}
	return *this;
}

inline double VectorRn::NormSq() const 
{
	double* target = x;
	double res = 0.0;
	for ( long i=length; i>0; i-- ) {
		res += (*target)*(*target);
		target++;
	}
	return res;
}

inline double Dot( const VectorRn& u, const VectorRn& v ) 
{
	assert ( u.GetLength() == v.GetLength() );
	double res = 0.0;
	const double* p = u.GetPtr();
	const double* q = v.GetPtr();
	for ( long i = u.GetLength(); i>0; i-- ) {
		res += (*(p++))*(*(q++));
	}
	return res;
}

#endif VECTOR_RN_H