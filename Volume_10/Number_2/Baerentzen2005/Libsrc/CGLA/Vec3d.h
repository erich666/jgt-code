#ifndef __Vec3d_H
#define __Vec3d_H

#include "ArithVec.h"
#include "Vec3i.h"
#include "Vec3f.h"


namespace CGLA {

	/** A 3D double vector. Useful for high precision arithmetic. */

	class Vec3d: public ArithVec<double,Vec3d,3>
	{
	public:

		/// Construct 0 vector
		Vec3d(){}

		/// Construct vector
		Vec3d(double a, double b, double c): ArithVec<double,Vec3d,3>(a,b,c) {}

		/// Construct vector where all coords = a 
		explicit Vec3d(double a): ArithVec<double,Vec3d,3>(a,a,a) {}

		/// Convert from int vector
		explicit Vec3d(const Vec3i& v): ArithVec<double,Vec3d,3>(v[0],v[1],v[2]) {}

		/// Convert from float vector
		explicit Vec3d(const Vec3f& v): ArithVec<double,Vec3d,3>(v[0],v[1],v[2]) {}
  
		/// Returns euclidean length
		double length() const 
    {
      return sqrt(data[0]*data[0]+
                  data[1]*data[1]+
                  data[2]*data[2]);
    }
    
		
		/// Normalize vector.
		void normalize() 
    {
      (*this) /= length();
    }
	  
		/** Get the vector in spherical coordinates.
				The first argument (theta) is inclination from the vertical axis.
				The second argument (phi) is the angle of rotation about the vertical 
				axis. The third argument (r) is the length of the vector. */
		void get_spherical( double&, double&, double& ) const;

		/** Assign the vector in spherical coordinates.
				The first argument (theta) is inclination from the vertical axis.
				The second argument (phi) is the angle of rotation about the vertical 
				axis. The third argument (r) is the length of the vector. */
		bool set_spherical( double, double, double);

	};

	/// Compute dot product
	inline double dot( const Vec3d& x, const Vec3d& y ) 
	{
		return x[0]*y[0] + x[1]*y[1] + x[2]*y[2];
	}


	/// Compute cross product
	inline Vec3d cross( const Vec3d& x, const Vec3d& y ) 
	{
		return Vec3d( x[1] * y[2] - x[2] * y[1], 
									x[2] * y[0] - x[0] * y[2], 
									x[0] * y[1] - x[1] * y[0] );
	}


}
#endif
