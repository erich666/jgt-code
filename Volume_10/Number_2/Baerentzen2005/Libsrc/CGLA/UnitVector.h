#ifndef __UNITVECTOR_H
#define __UNITVECTOR_H

#ifndef OLD_C_HEADERS
#include <cmath>
#else
#include <math.h>
#endif
#include "CGLA.h"
#include "Vec3f.h"
#include "TableTrigonometry.h"

namespace CGLA
{
	namespace TT=TableTrigonometry;

	/** The UnitVector stores a unit length vector as two angles.

	A vector stored as two (fix point) angles is much smaller than
	vector stored in the usual way. On a 32 bit architecture this
	class should take up four bytes. not too bad. */
	class UnitVector
	{
		TT::Angle theta, phi;

		void encode(const Vec3f& v)
		{
#ifndef OLD_C_HEADERS
			theta = TT::t_atan(std::sqrt(CMN::sqr(v[0])+CMN::sqr(v[1])), v[2]);
#else
			theta = TT::t_atan(sqrt(CMN::sqr(v[0])+CMN::sqr(v[1])), v[2]);
#endif
			phi   = TT::t_atan(v[0],v[1]);
		}
	
	public:

		/// Construct unitvector from normal vector
		explicit UnitVector(const Vec3f& v) {encode(v);}

		/// Construct default unit vector
		explicit UnitVector(): theta(0), phi(0) {}

		/// Get theta angle
		float t() const {return TT::angle2float(theta);}

		/// Get phi angle
		float f() const {return TT::angle2float(phi);}

		/// Reconstruct Vec3f from unit vector
		operator Vec3f() const
		{
			float costf = TT::t_cos(theta);
			return Vec3f(TT::t_cos(phi)*costf , 
									 TT::t_sin(phi)*costf, 
									 TT::t_sin(theta));
		}

		/// Test for equality.
		bool operator==(const UnitVector& u) const
		{
			return theta == u.theta && phi == u.phi;
		}
	};


	/// Inline output operator.
	inline std::ostream& operator<<(std::ostream& os, const UnitVector& u)
	{
		os << "<v=" << u.t() << " h=" << u.f() << ">";
		return os;
	}

}
#endif
