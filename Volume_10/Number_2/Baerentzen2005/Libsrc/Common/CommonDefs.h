#ifndef __MATHDEFAULT_H
#define __MATHDEFAULT_H

#ifndef OLD_C_HEADERS
#include <cmath>
#include <climits>
#else
#include <math.h>
#include <limits.h>
#endif

#include <algorithm>
#include "Common/ExceptionStandard.h"
#include "CGLA/CGLA.h"

#if defined(_MSC_VER)
#define M_E		2.7182818284590452354
#define M_LOG2E		1.4426950408889634074
#define M_LOG10E	0.43429448190325182765
#define M_LN2		0.69314718055994530942
#define M_LN10		2.30258509299404568402
#define M_PI		3.14159265358979323846
#define M_PI_2		1.57079632679489661923
#define M_PI_4		0.78539816339744830962
#define M_1_PI		0.31830988618379067154
#define M_2_PI		0.63661977236758134308
#define M_2_SQRTPI	1.12837916709551257390
#define M_SQRT2		1.41421356237309504880
#define M_SQRT1_2	0.70710678118654752440
#endif

#if defined(_MSC_VER) && _MSC_VER < 1300
#undef min
#undef max
template <class _Tp>
inline const _Tp& min(const _Tp& __a, const _Tp& __b)
{
  return __b < __a ? __b : __a;
}
template <class _Tp>
inline const _Tp& max(const _Tp& __a, const _Tp& __b) 
{
  return  __a < __b ? __b : __a;
}
#endif

namespace Common
{

	/** Numerical constant representing something large.
			value is a bit arbitrary */
	const float BIG=10e+30f;

	/** Numerical constant represents something extremely small.
			value is a bit arbitrary */
	const float MINUTE=10e-30f;

	/** Numerical constant represents something very small.
			value is a bit arbitrary */
	const float TINY=3e-7f;
	
	/** Numerical constant represents something small.
			value is a bit arbitrary */
	const float SMALL=10e-2f;

	const float SQRT3=sqrt(3.0f);
	
/** We derive a `math' exception from mother. To be used for the 
    derivation of specialized math exceptions */
DERIVEEXCEPTION(MathException, MotherException);

///Template for a function that squares the argument.
template <class T>
inline T sqr(T x) {///
return x*x;}

/// Template for a function that returns the cube of the argument.
template <class T>
inline T qbe(T x) {///
return x*x*x;}

/** What power of 2 ?. if x is the argument, find the largest 
    y so that 2^y <= x */
inline int two_to_what_power(int x) 
{
  if (x<1) 
    return -1;
  int i = 0;
  while (x != 1) {x>>=1;i++;}
  return i;
}

#ifdef __sgi
inline int round(float x) {return int(rint(x));}
#else
inline int round(float x) {return int(x+0.5);}
#endif

template<class T>
inline T sign(T x) {return x>=T(0) ? 1 : -1;}

inline bool is_zero(float x)	{return (x > -MINUTE && x < MINUTE);}
inline bool is_tiny(float x)	{return (x > -TINY && x < TINY);}


template<class T>
inline T int_pow(T x, int k) 
{
	T y = static_cast<T>(1);
	for(int i=0;i<k;++i)
		y *= x;
	return y;
}


}
namespace CMN = Common;

#endif
