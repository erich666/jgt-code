#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI    3.14159265358979323846
#define M_PI_2  1.57079632679489661923
#endif

// ========= OLD FUNCTIONS (avoid using) =========

extern void error_exit( const char *errloc, char *format, ... );
    // outputs an error message to the stderr and exits program.

extern void copy4double( const double src[4], double dest[4] );

extern void copy3double( const double src[3], double dest[3] );

extern void copy2double( const double src[2], double dest[2] );

extern void copy4float( const float src[4], float dest[4] );

extern void copy3float( const float src[3], float dest[3] );

extern void copy2float( const float src[2], float dest[2] );

extern double random01( void );
    // returns a random value in the range 0.0 to 1.0.

extern double random_in_range( double min, double max );
    // returns a random value in the range min to max.



// ========= NEW FUNCTIONS =========

extern void error_exit( const char *srcfile, int lineNum, char *format, ... );
    // outputs an error message to the stderr and exits program.

extern void show_warning( const char *srcfile, int lineNum, char *format, ... );
    // outputs a warning message to the stderr.


extern double get_curr_real_time( void );
	// returns time in seconds (plus fraction of a second) since midnight (00:00:00), 
	// January 1, 1970, coordinated universal time (UTC).
	// Up to millisecond precision.

extern double get_curr_cpu_time( void );
	// returns cpu time in seconds (plus fraction of a second) since the 
    // start of the current process.

extern void delay_real_time( int milliseconds );
	// pause process for the number of specified milliseconds.


inline void *checked_malloc( size_t size )
	// same as malloc(), but checks for out-of-memory.
{
	void *p = malloc( size );
	if ( p == NULL ) error_exit( __FILE__, __LINE__, "Cannot allocate memory" );
	return p;
}


inline double fsqr( double f )
    // returns the square of f.
{
    return ( f * f );
}


template <typename Type1, typename Type2>
inline void copyArray( Type1 dest[], const Type2 src[], size_t size )
{
	for ( size_t i = 0; i < size; i++ ) dest[i] = src[i];
}


template <typename Type1, typename Type2>
inline void copyArray4( Type1 dest[4], const Type2 src[4] )
{
	dest[0] = src[0];
	dest[1] = src[1];
	dest[2] = src[2];
	dest[3] = src[3];
}


template <typename Type1, typename Type2>
inline void copyArray3( Type1 dest[3], const Type2 src[3] )
{
	dest[0] = src[0];
	dest[1] = src[1];
	dest[2] = src[2];
}


template <typename Type1, typename Type2>
inline void copyArray2( Type1 dest[2], const Type2 src[2] )
{
	dest[0] = src[0];
	dest[1] = src[1];
}


template <typename Type>
inline Type min2( Type a, Type b )
{
	if ( a < b ) return a; else return b;
}


template <typename Type>
inline Type max2( Type a, Type b )
{
	if ( a > b ) return a; else return b;
}


template <typename Type>
inline Type min3( Type a, Type b, Type c )
{
    Type t = a;
    if ( b < t ) t = b;
    if ( c < t ) t = c;
    return t;
}


template <typename Type>
inline Type max3( Type a, Type b, Type c )
{
    Type t = a;
    if ( b > t ) t = b;
    if ( c > t ) t = c;
    return t;
}


inline double uniformRandom( void )
    // returns a random value in the range [0, 1] from a uniform distribution.
{
    return ((double)rand()) / RAND_MAX;
}


inline double uniformRandom( double min, double max )
    // returns a random value in the range [min, max] from a uniform distribution.
{
    return ( ((double)rand()) / RAND_MAX ) * (max - min) + min;
}


inline double normalRandom( void )
	// return a random number from a normal distribution with mean=0 and s.d.=1.
{
	double R1 = ((double)(rand() + 1)) / (RAND_MAX + 1);
	double R2 = ((double)(rand() + 1)) / (RAND_MAX + 1);

	return sqrt( -2 * log( R1 ) ) * cos( 2 * M_PI * R2 );
}


inline double normalRandom( double mean, double stddev )
	// return a random number from a normal distribution with mean and stddev.
{
	double R1 = ((double)(rand() + 1)) / (RAND_MAX + 1);
	double R2 = ((double)(rand() + 1)) / (RAND_MAX + 1);

	return normalRandom() * stddev + mean;
}


#endif
