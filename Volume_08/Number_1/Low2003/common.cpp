#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <sys/types.h>
#include <sys/timeb.h>
#include <time.h>
#include "common.h"


void error_exit( const char *errloc, char *format, ... )
    // outputs an error message to the stderr and exits program.
{
	va_list args;
	static char buffer[1024];
	va_start( args, format );
	vsprintf( buffer, format, args );
	va_end( args );

    fprintf( stderr, "\n%s - %s.\n\n", errloc, buffer );
    exit( 1 );
}


void error_exit( const char *srcfile, int lineNum, char *format, ... )
    // outputs an error message to the stderr and exits program.
{
	va_list args;
	static char buffer[1024];
	va_start( args, format );
	vsprintf( buffer, format, args );
	va_end( args );

    fprintf( stderr, "\nABORT at \"%s\", line %d\nERROR: %s.\n\n", srcfile, lineNum, buffer );
    exit( 1 );
}


void show_warning( const char *srcfile, int lineNum, char *format, ... )
    // outputs a warning message to the stderr.
{
	va_list args;
	static char buffer[1024];
	va_start( args, format );
	vsprintf( buffer, format, args );
	va_end( args );

    fprintf( stderr, "\nWARNING at \"%s\", line %d\nISSUE: %s.\n\n", srcfile, lineNum, buffer );
}



void copy4double( const double src[4], double dest[4] )
{
	dest[0] = src[0];
	dest[1] = src[1];
	dest[2] = src[2];
	dest[3] = src[3];
}


void copy3double( const double src[3], double dest[3] )
{
	dest[0] = src[0];
	dest[1] = src[1];
	dest[2] = src[2];
}


void copy2double( const double src[2], double dest[2] )
{
	dest[0] = src[0];
	dest[1] = src[1];
}


void copy4float( const float src[4], float dest[4] )
{
	dest[0] = src[0];
	dest[1] = src[1];
	dest[2] = src[2];
	dest[3] = src[3];
}


void copy3float( const float src[3], float dest[3] )
{
	dest[0] = src[0];
	dest[1] = src[1];
	dest[2] = src[2];
}


void copy2float( const float src[2], float dest[2] )
{
	dest[0] = src[0];
	dest[1] = src[1];
}


double random01( void )
    // returns a random value in the range 0.0 to 1.0.
{
    return ( (double) rand() ) / RAND_MAX;
}


double random_in_range( double min, double max )
    // returns a random value in the range min to max.
{
    return random01() * (max - min) + min;
}


double get_curr_real_time( void )
	// returns time in seconds (plus fraction of a second) since midnight (00:00:00), 
	// January 1, 1970, coordinated universal time (UTC).
	// Up to millisecond precision.
{
#ifdef _WIN32

	struct _timeb timebuffer;
	_ftime( &timebuffer );

#else

	struct timeb timebuffer;
	ftime( &timebuffer );

#endif

	return timebuffer.time + timebuffer.millitm / 1000.0;
}


double get_curr_cpu_time( void )
	// returns cpu time in seconds (plus fraction of a second) since the 
    // start of the current process.
{
	return ((double) clock() ) / CLOCKS_PER_SEC;
}


void delay_real_time( int milliseconds )
	// pause process for the number of specified milliseconds.
{
	double stopTime = get_curr_real_time() + ( milliseconds / 1000.0 );

	while ( get_curr_real_time() < stopTime ) 
	{
		// do nothing
	}
}