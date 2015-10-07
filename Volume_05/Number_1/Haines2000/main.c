/* ========== test harness for shaft.c ========= */
/*
 * Eric Haines, erich@acm.org, rev 1.1, 2/3/2000
 *
 * See shaft.c for how to call the shaft code.
 */

#include <stdio.h>
#include <stdlib.h>
#include "shaft.h"


void dumpBox( char *str, box *b )
{
	printf( "%s\n", str ) ;
	printf( "lo: %g   %g   %g\n",b->c[LO_X],b->c[LO_Y],b->c[LO_Z] ) ;
	printf( "hi: %g   %g   %g\n",b->c[HI_X],b->c[HI_Y],b->c[HI_Z] ) ;
}

void dumpPlaneSet( char *str, planeRec *pr )
{
	printf( "%s\n", str ) ;
	printf( "a: %g,  b: %g,  c: %g,  d: %g\n", pr->a, pr->b, pr->c, pr->d ) ;
#ifndef INSIDE_ONLY
	printf( "nearCorner: %d %d %d\n", pr->nearCorner[X], pr->nearCorner[Y], pr->nearCorner[Z] ) ;
#endif
#ifndef OUTSIDE_ONLY
	printf( "farCorner:  %d %d %d\n", pr->farCorner[X], pr->farCorner[Y], pr->farCorner[Z] ) ;
#endif
}

void dumpShaft( char *str, shaft *s )
{
	planeRec *pr ;
	char plane_no[256] ;
	int i = 0 ;

	printf( "\n%s\n", str ) ;
	dumpBox( "box", &s->bx ) ;
	pr = s->planeSet ;
	while ( pr ) {
		sprintf( plane_no, "plane %d: ", i ) ;
		dumpPlaneSet( plane_no, pr ) ;
		i++ ;
		pr = pr->next ;
	}
}

void convertIndex( int indx, float *lo, float *hi )
{
	switch (indx) {
	case 0:
		*lo = -1.0f ;
		*hi = -0.5f ;
		break;
	case 1:
		*lo = -1.0f ;
		*hi = 1.5f ;
		break;
	case 2:
		*lo = 0.5f ;
		*hi = 1.5f ;
		break ;
	case 3:
		*lo = 0.25f ;
		*hi = 0.75f ;
		break ;
	case 4:
		*lo = 0.0f ;
		*hi = 1.5f ;
		break ;
	case 5:
		*lo = -0.5f ;
		*hi = 1.0f ;
		break ;
	case 6:
		*lo = 0.0f ;
		*hi = 0.75f ;
		break ;
	case 7:
		*lo = 0.25f ;
		*hi = 1.0f ;
		break ;
	}
}

/* samples the cubic volume defined by lo and hi (in X,Y,Z), showing the
 * intersection conditions in 2D (the third dimension results are OR'ed together).
 */
void makeGraph( float lo, float hi, int i1n, int i2n, box *box1, box *box2, shaft *s )
{
#define res 20
#ifdef INCLUDE_SPHERE_TESTING
	sphere spht ;
	int sphOut, sphIn ;
#endif
	int grid[res+1][res+1] ;
	int xg, yg, zg, yng, i3n ;
	box boxt ;
	/* give the boxsize a positive value to make the test boxes have a volume;
	 * this tests the overlap condition (neither fully in nor fully out).
	 */
	float boxsize = 0.0f ;	/* box "radius" */
	float offset = 0.001f ;	/* avoid being exactly on a value by offsetting */

	printf( "%s\n",(i2n==X)?"X":((i2n==Y)?"Y":"Z") ) ;

	if ( (i1n+1)%3 == i2n ) {
		i3n = (i2n+1)%3 ;
	} else {
		i3n = (i1n+1)%3 ;
	}

	for ( xg = 0 ; xg <= res ; xg++ ) { 
		for ( yg = 0 ; yg <= res ; yg++ ) {
			grid[xg][yg] = 0 ;
			for ( zg = 0 ; zg <= res ; zg++ ) {
				boxt.c[i1n] = xg * (hi-lo)/res + lo + offset - boxsize ;
				boxt.c[i2n] = yg * (hi-lo)/res + lo + offset - boxsize ;
				boxt.c[i3n] = zg * (hi-lo)/res + lo + offset - boxsize ;
				boxt.c[i1n+3] = xg * (hi-lo)/res + lo + offset + boxsize ;
				boxt.c[i2n+3] = yg * (hi-lo)/res + lo + offset + boxsize ;
				boxt.c[i3n+3] = zg * (hi-lo)/res + lo + offset + boxsize ;

#ifdef INCLUDE_SPHERE_TESTING
				if ( boxsize == 0.0f ) {
					spht.center[i1n] = boxt.c[i1n] ;
					spht.center[i2n] = boxt.c[i2n] ;
					spht.center[i3n] = boxt.c[i3n] ;
					spht.radius = 0.0f ;
					sphOut = sphereOutside( &spht, s ) ;
					sphIn  = sphereInside ( &spht, s ) ;
				}
#endif

				if ( !boxOutside( &boxt, s ) ) {
#ifdef INCLUDE_SPHERE_TESTING
					if ( boxsize == 0.0f && sphOut )
						printf("ERROR! box not out, sphere out\n" ) ;
#endif
					if ( boxInside( &boxt, s ) ) {
						grid[xg][yg] |= 0x04 ;
#ifdef INCLUDE_SPHERE_TESTING
						if ( boxsize == 0.0f && !sphIn )
							printf("ERROR! box in, sphere not in\n" ) ;
#endif
					} else {
						grid[xg][yg] |= 0x02 ;
#ifdef INCLUDE_SPHERE_TESTING
						if ( boxsize == 0.0f && sphIn )
							printf("ERROR! box not in, sphere in\n" ) ;
#endif
					}
				} else {
#ifdef INCLUDE_SPHERE_TESTING
					if ( boxsize == 0.0f && !sphOut )
						printf("ERROR! box out, sphere not out\n" ) ;
#endif
					grid[xg][yg] |= 0x01 ;
				}

				/* box/box test (actually, point/box test;
				 * we use this test just to mark the existence
				 * of the two shaft-forming boxes)
				 */
				if ( boxt.c[LO_X] >= box1->c[LO_X] &&
					 boxt.c[LO_Y] >= box1->c[LO_Y] &&
					 boxt.c[LO_Z] >= box1->c[LO_Z] &&
					 boxt.c[LO_X] <= box1->c[HI_X] &&
					 boxt.c[LO_Y] <= box1->c[HI_Y] &&
					 boxt.c[LO_Z] <= box1->c[HI_Z] ) {
						grid[xg][yg] |= 0x08 ;
				}
				if ( boxt.c[LO_X] >= box2->c[LO_X] &&
					 boxt.c[LO_Y] >= box2->c[LO_Y] &&
					 boxt.c[LO_Z] >= box2->c[LO_Z] &&
					 boxt.c[LO_X] <= box2->c[HI_X] &&
					 boxt.c[LO_Y] <= box2->c[HI_Y] &&
					 boxt.c[LO_Z] <= box2->c[HI_Z] ) {
						grid[xg][yg] |= 0x10 ;
				}
			}
		}
	}
	
	for ( yg = 0 ; yg <= res ; yg++ ) {
		yng = res - yg ;
		printf ("%4.1f: ",(float)(yng * (hi-lo)/res + lo) ) ;
		for ( xg = 0 ; xg <= res ; xg++ ) {
			/* check for error */
			if ( boxsize == 0.0f && (grid[xg][yng] & 0x18) && !(grid[xg][yng] & 0x06) ) {
				/* error! inside a box, but not inside the shaft */
				printf( "ERROR! inside a box, but not the shaft\n" ) ;
			}

			if ( (grid[xg][yng] & 0x18) == 0x18 ) {
				printf( "3 " ) ;	/* inside box 1 & 2 */
			} else if ( grid[xg][yng] & 0x10 ) {
				printf( "2 " ) ;
			} else if ( grid[xg][yng] & 0x08 ) {
				printf( "1 " ) ;
			} else if ( grid[xg][yng] & 0x04 ) {
				printf( "* " ) ;	/* inside shaft */
			} else if ( grid[xg][yng] & 0x02 ) {
				printf( "O " ) ;	/* overlaps */
			} else {
				printf( ". " ) ;
			}
		}
		printf( "\n" ) ;
	}

	printf("    ");
	for ( xg = 0 ; xg <= res ; xg+=4 ) {
		printf ("%4.1f    ",(float)(xg * (hi-lo)/res + lo) ) ;
	}

	printf( "\n                     %s\n\n",(i1n==X)?"X":((i1n==Y)?"Y":"Z") ) ;
}

/* similar to makeGraph, but no output */
void testPoints( float lo, float hi, int i1n, int i2n, box *box1, box *box2, shaft *s )
{
#define res 20
	int grid[res+1][res+1] ;
	int xg, yg, zg, i3n ;
	box boxt ;
	/* give the boxsize a positive value to make the test boxes have a volume;
	 * this tests the overlap condition (neither fully in nor fully out).
	 */
	float boxsize = 0.125f ;	/* box "radius" */
	float offset = 0.001f ;	/* avoid being exactly on a value by offsetting */

	if ( (i1n+1)%3 == i2n ) {
		i3n = (i2n+1)%3 ;
	} else {
		i3n = (i1n+1)%3 ;
	}

	for ( xg = 0 ; xg <= res ; xg++ ) { 
		for ( yg = 0 ; yg <= res ; yg++ ) {
			grid[xg][yg] = 0 ;
			for ( zg = 0 ; zg <= res ; zg++ ) {
				boxt.c[i1n] = xg * (hi-lo)/res + lo + offset - boxsize ;
				boxt.c[i2n] = yg * (hi-lo)/res + lo + offset - boxsize ;
				boxt.c[i3n] = zg * (hi-lo)/res + lo + offset - boxsize ;
				boxt.c[i1n+3] = xg * (hi-lo)/res + lo + offset + boxsize ;
				boxt.c[i2n+3] = yg * (hi-lo)/res + lo + offset + boxsize ;
				boxt.c[i3n+3] = zg * (hi-lo)/res + lo + offset + boxsize ;


				if ( !boxOutside( &boxt, s ) ) {
					if ( boxInside( &boxt, s ) ) {
						grid[xg][yg] |= 0x04 ;
					} else {
						grid[xg][yg] |= 0x02 ;
					}
				} else {
					grid[xg][yg] |= 0x01 ;
				}

			}
		}
	}
}

/* change these to whatever random number seed and generator
 * is available on your system. */
#define myseedrand(x)	srand(x)
/* myrand() returns a float in the range [0..1) */
#define myrand()	((double)rand()/(double)0x7fff)

/* test a number of shaft configurations and gather statistics */
int main( int argc, char *argv[] )
{
	box box1, box2, box3 ;
	shaft *s ;
	int i,v ;
	int x1, y1, z1 ;
	float temp ;
	float min[3], max[3], len[3] ;

	/* test a simple box/box combination */
	box1.c[LO_X] = 0.0f ;
	box1.c[LO_Y] = 0.0f ;
	box1.c[LO_Z] = 0.0f ;
	box1.c[HI_X] = 1.0f ;
	box1.c[HI_Y] = 1.0f ;
	box1.c[HI_Z] = 1.0f ;

	box2.c[LO_X] = -1.0f ;
	box2.c[LO_Y] = -1.0f ;
	box2.c[LO_Z] = 0.5f ;
	box2.c[HI_X] = -0.5f ;
	box2.c[HI_Y] = 1.5f ;
	box2.c[HI_Z] = 1.5f ;

	s = formShaft( &box1, &box2 ) ;

	dumpShaft( "test shaft:", s ) ;

	box3.c[LO_X] = 0.5f ;
	box3.c[LO_Y] = 0.5f ;
	box3.c[LO_Z] = 0.5f ;
	box3.c[HI_X] = 3.5f ;
	box3.c[HI_Y] = 3.5f ;
	box3.c[HI_Z] = 3.5f ;

	dumpBox( "\ntest box", &box3 ) ;

	printf( "test box is %sfully outside shaft\n",
		boxOutside(&box3,s)?"":"not " ) ;
	printf( "test box is %sfully inside shaft\n",
		boxInside(&box3,s)?"":"not " ) ;

	makeGraph( -1.5f, 2.0f, 0, 1, &box1, &box2, s ) ;
	makeGraph( -1.5f, 2.0f, 0, 2, &box1, &box2, s ) ;
	makeGraph( -1.5f, 2.0f, 1, 2, &box1, &box2, s ) ;

	freeShaft(s) ;

#define SHOW_SHAFT_TESTS
#ifdef SHOW_SHAFT_TESTS
	/* loop through many box/box combinations, see if reasonable;
	 * generates much ASCII */
	/* If you really like spewage, set x1 and y1 <= 7 */
	for ( x1 = 0 ; x1 <= 3 ; x1++ ) {
		convertIndex( x1, &box2.c[LO_X], &box2.c[HI_X] ) ;
		for ( y1 = 0 ; y1 <= 3 ; y1++ ) {
			convertIndex( y1, &box2.c[LO_Y], &box2.c[HI_Y] ) ;
			for ( z1 = 0 ; z1 <= 7 ; z1++ ) {
				convertIndex( z1, &box2.c[LO_Z], &box2.c[HI_Z] ) ;
				dumpBox( "box:", &box2 ) ;
				
				s = formShaft( &box1, &box2 ) ;

				makeGraph( -1.5f, 2.0f, 0, 1, &box1, &box2, s ) ;
				makeGraph( -1.5f, 2.0f, 0, 2, &box1, &box2, s ) ;
				makeGraph( -1.5f, 2.0f, 1, 2, &box1, &box2, s ) ;
				freeShaft(s) ;
			}
		}
	}
#endif
	
	
#ifdef GATHER_STATISTICS
	clearStats();
	myseedrand(12345);

	/* generate random boxes from 0,0,0 to 1,1,1,
	 * then rescale them to fill the volume 0,0,0 to 1,1,1,
	 * then create the shaft and test against it.
	 */
	for ( i = 0 ; i < 10000 ; i++ ) {

		for ( v = 0 ; v < 6 ; v++ ) {
			box1.c[v] = (float)myrand() ;
			box2.c[v] = (float)myrand() ;
		}

		if ( box1.c[LO_X] > box1.c[HI_X] ) {
			temp = box1.c[LO_X] ;
			box1.c[LO_X] = box1.c[HI_X] ;
			box1.c[HI_X] = temp ;
		}
		if ( box1.c[LO_Y] > box1.c[HI_Y] ) {
			temp = box1.c[LO_Y] ;
			box1.c[LO_Y] = box1.c[HI_Y] ;
			box1.c[HI_Y] = temp ;
		}
		if ( box1.c[LO_Z] > box1.c[HI_Z] ) {
			temp = box1.c[LO_Z] ;
			box1.c[LO_Z] = box1.c[HI_Z] ;
			box1.c[HI_Z] = temp ;
		}

		if ( box2.c[LO_X] > box2.c[HI_X] ) {
			temp = box2.c[LO_X] ;
			box2.c[LO_X] = box2.c[HI_X] ;
			box2.c[HI_X] = temp ;
		}
		if ( box2.c[LO_Y] > box2.c[HI_Y] ) {
			temp = box2.c[LO_Y] ;
			box2.c[LO_Y] = box2.c[HI_Y] ;
			box2.c[HI_Y] = temp ;
		}
		if ( box2.c[LO_Z] > box2.c[HI_Z] ) {
			temp = box2.c[LO_Z] ;
			box2.c[LO_Z] = box2.c[HI_Z] ;
			box2.c[HI_Z] = temp ;
		}

		for ( v = 0 ; v < 3 ; v++ ) {
			if ( box1.c[v] < box2.c[v] ) {
				min[v] = box1.c[v] ;
			} else {
				min[v] = box2.c[v] ;
			}
		}

		for ( v = 3 ; v < 6 ; v++ ) {
			if ( box1.c[v] > box2.c[v] ) {
				max[v-3] = box1.c[v] ;
			} else {
				max[v-3] = box2.c[v] ;
			}
		}
		for ( v = 0 ; v < 3 ; v++ ) {
			if ( max[v] == min[v] ) {
				max[v] += 0.0001f ;
			}
			len[v] = max[v] - min[v] ;
		}
		for ( v = 0 ; v < 6 ; v++ ) {
			box1.c[v] = ( box1.c[v] - min[v%3] ) / len[v%3] ;
			box2.c[v] = ( box2.c[v] - min[v%3] ) / len[v%3] ;
		}

		s = formShaft( &box1, &box2 ) ;

		testPoints( -0.0f, 1.0f, 0, 1, &box1, &box2, s ) ;
		testPoints( -0.0f, 1.0f, 0, 2, &box1, &box2, s ) ;
		testPoints( -0.0f, 1.0f, 1, 2, &box1, &box2, s ) ;
		

		/* dumpShaft( "test shaft:", s ) ; */
		freeShaft(s) ;
	}
	dumpStats();
#endif

}
