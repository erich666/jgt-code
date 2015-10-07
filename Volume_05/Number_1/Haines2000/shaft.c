/*
 * Shaft creation and query code
 *
 * Eric Haines, erich@acm.org, rev 1.3, 3/5/2000
 *
 * see shaft.h for how to use this code.
 */

#include <stdio.h>
#include <malloc.h>
#include <math.h>

#include "shaft.h"

#ifdef USE_TABLE
#include "shafttab.h"
#endif


#ifndef TRUE
#define TRUE 1
#define FALSE 0
#endif


#ifdef GATHER_STATISTICS

/* Some statistics found while testing a set of random shafts:
 * A shaft has an average of about 6 planes associated with it.
 * The breakdown is:
 *     shafts with 0 planes (i.e. box in box): 1%
 *     shafts with 4 planes: 11%
 *     shafts with 6 planes: 74%
 *     shafts with 8 planes: 14%
 * Shafts with 1,2,3, and 5 planes are possible, these occur
 * when one or more coordinates are equal between the two boxes.
 */
static int gNumShaft = 0 ;
static int gNumPlanesMade = 0 ;
static int gNumOutShaftsTested = 0 ;
static int gNumOutPlanesTested = 0 ;
static int gNumInShaftsTested = 0 ;
static int gNumInPlanesTested = 0 ;
static int gInitHisto = 1 ;
static int gHisto[9] ;

#define STATS(v)	v

#else
#define STATS(v)

#endif


typedef struct {
	int c[6] ;
} boxI ;


/* Ax + By + Cz = D */
void addPlaneToShaft( float a, float b, float c, float d, shaft *s )
{
	/* possible efficiency: the planeRecs could be allocated once
	 * for the shaft structure, i.e. always have a set of 8 or 10
	 * planes for a given shaft and keep track of the number
	 * currently in use. This saves on allocs and may help caching.
	 * That said, in one set of tests by Martin Blais the savings
	 * amounted to a 1% speedup on a MIPS, so...
	 */
	planeRec *pr = (planeRec *)malloc( sizeof(planeRec) ) ;
	STATS(gNumPlanesMade++;)
	pr->next = s->planeSet ;
	s->planeSet = pr ;

	pr->a = a ;
	pr->b = b ;
	pr->c = c ;
	pr->d = d ;

	/* If the plane is pointing in the -X direction, set the
	 * index of nearCorner[0] = 0 (the lo[0] coord location),
	 * else set nearCorner[0] = 3 (the hi[0] coord location in union).
	 * Do for Y and Z also.
	 */

/* see shaft.h for information on these defines */
#ifndef INSIDE_ONLY
	pr->nearCorner[X] = ( a >= 0.0f ) ? LO_X : HI_X ;
	pr->nearCorner[Y] = ( b >= 0.0f ) ? LO_Y : HI_Y ;
	pr->nearCorner[Z] = ( c >= 0.0f ) ? LO_Z : HI_Z ;
#endif
	
#ifndef OUTSIDE_ONLY
	pr->farCorner[X] = ( a < 0.0f ) ? LO_X : HI_X ;
	pr->farCorner[Y] = ( b < 0.0f ) ? LO_Y : HI_Y ;
	pr->farCorner[Z] = ( c < 0.0f ) ? LO_Z : HI_Z ;

	/* An alternative when using both inside and outside testing:
	 * doing "+3)%6" changes LO_X to HI_X or vice versa, etc.
	 * This may be a little more efficient, as it avoids branches. */
	/*
	 * pr->farCorner[X] = (pr->nearCorner[X]+3)%6 ;
	 * pr->farCorner[Y] = (pr->nearCorner[Y]+3)%6 ;
	 * pr->farCorner[Z] = (pr->nearCorner[Z]+3)%6 ;
	 */
#endif
}

#ifdef PLANE_SORTING
typedef struct {
	planeRec *pr ;
	float cost ;
} planeSortRec ;

static float planeValue( shaft *s, planeRec *pr )
{
	float val ;

	/* normalizes plane, which should not affect anything */
#ifndef INCLUDE_SPHERE_TESTING
	float len = (float)sqrt(pr->a*pr->a + pr->b*pr->b + pr->c*pr->c);
	len = 1.0f / len ;
	pr->a *= len ;
	pr->b *= len ;
	pr->c *= len ;
	pr->d *= len ;
#endif

	/* note that far_corner needs to exist in the record to compute this value */
	/* The formula is to evaluate the farthest corner of the shaft with the plane
	 * equation; this is the distance to this corner. Squaring it and dividing by
	 * a*b (or b*c or a*c) gives 2x the area of the triangle cut off the square.
	 */
	val = pr->a * s->bx.c[pr->farCorner[X]] +
		  pr->b * s->bx.c[pr->farCorner[Y]] +
		  pr->c * s->bx.c[pr->farCorner[Z]] - pr->d ;

	val *= val ;
	if ( pr->a == 0.0f ) {
		val /= (pr->b * pr->c ) ;
	} else if ( pr->b == 0.0f ) {
		val /= (pr->a * pr->c ) ;
	} else {
		val /= (pr->a * pr->b ) ;
	}
	if ( val < 0.0f ) {
		val = -val ;
	}
	return val ;
}

/* Sorting the planes always helped on the average, but the savings are not that large.
 * For a set of tests on randomly generated shafts, the average number of planes
 * tested (when the shaft box test was passed) varied from 3.3 to 5.5 for boxOutside
 * testing, 1.1 to 5.9 for boxInside testing. Results were highly dependent on how
 * far the boxes forming the shaft were from each other, relative positioning, size
 * and dimensions of the test boxes, etc. The best test is to use sorting in your
 * application and see if it helps.
 *
 * That said, sorting the planes in advance saved a range of 0.1 to 0.6 plane tests
 * vs. unsorted for boxOutside tests. For boxInside tests the range was 0.1 to 1.1.
 * Not huge savings, but possibly worth it if the shaft is used against many boxes.
 *
 * Another minor sorting variant is to try to group opposite culling planes together
 * at the front of the list. For example, the pair of planes for the -X,+Y and +X,-Y
 * edges will, as a pair, potentially cut off more volume of the shaft box than the
 * two "best" planes. This is because this pair of planes cuts off volumes which do
 * not ever overlap, while two arbitrary planes often intersect inside the shaft box
 * and so cut off the same volume of space twice. Given that the savings were minor
 * from just sorting, adding additional code and trying to somehow bias the sort to
 * favor pairings did not seem worth the effort. I mention it here in case a huge
 * number of boxes will be compared to each shaft; this would justify a more careful
 * preprocess sort.
 */
static void	sortPlanes( shaft *s )
{
	int np = 0 ;
	planeRec *pr ;
	planeSortRec plns[24] ;	/* there can be up to 8 planes, but just in case more are added */
	int sortList[24], temp, i, sorted ;

#ifdef GATHER_STATISTICS
	if ( gInitHisto ) {
		for ( i = 0 ; i <= 8 ; i++ ) {
			gHisto[i] = 0 ;
		}
		gInitHisto = 0 ;
	}
#endif

	pr = s->planeSet ;
	/* check if 0 or 1 planes on list */
	if ( pr == NULL || pr->next == NULL ) {
#ifdef GATHER_STATISTICS
		if ( pr == NULL ) {
			gHisto[0]++ ;
		} else {
			gHisto[1]++ ;
		}
#endif
		return ;
	}

	while ( pr ) {
		plns[np].pr = pr ;
		plns[np].cost = planeValue( s, pr ) ;
		sortList[np] = np ;
		np++ ;
		pr = pr->next ;
	}
	STATS( gHisto[np]++ ;)

	do {
		sorted = TRUE ;
		for ( i = 0 ; i < np-1 ; i++ ) {
			if ( plns[sortList[i]].cost < plns[sortList[i+1]].cost ) {
				/* swap */
				temp = sortList[i+1] ;
				sortList[i+1] = sortList[i] ;
				sortList[i] = temp ;
				sorted = FALSE ;
			}
		}
	} while ( !sorted ) ;

	pr = s->planeSet = plns[sortList[0]].pr ;
	for ( i = 1 ; i < np ; i++ ) {
		pr->next = plns[sortList[i]].pr ;
		pr = pr->next ;
	}
	pr->next = NULL ;
}
#endif
	
/* Pass in two boxes, create a shaft between them */
/* A shaft is formed by a set of planes connecting two boxes.
 * The shaft itself has a bounding box, formed by the union of
 * the two boxes it is derived from. Each face of the shaft box
 * touches one or both of these two boxes. Another way of looking
 * at it is that the shaft box's faces get categorized as belonging
 * to one box or the other, or both - think of a cube in which
 * each face is painted blue or red (or purple, for both), to denote
 * which of the two boxes touches it. Each of the twelve cube edges
 * of this painted cube are examined: if the adjoining faces are
 * one red and one blue, then a shaft plane is formed to join the
 * two different boxes. If one of the adjoining faces is purple
 * (i.e. shared by both boxes), or both faces are the same color,
 * then no plane has to be formed.
 */
shaft * formShaft( box *box0, box *box1 )
{


#ifdef INCLUDE_SPHERE_TESTING
	float len ;
#endif
	int i, i1, i2, i1n, i2n, i3n ;
	float pn[4] ;


#ifdef USE_TABLE
	/* Use a predefined lookup table. For two boxes, we set bits in a
	 * 2^6 (64) value bit string depending on which box, box0 or box1,
	 * is at the maximum extent. This gives a set of from 0 to 8 planes
	 * which must be formed for this combination of boxes.
	 *
	 * One minor inefficiency of using table lookup is that if the two
	 * boxes have an identical coordinate (e.g. HI_Y), normally a plane
	 * between the two boxes that touches this face would not be needed.
	 * To keep the lookup table small (2^6, instead of 3^6, entries) this
	 * special case is ignored, meaning that the shaft itself is less
	 * efficient since it may include and use unnecessary planes (i.e.
	 * planes which are coincident with the sides of the shaft's overall
	 * bounding box).
	 *
	 * Interestingly, on a Pentium II (450 MHz) using the table method
	 * for forming shafts was only 3% faster than doing it by searching
	 * through all edges (see the #else area that follows). The savings
	 * of simply walking through a table may be offset by the costs of
	 * bringing table values into the cache. Or my test conditions may
	 * not have been realistic. The gist: it's worth testing to see
	 * which shaft formation method is faster on your machine.
	 */
	planeSetPair *ps ;
	intPair *ip ;
	int pairIndex ;

	shaft *s = (shaft *)malloc( sizeof(shaft) ) ;
	STATS(gNumShaft++;)
	s->planeSet = NULL ;

	pairIndex = 0 ;

	for ( i = LO_X ; i <= LO_Z ; i++ ) {
		if ( box0->c[i] < box1->c[i] ) {
			s->bx.c[i] = box0->c[i] ;
		} else {
			s->bx.c[i] = box1->c[i] ;
			pairIndex |= 1<<i ;
		}
	}

	for ( i = HI_X ; i <= HI_Z ; i++ ) {
		if ( box0->c[i] > box1->c[i] ) {
			s->bx.c[i] = box0->c[i] ;
		} else {
			s->bx.c[i] = box1->c[i] ;
			pairIndex |= 1<<i ;
		}
	}

	ps = &planeSetTable[pairIndex] ;
	ip = ps->pairs ;

	for ( i = 0 ; i < ps->numPairs ; i++, ip++ ) {
		i1 = ip->i1 ;
		i2 = ip->i2 ;
		i3n = ip->i3n ;
		i1n = i1 % 3 ;
		i2n = i2 % 3 ;

		/* Plane's normal: made by rotating the plane
		 * joining the two faces 90 degrees clockwise.
		 */
		pn[i1n] = box0->c[i2] - box1->c[i2] ;
		pn[i2n] = box1->c[i1] - box0->c[i1] ;
		pn[i3n] = 0.0f ;

#ifdef INCLUDE_SPHERE_TESTING
		/* for sphere testing, the normal must be normalized */
		len = (float)sqrt(pn[i1n]*pn[i1n] + pn[i2n]*pn[i2n]);
		pn[i1n] /= len ;
		pn[i2n] /= len ;
#endif

		/* compute d, offset of plane from origin */
		/* note that pn[i3n] is not used since it is always 0 */
		pn[3] = box0->c[i1] * pn[i1n] + box0->c[i2] * pn[i2n] ;

		/* add the plane to the shaft */
		/* Note: an alternate way to store and access the plane
		 * is by saving i1n and i2n and the two related pn
		 * normal factors (of a,b,c) along with d. This can be
		 * done because shaft planes always have a zero component
		 * in the plane normal. Then in boxOutside you could do:
				if ( box->c[pr->farCorner[i1n]] * pr->pn[i1n] +
					 box->c[pr->farCorner[i2n]] * pr->pn[i2n] > pr->d ) {
		 *
		 * I avoided this optimization here (which saves an add
		 * and multiply), as doing so means that arbitrary planes
		 * (i.e. planes where all three components of the normal
		 * are nonzero) could then not be added to shafts. Also,
		 * it is not proven that this optimization is actually
		 * faster; i1n and i2n have to be accessed vs. fixed indices.
		 */
		addPlaneToShaft( pn[0], pn[1], pn[2], pn[3], s ) ;
	}

#else
	/* not USE_TABLE */

	boxI match, faceTally ;

	shaft *s = (shaft *)malloc( sizeof(shaft) ) ;
	STATS(gNumShaft++;)
	s->planeSet = NULL ;

	/* Store union of the two bounding boxes in the shaft's box structure.
	 * Also set up "match", which tells whether two box coordinates are
	 * the same value, and "faceTally", which says whether box b0 or b1 is
	 * the one which defined the shaft's box (i.e. was the farther one out
	 * in the given X/Y/Z -/+ coordinate direction).
	 */
	for ( i = LO_X ; i <= LO_Z ; i++ ) {
		if ( box0->c[i] == box1->c[i] ) {
			/* low coordinates of bounding boxes match,
			 * so there needs to be no explicit joining
			 * shaft plane for this face, as such a plane
			 * is the same as the side of the shaft's box */
			s->bx.c[i] = box1->c[i] ;
			match.c[i] = (int)TRUE ;
			/* not needed, but so you know:faceTally.c[i] = 0 or 1 ; */
		}
		else {
			match.c[i] = (int)FALSE ;
			if ( box0->c[i] < box1->c[i] ) {
				s->bx.c[i] = box0->c[i] ;
				faceTally.c[i] = 0 ;
			} else {
				s->bx.c[i] = box1->c[i] ;
				faceTally.c[i] = 1 ;
			}
		}
	}

	for ( i = HI_X ; i <= HI_Z ; i++ ) {
		if ( box0->c[i] == box1->c[i] ) {
			/* high coordinates of bounding boxes match,
			 * so there needs to be no explicit joining
			 * shaft plane for this face, as such a plane
			 * is the same as the side of the shaft's box  */
			s->bx.c[i] = box1->c[i] ;
			match.c[i] = (int)TRUE ;
			/* not needed, but so you know:faceTally.c[i] = 0 or 1 ; */
		}
		else {
			match.c[i] = (int)FALSE ;
			if ( box0->c[i] > box1->c[i] ) {
				s->bx.c[i] = box0->c[i] ;
				faceTally.c[i] = 0 ;
			} else {
				s->bx.c[i] = box1->c[i] ;
				faceTally.c[i] = 1 ;
			}
		}
	}

	/* Search through all adjacent cube faces and see if they
	 * should be joined by a plane. If faceTally differs, then
	 * a plane (could be) added.
	 */
	/* loop through all cube faces, -X, -Y, -Z, +X, +Y, +Z */
	for ( i1 = LO_X ; i1 <= HI_Y /* HI_Z tested below */ ; i1++ ) {
		/* if a face's b0 and b1 coordinates matched, no plane needed (purple) */
		if ( !match.c[i1] ) {
			/* loop through cube faces above current face */
			for ( i2 = i1+1 ; i2 <= HI_Z ; i2++ ) {
				/* again, avoid faces with matching coordinates (purple) */
				if ( !match.c[i2] ) {
					/* and do not bother with opposite faces,
					 * e.g. -X and +X share no edges, so ignore.
					 */
					if ( i1+3 != i2 ) {
						/* OK, is there a split? (red and blue) */
						if ( faceTally.c[i1] != faceTally.c[i2] ) {

							/* A real split exists */
							i1n = i1 % 3 ;
							i2n = i2 % 3 ;
							if ( (i1n+1)%3 == i2n ) {
								i3n = (i2n+1)%3 ;
							} else {
								i3n = (i1n+1)%3 ;
							}

							/* Plane's normal: made by rotating the plane
							 * joining the two faces 90 degrees clockwise.
							 */
							pn[i1n] = box0->c[i2] - box1->c[i2] ;

							/* if the face is negative and the normal component points positive,
							 * or face/normal is positive/negative, then the normal must be flipped.
							 */
							if ( ( i1 <= LO_Z ) != ( pn[i1n] < 0.0f ) ) {
								pn[i1n] = -pn[i1n] ;
								pn[i2n] = box0->c[i1] - box1->c[i1] ;
							} else {
								pn[i2n] = box1->c[i1] - box0->c[i1] ;
							}
							pn[i3n] = 0.0f ;

#ifdef INCLUDE_SPHERE_TESTING
							/* for sphere testing, the normal must be normalized */
							len = (float)sqrt(pn[i1n]*pn[i1n] + pn[i2n]*pn[i2n]);
							pn[i1n] /= len ;
							pn[i2n] /= len ;
#endif

							/* compute d, offset of plane from origin */
							/* note that pn[i3n] is not used since it is always 0 */
							pn[3] = box0->c[i1] * pn[i1n] + box0->c[i2] * pn[i2n] ;

							/* add the plane to the shaft */
							/* Note: an alternate way to store and access the plane
							 * is by saving i1n and i2n and the two related pn
							 * normal factors (of a,b,c) along with d. This can be
							 * done because shaft planes always have a zero component
							 * in the plane normal. Then in boxOutside you could do:
							 		if ( box->c[pr->farCorner[i1n]] * pr->pn[i1n] +
										 box->c[pr->farCorner[i2n]] * pr->pn[i2n] > pr->d ) {
							 *
							 * I avoided this optimization here (which saves an add
							 * and multiply), as doing so means that arbitrary planes
							 * (i.e. planes where all three components of the normal
							 * are nonzero) could then not be added to shafts. Also,
							 * it is not proven that this optimization is actually
							 * faster; i1n and i2n have to be accessed vs. fixed indices.
							 */
							addPlaneToShaft( pn[0], pn[1], pn[2], pn[3], s ) ;
						}
					}
				}
			}
		}
	}
#endif

	/* Possible acceleration: look at each plane and determine how much
	 * of the shaft's bounding box is chopped off by it. Put the plane
	 * which chops the most, first. This should increase the probability
	 * of an early out when testing. The downside is that shaft formation
	 * takes longer; the time saved testing needs to outweigh the time
	 * spent building the shaft.
	 */
#ifdef PLANE_SORTING
	sortPlanes( s ) ;
#endif

	return s ;
}

#ifndef INSIDE_ONLY
int boxOutside( box *box, shaft *s )
{
	planeRec *pr ;

	/* first test if box does not overlap shaft's box */
	if ( box->c[LO_X] > s->bx.c[HI_X] ||
		 box->c[LO_Y] > s->bx.c[HI_Y] ||
		 box->c[LO_Z] > s->bx.c[HI_Z] ||
		 box->c[HI_X] < s->bx.c[LO_X] ||
		 box->c[HI_Y] < s->bx.c[LO_Y] ||
		 box->c[HI_Z] < s->bx.c[LO_Z] ) {
		return TRUE ;
	}
	STATS(gNumOutShaftsTested++;)

	/* Test if "nearest in" (compared to each plane) corner of box
	 * is outside (i.e. out of the shaft) of the plane. If so, then
	 * the box must be fully outside the plane and so the box is
	 * outside the shaft.
	 */
	pr = s->planeSet ;
	while ( pr ) {
		STATS(gNumOutPlanesTested++;)
		if ( box->c[pr->nearCorner[X]] * pr->a +
			 box->c[pr->nearCorner[Y]] * pr->b +
			 box->c[pr->nearCorner[Z]] * pr->c > pr->d ) {
			return TRUE ;
		}
		pr = pr->next ;
	}
	return FALSE ;
}
#endif

#ifndef OUTSIDE_ONLY
int boxInside( box *box, shaft *s )
{
	planeRec *pr ;

	/* first test if box is fully inside shaft box */
	if ( box->c[LO_X] < s->bx.c[LO_X] ||
		 box->c[LO_Y] < s->bx.c[LO_Y] ||
		 box->c[LO_Z] < s->bx.c[LO_Z] ||
		 box->c[HI_X] > s->bx.c[HI_X] ||
		 box->c[HI_Y] > s->bx.c[HI_Y] ||
		 box->c[HI_Z] > s->bx.c[HI_Z] ) {
		return FALSE ;
	}
	STATS(gNumInShaftsTested++;)

	/* Test if "farthest out" (compared to each plane) corner of box
	 * is outside (i.e. outside the shaft) of the plane. If so, then
	 * the box is not fully inside the shaft.
	 */
	pr = s->planeSet ;
	while ( pr ) {
		STATS(gNumInPlanesTested++;)
		if ( box->c[pr->farCorner[X]] * pr->a +
			 box->c[pr->farCorner[Y]] * pr->b +
			 box->c[pr->farCorner[Z]] * pr->c > pr->d ) {
			return FALSE ;
		}
		pr = pr->next ;
	}
	return TRUE ;
}
#endif



#ifdef INCLUDE_SPHERE_TESTING
#ifndef INSIDE_ONLY
/* note: the sphereOutside test is imperfect (unlike the boxOutside test).
 * It is, however, conservative. In other words, this test can occasionally
 * categorize a sphere as not being outside the shaft when, in fact, it is.
 * For most applications this simply results in the miscategorized sphere
 * and its contents undergoing further (useless) testing. This wastes time,
 * but no error occurs. sphereInside does not have this problem.
 */
int sphereOutside( sphere *sph, shaft *s )
{
	planeRec *pr ;

	/* first test if sphere does not overlap shaft's box */
	if ( sph->center[X] - sph->radius > s->bx.c[HI_X] ||
		 sph->center[Y] - sph->radius > s->bx.c[HI_Y] ||
		 sph->center[Z] - sph->radius > s->bx.c[HI_Z] ||
		 sph->center[X] + sph->radius < s->bx.c[LO_X] ||
		 sph->center[Y] + sph->radius < s->bx.c[LO_Y] ||
		 sph->center[Z] + sph->radius < s->bx.c[LO_Z] ) {
		return TRUE ;
	}

	/* Test if the center's distance to each plane is
	 * greater than the radius; if so, the sphere is
	 * fully outside the plane and so outside the shaft.
	 */
	pr = s->planeSet ;
	while ( pr ) {
		if ( sph->center[X] * pr->a +
			 sph->center[Y] * pr->b +
			 sph->center[Z] * pr->c - sph->radius > pr->d ) {
			return TRUE ;
		}
		pr = pr->next ;
	}
	return FALSE ;
}
#endif

#ifndef OUTSIDE_ONLY
int sphereInside( sphere *sph, shaft *s )
{
	planeRec *pr ;

	/* first test if box is fully inside shaft box */
	if ( sph->center[X] - sph->radius < s->bx.c[LO_X] ||
		 sph->center[Y] - sph->radius < s->bx.c[LO_Y] ||
		 sph->center[Z] - sph->radius < s->bx.c[LO_Z] ||
		 sph->center[X] + sph->radius > s->bx.c[HI_X] ||
		 sph->center[Y] + sph->radius > s->bx.c[HI_Y] ||
		 sph->center[Z] + sph->radius > s->bx.c[HI_Z] ) {
		return FALSE ;
	}

	/* Test if the distance to the point furthest along
	 * the plane's normal is outside the plane. If so, then
	 * the sphere is not fully inside the shaft.
	 */
	pr = s->planeSet ;
	while ( pr ) {
		if ( sph->center[X] * pr->a +
			 sph->center[Y] * pr->b +
			 sph->center[Z] * pr->c + sph->radius > pr->d ) {
			return FALSE ;
		}
		pr = pr->next ;
	}
	return TRUE ;
}
#endif
#endif /* #ifdef INCLUDE_SPHERE_TESTING */


void freeShaft( shaft *s )
{
	planeRec *pr, *nextPr ;
	if ( s ) {
		pr = s->planeSet ;
		while ( pr ) {
			nextPr = pr->next ;
			free( pr ) ;
			pr = nextPr ;
		}
		free( s ) ;
	}
}


#ifdef GATHER_STATISTICS
void clearStats()
{
	int i ;

	gNumShaft = 0 ;
	gNumPlanesMade = 0 ;
	gNumOutShaftsTested = 0 ;
	gNumOutPlanesTested = 0 ;
	gNumInShaftsTested = 0 ;
	gNumInPlanesTested = 0 ;
	gInitHisto = 0 ;	/* since we do this below */

	for ( i = 0 ; i <= 8 ; i++ ) {
		gHisto[i] = 0 ;
	}
}

void dumpStats()
{
	int i ;

	printf( "number of shafts made: %d, number of planes: %d\n    planes/shaft avg: %g\n",
		gNumShaft, gNumPlanesMade, (float)gNumPlanesMade/(float)gNumShaft ) ;
	printf( "outside shafts tested: %d, outside planes: %d\n    planes/shaft avg: %g\n",
		gNumOutShaftsTested, gNumOutPlanesTested, (float)gNumOutPlanesTested/(float)gNumOutShaftsTested ) ;
	printf( "inside shafts tested: %d, inside planes: %d\n    planes/shaft avg: %g\n",
		gNumInShaftsTested, gNumInPlanesTested, (float)gNumInPlanesTested/(float)gNumInShaftsTested ) ;

	printf( "\nHistogram for %d shafts\n# planes: # having this many planes\n", gNumShaft ) ;
	for ( i = 0 ; i <= 8 ; i++ ) {
		printf( "%d: %d, percent is %g\n", i, gHisto[i], (float)gHisto[i]*100.0f/gNumShaft ) ;
	}
}
#endif
