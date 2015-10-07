/* shaft.h - include file for shaft culling code shaft.c
 *
 * Eric Haines, erich@acm.org, rev 1.3, 3/5/2000
 * thanks to Martin Blais for noting a variety of code problems
 *
 * Implementation of "Shaft Culling for Efficient Ray-Traced Radiosity,"
 * Eric A. Haines and John R. Wallace, Photorealistic Rendering in
 * Computer Graphics (Proceedings of the Second Eurographics Workshop
 * on Rendering), Springer-Verlag, New York, 1994, p.122-138. Paper was
 * also in SIGGRAPH '91 Frontiers in Rendering course notes.
 *
 * A Postscript version of this paper can be found online at:
 * http://www.acm.org/pubs/tog/editors/erich/
 *
 *
 * Other relevant code:
 * shaft.h - include file
 * main.c - test harness for this code
 *
 * Main entry points:
 * formShaft - pass in the two boxes you want to form a shaft
 * addPlaneToShaft - if you want to add more cutting planes (e.g.
 *     the emitter or receiver polygon planes)
 *
 * boxOutside - test if a box is outside of the shaft
 * boxInside - test if a box is fully inside the shaft
 * sphereOutside - test if a sphere is outside of the shaft (see shaft.h)
 * sphereInside - test if a sphere is fully inside the shaft (see shaft.h)
 *
 * freeShaft - dealloc the shaft structure.
 *
 * A normal calling sequence is to create a shaft by passing in two boxes:
 *      box box1, box2 ;
 *      shaft *s ;
 *      ... fill in box.c[] data ...
 *      s = formShaft( &box1, &box2 ) ;
 *
 * Additional culling planes can be added to the shaft, if desired. For
 * example, say the two objects in your bounding boxes are polygons. The
 * planes of these two polygons, made to face outwards, could be added to
 * the shaft. Then, any bounding volume on the far side of either of these
 * polygons' planes will be culled out. Here is how to add a plane:
 *      float a,b,c,d ;
 *      ... fill in plane equation ax + by + cz = d ...
 *      addPlaneToShaft( a, b, c, d, s ) ;
 *
 * To test a given bounding box against the shaft, a typical sequence is:
 *      box boxt ;
 *      ... fill in test box ...
 *      if ( !boxOutside( &boxt, s ) ) {
 *          if ( boxInside( &boxt, s ) ) {
 *              ... box is fully inside shaft ...
 *          } else {
 *              ... box overlaps shaft, but not fully inside ...
 *          }
 *      } else {
 *          ... box is fully outside shaft ...
 *      }
 *
 * A similar sequence is used to test a sphere against the shaft. One
 * important caveat: occasionally a sphere which is categorized as
 * overlapping is actually fully outside the shaft. This miscategorization
 * cannot happen with the box tests, but can with spheres. If a sphere is
 * marked as outside or inside, these categorizations are always true.
 *
 * When done with a shaft, free it with:
 *      freeShaft( s ) ;
 *
 *
 * OPTIMIZATIONS:
 * - Read over the #defines below; if you do not need a test, e.g.
 *   boxInside(), then turn on OUTSIDE_ONLY and shaft formation will take
 *   less time. Test on your own machine whether USE_TABLE is faster or not.
 * - Read over the comments throughout this code for a number of possible
 *   enhancements to the code. These were not done here as they detracted
 *   from generality or readability. However, they may be worth doing for
 *   your particular use.
 * - If you are testing a hierarchy of boxes or spheres against a shaft,
 *   you may wish to inherit information from parent to child. Specifically,
 *   if a parent box or sphere overlaps a shaft (i.e. is neither fully inside
 *   or outside), there is probably information which can be reused. For
 *   example, the parent might be fully inside the shaft box, in which case
 *   all children will be inside the shaft box and so do not have to be
 *   tested against it. Also, the parent may be fully inside a given plane, in
 *   which case this plane does not have to be retested against the children.
 */

/* define USE_TABLE to use the lookup table of planes to generate for any
 * pair of boxes. 3% (that's all) faster on a Pentium II. */
/* #define USE_TABLE */

/* define OUTSIDE_ONLY if you only want to use the *Outside() routines */
/* #define OUTSIDE_ONLY */

/* define INSIDE_ONLY if you only want to use the *Inside() routines */
/* #define INSIDE_ONLY */

/* define INCLUDE_SPHERE_TESTING if you also want to test spheres against shafts;
 * note that this will make shaft creation more costly. */
/* #define INCLUDE_SPHERE_TESTING */

/* define PLANE_SORTING if you want to sort the planes in order of effectiveness */
/* #define PLANE_SORTING */

/* define GATHER_STATISTICS if you want to gather information on how many shafts
 * and planes were generated. Note: also define PLANE_SORTING if you use this. */
/* #define GATHER_STATISTICS */

#ifndef SHAFT_H
#define SHAFT_H

#define X 0
#define Y 1
#define Z 2

#define LO_X 0
#define LO_Y 1
#define LO_Z 2
#define HI_X 3
#define HI_Y 4
#define HI_Z 5


typedef struct {
	float c[6] ;	/* lo x,y,z; hi x,y,z */
} box ;

typedef struct {
	float center[3] ;
	float radius ;
} sphere ;


typedef struct planeRec_t {
	float a,b,c,d ;	/* Ax + By + Cz = D */
#ifndef INSIDE_ONLY
	int nearCorner[3] ;	/* corner indices, 0/4;1/5;2/6 */
#endif
#ifndef OUTSIDE_ONLY
	int farCorner[3] ;	/* corner indices, 0/4;1/5;2/6 */
#endif
	struct planeRec_t *next ;
} planeRec ;

typedef struct {
	box	bx ;
	planeRec *planeSet ;
} shaft ;


void addPlaneToShaft( float a, float b, float c, float d, shaft *s );
shaft * formShaft( box *box0, box *box1 );

#ifndef INSIDE_ONLY
int boxOutside( box *box, shaft *s );
#endif

#ifndef OUTSIDE_ONLY
int boxInside( box *box, shaft *s );
#endif

#ifdef INCLUDE_SPHERE_TESTING
#ifndef INSIDE_ONLY
int sphereOutside( sphere *sph, shaft *s );
#endif

#ifndef OUTSIDE_ONLY
int sphereInside( sphere *sph, shaft *s );
#endif
#endif	/* INCLUDE_SPHERE_TESTING */

void freeShaft( shaft *s );

#ifdef GATHER_STATISTICS
void clearStats();
void dumpStats();
#endif

#endif	/* SHAFT_H */
