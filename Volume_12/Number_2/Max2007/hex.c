/*
NOTICE 1


This work was produced at the University of California, Lawrence Livermore National Laboratory
(UC LLNL) under contract no. W-7405-ENG-48 (Contract 48) between the U.S. Department of Energy
(DOE) and The Regents of the University of California (University) for the operation of UC
LLNL. The rights of the Federal Government are reserved under Contract 48 subject to the
restrictions agreed upon by the DOE and University as allowed under DOE Acquisition Letter
97-1.


DISCLAIMER

This work was prepared as an account of work sponsored by an agency of the United States
Government. Neither the United States Government nor the University of California nor any
of their employees, makes any warranty, express or implied, or assumes any liability or
responsibility for the accuracy, completeness, or usefulness of any information, apparatus,
product, or process disclosed, or represents that its use would not infringe privately-owned
rights.  Reference herein to any specific commercial products, process, or service by trade
name, trademark, manufacturer or otherwise does not necessarily constitute or imply its
endorsement, recommendation, or favoring by the United States Government or the University
of California. The views and opinions of authors expressed herein do not necessarily state
or reflect those of the United States Government or the University of California, and shall
not be used for advertising or product endorsement purposes.


NOTIFICATION OF COMMERCIAL USE

Commercialization of this product is prohibited without notifying the Department of Energy
(DOE) or Lawrence Livermore National Laboratory (LLNL).

*/
// The triangle fans specification begins on line 1818
// The tests for non-convex quadrilaterals begin on line 875
// The vertex permutation for standard numbering begins on line 977
// The classification of cases and extra vertex computation begin on line 1076

#include <GL/gl.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "hex.h"

#define VSIZE 14
#define TSCALE 5.   /* higher means less linear texture. Was 5. */

/* If 1, use routines in projection.c, otherwise, use original code written
 * by nelson.
 */
#define USE_PROJECTION_ROUTINES 1

extern int ipersp;

#if USE_PROJECTION_ROUTINES != 1
extern double  vpdist;
extern double facbx, facby, xdisp, ydisp;
#endif


/* Homogeneous screen coordinates, another way of viewing
 * the xv,yv arrays which are local to har_hexProject..
 * The seem to be used by the splat code. Since splat just
 * does tets and computes its own edge intersections, we
 * do not need to copy in the additional vertices stored in
 * xv,yv to this array.  Only the initial projection is copyied
 * into this array at the beginning of har_hexProject.
 * IMPORTANT:  This array must be eliminated when the hex code
 * is validated and the tet splatting code is refactored.
 */
//extern double vv[8][3];
//extern int id;
extern float xvmin, xvmax, yvmin, yvmax;

/* DLDLDL: 
 * These variables MUST go away, but while refactoring, they stay and
 * are eliminated as we progress.  They are DEFINED in simple.c.
 */

/*
extern long int nv8 , nv9 , nv10 , nonconseq , eq3 , bowties ,
     nvsecond8 , nvsecond9 , nosecond9 , 
     count0 , draw0 , sv810 , sv910 ,
     no8sv10 , no9sv10 , normal1 , normal2 , normal4 ;
extern int test_degenerate, ndegen , nalldegen , toteq ;
extern long int badcount , totalcount ;
     */

static long int badexit, intrpr = 0, second10 = 0;
static int rr[VSIZE], rrinv[VSIZE], ppinv[VSIZE], pp[VSIZE], outline[8];
static double ang[8];
static FILE *fp;

static int faceverts[6][4] =
{{0,1,2,3}, {4,7,6,5}, {0,4,5,1}, {1,5,6,2}, {2,6,7,3}, {3,7,4,0}};

/* --------------------------------------------------------------------------
 *
 * viewcell
 *
 * --------------------------------------------------------------------------
 * This routine is UNTESTED.
 * It used to use global variables, now it is self contained.
 * It should work fine.
 * Its purpose is to produce inventor files of either the whole grid or selected
 * problem cells, for aid in debugging.
 */
static void
viewcell(FILE* fp, const har_Zone* zone, 
         const har_HexProjectionStatistics* stat)
{
    int i, j, l;

    assert(zone->type == HAR_ZONETYPE_HEX);

    fprintf((FILE*)fp,"#Inventor V2.0 ascii\n");
    if (stat != NULL) fprintf((FILE*)fp,"# totalcount %ld\n", stat->totalcount);
    fprintf((FILE*)fp,"DirectionalLight {on TRUE intensity  .8\n");
    fprintf((FILE*)fp,"    color   1 1 1  direction  0 1 -1 }\n");
    fprintf((FILE*)fp,"DirectionalLight {on TRUE intensity  .8\n");
    fprintf((FILE*)fp,"    color   1 1 1  direction  0 -1 1 }\n");
    fprintf((FILE*)fp,"Environment { ambientIntensity  1.1 \n");
    fprintf((FILE*)fp,"   ambientColor      1 1 1}\n");
    fprintf((FILE*)fp,"Separator {\n");
    fprintf((FILE*)fp,"DrawStyle { style LINES }\n");
    fprintf((FILE*)fp,"Material { ambientColor 0 1 1\n");
    fprintf((FILE*)fp,"           diffuseColor 0 0 0 }\n");
    fprintf((FILE*)fp, "Coordinate3 {  point [\n"); 
    for (i = 0; i < 8; ++i) {
        fprintf((FILE*)fp, "%15.9g %15.9g %15.9g,\n", 
                           zone->x[i], zone->y[i], zone->z[i]);
    }
    fprintf((FILE*)fp, "   ] }\n");
    fprintf((FILE*)fp, "IndexedFaceSet { coordIndex [\n");

    for (i = 0; i < 6; ++i) {
        for (j = 0; j < 4; ++j) {
            l = faceverts[i][j];
            fprintf((FILE*)fp,"%d,", l);
        }
        fprintf((FILE*)fp,"%d,\n", -1);
    }
    fprintf((FILE*)fp, "] } }\n");
}

/* --------------------------------------------------------------------------
 *
 * har_hexProjectionResultString
 *
 * --------------------------------------------------------------------------
 */
const char*
har_hexProjectionResultString(int returnval)
{
    switch (returnval) {
        case HEX_SUCCESS: return "HEX_SUCCESS";
        case HEX_FAIL_DEGENERATE: return "HEX_FAIL_DEGENERATE";
        case HEX_FAIL_DEGENERATE_TO_SINGLE_POINT: return "HEX_FAIL_DEGENERATE_TO_SINGLE_POINT";
        case HEX_FAIL_BOWTIE: return "HEX_FAIL_BOWTIE";
        case HEX_FAIL_EQ3: return  "HEX_FAIL_EQ3";
        case HEX_FAIL_INTERSECTING_EDGES_NOT_CONSECUTIVE: return  "HEX_FAIL_INTERSECTING_EDGES_NOT_CONSECUTIVE";
        case HEX_FAIL_NO_SECOND_CASE8_EDGE_INTERSECTION: return  "HEX_FAIL_NO_SECOND_CASE8_EDGE_INTERSECTION";
        case HEX_FAIL_NO_SECOND_TRY_FOR_CASE8_VERTEX10_INTERSECTION: return  "HEX_FAIL_NO_SECOND_TRY_FOR_CASE8_VERTEX10_INTERSECTION";
        case HEX_FAIL_NO_SECOND_TRY_FOR_CASE9_EDGE: return  "HEX_FAIL_NO_SECOND_TRY_FOR_CASE9_EDGE";
        case HEX_FAIL_BOTH_SECOND8_AND_SECOND9_USED: return  "HEX_FAIL_BOTH_SECOND8_AND_SECOND9_USED";
        case HEX_FAIL_NO_SECOND_TRY_FOR_CASE9_VERTEX10_INTERSECTION: return  "HEX_FAIL_NO_SECOND_TRY_FOR_CASE9_VERTEX10_INTERSECTION";
        case HEX_FAIL_COUNT2_AND_SECOND8_OR_SECOND9: return  "HEX_FAIL_COUNT2_AND_SECOND8_OR_SECOND9";
        case HEX_FAIL_NEITHER_OCTAGON_DIAGONAL_FOUND_FOR_VERT0: return  "HEX_FAIL_NEITHER_OCTAGON_DIAGONAL_FOUND_FOR_VERT0";
        case HEX_FAIL_MISSED_0_3_AND_1_6_INTERSECTIONS_FOR_OCTAGON: return  "HEX_FAIL_MISSED_0_3_AND_1_6_INTERSECTIONS_FOR_OCTAGON";
        case HEX_FAIL_MISSED_2_5_AND_3_0_INTERSECTIONS_FOR_OCTAGON: return  "HEX_FAIL_MISSED_2_5_AND_3_0_INTERSECTIONS_FOR_OCTAGON";
        case HEX_FAIL_MISSED_2_5_AND_4_7_INTERSECTIONS_FOR_OCTAGON: return  "HEX_FAIL_MISSED_2_5_AND_4_7_INTERSECTIONS_FOR_OCTAGON";
        case HEX_FAIL_MISSED_1_6_AND_4_7_INTERSECTIONS_FOR_OCTAGON: return  "HEX_FAIL_MISSED_1_6_AND_4_7_INTERSECTIONS_FOR_OCTAGON";
        case HEX_FAIL_TOP_AND_BOTTOM_VERTS_IDENTICAL: return  "HEX_FAIL_TOP_AND_BOTTOM_VERTS_IDENTICAL";
        case HEX_FAIL_SECOND8_OR_SECOND9_BUT_NO_CROSS: return  "HEX_FAIL_SECOND8_OR_SECOND9_BUT_NO_CROSS";
        case HEX_FAIL_UNKNOWN_TRIANGLE_FAN_CASE: return  "HEX_FAIL_UNKNOWN_TRIANGLE_FAN_CASE";
    }
    return "UNKNOWN CODE RETURNED";
}

/* --------------------------------------------------------------------------
 *
 * har_initHexProjectionStatistics
 *
 * --------------------------------------------------------------------------
 */
void
har_initHexProjectionStatistics(har_HexProjectionStatistics* stats)
{
    stats->nv8 = 0;
    stats->nv9 = 0;
    stats->nv10 = 0;

    stats->nonconseq = 0;
    stats->eq3 = 0;
    stats->bowties = 0;

    stats->nvsecond8 = 0;
    stats->nvsecond9 = 0;
    stats->nosecond9 = 0;

    stats->count0 = 0;
    stats->draw0 = 0;
    stats->sv810 = 0;
    stats->sv910 = 0;
    stats->no8sv10 = 0;
    stats->no9sv10 = 0;

    stats->normal1 = 0;
    stats->normal2 = 0;
    stats->normal4 = 0;

    stats->ndegen = 0;
    stats->nalldegen = 0;
    stats->toteq = 0;
    stats->badcount = 0;
    stats->totalcount = 0;
}

/* --------------------------------------------------------------------------
 *
 * har_printHexProjectionStatistics
 *
 * --------------------------------------------------------------------------
 */
void
har_printHexProjectionStatistics(har_HexProjectionStatistics* s, const char* str)
{
    printf("%s nv8: %ld %ld  nv9: %ld %ld %ld  nv10: %ld sv810: %ld sv910: %ld\n"
           "  nonconseq %ld  eq3 %ld  bowties %ld\n, nov8sv10 %ld nov9sv10 %ld\n"
           "  count0 %ld draw0 %ld normal1 %ld normal2 %ld normal4 %ld\n", str,
           s->nv8, s->nvsecond8, s->nv9, s->nvsecond9, s->nosecond9, s->nv10, s->sv810, 
           s->sv910, s->nonconseq, s->eq3, s->bowties, s->no8sv10, s->no9sv10, 
           s->count0, s->draw0, s->normal1, s->normal2, s->normal4);
    printf("  badcount %ld  some degenerate %ld  all degenerate %ld  avgeq %f\n",
            s->badcount, s->ndegen, s->nalldegen, (float)s->toteq/(float)s->ndegen);
}

/* --------------------------------------------------------------------------
 *
 * har_initHexProjectionOptions
 *
 * --------------------------------------------------------------------------
 */
void
har_initHexProjectionOptions(har_HexProjectionOptions* options)
{
    options->test_degenerate = 0;
    options->tet_only = 0;
    options->save_crossed = 0;
    options->outlines = 0;
}


/* --------------------------------------------------------------------------
 *
 * poly_eqn
 *
 * --------------------------------------------------------------------------
 */
static int 
poly_eqn(int i0, int i1, int i2, int i3,
         double xv[], double yv[], double a[4], double b[4], double c[4])
{

    /* Compute line equations for the four sides of the quadrilateral
       with vertex indices i0, i1, i2, and i3, and store them in the
       arrays a, b, and c. The function ax + by +c should be negative
       for points inside the polygon. Return 0 if bowtie projection.   */

    int verts[7];
    int i, j, k, l, ii, jj, kk, ll, in[4];
    double dx, dy;

    verts[0] = i0;
    verts[1] = i1;
    verts[2] = i2;
    verts[3] = i3;
    verts[4] = i0;
    verts[5] = i1;
    verts[6] = i2;

    for (i = 0; i < 4; ++i) {
        j = i+1;
        k = i+2; 
        l = i+3;
        ii = verts[i];
        jj = verts[j];
        kk = verts[k];
        ll = verts[l];
        dx = xv[jj] - xv[ii];
        dy = yv[jj] - yv[ii];
        a[i] = -dy;
        b[i] =  dx;
        c[i] = -a[i]*xv[ii] - b[i]*yv[ii];
        in[i] = 0;
        if (a[i]*xv[kk] + b[i]*yv[kk] + c[i] < 0) in[i] = 1;
        if (a[i]*xv[ll] + b[i]*yv[ll] + c[i] < 0) ++in[i];
    }
    if (in[0] == 2 && in[1] == 2 && in[2] == 2 && in[3] == 2) return 1;
    if (in[0] == 0 && in[1] == 0 && in[2] == 0 && in[3] == 0) {
        for (i = 0; i < 4; ++i) {
            a[i] = -a[i];
            b[i] = -b[i];
            c[i] = -c[i];
            if(intrpr > 1) {
                printf("equation %d a %f b %f c %f\n", i, a[i], b[i], c[i]);
            }
        }
        return 1;
    }
    return 0;
}

/* --------------------------------------------------------------------------
 *
 * poly_test
 *
 * --------------------------------------------------------------------------
 */
static int 
poly_test(int i0, int i1, int i2, int i3,
          double xv[], double yv[], double a[4], double b[4], double c[4], 
          int in[4] )
{

    /* Count the number of vertices among i0, i1, i2, and i3 that lie
       inside the polygon with edge equation s given by a, b, and c.   */

    int verts[4], i, k, kk, total = 0;

    verts[0] = i0;
    verts[1] = i1;
    verts[2] = i2;
    verts[3] = i3;

    for (k = 0; k < 4; ++k) {
        kk = verts[k];
        in[k] = 1;
        for (i = 0; i < 4; ++i) {
            if (a[i]*xv[kk] + b[i]*yv[kk] + c[i] >= 0) {
                in[k] = 0;
                break;
            }
        }
        total += in[k];
    }
    assert(total >= 0);
    return total;
}

/* --------------------------------------------------------------------------
 *
 * makepp
 *
 * --------------------------------------------------------------------------
 */
static void 
makepp(int pp[VSIZE], int i0, int i1, int i2, int i3, int i4,
       int i5, int i6, int i7) 
{
    pp[0] = i0;
    pp[1] = i1;
    pp[2] = i2;
    pp[3] = i3;
    pp[4] = i4;
    pp[5] = i5;
    pp[6] = i6;
    pp[7] = i7;
    /* DLDL: In several places of the code, 12 vertices are assumed possible,
     *       but in other cases, VSIZE (14) vertices are possible.
     *       Here, I am adding a loop to set the additional indices to 
     *       bogus values, since I don't know which number to trust, and 
     *       it makes sense to set a globally defined size for these arrays.
     */
    {
        int j;
        for (j = 8; j < VSIZE; ++j) pp[j] = j;
    }

}
/* --------------------------------------------------------------------------
 *
 * intersect
 *
 * --------------------------------------------------------------------------
 */
/* DL: I think this routine intersects edges of a given hexahedron. 
 *     I have documented the parameters as best as I can figure them.
 */
static int 
intersect(
          /* INPUT ARGUMENTS */
          const har_Projection* proj,
          int use_abc,
          const har_Zone* hex,
          int v1, int v2,               /* first edge */
          int w1, int w2,               /* second edge */
          int out,                      /* index of [xyz]v to update with intersection */
          const double a[4],            /* coeff. of line equations of edges (poly_eqn)*/
          const double b[4],
          const double c[4],
          const int *pp,                /* I don't know what this is */
          const double rcol[],          /* vertex RGBA values */
          const double gcol[],
          const double bcol[],
          const double av[],

          /* INPUT/OUTPUT ARGUMENTS */
          double xv[], double yv[], double zv[], 
          double rfb[VSIZE][2], double gfb[VSIZE][2],
          double bfb[VSIZE][2], double afb[VSIZE][2], 
          double dst[VSIZE])
{

    /* Find the intersection of line segments v1-v2 and w1-w2, and put
       the result in vertex out, with interpolated colors and opacities. */

    double zvtempv, zvtempw, d1, d2, dx, dy, aa, bb, cc, u, ou, t, ot, e1, e2;

    assert(out >= 0);
    assert(out < VSIZE);

    if(use_abc) {
        d1 = xv[v1]*a[w1] + yv[v1]*b[w1] + c[w1];
        d2 = xv[v2]*a[w1] + yv[v2]*b[w1] + c[w1];
        if(intrpr) {
            dx = xv[pp[w2]] - xv[pp[w1]];
            dy = yv[pp[w2]] - yv[pp[w1]];
            aa = -dy;
            bb = dx;
            cc = -aa*xv[pp[w1]] - bb*yv[pp[w1]];
            e1 = xv[v1]*aa + yv[v1]*bb + cc;
            e2 = xv[v2]*aa + yv[v2]*bb + cc;
            printf("used_abc d1 %f d2 %f\n         e1 %f e2 %f\n",
                    d1, d2, e1, e2);
            printf("aa %f bb %f cc %f\n", aa, bb, cc);
            printf(" a %f  b %f  c %f\n", a[w1], b[w1], c[w1]);
        }
    }
    else {
        dx = xv[pp[w2]] - xv[pp[w1]];
        dy = yv[pp[w2]] - yv[pp[w1]];
        aa = -dy;
        bb = dx;
        cc = -aa*xv[pp[w1]] - bb*yv[pp[w1]];
        d1 = xv[v1]*aa + yv[v1]*bb + cc;
        d2 = xv[v2]*aa + yv[v2]*bb + cc;
    }
    t = d1/(d1 - d2);
    ot = 1. - t;
    xv[out] = t*xv[v2] +ot*xv[v1];
    yv[out] = t*yv[v2] +ot*yv[v1];
    /* Recall that it is assumed  in nelson's code that zv holds the value -1/z
     * as set in the projection section of the simple() routine in simple.c, or
     * the har_projectHex routine in this file.
     */
    zvtempv = t*zv[v2] +ot*zv[v1];

    dx = xv[v2] - xv[v1];
    dy = yv[v2] - yv[v1];
    aa = -dy;
    bb = dx;
    cc = -aa*xv[v1] - bb*yv[v1];
    d1 = xv[pp[w1]]*aa + yv[pp[w1]]*bb + cc;
    d2 = xv[pp[w2]]*aa + yv[pp[w2]]*bb + cc;
    u = d1/(d1 - d2);
    ou = 1. - u;
    if(intrpr) {
        // intrpr = 0;
        printf("t %f u %f d1 %g d2 %g\n", t, u, d1, d2);
        printf("xyv1 %f %f xyv2 %f %f\n", xv[v1], yv[v1], xv[v2], yv[v2]);
        printf("xyw1 %f %f xyw2 %f %f\n",
                xv[pp[w1]], yv[pp[w1]], xv[pp[w2]], yv[pp[w2]]);
        printf("xyv1 %f %f xyv2 %f %f\n", hex->x[v1], hex->y[v1], 
                                          hex->x[v2], hex->y[v2]);
        printf("xyw1 %f %f xyw2 %f %f\n", hex->x[pp[w1]], hex->y[pp[w1]], 
                                          hex->x[pp[w2]], hex->y[pp[w2]]);
        printf(" out %f %f   %f %f\n", xv[out], yv[out],
                t*hex->x[v2] + ot*hex->x[v1], t*hex->y[v2] + ot*hex->y[v1]);
        printf("out %d v1 %d %d %d v2 %d %d %d w1 %d %d %d w2 %d %d %d\n\n",
                out, v1, ppinv[v1], rrinv[v1], v2, ppinv[v2], rrinv[v2], 
                w1, pp[w1], rrinv[pp[w1]], w2, pp[w2], rrinv[pp[w2]]);
    }
    if(u < 0. || u > 1. || t < 0. || t > 1.) return 0;

    zvtempw = u*zv[pp[w2]] + ou*zv[pp[w1]];

    /* I guess we set to some known value, but we use the dst[] array for
     * the compositing anyway, so that should be correct.
     */
    zv[out] = .5*(zvtempv + zvtempw);
    if ( ipersp == 0) {
        /* in orthogonal case zv == camera frame z coordinates */
        dst[out] = zvtempv - zvtempw;
    } else {
#if USE_PROJECTION_ROUTINES
        Point3 closer, farther;
        assert(proj != NULL);
        closer.x = farther.x = xv[out];
        closer.y = farther.x = yv[out];
        /* We undo the inversion needed for screen space interpolation since
         * the distance computation requires the Z coordinates in detector space.
         * The original inversion was done in the hexProjection routine.
         */
        closer.z = PROJ_Z_INV(zvtempv);
        farther.z = PROJ_Z_INV(zvtempw);
        dst[out] = projPerspectiveCorrectDistance(proj, &closer, &farther);
        printf("\nintersect(): %8.4f %8.4f:  closer: %8.4f   farther %8.4f   distance: %8.4f\n\n",
                xv[out], yv[out], closer.z, farther.z, dst[out]);
#else
        /* Perspective correction */
        /* First compute difference in Z values (in camera space) of the two points. */
        dst[out] = -1./zvtempv + 1./zvtempw;

        /* We have two similar triangles:
         *
         *                             dz  . v
         *                             .     |
         *                         w----------
         *                         ^         ^
         *       d   .  x          zw        zv
         *       .      |
         *  E...........|
         *      D=1
         *
         *      d/D  = dz/(zw-zv)
         *      dz = d(zw-zv)/D
         *      d = sqrt(x^2 + y^2 + D^2)
         *
         *  Hence the multiplication factor below:
         *
         */
        dst[out] *= sqrt( ( (xv[out]-xdisp)*(xv[out]-xdisp) +
                    (yv[out]-ydisp)*(yv[out]-ydisp) )/(facbx*facby) + 1.);
        printf("intersect() old distance computation used.\n");
#endif
    }

    if (dst[out] < 0) dst[out] = -dst[out];
    printf("intersect: dst[%d] = %f\n", out, dst[out]);

    return 1;
}
/* --------------------------------------------------------------------------
 *
 * poly_eqn
 *
 * --------------------------------------------------------------------------
 */

/* Set vertex position, color, and texture coordinates. */
static void 
myglArrayElement(const har_Projection* proj,
                 int i, double cd[VSIZE][6], double xyzv[VSIZE][3], double dst[VSIZE]) 
{
    int mypr = 3, j;
    double mycd[VSIZE][6];

    /* RADIOGRAPHY:
     * cd[i][0..3]  are attenuation_PER_LENGTH coefficients of three energy groups
     *              (that is, density is already multiplied into them).
     * cd[i][4] is  geometric_distance.
     * dst[i] should be equal to cd[i][4]
     * So this loop computes the path lengths.
     */

    /* compute path length, either using dst or cd[i][4], check rest of
     * code in this file to make sure you are using the right one! 
     * We need to use the cd[i][4], since distance is not 'dst' for every vertex,
     * in fact, most vertices will have dst == 0 (thin-vertices!).
     */
    assert(dst[i] == cd[i][4]);
    for(j = 0; j < 4; ++j) mycd[i][j] = cd[i][j]*dst[i];
    if (mypr>1) {
        printf("dst [%4d] = %f\n", i, dst[i]);
        printf("cd  [%4d] = (%8.4e %8.4e %8.4e %8.4e %8.4e)\n", 
                i, cd[i][0],cd[i][1],cd[i][2],cd[i][3], cd[i][4]);
        printf("mycd[%4d] = (%8.4e %8.4e %8.4e %8.4e)\n", 
                i, mycd[i][0],mycd[i][1],mycd[i][2],mycd[i][3]);

    }

    glColor4dv(mycd[i]);
    // glColor4d(1., 1., 1., 1.);
    if(mypr > 1) {
        if (cd[i][0] < 0 || cd[i][1] < 0 || cd[i][2] < 0) {
            printf("-----------------\n");
            printf("         Color %f %f %f %f   dst %f\n",
                    mycd[i][0], mycd[i][1], mycd[i][2], mycd[i][3], dst[i]);
            printf("-----------------\n");
        }
    }

#if USE_PROJECTION_ROUTINES
    {
        GLdouble x, y;
        Point3 p;
        p.x = xyzv[i][0];
        p.y = xyzv[i][1];
        p.z = xyzv[i][2];

        /* DLDLD:  This is badly inaccurate, we need to retain floating
         *         point accuracy.  This can be fixed by simply leaving
         *         point in float (not truncating to integer).
         *         The regression test needs to be changed though.
         */
        projProjectedPointToPixel(proj, &p, &x, &y);
        glVertex2d(x,y);

        if(mypr) {
            printf("Vertex: %3f %3f 0\n", x,y);
        }
    }
#else
    glVertex3dv(xyzv[i]);
    if(mypr) {
        printf("Vertex %2d %f %f %f\n", 
               rrinv[i], xyzv[i][0], xyzv[i][1], xyzv[i][2]);
    }
    if(mypr) fflush(stdout);
#endif
}

static int 
compar(const void *ip, const void *jp) 
{
    const int* i = (const int*) ip;
    const int* j = (const int*) jp;
    double d;

    d = ang[*j] - ang[*i];
    if(d < 0) return -1;
    else if (d == 0) return 0;
    return 1;
}
/* --------------------------------------------------------------------------
 *
 * poly_eqn
 *
 * --------------------------------------------------------------------------
 */

/* This is only used in some code that has been commented out.
static const int s_diagonal[4] = {2, 3, 0, 1};
*/
static const int s_opposite[6] = {1, 0, 4, 5, 2, 3};

/* atten_per_length is   density * mass_attenuation.
 * So, path length is simply atten_per_length * distance.
 */
int
har_hexProject (
        const har_Projection* proj,
        const har_Zone* hex, int drawnow, char *crossed_count_turn,
        char *save_in, int *lineword, const double atten_per_length[4],
        const har_HexProjectionOptions* options, har_HexProjectionStatistics* stats)
{

    /* If hexahedron has a simple enough projection, generate triangle
       fans for it; otherwise return error code.                    */

    double dst[VSIZE], cd[VSIZE][6], xyzv[VSIZE][3];  
    int i, j, k, l, m, n, m1, imax, i0, i1, i2, i3, j1, j2, turn,
        in[4], crossed, count, ivert[2][3], v, vsmall, jsmall,
        ii, jj, kk, ll, fbint, fbkk, istop, loop, cross, nverts, 
        tcount, qq[VSIZE], qqinv[VSIZE], dummy[4], ivview = 0, extraview = 0,
        tt[VSIZE], uu[8];
    double av[VSIZE], xv[VSIZE], topy, boty,
           yv[VSIZE], zv[VSIZE], rcol[VSIZE], gcol[VSIZE], bcol[VSIZE];
    double a[4], b[4], c[4]; 
    /* DLDL: The following arrays were defined with size [10][2], but this is a 
       mismatch with the [rgb]col arrays above, defined as [VSIZE].
       So, I am making these VSIZE as well.
       */
    double rfb[VSIZE][2], gfb[VSIZE][2], bfb[VSIZE][2], afb[VSIZE][2];
    double dz[3], da, db, dc, opac, opmax, dmax, dfac, fdist;
    double dx[3], dy[3],  ztemp, cgx, cgy,
           det; 
    int use_abc, mmpr = 3, mypr = 3, initial_bowtie_test = 1;
    int second8, second9, num_vertices1, num_vertices2;
    static int firstno8sv10 = 1, firstbad = 1;
    float dens;
    int im, ip, itop, ibot, om, op, otop, sign, nleft, nright, ileft[8],
        iright[8], shift, lineverts;

    assert(hex->type == HAR_ZONETYPE_HEX);
    assert(stats != NULL);

    if (mypr) printf("\n--------------------------------------\n"
                     " har_hexProject:   BEGIN.\n"
                     "--------------------------------------\n");

    ++(stats->totalcount);
    dens = 1.;
    if (intrpr) printf("\ncall %ld\n", stats->totalcount);
    if(drawnow && ivview) {
        if ((fp = fopen("simple.iv","a")) == 0) {
            fprintf(stderr,"reader() unable to open FILE %s\n","simple.iv");
            ivview = 0;
        }
        viewcell(fp, hex, stats);
        fclose(fp);
    }

    xvmin = 100000.;
    xvmax = -100000.;
    yvmin = 100000.;
    yvmax = -100000.;

    if (mypr) printf("atten_per_length: %8.4e %8.4e %8.4e\n", atten_per_length[0], atten_per_length[1], atten_per_length[2]); 

#if USE_PROJECTION_ROUTINES

    /* This code assumes that zone is now already in detector plane
     * coordinates. That is, we have already transformed from the 
     * detector coordinate system to the detector plane, either by
     * orthographic or perspective projection.
     */
    for (i = 0; i < 8; ++i) {
        xv[i] = hex->x[i];
        yv[i] = hex->y[i];
        zv[i] = hex->z[i];
    }

#if 0
    projPrintProjection(*proj);

    for (i = 0; i < 8; ++i) {
        Point3 p, q;
        p.x = hex->x[i];
        p.y = hex->y[i];
        p.z = hex->z[i];
        projProjectPoint(proj, &p, &q);
        xv[i] = q.x;
        yv[i] = q.y;
        /* The routines in hex.c and splat.c expect a z coordinate that can be
         * interpolated over the detector (that is, in screen space).
         */
        if (q.z > 0.0) {
            if (q.z >= proj->distance) return HEX_FAIL_CLIPPED_BEHIND_SOURCE;
            zv[i] = PROJ_Z_INV(q.z);
        } else {
            return HEX_FAIL_CLIPPED_BEHIND_DETECTOR;
        }
    }
#endif

#else

    /* Transform vertices to screen space 
     * The coordinate system assumed in this code, and in the setup code in
     * testsilo.c is as follows:
     *
     * The origin of the camera system is at the eye point.
     * The screen is located 1 unit from the eye point along the
     * positive Z axis.  This can be changed by changing the vpdist
     * global variable.
     *
     * Thus, the transform is:  x' = x * vpdist / z
     * Below, vpdist is multiplied into facbx and facby already.
     *
     */
    for (i = 0; i < 8; ++i) {
        if (ipersp == 0) {
            /* ax + b:  linearly transform orthographically projected vertices
             *          (projected along z axis onto xy detector plane.
             */
            xv[i] = facbx*hex->x[i] + xdisp;
            yv[i] = facby*hex->y[i] + ydisp;
            zv[i] = hex->z[i];
        } else {
            /* perspective: facbx facby have vpdist factored in.  It is one, but
             * that completes the perspective transform:
             *    projected_v = distance_to_screen * camera_v / world_v.
             *
             * The following is equivelent to performaing the perspective
             * projection:   xv = hex-x[i] * vpdist / hex->z[i]
             * Then translating:  xv * facbx' + xdisp, where
             * facbx' = facbx / vpdist.
             *
             * zv is the homogeneous coordinates.  It is negated to keep
             * it monotonic (farther from screen gives larger values).
             */
            xv[i] = facbx*hex->x[i]/hex->z[i] + xdisp;
            yv[i] = facby*hex->y[i]/hex->z[i] + ydisp;
            zv[i] = -1.0 / hex->z[i];
        }
    }

#endif /* if USE_PROJECTION_ROUTINES */

    /* compute detector plane bounding box of projection, and intiailize the
     * arrays that get used by the tet projection code should this hex need
     * to be subdivided.  DLDLDL:  this means that 'vv' couples the tet
     * routine to the hex routine, which should be changed.
     */
    for (i = 0; i < 8; ++i) {
        if(xv[i] > xvmax) xvmax = xv[i];
        if(xv[i] < xvmin) xvmin = xv[i];
        if(yv[i] > yvmax) yvmax = yv[i];
        if(yv[i] < yvmin) yvmin = yv[i];

        /* vv: homogenous screen coordinates, we already did the perspective
         *     divide above. These are used by the tet projection code.
         *     The tet code does not use the additional edge intersections
         *     computed in this routine, so only the 8 initial hex nodes
         *     are stored here. 
         */
         /*
        vv[i][0] = xv[i];
        vv[i][1] = yv[i];
        vv[i][2] = 1.;
        */

        if(mypr) {
            printf("projected vertex %8d %8.4f %8.4f %8.4f\n", i, xv[i], yv[i], zv[i]);
            /* These should always be the same for all verts, as long as the
             * code above continues to initialize them this way.
             printf("            rgba %8.4f %8.4f %8.4f %8.4f\n", rcol[i], gcol[i], bcol[i], av[i]);
             */
        }
    }

    /* Initialize arrays for eventual projection with additional vertices
     * introduced by intersecting projected edges.
     */
    for (i = 0; i < VSIZE; ++i) {
        rfb[i][0] = rfb[i][1] = rcol[i] = atten_per_length[0];
        gfb[i][0] = gfb[i][1] = gcol[i] = atten_per_length[1];
        bfb[i][0] = bfb[i][1] = bcol[i] = atten_per_length[2];
        afb[i][0] = afb[i][1] = av[i]   = atten_per_length[3];

        /* all distances start out as zero */
        dst[i] = 0;

        /* zero out the space for intersection vertices (computed later) */
        if (i >= 8) {
            xv[i] = 0.0;
            yv[i] = 0.0;
        }
    }

    if(options->test_degenerate) {
        int is_degenerate = 0;
        int counteq = 0;
        assert(hex->type == HAR_ZONETYPE_HEX);
        for(i = 1; i < 8; ++i) {
            // if(degenerate) break;
            for(j = 0; j < i; ++j) {
                if(xv[i] == xv[j] && yv[i] == yv[j] && zv[i] == zv[j]) {
                    is_degenerate = 1;
                    ++counteq;
                }
            }
        }
        stats->toteq += counteq;
        if (is_degenerate) {
            if(counteq == 28) {
                /* hex is collapsed to point */
                ++(stats->nalldegen);
                return HEX_FAIL_DEGENERATE_TO_SINGLE_POINT;
            } else {
                ++(stats->ndegen);
                return HEX_FAIL_DEGENERATE;
            }
        }
    }

    /* -------------------------------------------------
     * Test for bowtie hex; also discovers non-convex quadrilaterals.
     * -------------------------------------------------
     */

    if((drawnow == 0 || drawnow == 2) && initial_bowtie_test) {
        if (!poly_eqn(0, 1, 2, 3, xv, yv, a, b, c) ||
                !poly_eqn(4, 5, 6, 7, xv, yv, a, b, c) ||
                !poly_eqn(1, 5, 6, 2, xv, yv, a, b, c) ||
                !poly_eqn(0, 4, 7, 3, xv, yv, a, b, c) ||
                !poly_eqn(0, 4, 5, 1, xv, yv, a, b, c) ||
                !poly_eqn(3, 7, 6, 2, xv, yv, a, b, c) ) {
            ++(stats->bowties);
            return HEX_FAIL_BOWTIE;
        }
    }

    if (drawnow == 0 || drawnow == 2 || !options->save_crossed) {
        int bow = 0;
        if (poly_eqn(0, 1, 2, 3, xv, yv, a, b, c)) {
            count = poly_test(4, 5, 6, 7, xv, yv, a, b, c, in);
            if(count > 0) {
                crossed = 0;
                makepp(pp, 0, 1, 2, 3, 4, 5, 6, 7);
                goto done;
            }
        }
        else bow = 1;
        if (poly_eqn(4, 5, 6, 7, xv, yv, a, b, c)) {
            count = poly_test(0, 1, 2, 3, xv, yv, a, b, c, in);
            if(count > 0) {
                crossed = 1;
                makepp(pp, 4, 5, 6, 7, 0, 1, 2, 3);
                goto done;
            }
        }
        else bow = 1;
        if (poly_eqn(0, 4, 5, 1, xv, yv, a, b, c)) {
            count = poly_test(3, 7, 6, 2, xv, yv, a, b, c, in);
            if(count > 0) {
                crossed = 2;
                makepp(pp, 0, 4, 5, 1, 3, 7, 6, 2);
                goto done;
            }
        }
        else bow = 1;
        if (poly_eqn(1, 5, 6, 2, xv, yv, a, b, c)) {
            count = poly_test(0, 4, 7, 3, xv, yv, a, b, c, in);
            if(count > 0) {
                crossed = 3;
                makepp(pp, 1, 5, 6, 2, 0, 4, 7, 3);
                goto done;
            }
        }
        else bow = 1;
        if (poly_eqn(3, 7, 6, 2, xv, yv, a, b, c)) {
            count = poly_test(0, 4, 5, 1, xv, yv, a, b, c, in);
            if(count > 0) {
                crossed = 4;
                makepp(pp, 3, 7, 6, 2, 0, 4, 5, 1);
                goto done;
            }
        }
        else bow = 1;
        if (poly_eqn(0, 4, 7, 3, xv, yv, a, b, c)) {
            count = poly_test(1, 5, 6, 2, xv, yv, a, b, c, in);
            if(count > 0) {
                crossed = 5;
                makepp(pp, 0, 4, 7, 3, 1, 5, 6, 2);
                goto done;
            }
        }
        else bow = 1;
        crossed = 0;
        makepp(pp, 0, 1, 2, 3, 4, 5, 6, 7);
        if(bow) {
            ++(stats->bowties);
            return HEX_FAIL_BOWTIE;
        }

        if (0 && drawnow && ivview) fclose((FILE*)fp);
        /*
           ++(stats->badcount);
           if (firstbad) {
           firstbad = 0;
           if (1) {
           if ((fp = fopen("simple.iv","a")) == 0) 
           fprintf(stderr,
           "reader() unable to open FILE %s\n","simple.iv");
           else {
           viewcell(fp, hex, stats);
           fclose((FILE*)fp);
           }
           if(0) exit(-1);
           }
           }
           return 0;
         */

done:
        if(count == 3)  {
            if (drawnow && ivview) fclose((FILE*)fp);
            if(mmpr) printf("count == 3 in simple.c\n");
            ++(stats->eq3);
            return HEX_FAIL_EQ3;
        }

        for (i = 3; i >= 0; --i) {
            if(in[i]) {
                imax = i;
                break;
            }
        }
        if (imax == 3 && in[0] == 1) imax = 0;
        turn = 3 - imax;

        *crossed_count_turn = 20*crossed + 4*count + turn;
        *save_in = 8*in[3] + 4*in[2] + 2*in[1] + in[0];
    } else {
        crossed = *crossed_count_turn/20;
        tcount = *crossed_count_turn - 20*crossed;
        count = tcount/4;
        turn = tcount - count*4;
        in[3] = *save_in/8;
        in[2] = (*save_in%8)/4;
        in[2] = (*save_in%4)/2;
        in[0] = *save_in%2;
        switch(crossed) {
            case 0:
                makepp(pp, 0, 1, 2, 3, 4, 5, 6, 7);
                break;
            case 1:
                makepp(pp, 4, 5, 6, 7, 0, 1, 2, 3);
                break;
            case 2:
                makepp(pp, 0, 4, 5, 1, 3, 7, 6, 2);
                break;
            case 3:
                makepp(pp, 1, 5, 6, 2, 0, 4, 7, 3);
                break;
            case 4:
                makepp(pp, 3, 7, 6, 2, 0, 4, 5, 1);
                break;
            case 5:
                makepp(pp, 0, 4, 7, 3, 1, 5, 6, 2);
                break;
            default:
                printf("bad crossed = %d in simple's switch statement\n", crossed);
                exit(EXIT_FAILURE);
        }
    } /* end if (drawnow == 0 || drawnow == 2 || !options->save_crossed) */


    /*  pp takes the standard picture to the actual one. In the standard
        picture, the face known to contain an interior vertex is face 0123.
        Define a rotation qq to make this interior vertex be vertex 7,
        and then let rr be the product of qq followed by pp.             */

    for (i = 0; i < 4; ++i) {
        qq[i] = (i + 4 - turn)%4;
        qq[i+4] = qq[i] + 4;
        qqinv[(i + 4 - turn)%4] = i;
        qqinv[qq[i] + 4] = i + 4;
    }
    for (i = 0; i < 8; ++i) rr[i] = pp[qq[i]];
    rr[8] = 8;
    rr[9] = 9;
    rr[10] = 10;
    rr[11] = 11;
    for (i = 0; i < VSIZE; ++i) {
        rrinv[rr[i]] = i;
        ppinv[pp[i]] = i;
    }

    if(count == 2) {
        j1 = imax - 1;
        if (j1 < 0) j1 = 3;
        if (in[j1] == 0) {
            if(mmpr > 1) {
                printf("Intersecting edges not consecutive in simple.c.\n");
            }
            ++(stats->nonconseq);
            if (drawnow && ivview) fclose((FILE*)fp);
            return HEX_FAIL_INTERSECTING_EDGES_NOT_CONSECUTIVE;
        }
    }

    i0 = imax;
    i1 = imax+1;
    if(i1 > 3) i1 = 0;
    i2 = i1 + 1;
    if(i2 > 3) i2 = 0;
    badexit = 0;
    second8 = 0;
    second9 = 0;
    second10 = 0;
    // use_abc = 1; // This should work but it doesn't.
    use_abc = 0;
    if (drawnow == 1) use_abc = 0;

    if (count == 4) ++(stats->normal4);

    /* At this point I think we are computing intersections between
     * edges to figure out the additional vertices that need to be
     * used in the triangle fans.
     */
    if(count == 1 || count == 2) {
        nverts = 10;
        if ( !intersect(proj, use_abc, hex, pp[i0 + 4], pp[i1 + 4], i1, i2, 8, a, b, c, pp, 
                    rcol, gcol, bcol, av, xv, yv, zv, rfb, gfb, bfb, afb, dst) ) {
            if(mmpr > 1) {
                printf("No vertex 8  edge intersection in simple.c. %d %d %d %d\n",
                        pp[i0 + 4], pp[i1 + 4], i1, i2);
            }
            ++(stats->nv8);
            badexit = 1;
            i3 = i2 + 1;
            if(i3 > 3) i3 = 0;
            if ( !intersect(proj, use_abc, hex, pp[i0 + 4], pp[i1 + 4], i2, i3, 8, a,b,c,pp,
                        rcol, gcol, bcol, av, xv, yv, zv, rfb, gfb, bfb, afb, dst)){
                if(mmpr > 1) {
                    printf("No second 8  edge intersection in simple.c. %d %d %d %d\n",
                            pp[i0 + 4], pp[i1 + 4], i2, i3);
                }
                if (ivview && drawnow) {
                    if ((fp = fopen("simple.iv","a")) == 0) {
                        fprintf(stderr,
                                "reader() unable to open FILE %s\n","simple.iv");
                    } else {
                        viewcell(fp, hex, stats);
                        fclose((FILE*)fp);
                    }
                }
                ++(stats->nvsecond8);
                return HEX_FAIL_NO_SECOND_CASE8_EDGE_INTERSECTION;
            }
            second8 = 1;
            if (!intersect(proj, 0, hex, pp[i0 + 4], pp[i1 + 4], i2, i2+4, 10,a,b,c,pp,
                        rcol,gcol,bcol,av,xv,yv,zv, rfb, gfb, bfb, afb, dst) ) {
                if(mmpr > 1) {
                    printf("No 8 vertex 10 edge intersection in simple.c.\n");
                }
                ++(stats->nv10);
                //	    if(!intersect(proj, 0, pp[i0], pp[i0 + 4], i1, i2, 10, a,b,c,pp,
                if(!intersect(proj, 0, hex, pp[i1], pp[i1 + 4], i2, i3, 10, a,b,c,pp,
                            rcol, gcol, bcol, av, xv, yv, zv, rfb,gfb,bfb,afb,dst) ) {
                    printf("no second try for case 8 vertex 10 intersection.\n");
                    ++(stats->no8sv10);
                    if(firstno8sv10) {
                        firstno8sv10 = 0;
                        printf("i0 %d i1 %d i2 %d turn %d\n",
                                rrinv[pp[i0]], rrinv[pp[i1]], rrinv[pp[i2]], turn);
                        if ((fp = fopen("simple.iv","a")) == 0) {
                            fprintf(stderr,
                                    "reader() unable to open FILE %s\n","simple.iv");
                        } else {
                            viewcell(fp, hex, stats);
                            fclose((FILE*)fp);
                        }
                    }
                    return HEX_FAIL_NO_SECOND_TRY_FOR_CASE8_VERTEX10_INTERSECTION;
                } else {
                    second10 = 1;
                    ++(stats->sv810);
                }
                if (drawnow && ivview) {
                    if ((fp = fopen("simple.iv","a")) == 0) {
                        fprintf(stderr,
                                "reader() unable to open FILE %s\n","simple.iv");
                    } else {
                        viewcell(fp, hex, stats);
                        fclose((FILE*)fp);
                    }
                }
            }
            nverts = 11;
            if (drawnow && ivview) {
                if ((fp = fopen("simple.iv","a")) == 0) {
                    fprintf(stderr,
                            "reader() unable to open FILE %s\n","simple.iv");
                } else {
                    viewcell(fp,hex, stats);
                    // fclose((FILE*)fp);
                }
            }
        }

        /* DLDL: what does this do? */
        if (count == 1) {
            j1 = i0;
            i0 = i1;
            i1 = i2;
            i2 = i1 + 1;
            if(i2 > 3) i2 = 0;
        }
        j2 = j1 - 1;
        if (j2 < 0) j2 = 3;
        if (!intersect(proj, use_abc, hex, pp[j1 + 4], pp[j2 + 4], i1, i2, 9, a, b, c, pp,
                    rcol, gcol, bcol, av, xv, yv, zv, rfb, gfb, bfb, afb, dst) ) {
            if(mmpr > 1) {
                printf("No vertex 9  edge intersection in simple.c.\n");
            }
            ++(stats->nv9);
            if (!intersect(proj, use_abc, hex, pp[j1 + 4], pp[j2 + 4], i0, i1, 9, a,b,c, pp,
                        rcol, gcol, bcol, av, xv, yv, zv, rfb, gfb, bfb, afb,dst)) {
                ++(stats->nosecond9);
                if(mmpr > 1) {
                    printf("%ld No vertex second try 9 edge "
                           "intersection %d %d %d %d\n", (stats->nosecond9), 
                           qqinv[j1 + 4], qqinv[j2 + 4], qqinv[i0], qqinv[i1]);
                           /* nosecond9, pp[j1 + 4], pp[j2 + 4], i0, i1); */
                }
                if (1) {
                    if ((fp = fopen("simple.iv","a")) == 0) {
                        fprintf(stderr,
                                "reader() unable to open FILE %s\n","simple.iv");
                    } else {
                        viewcell(fp,hex, stats);
                        fclose((FILE*)fp);
                    }
                    if(1) exit(EXIT_FAILURE);
                }
                return HEX_FAIL_NO_SECOND_TRY_FOR_CASE9_EDGE;
            }
            second9 = 1;
            if(second8) {
                if(mmpr) printf("Both second8 and second9 used in simple.\n");
                return HEX_FAIL_BOTH_SECOND8_AND_SECOND9_USED;
            }
            ++(stats->nvsecond9);
            if(0) printf("Second 9 try %ld found %d %d %d %d\n",
                    (stats->nvsecond9), qqinv[j1 + 4], qqinv[j2 + 4], qqinv[i0], qqinv[i1]);
            if((stats->nvsecond9) == -130) {
                intrpr = 1;
            }
            if(0 && drawnow) intrpr = 1;
            if(!intersect(proj, 0, hex, pp[j1 + 4], pp[j2 + 4], i1, i1+4, 10, a,b,c,pp,
                        rcol, gcol, bcol, av, xv, yv, zv, rfb,gfb,bfb,afb,dst) ) {
                if(mmpr > 0) {
                    printf("No 9 vertex 10 edge intersection in "
                           "simple.c call %ld.\n", (stats->totalcount));
                }
                ++(stats->nv10);
                if(!intersect(proj, 0, hex, pp[j2], pp[j2 + 4], i1, i0, 10, a,b,c,pp,
                            rcol, gcol, bcol, av, xv, yv, zv, rfb,gfb,bfb,afb,dst) ) {
                    printf("no second try for case 9 vertex 10 intersection.\n");
                    if (0) {
                        if ((fp = fopen("simple.iv","a")) == 0) {
                            fprintf(stderr,
                                    "reader() unable to open FILE %s\n","simple.iv");
                        } else {
                            viewcell(fp, hex, stats);
                            fclose((FILE*)fp);
                        }
                        if(0) exit(EXIT_FAILURE);
                    }
                    ++(stats->no9sv10);
                    return HEX_FAIL_NO_SECOND_TRY_FOR_CASE9_VERTEX10_INTERSECTION;
                }
                else
                    if(0)
                        printf("second try for case 9 vertex 10 intersection.\n");
                second10 = 1;
                ++(stats->sv910);
                if (0 && drawnow) {
                    if ((fp = fopen("simple.iv","a")) == 0) 
                        fprintf(stderr,
                                "reader() unable to open FILE %s\n","simple.iv");
                    else {
                        viewcell(fp, hex, stats);
                        fclose((FILE*)fp);
                    }
                    if(0) exit(EXIT_FAILURE);
                }
            }
            nverts = 11;
            if (drawnow && ivview) {
                if ((fp = fopen("simple.iv","a")) == 0) 
                    fprintf(stderr,
                            "reader() unable to open FILE %s\n","simple.iv");
                else {
                    viewcell(fp, hex, stats);
                    /* fclose((FILE*)fp); */
                }
            }
            if (drawnow && ivview) {
                viewcell(fp, hex, stats);
                /* fclose((FILE*)fp); */
            }
        }
        if(mypr) {
            printf(".... intersections of projected edges ....\n");
            printf("projected vertex %8d %8.4f %8.4f %8.4f\n", 8, xv[8], yv[8], zv[8]);
            printf("projected vertex %8d %8.4f %8.4f %8.4f\n", 9, xv[9], yv[9], zv[9]);
        }
        if(count == 2 && (second8 || second9) ) {
            if(mmpr) printf("count == 2 && (second8 || second9)\n");
            return HEX_FAIL_COUNT2_AND_SECOND8_OR_SECOND9;
        }
        if(second8 == 0 && second9 == 0) {
            if(count == 1) ++(stats->normal1);
            if(count == 2) ++(stats->normal2);
        }
    } else if(count == 4) {
        nverts = 8;
    } else if(count == 0) { // convex octagon case, find counterclockwise order
        // about center of gravity.
        nverts = 12;  /* DLDL: Perhaps 12 is the maximum after all.  Then why
                       *       is VSIZE == 14?
                       */
        cgx = 0;
        cgy = 0;
        for(i = 0; i < 8; ++i) {
            cgx += xv[i];
            cgy += yv[i];
        }
        cgx /= 8.;
        cgy /= 8.;
        for (i = 0; i < 8; ++i)  ang[i] = atan2(yv[i] - cgy, xv[i] - cgx);
        for (i = 0; i < VSIZE; ++i) tt[i] = i;
        qsort(tt, 8, sizeof(int), compar); 
        for (i = 0; i < 8; ++i) {
            if(tt[i] == 0) break;
        }
        for (j = 0; j < 8; ++j) uu[j] = tt[(j + i)%8];
        if(0) {
            for (i = 0; i < 8; ++i) printf("uu[%d] %d\n", i, uu[i]);
        }
        if (uu[3] == 1 || uu[3] == 3 || uu[3] == 4) {
            /* DLDL: This is probably a bug? Note the comparison in stead of
             * assignment.  I have changed it to assignment below.
            for(j = 0; j < 8; ++j) tt[j] == uu[j];
             */
            for(j = 0; j < 8; ++j) tt[j] = uu[j];
        } else if (uu[5] == 1 || uu[5] == 3 || uu[5] == 4) {
            /* Again, this should probably be assignment instead of comparison. 
            for(j = 0; j < 8; ++j) tt[j] == uu[(8 - j)%8];
            */
            for(j = 0; j < 8; ++j) tt[j] = uu[(8 - j)%8];
        } else {
            printf("neither predicted octagon diagonal found from vertex 0\n");
            for(i = 0; i < 8; ++i) {
                printf("i %d  uu %d  tt %d  ", i, uu[i], tt[i]);
            }
            printf("call %ld\n", (stats->totalcount));
            ++(stats->badcount);
            if (firstbad) {
                firstbad = 0;
                if ((fp = fopen("simple.iv","a")) == 0) {
                    fprintf(stderr,"reader() unable to open FILE %s\n","simple.iv");
                    return 0;
                }
                viewcell(fp, hex, stats);
                fclose((FILE*)fp);
            }
            return HEX_FAIL_NEITHER_OCTAGON_DIAGONAL_FOUND_FOR_VERT0;
        }
        if (!intersect(proj, 0, hex, tt[0], tt[3], 1, 6, 8, a, b, c, tt,
                    rcol, gcol, bcol, av, xv, yv, zv, rfb, gfb, bfb, afb, dst) ) {
            printf("missed 0-3, 1-6 intersection for octagon\n");
            ++(stats->badcount);
            for(i = 0; i < 8; ++i) {
                printf("i %d  uu %d  tt %d\n", i, uu[i], tt[i]);
            }
            printf("call %ld\n", (stats->totalcount));
            ++(stats->badcount);
            if (firstbad) {
                firstbad = 0;
                if ((fp = fopen("simple.iv","a")) == 0) {
                    fprintf(stderr,"reader() unable to open FILE %s\n","simple.iv");
                    return 0;
                }
                viewcell(fp, hex, stats);
                fclose((FILE*)fp);
            }
            return HEX_FAIL_MISSED_0_3_AND_1_6_INTERSECTIONS_FOR_OCTAGON;
        }
        if (!intersect(proj, 0, hex, tt[2], tt[5], 3, 0, 9, a, b, c, tt,
                    rcol, gcol, bcol, av, xv, yv, zv, rfb, gfb, bfb, afb, dst) ) {
            printf("missed 2-5, 3-0 intersection for octagon\n");
            ++(stats->badcount);
            return HEX_FAIL_MISSED_2_5_AND_3_0_INTERSECTIONS_FOR_OCTAGON;
        }
        if (!intersect(proj, 0, hex, tt[2], tt[5], 4, 7, 10, a, b, c, tt,
                    rcol, gcol, bcol, av, xv, yv, zv, rfb, gfb, bfb, afb, dst) ) {
            printf("missed 2-5, 4-7 intersection for octagon\n");
            ++(stats->badcount);
            return HEX_FAIL_MISSED_2_5_AND_4_7_INTERSECTIONS_FOR_OCTAGON;
        }
        if (!intersect(proj, 0, hex, tt[1], tt[6], 4, 7, 11, a, b, c, tt,
                    rcol, gcol, bcol, av, xv, yv, zv, rfb, gfb, bfb, afb, dst) ) {
            printf("missed 1-6, 4-7 intersection for octagon\n");
            ++(stats->badcount);
            return HEX_FAIL_MISSED_1_6_AND_4_7_INTERSECTIONS_FOR_OCTAGON;
        }
        ++(stats->count0);
        if(drawnow) {
            if(0) {
                if ((fp = fopen("simple.iv","a")) == 0) {
                    fprintf(stderr,"reader() unable to open FILE %s\n","simple.iv");
                    return 1;
                }
                viewcell(fp, hex, stats);
                fclose((FILE*)fp);
            }
        }
    }
    else
        printf("count = 3 in drawnow call to simple.c\n");

    if(0 && ivview && (stats->totalcount) < 10) {
        printf("ivview for good cell\n");
        fprintf((FILE*)fp,"#Inventor V2.0 ascii\n");
        fprintf((FILE*)fp,"DirectionalLight {on TRUE intensity  .8\n");
        fprintf((FILE*)fp,"    color   1 1 1  direction  0 1 -1 }\n");
        fprintf((FILE*)fp,"DirectionalLight {on TRUE intensity  .8\n");
        fprintf((FILE*)fp,"    color   1 1 1  direction  0 -1 1 }\n");
        fprintf((FILE*)fp,"Environment { ambientIntensity  1.1 \n");
        fprintf((FILE*)fp,"   ambientColor      1 1 1}\n");
        fprintf((FILE*)fp,"Separator {\n");
        fprintf((FILE*)fp,"DrawStyle { style LINES }\n");
        fprintf((FILE*)fp,"Material { ambientColor 0 1 1\n");
        fprintf((FILE*)fp,"           diffuseColor 0 0 0 }\n");
        fprintf((FILE*)fp, "Coordinate3 {  point [\n"); 
        for (i = 0; i < 8; ++i) {
            fprintf((FILE*)fp, "%f %f %f,  # %d\n", 
                    hex->x[i], hex->y[i], hex->z[i], i);
        }
        fprintf((FILE*)fp, "   ] }\n");
        fprintf((FILE*)fp, "IndexedFaceSet { coordIndex [\n");

        for (i = 0; i < 6; ++i) {
            for (j = 0; j < 4; ++j) {
                l = faceverts[i][j];
                fprintf((FILE*)fp,"%d,", l);
            }
            fprintf((FILE*)fp,"%d,\n", -1);
        }
        fprintf((FILE*)fp, "] } }\n");
    }
    if(options->outlines && !drawnow) {
        if(count == 1 && !second10) {
            lineverts = 6;
            outline[0] = rr[0];
            outline[1] = rr[3];
            outline[2] = rr[2];
            outline[3] = rr[6];
            outline[4] = rr[5];
            outline[5] = rr[4];
        }
        else if (count == 1 && second8) {
            lineverts = 6;
            outline[0] = rr[0];
            outline[1] = rr[3];
            outline[2] = rr[2];
            outline[3] = rr[6];
            outline[4] = rr[5];
            outline[5] = rr[1];
        }
        else if (count == 1 && second9) {
            lineverts = 6;
            outline[0] = rr[0];
            outline[1] = rr[3];
            outline[2] = rr[2];
            outline[3] = rr[1];
            outline[4] = rr[5];
            outline[5] = rr[4];
        }
        else if (count == 2) {
            lineverts = 6;
            outline[0] = rr[0];
            outline[1] = rr[3];
            outline[2] = rr[2];
            outline[3] = rr[1];
            outline[4] = rr[5];
            outline[5] = rr[4];
        } else if (count == 4) {
            lineverts = 4;
            outline[0] = rr[0];
            outline[1] = rr[3];
            outline[2] = rr[2];
            outline[3] = rr[1];
        } else if (count == 0) {
            lineverts = 8;
            for (i = 0; i < 8; ++i)
                outline[i] = tt[i];
        } else printf("bad outline case\n");

        // find top and bottom vertices

        topy = -100000.;
        boty =  100000.;
        for (i = 0; i < lineverts; ++i) {
            if(yv[outline[i]] > topy) {
                topy = yv[outline[i]];
                itop = i;
            }
            if(yv[outline[i]] < boty) {
                boty = yv[outline[i]];
                ibot = i;
            }
        }
        if(topy == boty) return HEX_FAIL_TOP_AND_BOTTOM_VERTS_IDENTICAL;
        otop = outline[itop];

        // find which side is left side

        ip = (itop + 1) % lineverts;
        if (ip < 0) ip = lineverts + ip;
        while (yv[outline[ip]] == yv[otop]) {
            ip = (ip + 1) % lineverts;
            if (ip < 0) ip = lineverts + ip;
        }
        im = (itop - 1) % lineverts;
        if (im < 0) im = lineverts + im;
        while (yv[outline[im]] == yv[otop]) {
            im = (im - 1) % lineverts;
            if (im < 0) im = lineverts + im;
        }
        op = outline[ip];
        om = outline[im];
        if( (xv[op] - xv[otop]) / (yv[op] - yv[otop]) <
                (xv[om] - xv[otop]) / (yv[om] - yv[otop]) ) {
            sign = 1;
        } else {
            sign = -1;
        }

        // left side includes top and bottom

        nleft = 1;
        ileft[0] = otop;
        im = itop;
        for (i = 0; i < lineverts; ++i) {
            im = (im - sign) % lineverts;
            if (im < 0) im = lineverts + im;
            ileft[nleft] = outline[im];
            ++nleft;
            if(im == ibot) break;
        }

        // right side includes neither top nor bottom

        nright = 0;
        ip = (itop + sign) % lineverts;
        if (ip < 0) ip = lineverts + ip;
        while (ip != ibot) {
            iright[nright] = outline[ip];
            ++nright;
            ip = (ip + sign) % lineverts;
            if (ip < 0) ip = lineverts + ip;
        }
        if (nleft + nright != lineverts) {
            printf("nleft %d + nright %d != lineverts %d\n",
                    nleft, nright, lineverts);
        }

        // build lineword in octal "bytes", starting with lowest order bits

        *lineword = nleft + 8*nright;
        shift = 6;
        for (i = 0; i < nleft; ++i) {
            *lineword += ileft[i] << shift;
            shift += 3;
        }
        for (i = 0; i < nright; ++i) {
            *lineword += iright[i] << shift;
            shift += 3;
        }
    }
       
       // early return if called only to see if hexahedron can be successfully drawn
       // without subdivision into tetrahedra

    if(drawnow == 0) return HEX_SUCCESS;
    loop = (count == 1) ? 2 : 1;

    for (ll = 0; ll < loop; ++ll) {
        if(ll == 0) {
            cross = crossed;
            istop = 3;
        }
        else if(second8 == 0 && second9 == 0) {
            cross = s_opposite[crossed];
            istop = 0;
        }
        else {
            switch (crossed) {
                case 0:
                    if (poly_eqn(4, 5, 6, 7, xv, yv, a, b, c)) {
                        count = poly_test(0, 1, 2, 3, xv, yv, a, b, c, dummy);
                        if(count > 0) {
                            cross = 1;
                            goto donn;
                        }
                    }
                case 1:
                    if (poly_eqn(0, 4, 5, 1, xv, yv, a, b, c)) {
                        count = poly_test(3, 7, 6, 2, xv, yv, a, b, c, dummy);
                        if(count > 0) {
                            cross = 2;
                            goto donn;
                        }
                    }
                case 2:
                    if (poly_eqn(1, 5, 6, 2, xv, yv, a, b, c)) {
                        count = poly_test(0, 4, 7, 3, xv, yv, a, b, c, dummy);
                        if(count > 0) {
                            cross = 3;
                            goto donn;
                        }
                    }
                case 3:
                    if (poly_eqn(3, 7, 6, 2, xv, yv, a, b, c)) {
                        count = poly_test(0, 4, 5, 1, xv, yv, a, b, c, dummy);
                        if(count > 0) {
                            cross = 4;
                            goto donn;
                        }
                    }
                case 4:
                    if (poly_eqn(0, 4, 7, 3, xv, yv, a, b, c)) {
                        count = poly_test(1, 5, 6, 2, xv, yv, a, b, c, dummy);
                        if(count > 0) {
                            cross = 5;
                            goto donn;
                        }
                    }
                default:
                    printf("second8 or second9 was true but no cross found\n");
                    return HEX_FAIL_SECOND8_OR_SECOND9_BUT_NO_CROSS;
            }
donn:
            istop = 0;
            if(0) {
                printf("crossed %d cross %d faceverts %d %d %d %d  %d %d %d %d  %d %d\n",
                    crossed, cross, 
                    faceverts[cross][0], faceverts[cross][1],
                    faceverts[cross][2], faceverts[cross][3],
                    rrinv[faceverts[cross][0]], rrinv[faceverts[cross][1]],
                    rrinv[faceverts[cross][2]], rrinv[faceverts[cross][3]],
                    second8, second9);
            }
        }

        /*  Determine depth and color of vertex inside crossed face, by
            dividing this face into two triangles. First
            find the vertex of lowest index to draw diagonal from.
            There is no need for permutation pp here, because the
            necessary information is in crossed.                      */

        vsmall = 2000000000;
        for (j = 0; j < 4; ++j) {
            v = hex->node[faceverts[cross][j]];
            if (v < vsmall) {
                vsmall = v;
                jsmall = j;
            }
        }
        for (l = 0; l < 3; ++l) ivert[0][l] = faceverts[cross][(jsmall+l)%4];

        ivert[1][0] = faceverts[cross][jsmall];
        ivert[1][1] = faceverts[cross][(jsmall+2)%4];
        ivert[1][2] = faceverts[cross][(jsmall+3)%4];

        /*  Find equation of the diagonal separating the two triangles.
            Triangle 0 is the one on the negative side of the line.     */

        ii = faceverts[cross][jsmall];
        jj = faceverts[cross][(jsmall+2)%4];
        kk = faceverts[cross][(jsmall+1)%4];
        da = -yv[jj] + yv[ii];
        db =  xv[jj] - xv[ii];
        dc = -da*xv[ii] - db*yv[ii];
        if (da*xv[kk] + db*yv[kk] + dc > 0) {
            da = -da;
            db = -db;
            dc = -dc;
        }

        /*  Interpolate RGBAz values across appropriate triangles.           */

        for (ii = 0; ii <= istop; ++ii) {
            if(ll || in[ii]) {
                if(ll) {                        // second contained vertex
                    /*
                       if(!second10)
                       kk = pp[s_diagonal[imax]];  // diagonal of {0, 1, 2, 3}
                       else if (second9 && second10)
                       kk = rr[6];
                       else if(second8 && second10)
                       kk = rr[1];
                       else 
                       printf("bad case for interpolating color and depth\n");
                     */
                    if(!second8 && !second9) {
                        kk = rr[1];
                    } else if(second8 && !second10) {
                        kk = rr[1];
                    } else if(second8 && second10) {
                        kk = rr[4];
                    } else if(second9 && !second10) {
                        kk = rr[1];
                    } else {
                        kk = rr[6];
                    }
                } else {
                    kk = pp[ii + 4];
                }
                if (da*xv[kk] + db*yv[kk] + dc > 0) {
                    jj = 1;
                } else {
                    jj = 0;
                }
                n = ivert[jj][0];
                m = ivert[jj][1];
                m1 = ivert[jj][2];

                dx[1] = xv[m] - xv[n];
                dy[1] = yv[m] - yv[n];
                dz[1] = zv[m] - zv[n];

                dx[2] = xv[m1] - xv[n];
                dy[2] = yv[m1] - yv[n];
                dz[2] = zv[m1] - zv[n];

                det = dx[1]*dy[2] - dx[2]*dy[1];
                /*	    printf ("det %f   cross %d   crossed %d/n",
                        det, cross, crossed);                               */

                /*   Interpolate data from triangle vertices by getting linear
                     interpolation equation coefficients.                              */

                {
                    double zcoef[3];
                    zcoef[1] = (dz[1]*dy[2] - dz[2]*dy[1])/det;
                    zcoef[2] = (dx[1]*dz[2] - dx[2]*dz[1])/det;
                    zcoef[0] = zv[n] - zcoef[1]*xv[n] - zcoef[2]*yv[n];
                    ztemp = zcoef[1]*xv[kk] + zcoef[2]*yv[kk] + zcoef[0];
                }

                if ( ipersp == 0) {
                    dst[kk] = zv[kk] - ztemp;
                } else {
#if USE_PROJECTION_ROUTINES
                    Point3 closer, farther;
                    assert(proj != NULL);
                    closer.x = farther.x = xv[kk];
                    closer.y = farther.y = yv[kk];
                    /* Undo inversion for the distance computation */
                    closer.z = PROJ_Z_INV(zv[kk]);
                    farther.z = PROJ_Z_INV(ztemp);
                    dst[kk] = projPerspectiveCorrectDistance(proj, &closer, &farther);
                    printf("\nhex(): kk=%2d %8.4f %8.4f:  closer: %8.4f   farther %8.4f   distance: %8.4f\n\n",
                            kk, xv[kk], yv[kk], closer.z, farther.z, dst[kk]);

#else
                    /* perspective correction */
                    dst[kk] = -1./zv[kk] + 1./ztemp;
                    dst[kk] *= sqrt( ( (xv[kk]-xdisp)*(xv[kk]-xdisp) +
                                (yv[kk]-ydisp)*(yv[kk]-ydisp) )/(facbx*facby) + 1.);
                    if(mypr) printf("zv, ztemp  %f  %f\n", zv[kk], ztemp);
#endif
                }

                /*   The far end has index 0; the near end has index 1;    */

                if( dst[kk] < 0 ) {
                    dst[kk] = - dst[kk];
                    fbkk = 1;
                    fbint = 0;
                } else {
                    fbkk = 0;
                    fbint = 1;
                }
            }  /* end if(ll || in[ii]) */
        } /* end for ii */
    } /* end for ll */

    dmax = 0;
    opmax = 0;
    for (k = 0; k < nverts; ++k) {
        if (dst[k] > dmax) dmax = dst[k];
        opac = .5*(afb[k][0] + afb[k][1]);
        if(opac > opmax) opmax = opac;
    }

    if(dmax > 1.) {

        /* Clamp dist at 1 and make opacity factor larger to compensate.    */

        fdist = 1./dmax;
        dfac = dmax/TSCALE;
    } else if (opmax > TSCALE) {
        dfac = 1./opmax;
        fdist = opmax/TSCALE;
    } else {
        fdist = 1.;
        dfac = 1./TSCALE;
    }
    if (mypr) {
        printf("dmax %f  fdist %f  dfac %f\n", dmax, fdist, dfac);
        printf("nverts = %d\n", nverts);
    }
    for (k = 0; k < nverts; ++k) {

        /* RADIOGRAPHY: */
        /* STORE THE ATTENUATION COEFFICIENTS, and the PATH LENGTH INTO cd */
        cd[k][0] = rcol[k];
        cd[k][1] = gcol[k];
        cd[k][2] = bcol[k];
        cd[k][3] = av[k]; /* alpha channel is used to store fourth group */
        /* Since [rgb]col are atten_per_length, density is already multiplied
         * into them.  We do not need to put density in the thickness value.
         */

        /* dst[k] is zero for all but the thick vertices (it should have been 
         * initialized to zero at the start of this function.
         */
        cd[k][4] = dst[k];

        xyzv[k][0] = xv[k];
        xyzv[k][1] = yv[k];
        xyzv[k][2] = 1.;
    }

    /*  pp takes the standard picture to the actual one. In the standard
        picture, the face known to contain an interior vertex is face 0123.
        Define a rotation qq to make this interior vertex be vertex 7,
        and then let rr be the product of qq followed by pp.             */

    for (i = 0; i < 4; ++i) {
        qq[i] = (i + 4 - turn)%4;
        qq[i+4] = qq[i] + 4;
    }
    for (i = 0; i < 8; ++i) rr[i] = pp[qq[i]];
    rr[8] = 8;
    rr[9] = 9;
    rr[10] = 10;
    for (i = 0; i < VSIZE; ++i) {
        rrinv[rr[i]] = i;
        ppinv[pp[i]] = i;
    }

    /* Begin block of drawing code.
     * this could safely be moved into its own function.
     */
    {
        if(1 && count == 1 && !second8 && !second9) { // Standard count 1 case
            glBegin(GL_TRIANGLE_FAN);
            if(mypr) printf("glBegin(GL_TRIANGLE_FAN) 1\n");
            myglArrayElement(proj, rr[8], cd, xyzv, dst);
            myglArrayElement(proj, rr[1], cd, xyzv, dst);
            myglArrayElement(proj, rr[5], cd, xyzv, dst);
            myglArrayElement(proj, rr[4], cd, xyzv, dst);
            myglArrayElement(proj, rr[0], cd, xyzv, dst);
            myglArrayElement(proj, rr[3], cd, xyzv, dst);
            myglArrayElement(proj, rr[7], cd, xyzv, dst);
            myglArrayElement(proj, rr[1], cd, xyzv, dst);
            glEnd();

            glBegin(GL_TRIANGLE_FAN);
            if(mypr) printf("glBegin(GL_TRIANGLE_FAN) 2\n");
            myglArrayElement(proj, rr[9], cd, xyzv, dst);
            myglArrayElement(proj, rr[7], cd, xyzv, dst);
            myglArrayElement(proj, rr[3], cd, xyzv, dst);
            myglArrayElement(proj, rr[2], cd, xyzv, dst);
            myglArrayElement(proj, rr[6], cd, xyzv, dst);
            myglArrayElement(proj, rr[5], cd, xyzv, dst);
            myglArrayElement(proj, rr[1], cd, xyzv, dst);
            myglArrayElement(proj, rr[7], cd, xyzv, dst);
            glEnd();
            if (mypr) { printf("Done with fans.\n"); fflush(stdout); }
        }
        else if(count == 1 && second9) { // Alternate B count = 1 case
            if(!second10) {
                if(1) {
                    if(mmpr) printf("Three Triangle Fans for second9 != 0\n");
                    glBegin(GL_TRIANGLE_FAN);
                    myglArrayElement(proj, rr[10], cd, xyzv, dst);
                    myglArrayElement(proj, rr[5], cd, xyzv, dst);
                    myglArrayElement(proj, rr[6], cd, xyzv, dst);
                    myglArrayElement(proj, rr[2], cd, xyzv, dst);
                    myglArrayElement(proj, rr[1], cd, xyzv, dst);
                    myglArrayElement(proj, rr[9], cd, xyzv, dst);
                    myglArrayElement(proj, rr[8], cd, xyzv, dst);
                    myglArrayElement(proj, rr[4], cd, xyzv, dst);
                    myglArrayElement(proj, rr[5], cd, xyzv, dst);
                    glEnd();

                    glBegin(GL_TRIANGLE_FAN);
                    myglArrayElement(proj, rr[9], cd, xyzv, dst);
                    myglArrayElement(proj, rr[1], cd, xyzv, dst);
                    myglArrayElement(proj, rr[2], cd, xyzv, dst);
                    myglArrayElement(proj, rr[3], cd, xyzv, dst);
                    myglArrayElement(proj, rr[7], cd, xyzv, dst);
                    glEnd();

                    glBegin(GL_TRIANGLE_FAN);
                    myglArrayElement(proj, rr[8], cd, xyzv, dst);
                    myglArrayElement(proj, rr[9], cd, xyzv, dst);
                    myglArrayElement(proj, rr[7], cd, xyzv, dst);
                    myglArrayElement(proj, rr[3], cd, xyzv, dst);
                    myglArrayElement(proj, rr[0], cd, xyzv, dst);
                    myglArrayElement(proj, rr[4], cd, xyzv, dst);
                    glEnd();
                }
            }
            else if(1) { // the alternate D count = 1 case
                if(mypr) {
                    printf("2 Triangle Fans for second9 %d and second10 %ld  (stats->totalcount) %ld\n",
                           second9, second10, (stats->totalcount));
                }
                glBegin(GL_TRIANGLE_FAN);
                myglArrayElement(proj, rr[8], cd, xyzv, dst);
                myglArrayElement(proj, rr[4], cd, xyzv, dst);
                myglArrayElement(proj, rr[0], cd, xyzv, dst);
                myglArrayElement(proj, rr[3], cd, xyzv, dst);
                myglArrayElement(proj, rr[7], cd, xyzv, dst);
                myglArrayElement(proj, rr[9], cd, xyzv, dst);
                myglArrayElement(proj, rr[6], cd, xyzv, dst);
                myglArrayElement(proj, rr[5], cd, xyzv, dst);
                myglArrayElement(proj, rr[4], cd, xyzv, dst);
                glEnd();

                glBegin(GL_TRIANGLE_FAN);
                myglArrayElement(proj, rr[10], cd, xyzv, dst);
                myglArrayElement(proj, rr[6], cd, xyzv, dst);
                myglArrayElement(proj, rr[9], cd, xyzv, dst);
                myglArrayElement(proj, rr[7], cd, xyzv, dst);
                myglArrayElement(proj, rr[3], cd, xyzv, dst);
                myglArrayElement(proj, rr[2], cd, xyzv, dst);
                myglArrayElement(proj, rr[1], cd, xyzv, dst);
                myglArrayElement(proj, rr[5], cd, xyzv, dst);
                myglArrayElement(proj, rr[6], cd, xyzv, dst);
                glEnd();
            }
        }
        else if(count == 1 && second8) { // the alternate A count = 1 case
            if(!second10) {
                if(mmpr) printf("3 Triangle Fans for second8 and ! second10\n");
                glBegin(GL_TRIANGLE_FAN);
                myglArrayElement(proj, rr[9], cd, xyzv, dst);
                myglArrayElement(proj, rr[6], cd, xyzv, dst);
                myglArrayElement(proj, rr[2], cd, xyzv, dst);
                myglArrayElement(proj, rr[3], cd, xyzv, dst);
                myglArrayElement(proj, rr[7], cd, xyzv, dst);
                myglArrayElement(proj, rr[8], cd, xyzv, dst);
                myglArrayElement(proj, rr[10], cd, xyzv, dst);
                myglArrayElement(proj, rr[5], cd, xyzv, dst);
                myglArrayElement(proj, rr[6], cd, xyzv, dst);
                glEnd();

                glBegin(GL_TRIANGLE_FAN);
                myglArrayElement(proj, rr[8], cd, xyzv, dst);
                myglArrayElement(proj, rr[7], cd, xyzv, dst);
                myglArrayElement(proj, rr[3], cd, xyzv, dst);
                myglArrayElement(proj, rr[0], cd, xyzv, dst);
                myglArrayElement(proj, rr[1], cd, xyzv, dst);
                glEnd();

                glBegin(GL_TRIANGLE_FAN);
                myglArrayElement(proj, rr[10], cd, xyzv, dst);
                myglArrayElement(proj, rr[8], cd, xyzv, dst);
                myglArrayElement(proj, rr[1], cd, xyzv, dst);
                myglArrayElement(proj, rr[0], cd, xyzv, dst);
                myglArrayElement(proj, rr[4], cd, xyzv, dst);
                myglArrayElement(proj, rr[5], cd, xyzv, dst);
                glEnd();
            }
            else if(1) { // the alternate B count = 1 case
                if(mypr) {
                    printf("2 Triangle Fans for second8 %d and second10 %ld  (stats->totalcount) %ld\n",
                           second8, second10, (stats->totalcount));
                }
                glBegin(GL_TRIANGLE_FAN);
                myglArrayElement(proj, rr[9], cd, xyzv, dst);
                myglArrayElement(proj, rr[2], cd, xyzv, dst);
                myglArrayElement(proj, rr[3], cd, xyzv, dst);
                myglArrayElement(proj, rr[7], cd, xyzv, dst);
                myglArrayElement(proj, rr[8], cd, xyzv, dst);
                myglArrayElement(proj, rr[4], cd, xyzv, dst);
                myglArrayElement(proj, rr[5], cd, xyzv, dst);
                myglArrayElement(proj, rr[6], cd, xyzv, dst);
                myglArrayElement(proj, rr[2], cd, xyzv, dst);
                glEnd();

                glBegin(GL_TRIANGLE_FAN);
                myglArrayElement(proj, rr[10], cd, xyzv, dst);
                myglArrayElement(proj, rr[7], cd, xyzv, dst);
                myglArrayElement(proj, rr[3], cd, xyzv, dst);
                myglArrayElement(proj, rr[0], cd, xyzv, dst);
                myglArrayElement(proj, rr[1], cd, xyzv, dst);
                myglArrayElement(proj, rr[5], cd, xyzv, dst);
                myglArrayElement(proj, rr[4], cd, xyzv, dst);
                myglArrayElement(proj, rr[8], cd, xyzv, dst);
                myglArrayElement(proj, rr[7], cd, xyzv, dst);
                glEnd();
            }
        }
        else if(1 && count == 2) {  // old three fan method for count = 2 case
            if (mypr) printf("3 Triangle fans for old three fan method.\n");
            glBegin(GL_TRIANGLE_FAN);
            myglArrayElement(proj, rr[8], cd, xyzv, dst);
            myglArrayElement(proj, rr[4], cd, xyzv, dst);
            myglArrayElement(proj, rr[0], cd, xyzv, dst);
            myglArrayElement(proj, rr[3], cd, xyzv, dst);
            myglArrayElement(proj, rr[7], cd, xyzv, dst);
            glEnd();

            glBegin(GL_TRIANGLE_FAN);
            myglArrayElement(proj, rr[7], cd, xyzv, dst);
            myglArrayElement(proj, rr[3], cd, xyzv, dst);
            myglArrayElement(proj, rr[2], cd, xyzv, dst);
            myglArrayElement(proj, rr[6], cd, xyzv, dst);
            glEnd();

            if(mypr) printf("glBegin(GL_TRIANGLE_FAN) 3\n");
            glBegin(GL_TRIANGLE_FAN);
            myglArrayElement(proj, rr[9], cd, xyzv, dst);
            myglArrayElement(proj, rr[7], cd, xyzv, dst);
            myglArrayElement(proj, rr[6], cd, xyzv, dst);
            myglArrayElement(proj, rr[2], cd, xyzv, dst);
            myglArrayElement(proj, rr[1], cd, xyzv, dst);
            myglArrayElement(proj, rr[5], cd, xyzv, dst);
            myglArrayElement(proj, rr[4], cd, xyzv, dst);
            myglArrayElement(proj, rr[8], cd, xyzv, dst);
            myglArrayElement(proj, rr[7], cd, xyzv, dst);
            glEnd();
        }           

        else if(1 && count == 2) {  // new fan-strip method for count = 2 case
            if (mypr) printf("2 triangle fans for new fan-strip method\n");
            glBegin(GL_TRIANGLE_FAN);
            myglArrayElement(proj, rr[7], cd, xyzv, dst);
            myglArrayElement(proj, rr[8], cd, xyzv, dst);
            myglArrayElement(proj, rr[0], cd, xyzv, dst);
            myglArrayElement(proj, rr[3], cd, xyzv, dst);
            myglArrayElement(proj, rr[2], cd, xyzv, dst);
            myglArrayElement(proj, rr[6], cd, xyzv, dst);
            myglArrayElement(proj, rr[9], cd, xyzv, dst);
            myglArrayElement(proj, rr[8], cd, xyzv, dst);
            glEnd();

            glBegin(GL_TRIANGLE_STRIP);
            myglArrayElement(proj, rr[0], cd, xyzv, dst);
            myglArrayElement(proj, rr[4], cd, xyzv, dst);
            myglArrayElement(proj, rr[8], cd, xyzv, dst);
            myglArrayElement(proj, rr[5], cd, xyzv, dst);
            myglArrayElement(proj, rr[9], cd, xyzv, dst);
            myglArrayElement(proj, rr[1], cd, xyzv, dst);
            myglArrayElement(proj, rr[6], cd, xyzv, dst);
            myglArrayElement(proj, rr[2], cd, xyzv, dst);
            glEnd();
        }

        else if(1 && count == 0) { // the count = 0 case
            printf("2 triangle fans for count == 0\n");
            ++(stats->draw0);
            glBegin(GL_TRIANGLE_FAN);
            myglArrayElement(proj, tt[8], cd, xyzv, dst);
            myglArrayElement(proj, tt[0], cd, xyzv, dst);
            myglArrayElement(proj, tt[1], cd, xyzv, dst);
            myglArrayElement(proj, tt[2], cd, xyzv, dst);
            myglArrayElement(proj, tt[9], cd, xyzv, dst);
            myglArrayElement(proj, tt[10], cd, xyzv, dst);
            myglArrayElement(proj, tt[11], cd, xyzv, dst);
            myglArrayElement(proj, tt[7], cd, xyzv, dst);
            myglArrayElement(proj, tt[0], cd, xyzv, dst);
            glEnd();

            glBegin(GL_TRIANGLE_STRIP);
            myglArrayElement(proj, tt[7], cd, xyzv, dst);
            myglArrayElement(proj, tt[6], cd, xyzv, dst);
            myglArrayElement(proj, tt[11], cd, xyzv, dst);
            myglArrayElement(proj, tt[5], cd, xyzv, dst);
            myglArrayElement(proj, tt[10], cd, xyzv, dst);
            myglArrayElement(proj, tt[4], cd, xyzv, dst);
            myglArrayElement(proj, tt[3], cd, xyzv, dst);
            myglArrayElement(proj, tt[9], cd, xyzv, dst);
            myglArrayElement(proj, tt[2], cd, xyzv, dst);
            glEnd();
        }

        else if(1 && count == 4) {   // the count = 4 case
            printf("2 triangle fans for count == 4\n");
            num_vertices1 = 8;
            num_vertices2 = 6;
            glBegin(GL_TRIANGLE_FAN);
            if(mypr) printf("glBegin(GL_TRIANGLE_FAN) 4\n");
            myglArrayElement(proj, rr[4], cd, xyzv, dst);
            myglArrayElement(proj, rr[0], cd, xyzv, dst);
            myglArrayElement(proj, rr[3], cd, xyzv, dst);
            myglArrayElement(proj, rr[7], cd, xyzv, dst);
            myglArrayElement(proj, rr[6], cd, xyzv, dst);
            myglArrayElement(proj, rr[5], cd, xyzv, dst);
            myglArrayElement(proj, rr[1], cd, xyzv, dst);
            myglArrayElement(proj, rr[0], cd, xyzv, dst);
            glEnd();

            glBegin(GL_TRIANGLE_FAN);
            if(mypr) printf("glBegin(GL_TRIANGLE_FAN) 5\n");
            myglArrayElement(proj, rr[6], cd, xyzv, dst);
            myglArrayElement(proj, rr[7], cd, xyzv, dst);
            myglArrayElement(proj, rr[3], cd, xyzv, dst);
            myglArrayElement(proj, rr[2], cd, xyzv, dst);
            myglArrayElement(proj, rr[1], cd, xyzv, dst);
            myglArrayElement(proj, rr[5], cd, xyzv, dst);
            glEnd();
        }
        else if(1) {
            printf("unknown triangle fan case\n");
            return HEX_FAIL_UNKNOWN_TRIANGLE_FAN_CASE;
        }
    } /* end drawing code block */

    if (0 && drawnow && ivview) {
        fclose((FILE*)fp);
    }
    if(0 && extraview && count == 2) {
        if ((fp = fopen("simple.iv","a")) == 0) {
            fprintf(stderr,"reader() unable to open FILE %s\n","simple.iv");
            return 1;
        }
        viewcell(fp, hex, stats);
        fclose((FILE*)fp);
    }
    if (mypr) {
        printf("har_hexProject() is done, returning success.\n"); 
        fflush(stdout);
    }
    return HEX_SUCCESS;
}

