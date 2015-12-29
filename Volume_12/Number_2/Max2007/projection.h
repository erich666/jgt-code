#ifndef HAR_PROJECTION_H_INCLUDED
#define HAR_PROJECTION_H_INCLUDED

#include "GraphicsGems.h"

/*
 *
 * Invert the z coordinate returned the projProjectPoint().
 * This inversion is done so that the Z coordinates *increase* monotonically
 * as one moves from the source to the detector.
 * Furthermore, the inversion enables the z coordinates to be interpolated in
 * detector space according to the projection x and y coordinates.
 *
 */

#define PROJ_Z_INV(zcoord) (1.0 / (zcoord))


/* ----------------------------------------------------------------
 *
 * har_Projection structure
 *
 * ----------------------------------------------------------------
 */
typedef struct har_Projection_tag {
    int    perspective;      /* 1 if perspective view, 0 if parallel view */
    Point3 sourcePos;
    Point3 sourcePosInDetectorFrame;
    Point3 projectionPos;    /* perpendicular projection in world coordinates of source */
    Vector3 detectorNormal;  /* Detector plane-normal (also parallel to projection) */
    double distance;        /* distance from source to detector, measured
                             * according to perpendicular projection of source
                             * onto detector plane (using plane normal).
                             */

     /* Detector extents in detector space (screen space).
      * We can convert the detector based coordinates
      * to 0 based pixel indices using these bounds.
      */
    double xmin, xmax, ymin, ymax;
    int nx, ny;      /* pixel counts in each direction */
    double pixel_width, pixel_height;  /* of pixels in the detector frame */

    Matrix4 worldToDetectorMat;  /* Matrix that takes world coordinates to
                                  * detector coordinates. 
                                  */
}   har_Projection;


/*
 * Allocate and return a new projection object.
 * It is initialized to zero and is unusable.
 * The client must insure that SetRadiographyGeometry
 * and SetDetectorResolution are called before any other
 * operations can be called.
 */
har_Projection*
projAlloc(void);

/*
 * coordinates of the detector are in world reference frame.
 * Return 0 on failure.
 */
int
projSetRadiographyGeometry(har_Projection* proj,
                           Point3 source_position,
                           Point3 LL,  /* lower left corner of detector */
                           Point3 LR,  /* lower right corner of detector */
                           Point3 UL   /* upper left corner of detector */
                           );

/*
 * coordinates of the detector are in world reference frame.
 * Return 0 on failure.
 */
int
projSetParallelRadiographyGeometry(har_Projection* proj,
                                   Point3 LL,  /* lower left corner of detector */
                                   Point3 LR,  /* lower right corner of detector */
                                   Point3 UL   /* upper left corner of detector */
                                   );

void
projSetDetectorResolution(har_Projection* proj, int npixels_x, int npixels_y);

/* Compute 0-origin pixel indices for orthographic projection. */
void
projProjectedPointToPixel(const har_Projection* proj, const Point3* p, 
                          double* x, double* y);

/* -------------------------------------------------------------------------
 *
 * projProjectPoint()
 *
 * Project p to the detector plane.  The result 'q' contains the
 * projected X and Y coordinates on the detector, and the
 * Z coordinate of the point in the detector frame of reference (with Z==0
 * meaning 'p' was located on the detector plane).
 *
 * -------------------------------------------------------------------------
 */
void
projProjectPoint(const har_Projection* proj, const Point3* p, Point3* q);

/* -------------------------------------------------------------------------
 *
 * projWorldToDetector()
 *
 * compute transformation of point in world frame to the detector frame.
 *
 * -------------------------------------------------------------------------
 */
void
projWorldToDetector(const har_Projection* proj, const Point3* world, Point3* d);

/* -------------------------------------------------------------------------
 *
 * projDetectorToImage()
 *
 * Compute transformation of point in detector frame to the image (detector
 * plane) using
 * a perspective transformation.  For parallel projection no computation
 * occurs.
 * p->z == d->z  (z coordinate is left in the detector frame).
 *
 * -------------------------------------------------------------------------
 */
void
projDetectorToImage(const har_Projection* proj,
                    const Point3* d, Point3* p);


/* -------------------------------------------------------------------------
 *
 * projPerspectiveCorrectDistance()
 *
 * For two points that project to same point on detector plane, compute
 * the actual geometric distance between them in the detector frame.
 *
 * 'p' and 'q' are points returned by projProjectPoint().
 * Their x and y coordinates are in screen space.
 * -------------------------------------------------------------------------
 */

double
projPerspectiveCorrectDistance(const har_Projection* proj,
                               const Point3* p, const Point3* q);

void
projPrintProjection(har_Projection p);

/* Run unit tests, 1 = Pass, 0 = fail */
int
projSelftest(int verbose);

#endif /* include guard */
