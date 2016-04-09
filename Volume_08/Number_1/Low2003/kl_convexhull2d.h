#ifndef _KL_CONVEXHULL2D_H_
#define _KL_CONVEXHULL2D_H_

extern void kl_ConvexHull2D( const double inPoints2D[], int numInPoints2D, 
							 int **hull, int *hullSize );
// PURPOSE:
//   Computes (using QuickHull approach) and returns a 2D convex hull.
//
// INPUTS:
//   inPoints2D[0..2*numInPoint2D-1] -- Array of input 2D points. 
//         Coordinates of point i (0 <= i <= numInPoint2D-1) is stored in 
//         inPoints2D[2*i] and inPoints2D[2*i+1].
//   numInPoints2D -- Number of input 2D points.
//
// OUTPUTS:
//   (*hull)[0..(*hullSize)-1] -- Array of IDs of points on the convex hull.
//          Each ID ranges from 0 to (numInPoints2D-1).
//          Points on convex hull are listed in counter-clockwise order.
//   *hullSize -- Number of points on the convex hull.
//


extern void kl_FreeHull2D( int *hull );
// PURPOSE:
//   Frre up memory allocated to the output of KL_ConvexHull2D().
//
// INPUTS:
//   hull -- pointer to the memory allocated for the array of IDs 
//           of points on the convex hull.


#endif
