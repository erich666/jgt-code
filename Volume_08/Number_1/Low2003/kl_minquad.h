#ifndef _KL_MINQUAD_H_
#define _KL_MINQUAD_H_


extern bool kl_MinQuad( const double inPoints2D[], int numInPoints2D, double outPoints2D[8] );
// PURPOSE:
//   Returns the approximate minimum-area enclosing quadrilateral.
//   It uses the following heuristics: 
//     In each iteration, remove the edge that adds the smallest area to the enclosing polygon.
//     Repeat until enclosing polygon has only 4 sides.
//
// INPUTS:
//   inPoints2D[0..2*numInPoint2D-1] -- Array of input 2D points. 
//         Coordinates of point i (0 <= i <= numInPoint2D-1) is stored in 
//         inPoints2D[2*i] and inPoints2D[2*i+1].
//   numInPoints2D -- Number of input 2D points.
//
// OUTPUTS:
//   outPoints2D[0..7] -- Array of 2D points of the 4 corners of the 
//         approximate minimum-area enclosing quadrilateral.
//         They are listed in counter-clockwise order.
//
// RETURNED VALUE:
//   True iff minimum-area enclosing quadrilateral is successfully computed.




extern bool kl_MinQuad( const double inPoints2D[], const int convexhull[], int hullSize, double outPoints2D[8] );
// PURPOSE:
//   Returns the approximate minimum-area enclosing quadrilateral of a 2D convex hull.
//   It uses the following heuristics: 
//     In each iteration, remove the edge that adds the smallest area to the enclosing polygon.
//     Repeat until enclosing polygon has only 4 sides.
//
// INPUTS:
//   inPoints2D[] -- Array of input 2D points, referenced by convexhull[]. 
//         Coordinates of point i (0 <= i <= numInPoint2D-1) is stored in 
//         inPoints2D[2*i] and inPoints2D[2*i+1].
//   convexhull[0..hullSize-1] -- Array of IDs of points on the convex hull.
//          Points on convex hull are listed in counter-clockwise order.
//   hullSize -- Number of points on the convex hull.

//
// OUTPUTS:
//   outPoints2D[0..7] -- Array of 2D points of the 4 corners of the 
//         approximate minimum-area enclosing quadrilateral.
//         They are listed in counter-clockwise order.
//
// RETURNED VALUE:
//   True iff minimum-area enclosing quadrilateral is successfully computed.




// ********************* ANOTHER MINQUAD HEURISTICS. EXPERIMENTAL *********************

extern bool kl_MinQuad2( const double inPoints2D[], int numInPoints2D, double outPoints2D[8] );
extern bool kl_MinQuad2( const double inPoints2D[], const int convexhull[], int hullSize, double outPoints2D[8] );
// returns the approximate minimum-area enclosing quadrilateral.
// It uses the following heuristics: 
//     In each iteration, remove the edge that adds the smallest sum of squared edge lengths
//     to the enclosing polygon.
//     Repeat until enclosing polygon has only 4 sides.


#endif