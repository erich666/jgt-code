#ifndef _KL_OPTIFRUSTUM_H_
#define _KL_OPTIFRUSTUM_H_

#include <GL/glut.h>


extern void kl_BoundingSphere( const double inPoints3D[],int numInPoints3D, 
							   double sphereCenter[3], double *sphereRadius );
// PURPOSE:
//   Computes a bounding sphere of the input 3D points.
//
// INPUTS:
//   inPoints3D[0..3*numInPoint3D-1] -- Array of input 3D points. 
//         Coordinates of point i (0 <= i <= numInPoint3D-1) is stored in 
//         inPoints3D[3*i], inPoints2D[3*i+1] and inPoints2D[3*i+2]
//   numInPoints3D -- Number of input 3D points.
//
// OUTPUTS:
//   sphereCenter[0..2] -- 3D coordinates of the center of the bounding sphere.
//   *sphereRadius -- Radius of the bounding sphere.



extern void kl_OptiFrustum( const double inPoints3D[],int numInPoints3D, 
					 const double sphereCenter[3], double sphereRadius,
					 const double viewpoint[3],
					 int viewportWidth, int viewportHeight,
					 GLdouble optiProjectionMat[16], GLdouble optiModelviewMat[16],
					 GLdouble *convProjectionMat = NULL, GLdouble *convModelviewMat = NULL,
					 double *minEnclosingQuad = NULL );
// PURPOSE:
//   Computes a frustum to maximize the image area of the input object.
//
// INPUTS:
//   inPoints3D[0..3*numInPoint3D-1] -- Array of input 3D points of the object's vertices. 
//         Coordinates of point i (0 <= i <= numInPoint3D-1) is stored in 
//         inPoints3D[3*i], inPoints2D[3*i+1] and inPoints2D[3*i+2]
//   numInPoints3D -- Number of input 3D points.
//   sphereCenter[0..2] -- 3D coordinates of the center of a bounding sphere.
//   sphereRadius -- Radius of the bounding sphere.
//   viewpoint[0..2] -- 3D coordinates of the viewpoint (eye).
//   viewportWidth -- width of viewport in pixels.
//   viewportHeight -- height of viewport in pixels.
//
// OUTPUTS:
//   projectionMat[0..15] -- OpenGL projection matrix of the optimized frustum.
//   modelviewMat[0..15] -- OpenGL model-view matrix of the optimized frustum.
//
// OPTIONAL OUTPUTS:
//   convProjectionMat[0..15] -- the intermediate OpenGL projection matrix used to
//			derive the minimum-enclosing quadrilateral. Pass in a NULL pointer
//          if output is not needed.
//   convModelviewMat[0..15] -- the intermediate OpenGL modelview matrix used to
//			derive the minimum-enclosing quadrilateral. Pass in a NULL pointer
//          if output is not needed.
//   minEnclosingQuad[0..8] -- the 2D positions of the 4 corners of the minimum-enclosing quadrilateral. 
//			Pass in a NULL pointer if output is not needed.


#endif