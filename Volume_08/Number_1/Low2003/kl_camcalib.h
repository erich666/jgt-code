#ifndef _KL_CAMCALIB_H_
#define _KL_CAMCALIB_H_


void compIntExtMats( double *const *pt3D, double *const *pt2D, 
				     int numPoints, double **intMat, double **extMat );
	// Compute the 3x3 intrinsic matrix and the 3x4 extrinsic matrix given that
	// the camera's COP position is unknown.
	// INPUT:
	//	 - pt3D[1..numPoints][1..3] -- 3D coordinates of 3D points.
	//	 - pt2D[1..numPoints][1..2] -- 2D pixel coordinates of image points.
	//	 - numPoints -- number of pair-correspondences (must be at least 6).
	// OUTPUT:
	//   - intMat[1..3][1..3] -- the intrinsic matrix.
	//   - extMat[1..3][1..4] -- the extrinsic matrix.


void compIntExtMats( const double camPos[], double *const *pt3D, double *const *pt2D, 
				     int numPoints, double **intMat, double **extMat );
	// Compute the 3x3 intrinsic matrix and the 3x4 extrinsic matrix given that
	// the camera's COP position is known.
	// INPUT:
	//	 - camPos[1..3] -- camera's COP position.
	//	 - pt3D[1..numPoints][1..3] -- 3D coordinates of 3D points.
	//	 - pt2D[1..numPoints][1..2] -- 2D pixel coordinates of image points.
	//	 - numPoints -- number of pair-correspondences (must be at least 4).
	// OUTPUT:
	//   - intMat[1..3][1..3] -- the intrinsic matrix.
	//   - extMat[1..3][1..4] -- the extrinsic matrix.


void compProjMatFromCam( double *const *pt3D, double *const *pt2D, 
					     int numPoints, double **projMat );
	// Compute the 3x4 projection matrix given that the camera's position is unknown.  
	// INPUT:
	//   - pt3D[1..numPoints][1..3] -- 3D coordinates of 3D points.
	//	 - pt2D[1..numPoints][1..2] -- 2D pixel coordinates of image points.
	//	 - numPoints -- number of pair-correspondences (must be at least 6).
	// OUTPUT:
	//	 - projMat[1..3][1..4] -- the projection matrix (up to an arbitrary sign).


void compProjMatFromOrig( double *const *pt3D, double *const *pt2D, 
					      int numPoints, double **projMat );
	// Compute the 3x3 projection matrix given that the camera is already at the origin.  
	// INPUT:
	//   - pt3D[1..numPoints][1..3] -- 3D coordinates of 3D points.
	//	 - pt2D[1..numPoints][1..2] -- 2D pixel coordinates of image points.
	//	 - numPoints -- number of pair-correspondences (must be at least 4).
	// OUTPUT:
	//	 - projMat[1..3][1..3] -- the projection matrix (up to an arbitrary sign).


void reprojectionError( double *const *pt3D, double *const *pt2D, 
				        int numPoints, double *const *intMat, double *const *extMat,
						double *pixelError, double *totalError );
	// Re-project the 3D points using the input intrinsic and extrinsic matrices,
	// and find their pixel errors from the input 2D points.
	// INPUT:
	//   - pt3D[1..numPoints][1..3] -- 3D coordinates of 3D points.
	//	 - pt2D[1..numPoints][1..2] -- 2D pixel coordinates of image points.
	//	 - numPoints -- number of pair-correspondences (must be at least 4).
	//   - intMat[1..3][1..3] -- the intrinsic matrix.
	//   - extMat[1..3][1..4] -- the extrinsic matrix.
	// OUTPUT:
	//   - pixelError[1..numPoints] -- pixel error of each re-projected 3D point.
	//   - totalError -- sum of all pixel errors.


void convToOpenGLMats( double *const *intMat, double *const *extMat, 
					   double nearPlaneDist, double farPlaneDist,
					   int viewportWidth, int viewportHeight,
					   double projectionMat[16], double modelviewMat[16] );
	// Convert a 3x3 intrinsic matrix to a OpenGL PROJECTION matrix, 
	// and a 3x4 extrinsic matrix to a OpenGL MODELVIEW matrix.
	// INPUT:
	//   - intMat[1..3][1..3] -- the intrinsic matrix.
	//   - extMat[1..3][1..4] -- the extrinsic matrix.
	//   - nearPlaneDist -- distance to near clipping plane ( > 0 ).
	//   - farPlaneDist -- distance to far clipping plane ( > nearPlaneDist ).
	//   - viewportWidth -- width of viewport in pixels.
	//   - viewportHeight -- height of viewport in pixels.
	// OUTPUT:
	//   - projectionMat[0..15] -- OpenGL PROJECTION matrix.
	//   - modelviewMat[0..15] -- OpenGL MODELVIEW matrix.



#endif
