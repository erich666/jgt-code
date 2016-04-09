#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "kl_camcalib.h"


#ifdef __cplusplus
extern "C" {
#endif

#include "nrutil.h"

extern void dsvdcmp(double **a, int m, int n, double w[], double **v);

#ifdef __cplusplus
}
#endif



static double sqr( double a )
{
	return a * a;
}



void convToOpenGLMats( double *const *intMat, double *const *extMat, 
					   double nearPlaneDist, double farPlaneDist,
					   int viewportWidth, int viewportHeight,
					   double projectionMat[16], double modelviewMat[16] )
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
{

	modelviewMat[0] = extMat[1][1];
	modelviewMat[1] = extMat[2][1];
	modelviewMat[2] = extMat[3][1];
	modelviewMat[3] = 0;
	modelviewMat[4] = extMat[1][2];
	modelviewMat[5] = extMat[2][2];
	modelviewMat[6] = extMat[3][2];
	modelviewMat[7] = 0;
	modelviewMat[8] = extMat[1][3];
	modelviewMat[9] = extMat[2][3];
	modelviewMat[10] = extMat[3][3];
	modelviewMat[11] = 0;
	modelviewMat[12] = extMat[1][4];
	modelviewMat[13] = extMat[2][4];
	modelviewMat[14] = extMat[3][4];
	modelviewMat[15] = 1;


	projectionMat[0] = -2 * intMat[1][1] / viewportWidth;
	projectionMat[1] = 0;
	projectionMat[2] = 0;
	projectionMat[3] = 0;
	projectionMat[4] = 0;
	projectionMat[5] = -2 * intMat[2][2] / viewportHeight;
	projectionMat[6] = 0;
	projectionMat[7] = 0;
	projectionMat[8] = 1 - 2 * intMat[1][3] / viewportWidth;
	projectionMat[9] = 1 - 2 * intMat[2][3] / viewportHeight;
	//projectionMat[10] = -( farPlaneDist + nearPlaneDist ) / ( farPlaneDist - nearPlaneDist );
	projectionMat[10] = -( farPlaneDist + nearPlaneDist ) / ( farPlaneDist - nearPlaneDist );
	projectionMat[11] = -1;
	projectionMat[12] = 0;
	projectionMat[13] = 0;
	//projectionMat[14] = -2 * farPlaneDist * nearPlaneDist / ( farPlaneDist - nearPlaneDist );
	projectionMat[14] = -2 * farPlaneDist * nearPlaneDist / ( farPlaneDist - nearPlaneDist );
	projectionMat[15] = 0;
}



void reprojectionError( double *const *pt3D, double *const *pt2D, 
				        int numPoints, double *const *intMat, double *const *extMat,
						double *pixelError, double *totalError )
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
{

	double **M = dmatrix( 1, 3, 1, 4 );

	int i, j, k;

	for ( i = 1; i <= 3; i++ )
		for ( j = 1; j <= 4; j++ )
	{
		M[i][j] = 0;
		for ( k = 1; k <= 3; k++ ) M[i][j] += intMat[i][k] * extMat[k][j];
	}


	*totalError = 0;

	for ( i = 1; i <= numPoints; i++ )
	{
		double x = M[1][1]*pt3D[i][1] + M[1][2]*pt3D[i][2] + M[1][3]*pt3D[i][3] + M[1][4];
		double y = M[2][1]*pt3D[i][1] + M[2][2]*pt3D[i][2] + M[2][3]*pt3D[i][3] + M[2][4];
		double w = M[3][1]*pt3D[i][1] + M[3][2]*pt3D[i][2] + M[3][3]*pt3D[i][3] + M[3][4];

		x /= w;
		y /= w;

		pixelError[i] = sqrt( sqr( x - pt2D[i][1] ) + sqr( y - pt2D[i][2] ) );

		(*totalError) += pixelError[i];
	}

	free_dmatrix( M, 1, 3, 1, 4 );
}



void compIntExtMats( const double camPos[], double *const *pt3D, double *const *pt2D, 
				     int numPoints, double **intMat, double **extMat )
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
{
	if ( numPoints < 4 )
	{
		fprintf( stderr, "compIntExtMats: must have at least 4 pair-correspondences.\n" );
		exit(1);
	}

	int i, j;

	double **new_pt3D = dmatrix( 1, numPoints, 1, 3 );

	for ( i = 1; i <= numPoints; i++ )
		for ( j = 1; j <= 3; j++ )
			new_pt3D[i][j] = pt3D[i][j] - camPos[j];

	double **P = dmatrix( 1, 3, 1, 3 );	// 3x3 projection matrix

	// Compute a 3x3 projection matrix. 
	compProjMatFromOrig( new_pt3D, pt2D, numPoints, P );

	// Compute camera parameters.
	double ox = P[1][1]*P[3][1] + P[1][2]*P[3][2] + P[1][3]*P[3][3];
	double oy = P[2][1]*P[3][1] + P[2][2]*P[3][2] + P[2][3]*P[3][3];

	double a = -sqrt( P[1][1]*P[1][1] + P[1][2]*P[1][2] + P[1][3]*P[1][3] - ox*ox );
	double b = -sqrt( P[2][1]*P[2][1] + P[2][2]*P[2][2] + P[2][3]*P[2][3] - oy*oy );

	double r11 = ( P[1][1] - ox * P[3][1] ) / a;
	double r12 = ( P[1][2] - ox * P[3][2] ) / a;
	double r13 = ( P[1][3] - ox * P[3][3] ) / a;
	double r21 = ( P[2][1] - oy * P[3][1] ) / b;
	double r22 = ( P[2][2] - oy * P[3][2] ) / b;
	double r23 = ( P[2][3] - oy * P[3][3] ) / b;
	double r31 = P[3][1];
	double r32 = P[3][2];
	double r33 = P[3][3];

	// Test if all 3D points have -ve Z in camera coordinate system.
	// If not, then negate all rij.

	for ( i = 1; i <= numPoints; i++ )
	{
		double Zcam = r31*new_pt3D[i][1] + r32*new_pt3D[i][2] + r33*new_pt3D[i][3];
		if ( Zcam >= 0.0 ) break;
	}

	if ( i <= numPoints )
	{
		r11 *= -1;
		r12 *= -1;
		r13 *= -1;
		r21 *= -1;
		r22 *= -1;
		r23 *= -1;
		r31 *= -1;
		r32 *= -1;
		r33 *= -1;
	}


	// intMat = [ a 0 ox; 0 b oy; 0 0 1 ]

	intMat[1][1] = a;
	intMat[1][2] = 0;
	intMat[1][3] = ox;
	intMat[2][1] = 0;
	intMat[2][2] = b;
	intMat[2][3] = oy;
	intMat[3][1] = 0;
	intMat[3][2] = 0;
	intMat[3][3] = 1;
	

	// extMat = [ r11 r12 r13 t1; r21 r22 r23 t2; r31 r32 r33 t3 ]

	extMat[1][1] = r11;
	extMat[1][2] = r12;
	extMat[1][3] = r13;
	extMat[1][4] = r11*(-camPos[1]) + r12*(-camPos[2]) + r13*(-camPos[3]);
	extMat[2][1] = r21;
	extMat[2][2] = r22;
	extMat[2][3] = r23;
	extMat[2][4] = r21*(-camPos[1]) + r22*(-camPos[2]) + r23*(-camPos[3]);
	extMat[3][1] = r31;
	extMat[3][2] = r32;
	extMat[3][3] = r33;
	extMat[3][4] = r31*(-camPos[1]) + r32*(-camPos[2]) + r33*(-camPos[3]);


	free_dmatrix( P, 1, 3, 1, 3 );
	free_dmatrix( new_pt3D, 1, numPoints, 1, 3 );
}




void compProjMatFromOrig( double *const *pt3D, double *const *pt2D, 
					      int numPoints, double **projMat )
	// Compute the 3x3 projection matrix given that the camera is already at the origin.  
	// INPUT:
	//   - pt3D[1..numPoints][1..3] -- 3D coordinates of 3D points.
	//	 - pt2D[1..numPoints][1..2] -- 2D pixel coordinates of image points.
	//	 - numPoints -- number of pair-correspondences (must be at least 4).
	// OUTPUT:
	//	 - projMat[1..3][1..3] -- the projection matrix (up to an arbitrary sign).
{

// Solve for vector M in the homogeneous system A*M=0.
// Vector M contains the 9 entries of the projection matrix.

	if ( numPoints < 4 )
	{
		fprintf( stderr, "compProjMatFromOrig: must have at least 4 pair-correspondences.\n" );
		exit(1);
	}

	int i;

	// Setup matrix A.
	double **A = dmatrix( 1, 2 * numPoints + 1, 1, 9 );

	for ( i = 1; i <= numPoints; i++ )
	{
		A[2*i-1][1] = pt3D[i][1];
		A[2*i-1][2] = pt3D[i][2];
		A[2*i-1][3] = pt3D[i][3];
		A[2*i-1][4] = 0;
		A[2*i-1][5] = 0;
		A[2*i-1][6] = 0;
		A[2*i-1][7] = -pt2D[i][1] * pt3D[i][1];
		A[2*i-1][8] = -pt2D[i][1] * pt3D[i][2];
		A[2*i-1][9] = -pt2D[i][1] * pt3D[i][3];

		A[2*i][1] = 0;
		A[2*i][2] = 0;
		A[2*i][3] = 0;
		A[2*i][4] = pt3D[i][1];
		A[2*i][5] = pt3D[i][2];
		A[2*i][6] = pt3D[i][3];
		A[2*i][7] = -pt2D[i][2] * pt3D[i][1];
		A[2*i][8] = -pt2D[i][2] * pt3D[i][2];
		A[2*i][9] = -pt2D[i][2] * pt3D[i][3];
	}

	A[2 * numPoints + 1][1] = 0;
	A[2 * numPoints + 1][2] = 0;
	A[2 * numPoints + 1][3] = 0;
	A[2 * numPoints + 1][4] = 0;
	A[2 * numPoints + 1][5] = 0;
	A[2 * numPoints + 1][6] = 0;
	A[2 * numPoints + 1][7] = 0;
	A[2 * numPoints + 1][8] = 0;
	A[2 * numPoints + 1][9] = 0;


	double *W = dvector( 1, 9 );		   // will store the sigular values.
	double **V = dmatrix( 1, 9, 1, 9 );	

	// Singular-Value Decompose A into U*W*transpose(V). U will be stored in A.
	dsvdcmp( A, 2 * numPoints + 1, 9, W, V );

	// Look for the smallest singular value in W. 
	double minw = W[1];
	int minw_index = 1;

	for ( i = 2; i <= 9; i++ )
		if ( W[i] < minw )
		{
			minw = W[i];
			minw_index = i;
		}

	// The column of V corresponding to the smallest singular value is the 
	// closest solution to A*M=0.

	projMat[1][1] = V[1][minw_index];
	projMat[1][2] = V[2][minw_index];
	projMat[1][3] = V[3][minw_index];
	projMat[2][1] = V[4][minw_index];
	projMat[2][2] = V[5][minw_index];
	projMat[2][3] = V[6][minw_index];
	projMat[3][1] = V[7][minw_index];
	projMat[3][2] = V[8][minw_index];
	projMat[3][3] = V[9][minw_index];

	// Scale projMat so that its last row has magnitude 1.

	double last_row_len = sqrt( sqr( projMat[3][1] ) + sqr( projMat[3][2] ) + 
		                        sqr( projMat[3][3] ) );

	projMat[1][1] /= last_row_len;
	projMat[1][2] /= last_row_len;
	projMat[1][3] /= last_row_len;
	projMat[2][1] /= last_row_len;
	projMat[2][2] /= last_row_len;
	projMat[2][3] /= last_row_len;
	projMat[3][1] /= last_row_len;
	projMat[3][2] /= last_row_len;
	projMat[3][3] /= last_row_len;


	free_dmatrix( V, 1, 9, 1, 9 );
	free_dvector( W, 1, 9 );
	free_dmatrix( A, 1, 2 * numPoints+1, 1, 9 );
}




void compIntExtMats( double *const *pt3D, double *const *pt2D, 
				     int numPoints, double **intMat, double **extMat )
	// Compute the 3x3 intrinsic matrix and the 3x4 extrinsic matrix given that
	// the camera's COP position is unknown.
	// INPUT:
	//	 - pt3D[1..numPoints][1..3] -- 3D coordinates of 3D points.
	//	 - pt2D[1..numPoints][1..2] -- 2D pixel coordinates of image points.
	//	 - numPoints -- number of pair-correspondences (must be at least 6).
	// OUTPUT:
	//   - intMat[1..3][1..3] -- the intrinsic matrix.
	//   - extMat[1..3][1..4] -- the extrinsic matrix.
{
	if ( numPoints < 6 )
	{
		fprintf( stderr, "compIntExtMats: must have at least 6 pair-correspondences.\n" );
		exit(1);
	}

	int i;

	double **P = dmatrix( 1, 3, 1, 4 );	// 3x4 projection matrix

	// Compute a 3x4 projection matrix. 
	compProjMatFromCam( pt3D, pt2D, numPoints, P );

	// Compute camera parameters.
	double ox = P[1][1]*P[3][1] + P[1][2]*P[3][2] + P[1][3]*P[3][3];
	double oy = P[2][1]*P[3][1] + P[2][2]*P[3][2] + P[2][3]*P[3][3];

	double a = -sqrt( P[1][1]*P[1][1] + P[1][2]*P[1][2] + P[1][3]*P[1][3] - ox*ox );
	double b = -sqrt( P[2][1]*P[2][1] + P[2][2]*P[2][2] + P[2][3]*P[2][3] - oy*oy );

	double r11 = ( P[1][1] - ox * P[3][1] ) / a;
	double r12 = ( P[1][2] - ox * P[3][2] ) / a;
	double r13 = ( P[1][3] - ox * P[3][3] ) / a;
	double r21 = ( P[2][1] - oy * P[3][1] ) / b;
	double r22 = ( P[2][2] - oy * P[3][2] ) / b;
	double r23 = ( P[2][3] - oy * P[3][3] ) / b;
	double r31 = P[3][1];
	double r32 = P[3][2];
	double r33 = P[3][3];

	double t3 = P[3][4];
	double t1 = (P[1][4] - ox * t3 ) / a;
	double t2 = (P[2][4] - oy * t3 ) / b;


	// Test if all 3D points have -ve Z in camera coordinate system.
	// If not, then negate all rij and ti.

	for ( i = 1; i <= numPoints; i++ )
	{
		double Zcam = r31*pt3D[i][1] + r32*pt3D[i][2] + r33*pt3D[i][3] + t3;
		if ( Zcam >= 0.0 ) break;
	}

	if ( i <= numPoints )
	{
		r11 *= -1;
		r12 *= -1;
		r13 *= -1;
		r21 *= -1;
		r22 *= -1;
		r23 *= -1;
		r31 *= -1;
		r32 *= -1;
		r33 *= -1;
		t1 *= -1;
		t2 *= -1;
		t3 *= -1;
	}


	// intMat = [ a 0 ox; 0 b oy; 0 0 1 ]

	intMat[1][1] = a;
	intMat[1][2] = 0;
	intMat[1][3] = ox;
	intMat[2][1] = 0;
	intMat[2][2] = b;
	intMat[2][3] = oy;
	intMat[3][1] = 0;
	intMat[3][2] = 0;
	intMat[3][3] = 1;
	

	// extMat = [ r11 r12 r13 t1; r21 r22 r23 t2; r31 r32 r33 t3 ]

	extMat[1][1] = r11;
	extMat[1][2] = r12;
	extMat[1][3] = r13;
	extMat[1][4] = t1;
	extMat[2][1] = r21;
	extMat[2][2] = r22;
	extMat[2][3] = r23;
	extMat[2][4] = t2;
	extMat[3][1] = r31;
	extMat[3][2] = r32;
	extMat[3][3] = r33;
	extMat[3][4] = t3;


	free_dmatrix( P, 1, 3, 1, 4 );
}




void compProjMatFromCam( double *const *pt3D, double *const *pt2D, 
					     int numPoints, double **projMat )
	// Compute the 3x4 projection matrix given that the camera's position is unknown.  
	// INPUT:
	//   - pt3D[1..numPoints][1..3] -- 3D coordinates of 3D points.
	//	 - pt2D[1..numPoints][1..2] -- 2D pixel coordinates of image points.
	//	 - numPoints -- number of pair-correspondences (must be at least 6).
	// OUTPUT:
	//	 - projMat[1..3][1..4] -- the projection matrix (up to an arbitrary sign).
{

// Solve for vector M in the homogeneous system A*M=0.
// Vector M contains the 12 entries of the projection matrix.

	if ( numPoints < 6 )
	{
		fprintf( stderr, "compProjMatFromCam: must have at least 6 pair-correspondences.\n" );
		exit(1);
	}

	int i;

	// Setup matrix A.
	double **A = dmatrix( 1, 2*numPoints, 1, 12 );

	for ( i = 1; i <= numPoints; i++ )
	{
		A[2*i-1][1] = pt3D[i][1];
		A[2*i-1][2] = pt3D[i][2];
		A[2*i-1][3] = pt3D[i][3];
		A[2*i-1][4] = 1;
		A[2*i-1][5] = 0;
		A[2*i-1][6] = 0;
		A[2*i-1][7] = 0;
		A[2*i-1][8] = 0;
		A[2*i-1][9] = -pt2D[i][1] * pt3D[i][1];
		A[2*i-1][10] = -pt2D[i][1] * pt3D[i][2];
		A[2*i-1][11] = -pt2D[i][1] * pt3D[i][3];
		A[2*i-1][12] = -pt2D[i][1];

		A[2*i][1] = 0;
		A[2*i][2] = 0;
		A[2*i][3] = 0;
		A[2*i][4] = 0;
		A[2*i][5] = pt3D[i][1];
		A[2*i][6] = pt3D[i][2];
		A[2*i][7] = pt3D[i][3];
		A[2*i][8] = 1;
		A[2*i][9] = -pt2D[i][2] * pt3D[i][1];
		A[2*i][10] = -pt2D[i][2] * pt3D[i][2];
		A[2*i][11] = -pt2D[i][2] * pt3D[i][3];
		A[2*i][12] = -pt2D[i][2];
	}


	double *W = dvector( 1, 12 );		   // will store the sigular values.
	double **V = dmatrix( 1, 12, 1, 12 );	

	// Singular-Value Decompose A into U*W*transpose(V). U will be stored in A.
	dsvdcmp( A, 2 * numPoints, 12, W, V );

	// Look for the smallest singular value in W. 
	double minw = W[1];
	int minw_index = 1;

	for ( i = 2; i <= 12; i++ )
		if ( W[i] < minw )
		{
			minw = W[i];
			minw_index = i;
		}

	// The column of V corresponding to the smallest singular value is the 
	// closest solution to A*M=0.

	projMat[1][1] = V[1][minw_index];
	projMat[1][2] = V[2][minw_index];
	projMat[1][3] = V[3][minw_index];
	projMat[1][4] = V[4][minw_index];
	projMat[2][1] = V[5][minw_index];
	projMat[2][2] = V[6][minw_index];
	projMat[2][3] = V[7][minw_index];
	projMat[2][4] = V[8][minw_index];
	projMat[3][1] = V[9][minw_index];
	projMat[3][2] = V[10][minw_index];
	projMat[3][3] = V[11][minw_index];
	projMat[3][4] = V[12][minw_index];

	// Scale projMat so that its last row has magnitude 1.

	double last_row_len = sqrt( sqr( projMat[3][1] ) + sqr( projMat[3][2] ) + 
		                        sqr( projMat[3][3] ) );

	projMat[1][1] /= last_row_len;
	projMat[1][2] /= last_row_len;
	projMat[1][3] /= last_row_len;
	projMat[1][4] /= last_row_len;
	projMat[2][1] /= last_row_len;
	projMat[2][2] /= last_row_len;
	projMat[2][3] /= last_row_len;
	projMat[2][4] /= last_row_len;
	projMat[3][1] /= last_row_len;
	projMat[3][2] /= last_row_len;
	projMat[3][3] /= last_row_len;
	projMat[3][4] /= last_row_len;


	free_dmatrix( V, 1, 12, 1, 12 );
	free_dvector( W, 1, 12 );
	free_dmatrix( A, 1, 2*numPoints, 1, 12 );
}

