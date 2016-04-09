#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <GL/glut.h>
#include "kl_convexhull2d.h"
#include "kl_minquad.h"
#include "kl_camcalib.h"
#include "kl_optifrustum.h"


#ifdef __cplusplus
extern "C" {
#endif

#include "nrutil.h"

#ifdef __cplusplus
}
#endif


#ifndef M_PI
#define M_PI    3.14159265358979323846
#endif


static 
void *checked_malloc( size_t size )
	// same as malloc(), but checks for out-of-memory.
{
	void *p = malloc( size );
	if ( p == NULL )
	{
		fprintf( stderr, "Cannot allocate memory\n." );
		exit( 1 );
	}
	return p;
}


static
double fsqr( double f )
    // returns the square of f.
{
    return ( f * f );
}


static
double sqrDist3( const double p1[3], const double p2[3] )
	// returns square of the distance between 2 points
{
	return fsqr( p1[0] - p2[0] ) + fsqr( p1[1] - p2[1] ) + fsqr( p1[2] - p2[2] );
}


static
double sqrDist2( const double p1[2], const double p2[2] )
	// returns square of the distance between 2 points
{
	return fsqr( p1[0] - p2[0] ) + fsqr( p1[1] - p2[1] );
}



static 
void CompBoundingBox( const double *pt, int numPts, double min_xyz[3], double max_xyz[3] )
{
	min_xyz[0] = FLT_MAX;
	min_xyz[1] = FLT_MAX;
	min_xyz[2] = FLT_MAX;
	max_xyz[0] = -FLT_MAX;
	max_xyz[1] = -FLT_MAX;
	max_xyz[2] = -FLT_MAX;

	for ( int i = 0; i < numPts; i++ )
	{
		if ( pt[3*i+0] < min_xyz[0] ) min_xyz[0] = pt[3*i+0];
		if ( pt[3*i+1] < min_xyz[1] ) min_xyz[1] = pt[3*i+1];
		if ( pt[3*i+2] < min_xyz[2] ) min_xyz[2] = pt[3*i+2];
		if ( pt[3*i+0] > max_xyz[0] ) max_xyz[0] = pt[3*i+0];
		if ( pt[3*i+1] > max_xyz[1] ) max_xyz[1] = pt[3*i+1];
		if ( pt[3*i+2] > max_xyz[2] ) max_xyz[2] = pt[3*i+2];
	}
}




void kl_BoundingSphere( const double inPoints3D[],int numInPoints3D, 
					    double sphereCenter[3], double *sphereRadius )
{
	double min_xyz[3], max_xyz[3];
	CompBoundingBox( inPoints3D, numInPoints3D, min_xyz, max_xyz );

	sphereCenter[0] = 0.5 * (max_xyz[0] + min_xyz[0]);
	sphereCenter[1] = 0.5 * (max_xyz[1] + min_xyz[1]);
	sphereCenter[2] = 0.5 * (max_xyz[2] + min_xyz[2]);

	*sphereRadius = sqrt( sqrDist3( sphereCenter, min_xyz ) );
}


//============================================================================================


static
void mat16dMult( double mo[16], const double m1[16], const double m2[16] )
	// mo = m1 x m2.
{
	double M1[4][4], M2[4][4], MO[4][4];
	int i, j, k;

	for ( i = 0; i < 4; i++ )
		for ( j = 0; j < 4; j++ )
		{
			M1[i][j] = m1[ j * 4 + i ];
			M2[i][j] = m2[ j * 4 + i ];
		}

	for ( i = 0; i < 4; i++ )
		for ( j = 0; j < 4; j++ )
		{
			double sum = 0.0;
			for ( k = 0; k < 4; k++ )
				sum += M1[i][k] * M2[k][j];
			MO[i][j] = sum;
		}

	for ( i = 0; i < 4; i++ )
		for ( j = 0; j < 4; j++ )
			mo[ j * 4 + i ] = MO[i][j];
}



static 
void MyGluProject( const double obj[3], const GLdouble mat[16], const GLint viewport[4],
				   double winxy[2] )
{
	double tmp[4];
	tmp[0] = mat[0]*obj[0] + mat[4]*obj[1] + mat[8]*obj[2] + mat[12];
	tmp[1] = mat[1]*obj[0] + mat[5]*obj[1] + mat[9]*obj[2] + mat[13];
	tmp[3] = mat[3]*obj[0] + mat[7]*obj[1] + mat[11]*obj[2] + mat[15];
	tmp[0] /= tmp[3];
	tmp[1] /= tmp[3];
	winxy[0] = (tmp[0] + 1.0) * ( viewport[2] / 2.0 ) + viewport[0];
	winxy[1] = (tmp[1] + 1.0) * ( viewport[3] / 2.0 ) + viewport[1];
}



//============================================================================================


static 
void CompUpVector( const double eye[3], const double lookat[3], double upvec[3] )
{
	double frontvec[3] = { lookat[0] - eye[0], lookat[1] - eye[1], lookat[2] - eye[2] };

	if ( frontvec[0] != 0.0  ||  frontvec[1] != 0.0 )
	{
		upvec[0] = frontvec[1];
		upvec[1] = -frontvec[0];
		upvec[2] = 0.0;
	}
	else	
	{
		upvec[0] = 0.0;
		upvec[1] = 1.0;
		upvec[2] = 0.0;
	}
}


static
void SetUpPerspective( const double eye[3], const double sphereCenter[3], double sphereRadius,
					   double viewport_width, double viewport_height )
{
	double dist = sqrt( sqrDist3( eye, sphereCenter ) );
	double fovy, zNear, zFar;;

	if ( dist <= sphereRadius + 1.0E-4 )
	{
		fovy = 89.0;
		zNear = sphereRadius / 100.0;
		zFar = dist + sphereRadius;
	}
	else
	{
		fovy = 2.0 * asin( sphereRadius / dist ) / M_PI * 180.0;
		zNear = dist - sphereRadius;
		zFar = dist + sphereRadius;
	}

	gluPerspective( fovy, viewport_width / viewport_height, zNear, zFar );

// Note that not all points will appear in the viewport if the viewport is much taller
// than it is wide. But this does not affect the final optmized frustum.
}



void kl_OptiFrustum( const double inPoints3D[],int numInPoints3D, 
					 const double sphereCenter[3], double sphereRadius,
					 const double viewpoint[3],
					 int viewportWidth, int viewportHeight,
					 GLdouble optiProjectionMat[16], GLdouble optiModelviewMat[16],
					 GLdouble convProjectionMat[16], GLdouble convModelviewMat[16],
					 double minEnclosingQuad[8] )
{
// Save current OpenGL matrix mode, viewport and matrices
	GLint orig_mat_mode;
	GLdouble orig_projection[16], orig_modelview[16];
	GLint orig_viewport[4];
	glGetIntegerv( GL_MATRIX_MODE, &orig_mat_mode );
	glGetDoublev( GL_PROJECTION_MATRIX, orig_projection );
	glGetDoublev( GL_MODELVIEW_MATRIX, orig_modelview );
	glGetIntegerv( GL_VIEWPORT, orig_viewport );

// Set up viewport and symmetric perspective frustum
	glViewport( 0, 0, viewportWidth, viewportHeight );
	glMatrixMode( GL_PROJECTION  );
	glLoadIdentity();
	SetUpPerspective( viewpoint, sphereCenter, sphereRadius, viewportWidth, viewportHeight );
	glMatrixMode( GL_MODELVIEW );
	glLoadIdentity();
	double up_vec[3];
	CompUpVector( viewpoint, sphereCenter, up_vec );
	gluLookAt( viewpoint[0], viewpoint[1], viewpoint[2],
		       sphereCenter[0], sphereCenter[1], sphereCenter[2], up_vec[0], up_vec[1], up_vec[2] );
		       
// Read back the 2 transformation matrices and viewport from OpenGL
	GLdouble conv_projection[16], conv_modelview[16];
	GLint conv_viewport[4];
	glGetDoublev( GL_PROJECTION_MATRIX, conv_projection );
	glGetDoublev( GL_MODELVIEW_MATRIX, conv_modelview );
	glGetIntegerv( GL_VIEWPORT, conv_viewport );

// Restore saved OpenGL matrices, viewport and matrix mode
	glViewport( orig_viewport[0], orig_viewport[1], orig_viewport[2], orig_viewport[3] );
	glMatrixMode( GL_PROJECTION  );
	glLoadMatrixd( orig_projection );
	glMatrixMode( GL_MODELVIEW );
	glLoadMatrixd( orig_modelview );
	glMatrixMode( orig_mat_mode );


// Project 3D points to 2D image points
	int numPoints2D = numInPoints3D;
	double *points2D = (double *) checked_malloc( sizeof(double) * 2 * numPoints2D );

	GLdouble conv_mat[16];
	mat16dMult( conv_mat, conv_projection, conv_modelview );  // concat 2 matrices
	int i;
	for ( i = 0; i < numInPoints3D; i++ )
		MyGluProject( &inPoints3D[3*i], conv_mat, conv_viewport, &points2D[2*i] );

// Compute approx. smallest enclosing quadrilateral of the 2D image points
	double minQuad[8];
	kl_MinQuad( points2D, numPoints2D, minQuad );

	free( points2D );


// Find the longer dimension of the viewport and put it as the first edge
	double vpCornersW[8] = { 0.5, 0.5,                                viewportWidth-0.5, 0.5,
			                 viewportWidth-0.5, viewportHeight-0.5,   0.5, viewportHeight-0.5 };
	double vpCornersH[8] = { viewportWidth-0.5, 0.5,     viewportWidth-0.5, viewportHeight-0.5,   
		                     0.5, viewportHeight-0.5,    0.5, 0.5 };
	double *vpCorners;
	if ( viewportWidth > viewportHeight ) vpCorners = vpCornersW; else vpCorners = vpCornersH;


// Find the longest 2 opposite edges of the quadrilateral and put them as the first and third edges
	double qLen[4];
	qLen[0] = sqrt( sqrDist2( &minQuad[0], &minQuad[2] ) );
	qLen[1] = sqrt( sqrDist2( &minQuad[2], &minQuad[4] ) );
	qLen[2] = sqrt( sqrDist2( &minQuad[4], &minQuad[6] ) );
	qLen[3] = sqrt( sqrDist2( &minQuad[6], &minQuad[0] ) );

	if ( qLen[1] + qLen[3] > qLen[0] + qLen[2] )
	{
		double x0 = minQuad[0];
		double y0 = minQuad[1];
		for ( int i = 0; i < 6; i++ ) minQuad[i] = minQuad[i+2];
		minQuad[6] = x0;
		minQuad[7] = y0;
	}


// Project 2D quadrilateral into 3D
	double quad3D[3*4];
	for ( i = 0; i < 4; i++ )
	{
		gluUnProject( minQuad[2*i], minQuad[2*i+1], 0.5,
			          conv_modelview, conv_projection, conv_viewport,
					  &quad3D[3*i+0], &quad3D[3*i+1], &quad3D[3*i+2] );
	}


// Set up matrices to solve for camera parameters

	double **pt3D = dmatrix( 1, 4, 1, 3 );
	double **pt2D = dmatrix( 1, 4, 1, 2 );
	double **intMat = dmatrix( 1, 3, 1, 3 );
	double **extMat = dmatrix( 1, 3, 1, 4 );

	for ( i = 0; i < 4; i++ )
	{
		pt3D[i+1][1] = quad3D[3*i+0];
		pt3D[i+1][2] = quad3D[3*i+1];
		pt3D[i+1][3] = quad3D[3*i+2];
		pt2D[i+1][1] = vpCorners[2*i+0];
		pt2D[i+1][2] = vpCorners[2*i+1];
	}

	compIntExtMats( viewpoint - 1, pt3D, pt2D, 4, intMat, extMat );

	double xformSphereZ;
	xformSphereZ = extMat[3][1] * sphereCenter[0] + extMat[3][2] * sphereCenter[1] +
				   extMat[3][3] * sphereCenter[2] + extMat[3][4];

	double zNear = -xformSphereZ - sphereRadius;
	if ( zNear <= 0.0 ) zNear = sphereRadius / 100.0;
	double zFar = zNear + 2.0 * sphereRadius;

	convToOpenGLMats( intMat, extMat, zNear, zFar,
	 				  viewportWidth, viewportHeight, optiProjectionMat, optiModelviewMat );

	free_dmatrix( extMat, 1, 3, 1, 4 );
	free_dmatrix( intMat, 1, 3, 1, 3 );
	free_dmatrix( pt3D, 1, 4, 1, 3 );
	free_dmatrix( pt2D, 1, 4, 1, 2 );


// Copy to optional outputs
	if ( convProjectionMat != NULL )
		for ( i = 0; i < 16; i++ ) convProjectionMat[i] = conv_projection[i];

	if ( convModelviewMat != NULL )
		for ( i = 0; i < 16; i++ ) convModelviewMat[i] = conv_modelview[i];

	if ( minEnclosingQuad != NULL )
		for ( i = 0; i < 8; i++ ) minEnclosingQuad[i] = minQuad[i];
}


