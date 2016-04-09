#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <GL/glut.h>
#include "common.h"
#include "trimodel.h"
#include "kl_convexhull2d.h"
#include "kl_minquad.h"
#include "kl_optifrustum.h"


// scene model
static char *sceneFile = "teapot.dli";
static double sceneScaleFact = 1.0;
static TM_Model *scene;
static GLfloat sceneRadius;
static TM_DList *sceneDList;

// lights file
static char *lightFile = "lights.dat";
static GLint numLights;
static GLfloat lightPos[8][4];
static GLenum lightID[8] = { GL_LIGHT0, GL_LIGHT1, GL_LIGHT2, GL_LIGHT3,  
							 GL_LIGHT4, GL_LIGHT5, GL_LIGHT6, GL_LIGHT7 };


static GLint win_width = 1024, win_height = 768;


static bool drawScene = true;
static bool texMap = true;
static bool wire = false;
static bool hasLighting = true;
static bool backfaceCulling = true;
static int drawMode = 0;			// 0 -- user, 1 -- convenient, 2 -- optimized


static int numPoints = 0;
static double *points3D = NULL;
static double sphereCenter[3];
static double sphereRadius;

static double nearPlane, farPlane;
static double fovy;

static double eye[3] = { 0.0, 2.0, 2.0 };
static double minQuad[8];



static double sqrDist3( const double p1[3], const double p2[3] )
	// returns square of the distance between 2 points
{
	return fsqr( p1[0] - p2[0] ) + fsqr( p1[1] - p2[1] ) + fsqr( p1[2] - p2[2] );
}



static void InitLights( void )
{
	FILE *fp = fopen( lightFile, "r" );
	if ( fp == NULL ) error_exit( "InitLights()", "Cannot read \"%s\"", lightFile );

	float am[4], di[4], sp[4], pos[4];

	if ( fscanf( fp, "%d", &numLights ) < 1 )
		error_exit( "InitLights()", "Invalid file \"%s\"", lightFile );

	if ( numLights > 8 )
		error_exit( "InitLights()", "Too many lights (max = 8)" );

	for ( int i = 0; i < numLights; i++ )
	{
		if ( fscanf( fp, "%f %f %f %f", &am[0], &am[1], &am[2], &am[3] ) < 4 )
			error_exit( "InitLights()", "Invalid file \"%s\"", lightFile );
		if ( fscanf( fp, "%f %f %f %f", &di[0], &di[1], &di[2], &di[3] ) < 4 )
			error_exit( "InitLights()", "Invalid file \"%s\"", lightFile );
		if ( fscanf( fp, "%f %f %f %f", &sp[0], &sp[1], &sp[2], &sp[3] ) < 4 )
			error_exit( "InitLights()", "Invalid file \"%s\"", lightFile );
		if ( fscanf( fp, "%f %f %f %f", &pos[0], &pos[1], &pos[2], &pos[3] ) < 4 )
			error_exit( "InitLights()", "Invalid file \"%s\"", lightFile );

		glLightfv( lightID[i], GL_AMBIENT, am );
		glLightfv( lightID[i], GL_DIFFUSE, di );
		glLightfv( lightID[i], GL_SPECULAR, sp );

		lightPos[i][0] = pos[0];
		lightPos[i][1] = pos[1];
		lightPos[i][2] = pos[2];
		lightPos[i][3] = pos[3];

		glEnable( lightID[i] );
	}

	glLightModeli( GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE );

	fclose( fp );
}



static void init( void )
{
	scene = TM_ReadModel( sceneFile, sceneScaleFact );
	sceneRadius = 0.5 * sqrt( fsqr(scene->dim_xyz[0]) + 
		                      fsqr(scene->dim_xyz[1]) + fsqr(scene->dim_xyz[2]) );
	sceneDList = TM_MakeOGLDList( scene, true );


	numPoints = 3 * scene->numTris;
	points3D = (double *) checked_malloc( 3 * numPoints * sizeof( double ) );

	for ( int tc = 0; tc < scene->numTris; tc++ )
	{
		for ( int vc = 0; vc < 3; vc++ )
		{
			points3D[9*tc+3*vc+0] = scene->tri[tc].v[vc][0];
			points3D[9*tc+3*vc+1] = scene->tri[tc].v[vc][1];
			points3D[9*tc+3*vc+2] = scene->tri[tc].v[vc][2];
		}
	}

	fprintf( stderr, "Number of points = %d\n", numPoints ); 

	// Compute bounding sphere
	kl_BoundingSphere( points3D, numPoints, sphereCenter, &sphereRadius );

	eye[0] *= sphereRadius;
	eye[1] *= sphereRadius;
	eye[2] *= sphereRadius;

	double dist = sqrt( sqrDist3( eye, sphereCenter ) );
	fovy = 2.0 * asin( sphereRadius / dist ) / M_PI * 180.0;
	nearPlane = dist - sphereRadius;
	farPlane = dist + sphereRadius;

	glEnable( GL_DEPTH_TEST );
	glDisable( GL_DITHER );
	glDisable( GL_LIGHTING );
	glDisable( GL_BLEND );
	glDisable( GL_TEXTURE_2D );

	InitLights();
}



static void display( void )
{
	static double opti_modelview[16], opti_projection[16];
	static double conv_modelview[16], conv_projection[16];

	glViewport( 0, 0, win_width, win_height );
	glEnable( GL_DEPTH_TEST );

	if ( drawMode == 1 || drawMode == 2 )
		kl_OptiFrustum( points3D, numPoints, sphereCenter, sphereRadius,
						eye, win_width, win_height, opti_projection, opti_modelview,
						conv_projection, conv_modelview, minQuad );

	if ( drawMode == 2 )
	{
		printf( "Optimized frustum.\n" );
		glMatrixMode( GL_PROJECTION );
		glLoadMatrixd( opti_projection );
		glMatrixMode( GL_MODELVIEW );
		glLoadMatrixd( opti_modelview );
	}
	else if ( drawMode == 1 )
	{
		printf( "Intermediate \"convenient\" frustum, with approx minimal-enclosing quadrilateral.\n" );
		glMatrixMode( GL_PROJECTION );
		glLoadMatrixd( conv_projection );
		glMatrixMode( GL_MODELVIEW );
		glLoadMatrixd( conv_modelview );
	}
	else if ( drawMode == 0 )
	{
		printf( "User-defined frustum.\n" );
		glMatrixMode( GL_PROJECTION );
		glLoadIdentity();
		gluPerspective( fovy, (GLdouble) win_width / (GLdouble) win_height, nearPlane, farPlane );
		glMatrixMode( GL_MODELVIEW );
		glLoadIdentity();
		gluLookAt( eye[0], eye[1], eye[2], sphereCenter[0], sphereCenter[1], sphereCenter[2],
			       0.0, -1.0, 0.0 );
	}
			     
	glPushAttrib( GL_ALL_ATTRIB_BITS );

	glClearColor( 0.0F, 0.1F, 0.15F, 0.0F );
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	glColor3f( 1.0, 1.0, 1.0 );

	if ( hasLighting )
		for ( int i = 0; i < numLights; i++ ) glLightfv( lightID[i], GL_POSITION, lightPos[i] );

	if ( drawScene ) TM_DrawModel( sceneDList, texMap, hasLighting, wire, false, backfaceCulling );


	if ( drawMode == 1 )
	{
		glPushAttrib( GL_ALL_ATTRIB_BITS );
		glDisable( GL_DEPTH_TEST );
		glMatrixMode( GL_PROJECTION );
		glLoadIdentity();
		gluOrtho2D( 0, win_width, 0, win_height );
		glMatrixMode( GL_MODELVIEW );
		glLoadIdentity();

		glLineWidth( 3.0 );
		glBegin( GL_LINE_LOOP );
			glColor3f( 1.0, 0.0, 0.0 );
			glVertex2dv( &minQuad[0] );
			glColor3f( 0.0, 1.0, 0.0 );
			glVertex2dv( &minQuad[2] );
			glColor3f( 0.0, 0.0, 1.0 );
			glVertex2dv( &minQuad[4] );
			glColor3f( 1.0, 1.0, 1.0 );
			glVertex2dv( &minQuad[6] );

		glEnd();
		glPopAttrib();
	}
		
	glPopAttrib();

	glutSwapBuffers();

	printf( "Press 'm' to change view.\n" );
}


static void reshape( int width, int height )
{
	win_width = width;
	win_height = height;
}



static void keyboard( unsigned char key, int x, int y )
{
	int modi = glutGetModifiers();

	switch ( key )
	{
	case 'q':
	case 'Q':
			exit( 0 );
			break;
	case 't':
	case 'T':
			texMap = !texMap;
			break;
	case 'w':
	case 'W':
			wire = !wire;
			break;
	case 'l':
	case 'L':
			hasLighting = !hasLighting;
			break;

	case 'c':
	case 'C':
			backfaceCulling = !backfaceCulling;
			break;

	case 'm':
	case 'M':
			drawMode = (drawMode + 1) % 3;
			break;

	default:
			return;
	}

	glutPostRedisplay();
}



int main( int argc, char** argv )
{
	glutInit( &argc, argv );
	glutInitDisplayMode( GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE );

	glutInitWindowSize( win_width, win_height );
	glutCreateWindow( "" );
	glutReshapeFunc( reshape );
	glutDisplayFunc( display );
	glutKeyboardFunc( keyboard );
	init();

	glutMainLoop();
	return 0;
}
