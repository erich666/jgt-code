/////////////////////////////////////////////////////////////////////////
// 
// Pseudo Cursor Example Implementation
// http://www.acm.org/jgt/papers/Steed02/
//
// (c) Anthony Steed 2001-2003, A.Steed@cs.ucl.ac.uk, asteedd@acm.org
// http://www.cs.ucl.ac.uk/staff/A.Steed/
//
// This program is freely distributable without licensing fees and is
// provided without guarantee or warrantee expressed or implied. This
// program is -not- in the public domain.
//
// Since the original paper and code was written for JGT, Everitt and
// Kilgard published their paper "Practical and Robust Stenciled
// Shadow Volumes for Hardware-Accelerated Rendering" (available
// online at http://developer.nvidia.com). Their z-fail approach was
// adopted here as it simplified the code which otherwise had to deal
// with capping the shadow volumes on the near clipping plane since
// the infinite slab would often intersect the view plane window. To
// use the z-fail approach the "infinite slabs" are now large enough
// that they extend beyond the scene, but are not clipped by the far
// clipping plane.
//
// Compiling this example requires GLUI from
// http://www.cs.unc.edu/~rademach/glui/
//
// Plaforms other than Win32 and IRIX are as yet untested. It should
// be trivial to port to anything supporting OpenGL 1.0 for stencil
// volume version, or OpenGL 1.1 for texture volume version.
//
// $Id: pseudo_cursor.cpp,v 1.8 2003/06/15 21:11:49 ucacajs Exp ucacajs $
//
/////////////////////////////////////////////////////////////////////////


#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#include <GL/glut.h>
#include <GL/glext.h>
#else
#include <GL/glx.h>
#include <GL/glut.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "glui.h"

// Win32 symbol loading
#ifdef _WIN32
PFNGLTEXIMAGE3DEXTPROC glTexImage3DEXT;
#endif

// OpenGL ids
enum {
	DL_BOGUS = 0,
	DL_CUT_OUT,
	DL_CUT_OUT_VOLUME,
	DL_SPHERE,		
	DL_CONE,
	DL_TORUS,
	DL_TWO_PLANES,
	DL_TWO_PLANES_NARROW,
	DL_TWO_PLANES_VNARROW,
	DL_TWO_PLANES_CURSOR,
	DL_NV_CULL
};

// Rendering modes. We cheat below and assume they are added to GLUI menu in this order
enum {
	M_PLAIN=0,
	M_PLANE_SHADOW,
	M_PLANE_SHADOW_NARROW,
	M_PLANE_SHADOW_VNARROW,
	M_PLANE_TSHADOW,
	M_PLANE_TSHADOW_NARROW,
	M_PLANE_TSHADOW_VNARROW,
	M_PLANE_TRANSP1,
	M_PLANE_TRANSP3
};

// Window and mode configuration
int   main_win;
GLUI *glui_win;
GLUI_RadioGroup *radio;

// GLUI control ids
static int BUTTON_ID=1;
static int RADIO_ID=2;

int window_width=800, window_height=600;
int displayMode = M_PLAIN;

// Object configurations
float light_location[4] = { 7, 8, 6, 1 };
float cursor_translate[3] = { 2,4,2 };
float cursor_rotate[16] = { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 }; 
float eye_rotate[16] = { 1,0,0,0, 0,0.8944,-0.4471,0, 0,0.4471,0.8944,0, 0,0,0,1 };
float eye_dist = 12;

// GL id for voxel shadows
GLuint texid1;
GLuint texid2;
GLuint texid3;
int enable_texture3d=0;


#define FAR_CLIP 2000.0
#define CURSOR_SIZE 20

static GLfloat texmatrix[16] = { 1.0, 0.0, 0.0, 0.0,
0.0, 1.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0,
1.5, 1.5, 0.0, 1.0 };

static void
drawFloor(void) {

	float x, y;
	float d = 1.0;

	/* Draw ground. */
	glColor3f(1,1,0);
	glNormal3f(0,1,0);

	/* Tesselate floor so lighting looks reasonable. */
	for (x=-2.9; x<=7; x+=d) {
		glBegin(GL_QUAD_STRIP);
		for (y=-2.9; y<=8; y+=d) {
			glVertex3f(x+1, -1, y);
			glVertex3f(x, -1, y);
		}
		glEnd();
	}
	for (x=-0.9; x<=7; x+=d) {
		glBegin(GL_QUAD_STRIP);
		for (y=-2.9; y<=8; y+=d) {
			glVertex3f(-3, x, y);
			glVertex3f(-3, x+1, y);
		}
		glEnd();
	}
	for (x=-2.9; x<=7; x+=d) {
		glBegin(GL_QUAD_STRIP);
		for (y=-0.9; y<=8; y+=d) {
			glVertex3f(x, y, -3);
			glVertex3f(x+1, y, -3);
		}
		glEnd();
	}
}

static void
initObjects(void) {

	glNewList(DL_SPHERE, GL_COMPILE);
	glutSolidSphere(1.0, 15, 15);
	glEndList();

	glNewList(DL_CONE, GL_COMPILE);
	glutSolidCone(1.0, 4.0, 20, 20);
	glEndList();

	glNewList(DL_TORUS, GL_COMPILE);
	glutSolidTorus(0.5,1.3, 20, 20);
	glEndList();


	glNewList(DL_TWO_PLANES, GL_COMPILE);
	glDisable(GL_CULL_FACE);
	glScalef(CURSOR_SIZE*2,0.4,CURSOR_SIZE*2);
	glutSolidCube(1.0);
	glEnable(GL_CULL_FACE);
	glEndList();

	glNewList(DL_TWO_PLANES_NARROW, GL_COMPILE);
	glDisable(GL_CULL_FACE);
	glScalef(CURSOR_SIZE*2,0.05,CURSOR_SIZE*2);
	glutSolidCube(1.0);
	glEnable(GL_CULL_FACE);
	glEndList();

	glNewList(DL_TWO_PLANES_VNARROW, GL_COMPILE);
	glDisable(GL_CULL_FACE);
	glScalef(CURSOR_SIZE*2,0.005,CURSOR_SIZE*2);
	glutSolidCube(1.0);
	glEnable(GL_CULL_FACE);
	glEndList();

	glNewList(DL_TWO_PLANES_CURSOR, GL_COMPILE);
	glDisable(GL_CULL_FACE);
	glScalef(1, 0.05,1);
	glutSolidCube(1.0);
	glEnable(GL_CULL_FACE);

	glEndList();

#if defined(GL_EXT_texture3D) 
	if (enable_texture3d) {
		{
			GLubyte *img = new GLubyte[16*16*16];
			GLubyte *p = img;

			for(int i=0; i<16; i++) {
				for(int j=0; j<16; j++) {
					for(int k=0; k<16; k++) {
						if (
							(i<4 && j < 4 && k < 4) ||
							(i<4 && j < 4 && k > 11) ||
							(i<4 && j > 11 && k < 4) ||
							(i<4 && j > 11 && k > 11) ||
							(i>11 && j < 4 && k < 4) ||
							(i>11 && j < 4 && k > 11) ||
							(i> 11 && j > 11 && k < 4 ) ||
							(i> 11 && j > 11 && k > 11) ) {
								*p++ = (GLubyte) (0xff);
							}
						else {
							*p++ = (GLubyte) (0x22);
						}
					}
				}
			}

			glGenTextures(1, &texid1);
			glBindTexture(GL_TEXTURE_3D_EXT, texid1);

			glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP);
			glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP);
			glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_R_EXT, GL_CLAMP);

			glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

			glTexImage3DEXT(GL_TEXTURE_3D_EXT, 0, GL_LUMINANCE, 16, 16, 16, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, img);
			delete [] img;
		}

		{
			GLubyte *img = new GLubyte[16*16*16];
			GLubyte *p = img;

			for(int i=0; i<16; i++) {
				for(int j=0; j<16; j++) {
					for(int k=0; k<16; k++) {
						if (
							(i<6 && j < 6 && k < 6) ||
							(i<6 && j < 6 && k > 9) ||
							(i<6 && j > 9 && k < 6) ||
							(i<6 && j > 9 && k > 9) ||
							(i>9 && j < 6 && k < 6) ||
							(i>9 && j < 6 && k > 9) ||
							(i>9 && j > 9 && k < 6 ) ||
							(i>9 && j > 9 && k > 9) ) {
								*p++ = (GLubyte) (0xff);
							}
						else {
							*p++ = (GLubyte) (0x22);
						}
					}
				}
			}
			glGenTextures(1, &texid2);
			glBindTexture(GL_TEXTURE_3D_EXT, texid2);

			glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP);
			glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP);
			glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_R_EXT, GL_CLAMP);

			glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

			glTexImage3DEXT(GL_TEXTURE_3D_EXT, 0, GL_LUMINANCE, 16, 16, 16, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, img);

			delete [] img;
		}
		{
			GLubyte *img = new GLubyte[16*16*16];
			GLubyte *p = img;

			for(int i=0; i<16; i++) {
				for(int j=0; j<16; j++) {
					for(int k=0; k<16; k++) {
						if (
							(i<8 && j < 8 && k < 8) ||
							(i<8 && j < 8 && k > 8) ||
							(i<8 && j > 8 && k < 8) ||
							(i<8 && j > 8 && k > 8) ||
							(i>8 && j < 8 && k < 8) ||
							(i>8 && j < 8 && k > 8) ||
							(i>8 && j > 8 && k < 8 ) ||
							(i>8 && j > 8 && k > 8) ) {
								*p++ = (GLubyte) (0xff);
							}
						else {
							*p++ = (GLubyte) (0x22);
						}
					}
				}
			}
			glGenTextures(1, &texid3);
			glBindTexture(GL_TEXTURE_3D_EXT, texid3);

			glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP);
			glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP);
			glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_R_EXT, GL_CLAMP);

			glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

			glTexImage3DEXT(GL_TEXTURE_3D_EXT, 0, GL_LUMINANCE, 16, 16, 16, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, img);

			delete [] img;
		}
	}
#endif
}

static void
drawObjects(void) {


	glMatrixMode( GL_MODELVIEW);
	glPushMatrix();
	glMatrixMode( GL_TEXTURE );
	glPushMatrix();
	glCallList(DL_SPHERE);
	glMatrixMode( GL_MODELVIEW);
	glPopMatrix();
	glMatrixMode( GL_TEXTURE );
	glPopMatrix();


	glMatrixMode( GL_MODELVIEW);
	glPushMatrix();
	glTranslatef(1,-1, 5.0);
	glRotatef(270, 1,0,0);
	glMatrixMode( GL_TEXTURE );
	glPushMatrix();
	glTranslatef(1,-1, 5.0);
	glRotatef(270, 1,0,0);
	glCallList(DL_CONE);
	glMatrixMode( GL_MODELVIEW);
	glPopMatrix();
	glMatrixMode( GL_TEXTURE );
	glPopMatrix();

	glMatrixMode( GL_MODELVIEW);
	glPushMatrix();
	glTranslatef(3, 0.6,-1.5);
	glMatrixMode( GL_TEXTURE );
	glPushMatrix();
	glTranslatef(3, 0.6,-1.5);
	glCallList(DL_TORUS);
	glMatrixMode( GL_MODELVIEW);
	glPopMatrix();
	glMatrixMode( GL_TEXTURE );
	glPopMatrix();

	glMatrixMode( GL_MODELVIEW);
	glPushMatrix();
	glTranslatef(-0.5,1.0,2.5);
	glMatrixMode( GL_TEXTURE );
	glPushMatrix();
	glTranslatef(-0.5,1.0,2.5);
	glutSolidCube(2.5);
	glMatrixMode( GL_MODELVIEW);
	glPopMatrix();
	glMatrixMode( GL_TEXTURE );
	glPopMatrix();

	glMatrixMode( GL_MODELVIEW);

}

static void
drawPlanes(void) {

	glPushMatrix();

	glTranslatef(cursor_translate[0],cursor_translate[1],cursor_translate[2]);
	glMultMatrixf(&cursor_rotate[0]);

	glPushMatrix();
	glCallList(DL_TWO_PLANES);
	glPopMatrix();

	glPushMatrix();
	glRotatef(90, 1,0,0);
	glCallList(DL_TWO_PLANES);
	glPopMatrix();

	glPushMatrix();
	glRotatef(90, 0,0,1);
	glCallList(DL_TWO_PLANES);
	glPopMatrix();

	glPopMatrix();
}

static void
drawPlanesNarrow(void) {

	glPushMatrix();

	glTranslatef(cursor_translate[0],cursor_translate[1],cursor_translate[2]);
	glMultMatrixf(&cursor_rotate[0]);

	glPushMatrix();
	glCallList(DL_TWO_PLANES_NARROW);
	glPopMatrix();

	glPushMatrix();
	glRotatef(90, 1,0,0);
	glCallList(DL_TWO_PLANES_NARROW);
	glPopMatrix();

	glPushMatrix();
	glRotatef(90, 0,0,1);
	glCallList(DL_TWO_PLANES_NARROW);
	glPopMatrix();

	glPopMatrix();
}


static void
drawPlanesVNarrow(void) {

	glPushMatrix();

	glTranslatef(cursor_translate[0],cursor_translate[1],cursor_translate[2]);
	glMultMatrixf(&cursor_rotate[0]);

	glPushMatrix();
	glCallList(DL_TWO_PLANES_VNARROW);
	glPopMatrix();

	glPushMatrix();
	glRotatef(90, 1,0,0);
	glCallList(DL_TWO_PLANES_VNARROW);
	glPopMatrix();

	glPushMatrix();
	glRotatef(90, 0,0,1);
	glCallList(DL_TWO_PLANES_VNARROW);
	glPopMatrix();

	glPopMatrix();
}


static void
drawPlanesCursor(void) {

	glPushMatrix();

	glTranslatef(cursor_translate[0],cursor_translate[1],cursor_translate[2]);
	glMultMatrixf(&cursor_rotate[0]);

	glPushMatrix();
	glColor3f(1,0,0);
	glCallList(DL_TWO_PLANES_CURSOR);
	glPopMatrix();

	glPushMatrix();
	glRotatef(90, 1,0,0);
	glColor3f(0,1,0);
	glCallList(DL_TWO_PLANES_CURSOR);
	glPopMatrix();

	glPushMatrix();

	glRotatef(90, 0,0,1);
	glColor3f(0,0,1);
	glCallList(DL_TWO_PLANES_CURSOR);
	glPopMatrix();

	glPopMatrix();

}


static void
drawPlanesTransp1(void) {

	static float transpm[] = {1.0, 1.0, 1.0, 0.5};

	glPushMatrix();
	glTranslatef(cursor_translate[0],cursor_translate[1],cursor_translate[2]);
	glMultMatrixf(&cursor_rotate[0]);
	glPushMatrix();

	glDisable(GL_CULL_FACE);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_BLEND);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, (float *)transpm);

	glBegin(GL_QUADS);
	glVertex3f(0, -CURSOR_SIZE, -CURSOR_SIZE);
	glVertex3f(0,-CURSOR_SIZE, CURSOR_SIZE);
	glVertex3f(0, CURSOR_SIZE, CURSOR_SIZE);
	glVertex3f(0, CURSOR_SIZE, -CURSOR_SIZE);
	glEnd();

	glDisable(GL_BLEND);
	glDepthFunc(GL_LEQUAL);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);

	glPopMatrix();
	glPopMatrix();
}


// Some definitions to help with drawing our transparent planes since 
// they have to be sorted back to front

enum {
	UNKNOWN, POSITIVE, NEGATIVE, COPLANAR
};

#define dot(a,b) ((((a)[0])*((b)[0]))+(((a)[1])*((b)[1]))+(((a)[2])*((b)[2])))

float eye_obj_dir[3];

static void
drawPlanesTransp3Rec(int x, int y, int z) {
	if (x==UNKNOWN) {
		if (dot(eye_obj_dir, &cursor_rotate[0]) < 0) {
			drawPlanesTransp3Rec(POSITIVE, UNKNOWN, UNKNOWN);
			drawPlanesTransp3Rec(COPLANAR, NEGATIVE, NEGATIVE);
			drawPlanesTransp3Rec(COPLANAR, NEGATIVE, POSITIVE);
			drawPlanesTransp3Rec(COPLANAR, POSITIVE, POSITIVE);
			drawPlanesTransp3Rec(COPLANAR, POSITIVE, NEGATIVE);
			drawPlanesTransp3Rec(NEGATIVE, UNKNOWN, UNKNOWN);
		}
		else {
			drawPlanesTransp3Rec(NEGATIVE, UNKNOWN, UNKNOWN);
			drawPlanesTransp3Rec(COPLANAR, NEGATIVE, NEGATIVE);
			drawPlanesTransp3Rec(COPLANAR, NEGATIVE, POSITIVE);
			drawPlanesTransp3Rec(COPLANAR, POSITIVE, POSITIVE);
			drawPlanesTransp3Rec(COPLANAR, POSITIVE, NEGATIVE);
			drawPlanesTransp3Rec(POSITIVE, UNKNOWN, UNKNOWN);
		}
	}
	else if (y==UNKNOWN) {
		if (dot(eye_obj_dir, &cursor_rotate[4]) < 0) {
			drawPlanesTransp3Rec(x, POSITIVE, COPLANAR);
			drawPlanesTransp3Rec(x, COPLANAR, NEGATIVE);
			drawPlanesTransp3Rec(x, COPLANAR, POSITIVE);
			drawPlanesTransp3Rec(x, NEGATIVE, COPLANAR);
		}
		else {
			drawPlanesTransp3Rec(x, NEGATIVE, COPLANAR);
			drawPlanesTransp3Rec(x, COPLANAR, NEGATIVE);
			drawPlanesTransp3Rec(x, COPLANAR, POSITIVE);
			drawPlanesTransp3Rec(x, POSITIVE, COPLANAR);
		}
	}
	else {
		// By the time we are here we known which of 12 quads we want
		if (x == COPLANAR) {
			float a,b;
			a = (y==NEGATIVE)?-CURSOR_SIZE:CURSOR_SIZE;
			b = (z==NEGATIVE)?-CURSOR_SIZE:CURSOR_SIZE;
			glBegin(GL_QUADS);
			glVertex3f(0.0, 0.0, 0.0);
			glVertex3f(0.0, 0.0,   b);
			glVertex3f(0.0,   a,   b);
			glVertex3f(0.0,   a, 0.0);
			glEnd();

		}
		else if (y == COPLANAR) {
			float a,b;
			a = (x==NEGATIVE)?-CURSOR_SIZE:CURSOR_SIZE;
			b = (z==NEGATIVE)?-CURSOR_SIZE:CURSOR_SIZE;
			glBegin(GL_QUADS);
			glVertex3f(0.0, 0.0, 0.0);
			glVertex3f(0.0, 0.0,   b);
			glVertex3f(  a, 0.0,   b);
			glVertex3f(  a, 0.0, 0.0);
			glEnd();
		}
		else {
			float a,b;
			a = (x==NEGATIVE)?-CURSOR_SIZE:CURSOR_SIZE;
			b = (y==NEGATIVE)?-CURSOR_SIZE:CURSOR_SIZE;
			glBegin(GL_QUADS);
			glVertex3f(0.0, 0.0, 0.0);
			glVertex3f(0.0,   b, 0.0);
			glVertex3f(  a,   b, 0.0);
			glVertex3f(  a, 0.0, 0.0);
			glEnd();
		}
	}
}


static void
drawPlanesTransp3(void) {

	static float transpm[] = {1.0, 1.0, 1.0, 0.5};

	eye_obj_dir[0]=(eye_dist*eye_rotate[8])  - cursor_translate[0];
	eye_obj_dir[1]=(eye_dist*eye_rotate[9])  - cursor_translate[1];
	eye_obj_dir[2]=(eye_dist*eye_rotate[10]) - cursor_translate[2];

	glPushMatrix();

	glTranslatef(cursor_translate[0],cursor_translate[1],cursor_translate[2]);
	glMultMatrixf(&cursor_rotate[0]);
	glPushMatrix();
	glDisable(GL_CULL_FACE);

	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_BLEND);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, (float *)transpm);

	drawPlanesTransp3Rec(UNKNOWN,UNKNOWN,UNKNOWN);

	glDisable(GL_BLEND);
	glDepthFunc(GL_LEQUAL);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);

	glPopMatrix();
	glPopMatrix();
}




static void
render(void) {

	GLfloat Lf[4];
	GLfloat Lc[4] = { 1.0,  1.0,  1.0,  1.0 }; 
	GLfloat on[4] = { 0.92, 0.92, 0.92, 1.0 };

	/* Use a white light. */
	glLightfv(GL_LIGHT0, GL_DIFFUSE, &Lc[0]);
	glLightfv(GL_LIGHT0, GL_SPECULAR, &Lc[0]);

	glColorMask(1,1,1,1);
	glDepthMask(1);
	glStencilMask(~0u);

	glDepthFunc(GL_LEQUAL);
	glDisable(GL_STENCIL_TEST);
	glEnable(GL_LIGHT0);

	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, on);

	Lf[0] = light_location[0];
	Lf[1] = light_location[1];
	Lf[2] = light_location[2];
	Lf[3] = light_location[3];
	glLightfv(GL_LIGHT0, GL_POSITION, &Lf[0]);

	if ((displayMode == M_PLANE_TSHADOW) ||
		(displayMode == M_PLANE_TSHADOW_NARROW) ||
		(displayMode == M_PLANE_TSHADOW_VNARROW)) {
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		}
	else {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
	}

	glEnable(GL_LIGHTING);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);

	if ((displayMode == M_PLANE_TSHADOW) ||
		(displayMode == M_PLANE_TSHADOW_NARROW) ||
		(displayMode == M_PLANE_TSHADOW_VNARROW)) {

#if defined(GL_EXT_texture3D) 
			if (enable_texture3d) {

				// Set up texture coordinate generation
				// s,t,r planes go through centre of cursor

				double sPlane[4];
				double tPlane[4]; 
				double rPlane[4]; 


				sPlane[0] = 1.0;
				sPlane[1] = 0.0;
				sPlane[2] = 0.0;
				sPlane[3] = 0.0;
				tPlane[0] = 0.0;
				tPlane[1] = 1.0;
				tPlane[2] = 0.0;
				tPlane[3] = 0.0;
				rPlane[0] = 0.0;
				rPlane[1] = 0.0;
				rPlane[2] = 1.0;
				rPlane[3] = 0.0;


				glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
				glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
				glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

				glTexGeni( GL_S, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR );
				glTexGeni( GL_T, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR );
				glTexGeni( GL_R, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR );

				glMatrixMode(GL_MODELVIEW);
				glPushMatrix();
				glLoadIdentity();
				glTexGendv(GL_S, GL_OBJECT_PLANE, sPlane);
				glTexGendv(GL_T, GL_OBJECT_PLANE, tPlane);
				glTexGendv(GL_R, GL_OBJECT_PLANE, rPlane);
				glPopMatrix();

				glEnable(GL_TEXTURE_3D_EXT);

				if (displayMode == M_PLANE_TSHADOW) glBindTexture(GL_TEXTURE_3D_EXT, texid1);
				if (displayMode == M_PLANE_TSHADOW_NARROW) glBindTexture(GL_TEXTURE_3D_EXT, texid2);
				if (displayMode == M_PLANE_TSHADOW_VNARROW) glBindTexture(GL_TEXTURE_3D_EXT, texid3);

				glEnable( GL_TEXTURE_GEN_S );
				glEnable( GL_TEXTURE_GEN_T );
				glEnable( GL_TEXTURE_GEN_R );

				texmatrix[0]=cursor_rotate[0];
				texmatrix[1]=cursor_rotate[4];
				texmatrix[2]=cursor_rotate[8];
				texmatrix[3]=cursor_rotate[12];
				texmatrix[4]=cursor_rotate[1];
				texmatrix[5]=cursor_rotate[5];
				texmatrix[6]=cursor_rotate[9];
				texmatrix[7]=cursor_rotate[13];
				texmatrix[8]=cursor_rotate[2];
				texmatrix[9]=cursor_rotate[6];
				texmatrix[10]=cursor_rotate[10];
				texmatrix[11]=cursor_rotate[14];
				texmatrix[12]=cursor_rotate[3];
				texmatrix[13]=cursor_rotate[7];
				texmatrix[14]=cursor_rotate[11];
				texmatrix[15]=cursor_rotate[15];

				glMatrixMode( GL_TEXTURE );
				glLoadIdentity();
				glTranslatef(0.5,0.5,0.5); // Offset due to texture coordinates
				glMultMatrixf( texmatrix ); // Push on the partial inverse of the cursor matrix
				glTranslatef(-cursor_translate[0],-cursor_translate[1],-cursor_translate[2]);
				glPushMatrix();

			}
#endif

			glMatrixMode( GL_MODELVIEW);

			drawObjects();
			drawFloor();

#if defined(GL_EXT_texture3D)
			if (enable_texture3d) {
				glDisable(GL_TEXTURE_3D_EXT);
				glDisable(GL_TEXTURE_GEN_S );
				glDisable(GL_TEXTURE_GEN_T );
				glDisable(GL_TEXTURE_GEN_R );

			}
#endif

			drawPlanesCursor();

#if defined(GL_EXT_texture3D) 
			if (enable_texture3d) {
				glMatrixMode( GL_TEXTURE );
				glPopMatrix();	
				glMatrixMode( GL_MODELVIEW);
			}
#endif
			return;
		}

		drawObjects();	
		drawFloor();
		drawPlanesCursor();

		switch (displayMode) {
		case M_PLANE_TRANSP1:
			glPushMatrix();
			drawPlanesTransp1();
			break;
		case M_PLANE_TRANSP3:
			glPushMatrix();
			drawPlanesTransp3();
			break;
		case M_PLANE_SHADOW:
		case M_PLANE_SHADOW_NARROW:
		case M_PLANE_SHADOW_VNARROW:
			glColorMask(0,0,0,0);
			glDepthMask(0);
			glEnable(GL_STENCIL_TEST);

			//			glStencilOp(GL_KEEP, GL_KEEP, GL_INVERT);
			glStencilOp(GL_KEEP, GL_INVERT, GL_KEEP);

			// Note we only need one bit
			glStencilFunc(GL_ALWAYS, 0, 0x1);
			glStencilMask(0x1);

			glDisable(GL_CULL_FACE);
			glPushMatrix();

			switch (displayMode) {
			case M_PLANE_SHADOW:
				drawPlanes();
				break;		
			case M_PLANE_SHADOW_NARROW:
				drawPlanesNarrow();
				break;		
			case M_PLANE_SHADOW_VNARROW:
				drawPlanesVNarrow();
				break;		
			default:
				break;
			}
			glPopMatrix();

			glEnable(GL_CULL_FACE); 
			glColorMask(1,1,1,1);
			glStencilOp(GL_KEEP, GL_KEEP, GL_ZERO);
			glStencilFunc(GL_EQUAL, 0x1, 0x1);
			glDepthFunc(GL_EQUAL);
			glDisable(GL_LIGHT0);

			drawFloor();
			drawObjects();
			glEnable(GL_LIGHT0);	
		}
}

//
// GLUT/GLUI callbacks
//

static void 
reshape(int w, int h) {

	window_width = w;
	window_height = h;
	glViewport(0,0,w,h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (GLdouble)w/h, 1.0, FAR_CLIP);
	glPushMatrix();
	glMatrixMode(GL_MODELVIEW);  
}

static void 
display(void) {

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(
		eye_dist*eye_rotate[8],eye_dist*eye_rotate[9],eye_dist*eye_rotate[10],
		0,0,0,
		0,1,0);
	render();	  
	glutSwapBuffers();
}

static void updateWindowTitle(void) {	  
	switch (displayMode) {
	  case M_PLAIN:
		  glutSetWindowTitle("shadow_cursor: Plain cursor");
		  break;
	  case M_PLANE_SHADOW:
		  glutSetWindowTitle("shadow_cursor: Psuedo Shadowed (Stencil)");
		  break;
	  case M_PLANE_SHADOW_NARROW:
		  glutSetWindowTitle("shadow_cursor: Psuedo Shadowed (Stencil Narrow)");
		  break;
	  case M_PLANE_SHADOW_VNARROW:
		  glutSetWindowTitle("shadow_cursor: Psuedo Shadowed (Stencil VNarrow)");
		  break;
#if defined(GL_EXT_texture3D) 
	  case M_PLANE_TSHADOW:
		  if (enable_texture3d) {
			  glutSetWindowTitle("shadow_cursor: Psuedo Shadowed (Textured)");
		  }
		  else {
			  glutSetWindowTitle("shadow_cursor: Mode not supported in this run-time");
		  }
		  break;
	  case M_PLANE_TSHADOW_NARROW:
		  if (enable_texture3d) {
			  glutSetWindowTitle("shadow_cursor: Psuedo Shadowed (Textured Narrow)");
		  }
		  else {
			  glutSetWindowTitle("shadow_cursor: Mode not supported in this run-time");
		  }		 
		  break;
	  case M_PLANE_TSHADOW_VNARROW:
		  if (enable_texture3d) {
			  glutSetWindowTitle("shadow_cursor: Psuedo Shadowed (Textured VNarrow)");
		  }
		  else {
			  glutSetWindowTitle("shadow_cursor: Mode not supported in this run-time");
		  }
		  break;
#else
	  case M_PLANE_TSHADOW:
	  case M_PLANE_TSHADOW_NARROW:
	  case M_PLANE_TSHADOW_VNARROW:
		  glutSetWindowTitle("shadow_cursor: Mode not support in this build");
		  break;
#endif
	  default:
		  glutSetWindowTitle("shadow_cursor: No title for mode");
	}
}



static void 
idle( void ) {

	if ( glutGetWindow() != main_win ) 
		glutSetWindow(main_win);  


	GLUI_Master.sync_live_all();

	glutPostRedisplay();


}

static void 
visible(int vis)  {

	if (vis == GLUT_VISIBLE) {
		GLUI_Master.set_glutIdleFunc(idle);
	} else {
		GLUI_Master.set_glutIdleFunc(NULL);
	}
}

static void 
dumpDriverInfo(void) {

	int i;
	printf("Vendor:   %s\n", glGetString(GL_VENDOR));
	printf("Renderer: %s\n", glGetString(GL_RENDERER));
	printf("Version:  %s\n", glGetString(GL_VERSION));
	printf("Extensions: %s\n", glGetString(GL_EXTENSIONS));

#if defined(GL_EXT_texture3D) 
	if (enable_texture3d) {
		glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE_EXT, &i);
		printf("Max 3D texture size = %d\n", i);
	}
#endif
}

static void 
controlCallback( int control ) {
	if (control == BUTTON_ID) {
		exit(1);
	}
	else if (control == RADIO_ID) {
		updateWindowTitle();
	}
}


static int 
main(int argc, char **argv) {

	int i;

	// Window config
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH | GLUT_STENCIL);
	glutInitWindowSize(window_width, window_height);
	main_win = glutCreateWindow("shadow_cursor");
	updateWindowTitle();
	glutDisplayFunc(display);
	GLUI_Master.set_glutReshapeFunc(reshape);
	glutVisibilityFunc(visible);
	GLUI_Master.set_glutIdleFunc(idle);

	// GLUI sub-window config
	glui_win = GLUI_Master.create_glui_subwindow(main_win, 
		GLUI_SUBWINDOW_RIGHT );
	glui_win->set_main_gfx_window( main_win );

	// Eye controls
	GLUI_Rotation *eye_rot = glui_win->add_rotation( "Eye", eye_rotate );
	eye_rot->set_spin( 1.0 );
	glui_win->add_separator();
	GLUI_Translation *eye_z= 
		glui_win->add_translation( "Eye Dist", GLUI_TRANSLATION_Z, &eye_dist );
	eye_z->set_speed( .005 );
	glui_win->add_separator();

	// Cursor controls
	GLUI_Rotation *plane_rot = glui_win->add_rotation( "Cursor", cursor_rotate );
	plane_rot->set_spin( 1.0 );
	glui_win->add_separator();
	GLUI_Translation *trans_xy = 
		glui_win->add_translation( "Cursor XY", GLUI_TRANSLATION_XY, cursor_translate );
	trans_xy->set_speed( .005 );
	glui_win->add_separator();
	GLUI_Translation *trans_z = 
		glui_win->add_translation( "Cursor Z", GLUI_TRANSLATION_Z, &cursor_translate [2] );
	trans_z->set_speed( .005 );
	glui_win->add_separator();

	// Mode selection
	radio = glui_win->add_radiogroup(&displayMode, RADIO_ID, controlCallback);
	glui_win->add_radiobutton_to_group( radio, "No Highlight" );
	glui_win->add_radiobutton_to_group( radio, "Stencil" );
	glui_win->add_radiobutton_to_group( radio, "Stencil Narrow" );
	glui_win->add_radiobutton_to_group( radio, "Stencil VNarrow" );
	glui_win->add_radiobutton_to_group( radio, "Textured" );
	glui_win->add_radiobutton_to_group( radio, "Textured Narrow" );
	glui_win->add_radiobutton_to_group( radio, "Textured VNarrow" );
	glui_win->add_radiobutton_to_group( radio, "Transparent 1" );
	glui_win->add_radiobutton_to_group( radio, "Transparent 3" );
	glui_win->add_separator();

	glui_win->add_button( "QUIT", BUTTON_ID, controlCallback );
	glui_win->add_separator();

	enable_texture3d = glutExtensionSupported("GL_EXT_texture3D");

#ifdef _WIN32
	if (enable_texture3d) {
		glTexImage3DEXT = (PFNGLTEXIMAGE3DEXTPROC)
			wglGetProcAddress("glTexImage3DEXT");
	}
#endif

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, 1.0, 0.1, FAR_CLIP);
	glEnable(GL_DEPTH_TEST);
	glPointSize(4.0);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

	dumpDriverInfo();

	initObjects();
	glutMainLoop();
	return 0;
}

