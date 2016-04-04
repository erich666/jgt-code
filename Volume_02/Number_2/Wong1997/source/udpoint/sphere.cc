/*
 *
 * Copyright (c) Tien-Tsin Wong, 1997.
 * All Right Reserved.
 *
 * sphere.cc  (A simpler version of uniform.cc of demostration only)
 *
 * An interactive program to view the uniformity of pointset on sphere.
 * 11 Feb 1997
 *
 * Function of the keys:
 * "1"  Random point
 * "2"  Hammersley p_1 = 2
 * "3"  Hammersley p_1 = 3
 * "4"  Hammersley p_1 = 5
 * "5"  Hammersley p_1 = 7
 * "6"  Hammersley p_1 = 11
 * "7"  Halton p_1 = 2, p_2 = 3
 * "8"  Halton p_1 = 2, p_2 = 5
 * "9"  Halton p_1 = 2, p_2 = 7
 * "A"  Halton p_1 = 3, p_2 = 5
 * "B"  Halton p_1 = 3, p_2 = 7
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include "common.h"
#include "trackball.h"
#include "vecmath.h"
#include "udpoint.h"


/* Mode of working with left mouse button */
#define ZOOM         1
#define ROTATE	     2

/* ID of the menu options */
#define MENU_SWITCHLIGHT0      1
#define MENU_SWITCHLIGHT1      2
#define MENU_CHANGEMODE        3
#define MENU_RECORDTRANSFORM   4
#define MENU_SAVESCREEN        5
#define MENU_RELOAD	       6

/* Point generation Methods */
#define RANDOM       	       1
#define HAMM2                  2
#define HAMM3                  3
#define HAMM5                  4
#define HAMM7     	       5
#define HAMM11    	       6
#define HALTON2_3              7
#define HALTON2_5              8
#define HALTON2_7              9
#define HALTON3_5             10
#define HALTON3_7             11

/* Some Limits */
#define UF_MAXMETHOD 	      12
#define UF_SAMPLENO	    1000
#define UF_PLANESAMPLENO     500
#define UF_POINTSIZE           3

/******************************* Global Variables ***************************/
/* Global variables */
int   G_RightButtonMenu;
int   G_BoxObj = 20;
char  G_LightZeroSwitch = TRUE,
      G_LightOneSwitch = TRUE,
      G_Spinning = FALSE,
      G_Dragging = FALSE,
      G_NewModel = TRUE,
      G_RecordTransform=FALSE,
      G_Mode=ROTATE;            /* mode of working with left mouse button */
int   G_CurrX, G_CurrY, G_StartX, G_StartY;
int   G_W = 300, G_H = 300, G_HalfW = 150, G_HalfH = 150;
float G_CurQuat[4], G_LastQuat[4];
float G_Scale = 1.0, G_OldScale = 1.0, G_ZoomVector[2];
char  G_SphereBitmap[UF_MAXMETHOD]; 

GLfloat G_LightZeroPosition[] = {0.0, 1.0, 1.0, 0.0}; /* directional */ /*{10.0, 4.0, 10.0, 1.0};*/
GLfloat G_LightZeroColor[] = {0.9, 1.0, 0.9, 1.0}; /* green-tinted */
GLfloat G_LightOnePosition[] = {-1.0, -2.0, 1.0, 0.0};
GLfloat G_LightOneColor[] = {0.9, 0.6, 0.3, 1.0}; /* red-tinted */
GLfloat G_SkinColor[] = {1.0, 1.0, 1.0, 1.0};
GLfloat G_MatAmb[4] = {0.5, 0.5, 0.5, 1.0};
GLfloat G_MatDiff[4] = {0.5, 0.4, 0.4, 1.0};
GLfloat G_MatSpec[4] = {0.7, 0.4, 0.4, 1.0};
GLfloat G_GlobalAmbLight[4] = {0.2, 0.2, 0.2, 1.0};


/************************* Function Prototypes ****************************/
/**** Callback Functions ****/
void Redraw();
void Mouse(int button, int state, int x, int y);
void Animate();
void Zoom();
void Motion(int x, int y);
void Reshape(int width, int height);
void ControlMenu(int value);
void Vis(int visible);
void Key(unsigned char key, int x, int y);
/**** End of Callback Functions ****/

/**** Other Functions ****/
void DrawBox(GLdouble x0, GLdouble x1, GLdouble y0, GLdouble y1,
	     GLdouble z0, GLdouble z1, GLenum type);
void InitModel();
void InitRenderer();
void RecalcModelView();
void RecordTransform();
void SaveRGBImage(char *filename);


/************************** Function Implementation ************************/
/* DrawBox:
 *
 * draws a rectangular box with the given x, y, and z ranges.
 * The box is axis-aligned.
 */
void DrawBox(GLdouble x0, GLdouble x1, GLdouble y0, GLdouble y1,
	     GLdouble z0, GLdouble z1, GLenum type)
{
    static GLdouble n[6][3] = {
	{-1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {1.0, 0.0, 0.0},
	{0.0, -1.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, -1.0}
    };
    static GLint faces[6][4] = {
	{ 0, 1, 2, 3 }, { 3, 2, 6, 7 }, { 7, 6, 5, 4 },
	{ 4, 5, 1, 0 }, { 5, 6, 2, 1 }, { 7, 4, 0, 3 }
    };
    GLdouble v[8][3], tmp;
    GLint i;

    if (x0 > x1) {
	tmp = x0; x0 = x1; x1 = tmp;
    }
    if (y0 > y1) {
	tmp = y0; y0 = y1; y1 = tmp; 
    }
    if (z0 > z1) {
	tmp = z0; z0 = z1; z1 = tmp; 
    }
    v[0][0] = v[1][0] = v[2][0] = v[3][0] = x0;
    v[4][0] = v[5][0] = v[6][0] = v[7][0] = x1;
    v[0][1] = v[1][1] = v[4][1] = v[5][1] = y0;
    v[2][1] = v[3][1] = v[6][1] = v[7][1] = y1;
    v[0][2] = v[3][2] = v[4][2] = v[7][2] = z0;
    v[1][2] = v[2][2] = v[5][2] = v[6][2] = z1;

    for (i = 0; i < 6; i++) {
	glBegin(type);
	glNormal3dv(&n[i][0]);
	glVertex3dv(&v[faces[i][0]][0]);
	glNormal3dv(&n[i][0]);
	glVertex3dv(&v[faces[i][1]][0]);
	glNormal3dv(&n[i][0]);
	glVertex3dv(&v[faces[i][2]][0]);
	glNormal3dv(&n[i][0]);
	glVertex3dv(&v[faces[i][3]][0]);
	glEnd();
    }
}



void InitModel()
{
  int i, j;
  int p1, p2;
  float *point;

  for (i=0 ; i<UF_MAXMETHOD ; i++)
    G_SphereBitmap[i] = FALSE;

  if ((point=(float*)malloc(sizeof(float)*3*UF_SAMPLENO))==NULL)
    ERREXIT("[InitModel]: Not enough memory\n");

  glNewList(RANDOM, GL_COMPILE);
    glPointSize(UF_POINTSIZE);
    glDisable(GL_LIGHTING);
    glColor3f(0, 0, 0);
    SphereStripRnd(point, UF_SAMPLENO);
    glBegin(GL_POINTS);
      for (i=0 ; i<UF_SAMPLENO; i++)
        glVertex3f(point[i*3], point[i*3+1], point[i*3+2]);
    glEnd();
    glEnable(GL_LIGHTING);
  glEndList();

  glNewList(HAMM2, GL_COMPILE);
    glPointSize(UF_POINTSIZE);
    glDisable(GL_LIGHTING);
    glColor3f(0, 0, 0);
    SphereHammersley(point, UF_SAMPLENO);
    glBegin(GL_POINTS);
      for (i=0 ; i<UF_SAMPLENO; i++)
        glVertex3f(point[i*3], point[i*3+1], point[i*3+2]);
    glEnd();
    glEnable(GL_LIGHTING);
  glEndList();

  glNewList(HAMM3, GL_COMPILE);
    glPointSize(UF_POINTSIZE);
    glDisable(GL_LIGHTING);
    glColor3f(0, 0, 0);
    SphereHammersley2(point, UF_SAMPLENO, 3);
    glBegin(GL_POINTS);
      for (i=0 ; i<UF_SAMPLENO; i++)
        glVertex3f(point[i*3], point[i*3+1], point[i*3+2]);
    glEnd();
    glEnable(GL_LIGHTING);
  glEndList();

  glNewList(HAMM5, GL_COMPILE);
    glPointSize(UF_POINTSIZE);
    glDisable(GL_LIGHTING);
    glColor3f(0, 0, 0);
    SphereHammersley2(point, UF_SAMPLENO, 5);
    glBegin(GL_POINTS);
      for (i=0 ; i<UF_SAMPLENO; i++)
        glVertex3f(point[i*3], point[i*3+1], point[i*3+2]);
    glEnd();
    glEnable(GL_LIGHTING);
  glEndList();

  glNewList(HAMM7, GL_COMPILE);
    glPointSize(UF_POINTSIZE);
    glDisable(GL_LIGHTING);
    glColor3f(0, 0, 0);
    SphereHammersley2(point, UF_SAMPLENO, 7);
    glBegin(GL_POINTS);
      for (i=0 ; i<UF_SAMPLENO; i++)
        glVertex3f(point[i*3], point[i*3+1], point[i*3+2]);
    glEnd();
    glEnable(GL_LIGHTING);
  glEndList();

  glNewList(HAMM11, GL_COMPILE);
    glPointSize(UF_POINTSIZE);
    glDisable(GL_LIGHTING);
    glColor3f(0, 0, 0);
    SphereHammersley2(point, UF_SAMPLENO, 11);
    glBegin(GL_POINTS);
      for (i=0 ; i<UF_SAMPLENO; i++)
        glVertex3f(point[i*3], point[i*3+1], point[i*3+2]);
    glEnd();
    glEnable(GL_LIGHTING);
  glEndList();

  glNewList(HALTON2_3, GL_COMPILE);
    glPointSize(UF_POINTSIZE);
    glDisable(GL_LIGHTING);
    glColor3f(0, 0, 0);
    SphereHalton(point, UF_SAMPLENO, 3);
    glBegin(GL_POINTS);
      for (i=0 ; i<UF_SAMPLENO; i++)
        glVertex3f(point[i*3], point[i*3+1], point[i*3+2]);
    glEnd();
    glEnable(GL_LIGHTING);
  glEndList();

  glNewList(HALTON2_5, GL_COMPILE);
    glPointSize(UF_POINTSIZE);
    glDisable(GL_LIGHTING);
    glColor3f(0, 0, 0);
    SphereHalton(point, UF_SAMPLENO, 5);
    glBegin(GL_POINTS);
      for (i=0 ; i<UF_SAMPLENO; i++)
        glVertex3f(point[i*3], point[i*3+1], point[i*3+2]);
    glEnd();
    glEnable(GL_LIGHTING);
  glEndList();

  glNewList(HALTON2_7, GL_COMPILE);
    glPointSize(UF_POINTSIZE);
    glDisable(GL_LIGHTING);
    glColor3f(0, 0, 0);
    SphereHalton(point, UF_SAMPLENO, 7);
    glBegin(GL_POINTS);
      for (i=0 ; i<UF_SAMPLENO; i++)
        glVertex3f(point[i*3], point[i*3+1], point[i*3+2]);
    glEnd();
    glEnable(GL_LIGHTING);
  glEndList();

  glNewList(HALTON3_5, GL_COMPILE);
    glPointSize(UF_POINTSIZE);
    glDisable(GL_LIGHTING);
    glColor3f(0, 0, 0);
    SphereHalton2(point, UF_SAMPLENO, 3, 5);
    glBegin(GL_POINTS);
      for (i=0 ; i<UF_SAMPLENO; i++)
        glVertex3f(point[i*3], point[i*3+1], point[i*3+2]);
    glEnd();
    glEnable(GL_LIGHTING);
  glEndList();

  glNewList(HALTON3_7, GL_COMPILE);
    glPointSize(UF_POINTSIZE);
    glDisable(GL_LIGHTING);
    glColor3f(0, 0, 0);
    SphereHalton2(point, UF_SAMPLENO, 3, 7);
    glBegin(GL_POINTS);
      for (i=0 ; i<UF_SAMPLENO; i++)
        glVertex3f(point[i*3], point[i*3+1], point[i*3+2]);
    glEnd();
    glEnable(GL_LIGHTING);
  glEndList();

  free(point);
}


void RecalcModelView()
{
  GLfloat m[4][4];

  glPopMatrix();
  glPushMatrix();
  build_rotmatrix(m, G_CurQuat);
  glMultMatrixf(&m[0][0]);
  G_NewModel = FALSE;
}


void RecordTransform()
{
  FILE *fptr;
  int i;
  GLfloat matrix[16];
  GLint viewport[4];
  
  if ((fptr=fopen("transfrm.txt", "wt"))==NULL)
  {
    fprintf(stderr, "Cannot write text transformation file transform.txt\n");
    G_RecordTransform = FALSE;
    return;
  }
  glGetFloatv(GL_MODELVIEW_MATRIX, matrix);
  fprintf(fptr, "ModelViewMatrix");
  for (i=0 ; i<16 ; i++)
    fprintf(fptr," %f", matrix[i]);
  glGetFloatv(GL_PROJECTION_MATRIX, matrix);
  fprintf(fptr, "\nProjectionMatrix");
  for (i=0 ; i<16 ; i++)
    fprintf(fptr," %f", matrix[i]);  
  glGetIntegerv(GL_VIEWPORT, viewport);
  fprintf(fptr, "\nViewport");
  for (i=0 ; i<4 ; i++)
    fprintf(fptr, " %d", viewport[i]);
  glGetFloatv(GL_DEPTH_RANGE, matrix);
  fprintf(fptr, "\nDepthRange");
  for(i=0 ; i<2 ; i++)
    fprintf(fptr, " %f", matrix[i]);

  G_RecordTransform = FALSE;
  fclose(fptr);
}



void DrawSphere()
{
  glutSolidSphere(0.995, 20, 20);
  glLineWidth(5.0);
  glColor3f(0.1, 0.1, 0.8);
  glDisable(GL_LIGHTING);
  glBegin(GL_LINES);
    glVertex3f(0, 1.5, 0);
    glVertex3f(0, -1.5, 0);
  glEnd();
  glEnable(GL_LIGHTING);
  glPushMatrix();
  glTranslatef(0, 1.5, 0);
  glRotatef(-90, 1, 0, 0);
  glutSolidCone(0.1, 0.2, 10, 10);
  glPopMatrix();
}


void DrawPlane()
{
  glDisable(GL_LIGHTING);
  glColor3f(1, 1, 1);
  glBegin(GL_POLYGON);
    glVertex3f(-1, -1, 0);
    glVertex3f(1, -1, 0);
    glVertex3f(1, 1, 0);
    glVertex3f(-1, 1, 0);
  glEnd();
}

void Redraw()
{
  if (G_NewModel)
    RecalcModelView();
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glPushMatrix();
  if (G_RecordTransform)
    RecordTransform();

  glScalef(G_Scale, G_Scale, G_Scale);   // Zoom

  /* Insert object */

  int i;
  DrawSphere();
  for (i=1 ; i<UF_MAXMETHOD ; i++)
    if (G_SphereBitmap[i])
    {
        glClearColor(0.6, 0.6, 0.8, 0.0);
        glPushMatrix();
        glRotatef(-90, 1, 0, 0);
        glCallList(i); 
        glPopMatrix();
    }

  glPopMatrix();
  glutSwapBuffers();
}


void Mouse(int button, int state, int x, int y)
{
  if (button == GLUT_LEFT_BUTTON)
    if (state == GLUT_DOWN) 
    {
      G_Spinning = FALSE;
      glutIdleFunc(NULL);
      G_Dragging = TRUE;
      G_CurrX = x;
      G_CurrY = y;
      G_StartX = x;
      G_StartY = y;
      if (G_Mode==ZOOM)
      {
	G_ZoomVector[0] = x - G_HalfW;
	G_ZoomVector[1] = y - G_HalfH;
	if (G_ZoomVector[0]==0 && G_ZoomVector[1]==0)
	  G_ZoomVector[0] = G_ZoomVector[1] = 1.0;
      }
    }
    else if (state == GLUT_UP) 
    {
      G_Spinning = TRUE;
      G_Dragging = FALSE;
      if (G_Mode==ZOOM)  /* Stop object continue to zoom in zoom mode */
      {
        glutIdleFunc(NULL);
	G_OldScale = G_Scale;
      }
      else
      {
        if (G_StartX == x && G_StartY == y)  /* user click to stop spining */
          glutIdleFunc(NULL);
        else 
          glutIdleFunc(Animate);
      }
    }
}


void Key(unsigned char key, int x, int y)
{
  int idx;
  if (key >= '0' && key <= '9')
    idx = key - '0';
  else if (key >= 'a' && key <= 'b')
    idx = key - 'a' + 10;
  else if (key >= 'A' && key <= 'B')
    idx = key - 'A' + 10;
  else
    return;
  G_SphereBitmap[idx] = !G_SphereBitmap[idx];
  switch(idx)
  {
    case RANDOM:
      if (G_SphereBitmap[idx])
  	printf("Random sample on sphere toggle ON\n");
      else 
  	printf("Random sample on sphere toggle OFF\n");
      break;
    case HAMM2:
      if (G_SphereBitmap[idx])
  	printf("Hammersley points with p_1=2 on sphere toggle ON\n");
      else 
  	printf("Hammersley points with p_1=2 on sphere toggle OFF\n");
      break;
    case HAMM3:
      if (G_SphereBitmap[idx])
  	printf("Hammersley points with p_1=3 on sphere toggle ON\n");
      else 
  	printf("Hammersley points with p_1=3 on sphere toggle OFF\n");
      break;
    case HAMM5:
      if (G_SphereBitmap[idx])
  	printf("Hammersley points with p_1=5 on sphere toggle ON\n");
      else 
  	printf("Hammersley points with p_1=5 on sphere toggle OFF\n");
      break;
    case HAMM7:
      if (G_SphereBitmap[idx])
  	printf("Hammersley points with p_1=7 on sphere toggle ON\n");
      else 
  	printf("Hammersley points with p_1=7 on sphere toggle OFF\n");
      break;
    case HAMM11:
      if (G_SphereBitmap[idx])
  	printf("Hammersley points with p_1=11 on sphere toggle ON\n");
      else 
  	printf("Hammersley points with p_1=11 on sphere toggle OFF\n");
      break;
    case HALTON2_3:
      if (G_SphereBitmap[idx])
  	printf("Halton points with p_1=2, p_2=3 on sphere toggle ON\n");
      else 
  	printf("Halton points with p_1=2, p_2=3 on sphere toggle OFF\n");
      break;
    case HALTON2_5:
      if (G_SphereBitmap[idx])
  	printf("Halton points with p_1=2, p_2=5 on sphere toggle ON\n");
      else 
  	printf("Halton points with p_1=2, p_2=5 on sphere toggle OFF\n");
      break;
    case HALTON2_7:
      if (G_SphereBitmap[idx])
  	printf("Halton points with p_1=2, p_2=7 on sphere toggle ON\n");
      else 
  	printf("Halton points with p_1=2, p_2=7 on sphere toggle OFF\n");
      break;
    case HALTON3_5:
      if (G_SphereBitmap[idx])
  	printf("Halton points with p_1=3, p_2=5 on sphere toggle ON\n");
      else 
  	printf("Halton points with p_1=3, p_2=5 on sphere toggle OFF\n");
      break;
    case HALTON3_7:
      if (G_SphereBitmap[idx])
  	printf("Halton points with p_1=3, p_2=7 on sphere toggle ON\n");
      else 
  	printf("Halton points with p_1=3, p_2=7 on sphere toggle OFF\n");
      break;
  }
  if (idx < UF_MAXMETHOD)
    glutPostRedisplay();
}


void Animate()
{
  add_quats(G_LastQuat, G_CurQuat, G_CurQuat);
  G_NewModel = TRUE;
  glutPostRedisplay();
}


void Zoom()
{
  float distance, sign, dotproduct, motionvector[2];
  motionvector[0] = G_CurrX-G_StartX;
  motionvector[1] = G_CurrY-G_StartY;
  dotproduct = G_ZoomVector[0]*motionvector[0] + G_ZoomVector[1]*motionvector[1];
  if (dotproduct<0)
    sign = -1;
  else
    sign = 1;  
  distance = sqrt(motionvector[0]*motionvector[0] + motionvector[1]*motionvector[1]);
  G_Scale = G_OldScale * (1.0 + sign*(distance/MAX(G_HalfW,G_HalfH)));
  glutPostRedisplay();
}


void Motion(int x, int y)
{
  if (G_Dragging)
    if (G_Mode==ROTATE)
    {
      trackball(G_LastQuat,
		2.0 * (G_W - x) / G_W - 1.0,
		2.0 * y / G_H - 1.0,
		2.0 * (G_W - G_CurrX) / G_W - 1.0,
		2.0 * G_CurrY / G_H - 1.0);
      G_CurrX = x;
      G_CurrY = y;
   /* G_Spinning = TRUE; */
   /* glutIdleFunc(Animate); */
      Animate();
    }
    else if (G_Mode==ZOOM)
    {
      G_CurrX = x;
      G_CurrY = y;
      glutIdleFunc(Zoom);
    }
}


void Reshape(int width, int height)
{
  G_W = width;
  G_H = height;
  G_HalfW = G_W/2;
  G_HalfH = G_H/2;
  glViewport(0,0,width,height);
}


/*
 * Save framebuffer as an image in pbm format
 */
void SaveRGBImage(char *filename)
{
  unsigned char *buffer;
  int w, h;
  FILE *fptr;
  GLint viewport[4];
  
  glGetIntegerv(GL_VIEWPORT, viewport);  
  fprintf(stderr, "%d %d %d %d\n", viewport[0], viewport[1], viewport[2], viewport[3]);
  fprintf(stderr, "%d %d\n", G_W, G_H);
  w = G_W;
  h = G_H;
  if ((buffer=(unsigned char*)malloc(3*w*h))==NULL)
    ERREXIT("[SaveRGBImage]: not enough memory for saving image\n");
  if ((fptr=fopen(filename,"wb"))==NULL)
    ERREXIT("[SaveRGBImage]: cannot write image file\n");
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, buffer);
  fprintf(fptr, "P6\n%d %d\n255\n", w, h);
  fwrite(buffer, w*h*3, 1, fptr);
  fclose(fptr);
  free(buffer);
}



void ControlMenu(int value)
{
  int oldmenu;
  oldmenu = glutGetMenu();
  glutSetMenu(G_RightButtonMenu);
  switch (value)
  {
    case MENU_SWITCHLIGHT0:
      G_LightZeroSwitch = !G_LightZeroSwitch;
      if (G_LightZeroSwitch)
      {
        glutChangeToMenuEntry(value, "Turn off right light", value);
        glEnable(GL_LIGHT0);
      }
      else 
      {
        glutChangeToMenuEntry(value, "Turn on right light", value);
        glDisable(GL_LIGHT0);
      }
      break;

    case MENU_SWITCHLIGHT1:
      G_LightOneSwitch = !G_LightOneSwitch;
      if (G_LightOneSwitch)
      {
        glutChangeToMenuEntry(value, "Turn off left light", value);      
        glEnable(GL_LIGHT1);
      }
      else
      {
        glutChangeToMenuEntry(value, "Turn on left light", value);      
        glDisable(GL_LIGHT1);
      }
      break;

    case MENU_CHANGEMODE:
      if (G_Mode==ZOOM)
      {
	glutChangeToMenuEntry(value, "Zoom mode", value);
	G_Mode = ROTATE;
      }
      else
      {
	glutChangeToMenuEntry(value, "Rotate mode", value);
	G_Mode = ZOOM;
      }
      break;
  }
  glutPostRedisplay();
  glutSetMenu(oldmenu);
}


void Vis(int visible)
{
  if (visible == GLUT_VISIBLE) 
  {
    if (G_Spinning)
      glutIdleFunc(Animate);
  } 
  else 
  {
    if (G_Spinning)
      glutIdleFunc(NULL);
  }
}


void InitRenderer()
{
  glDisable(GL_CULL_FACE);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);
  glEnable(GL_LIGHTING);
  glMatrixMode(GL_PROJECTION);
  gluPerspective( /* field of view in degree */ 40.0,  
  		  /* aspect ratio */ 1.0,
                  /* Z near */ 1.0,  /* Z far */ 100.0);
  glMatrixMode(GL_MODELVIEW);
  gluLookAt(0.0, 0.0, 5.0,  /* eye is at (0,0,5) */
	    0.0, 0.0, 0.0,   /* center is at (0,0,0) */
	    0.0, 1.0, 0.0);  /* up is in postivie Y direction */
  glPushMatrix();       /* dummy push so we can pop on model recalc */
  glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, 1);
  glLightfv(GL_LIGHT0, GL_POSITION, G_LightZeroPosition);
  glLightfv(GL_LIGHT0, GL_DIFFUSE, G_LightZeroColor);
  glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 0.1);
  glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0.05);
  glLightfv(GL_LIGHT1, GL_POSITION, G_LightOnePosition);
  glLightfv(GL_LIGHT1, GL_DIFFUSE, G_LightOneColor);
  glEnable(GL_LIGHT0);
  glEnable(GL_LIGHT1);
  glLightModelfv(GL_LIGHT_MODEL_AMBIENT, G_GlobalAmbLight);
  glLineWidth(1.0);
  glFrontFace (GL_CW);
  glEnable(GL_AUTO_NORMAL);
  glEnable(GL_NORMALIZE); 
  glClearColor(0.6, 0.6, 0.8, 0.0);
                  
  trackball(G_CurQuat, 0.0, 0.0, 0.0, 0.0);  /* init current quaterion */
}


void main(int argc, char **argv)
{
  char title[1024], lidfile[100];

  glutInit(&argc, argv);  /* let glut extract its own argument */

  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
  sprintf(title, "Sample points on Sphere (Demo)\n");
  glutCreateWindow(title);
  glutDisplayFunc(Redraw);
  glutVisibilityFunc(Vis);
  glutKeyboardFunc(Key);
  glutMouseFunc(Mouse);
  glutMotionFunc(Motion);
  glutReshapeFunc(Reshape);

  G_RightButtonMenu = glutCreateMenu(ControlMenu);
  glutAddMenuEntry("Turn off right light", MENU_SWITCHLIGHT0);
  glutAddMenuEntry("Turn off left light", MENU_SWITCHLIGHT1);
  glutAddMenuEntry("Zoom mode", MENU_CHANGEMODE);
  glutAttachMenu(GLUT_RIGHT_BUTTON);

  InitModel();
  InitRenderer();
  glutMainLoop();
}
