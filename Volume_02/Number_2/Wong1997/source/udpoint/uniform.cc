/*
 *
 * Copyright (c) Tien-Tsin Wong, 1996.  
 * All Right Reserved.
 *
 * uniform.cc
 *
 * An interactive program to view the uniformity of pointset on sphere.
 *
 *
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
#define ONCUBE		       1
#define SPHERESTRIPREG         2
#define SPHERESTRIPRND         3
#define TWOPOLEREG             4
#define TWOPOLERND	       5
#define HAMMERSLEY	       6
#define HAMMERSLEY2            7
#define HALTON 	               8
#define HALTON2	               9
#define PHAMMERSLEY           10
#define PHAMMERSLEY2          11
#define PHALTON 	      12
#define PHALTON2              13
#define PRANDOM               14

/* Some Limits */
#define UF_MAXMETHOD 	      20
#define UF_SAMPLENO	    1000
#define UF_PLANESAMPLENO     500
#define UF_POINTSIZE           3

/******************************* Global Variables ***************************/
/* Global variables */
int   G_p2 = 3; 
int   G_p1 = 2; 
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

  glNewList(ONCUBE, GL_COMPILE);
    glPointSize(UF_POINTSIZE);
    glDisable(GL_LIGHTING);
    glColor3f(0, 0, 0);
    OnCube(point, UF_SAMPLENO);
    glBegin(GL_POINTS);
      for (i=0 ; i<UF_SAMPLENO; i++)
	glVertex3f(point[i*3], point[i*3+1], point[i*3+2]);
    glEnd();
    glEnable(GL_LIGHTING);
  glEndList();

  glNewList(SPHERESTRIPREG, GL_COMPILE);
    glPointSize(UF_POINTSIZE);
    glDisable(GL_LIGHTING);
    glColor3f(0, 0, 0);
    SphereStripReg(point, UF_SAMPLENO);
    glBegin(GL_POINTS);
      for (i=0 ; i<UF_SAMPLENO; i++)
	glVertex3f(point[i*3], point[i*3+1], point[i*3+2]);
    glEnd();
    glEnable(GL_LIGHTING);
  glEndList();

  glNewList(SPHERESTRIPRND, GL_COMPILE);
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

  glNewList(TWOPOLEREG, GL_COMPILE);
    glPointSize(UF_POINTSIZE);
    glDisable(GL_LIGHTING);
    glColor3f(0, 0, 0);
    TwoPoleReg(point, UF_SAMPLENO);
    glBegin(GL_POINTS);
      for (i=0 ; i<UF_SAMPLENO; i++)
        glVertex3f(point[i*3], point[i*3+1], point[i*3+2]);
    glEnd();
    glEnable(GL_LIGHTING);
  glEndList();

  glNewList(TWOPOLERND, GL_COMPILE);
    glPointSize(UF_POINTSIZE);
    glDisable(GL_LIGHTING);
    glColor3f(0, 0, 0);
    TwoPoleRnd(point, UF_SAMPLENO);
    glBegin(GL_POINTS);
      for (i=0 ; i<UF_SAMPLENO; i++)
        glVertex3f(point[i*3], point[i*3+1], point[i*3+2]);
    glEnd();
    glEnable(GL_LIGHTING);
  glEndList();

  glNewList(HAMMERSLEY, GL_COMPILE);
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

  printf("What is the value of p1 for SphereHammersley2()?");
  scanf("%d", &G_p1);
  glNewList(HAMMERSLEY2, GL_COMPILE);
    glPointSize(UF_POINTSIZE);
    glDisable(GL_LIGHTING);
    glColor3f(0, 0, 0);
    SphereHammersley2(point, UF_SAMPLENO, G_p1);
    glBegin(GL_POINTS);
      for (i=0 ; i<UF_SAMPLENO; i++)
        glVertex3f(point[i*3], point[i*3+1], point[i*3+2]);
    glEnd();
    glEnable(GL_LIGHTING);
  glEndList();

  printf("What is the value of p2 for SphereHalton (prime, not 2)?");
  scanf("%d", &G_p2);
  glNewList(HALTON, GL_COMPILE);
    glPointSize(UF_POINTSIZE);
    glDisable(GL_LIGHTING);
    glColor3f(0, 0, 0);
    SphereHalton(point, UF_SAMPLENO, G_p2);
    glBegin(GL_POINTS);
      for (i=0 ; i<UF_SAMPLENO; i++)
        glVertex3f(point[i*3], point[i*3+1], point[i*3+2]);
    glEnd();
    glEnable(GL_LIGHTING);
  glEndList();

  printf("What is the value of p1 for SphereHalton2?");
  scanf("%d", &p1);
  printf("What is the value of p2 for SphereHalton2?");
  scanf("%d", &p2);
  glNewList(HALTON2, GL_COMPILE);
    glPointSize(UF_POINTSIZE);
    glDisable(GL_LIGHTING);
    glColor3f(0, 0, 0);
    SphereHalton2(point, UF_SAMPLENO, p1, p2);
    glBegin(GL_POINTS);
      for (i=0 ; i<UF_SAMPLENO; i++)
        glVertex3f(point[i*3], point[i*3+1], point[i*3+2]);
    glEnd();
    glEnable(GL_LIGHTING);
  glEndList();

  ///////////////// Plane /////////////////////
  glNewList(PHAMMERSLEY, GL_COMPILE);
    glPointSize(UF_POINTSIZE);
    glDisable(GL_LIGHTING);
    glColor3f(0, 0, 0);
    PlaneHammersley(point, UF_PLANESAMPLENO);
    glBegin(GL_POINTS);
      for (i=0 ; i<UF_PLANESAMPLENO; i++)
        glVertex3f(point[i*2], point[i*2+1], 0.02);
    glEnd();
    glEnable(GL_LIGHTING);
  glEndList();

  printf("What is the value of p1 for PlaneHammersley2()?");
  scanf("%d", &p1);
  glNewList(PHAMMERSLEY2, GL_COMPILE);
    glPointSize(UF_POINTSIZE);
    glDisable(GL_LIGHTING);
    glColor3f(0, 0, 0);
    PlaneHammersley2(point, UF_PLANESAMPLENO, p1);
    glBegin(GL_POINTS);
      for (i=0 ; i<UF_PLANESAMPLENO; i++)
        glVertex3f(point[i*2], point[i*2+1], 0.02);
    glEnd();
    glEnable(GL_LIGHTING);
  glEndList();

  printf("What is the value of p2 for PlaneHalton (prime, not 2)?");
  scanf("%d", &p2);
  glNewList(PHALTON, GL_COMPILE);
    glPointSize(UF_POINTSIZE);
    glDisable(GL_LIGHTING);
    glColor3f(0, 0, 0);
    PlaneHalton(point, UF_PLANESAMPLENO, p2);
    glBegin(GL_POINTS);
      for (i=0 ; i<UF_PLANESAMPLENO; i++)
        glVertex3f(point[i*2], point[i*2+1], 0.02);
    glEnd();
    glEnable(GL_LIGHTING);
  glEndList();

  printf("What is the value of p1 for PlaneHalton2?");
  scanf("%d", &p1);
  printf("What is the value of p2 for PlaneHalton2?");
  scanf("%d", &p2);
  glNewList(PHALTON2, GL_COMPILE);
    glPointSize(UF_POINTSIZE);
    glDisable(GL_LIGHTING);
    glColor3f(0, 0, 0);
    PlaneHalton2(point, UF_PLANESAMPLENO, p1, p2);
    glBegin(GL_POINTS);
      for (i=0 ; i<UF_PLANESAMPLENO; i++)
        glVertex3f(point[i*2], point[i*2+1], 0.02);
    glEnd();
    glEnable(GL_LIGHTING);
  glEndList();


  glNewList(PRANDOM, GL_COMPILE);
    glPointSize(UF_POINTSIZE);
    glDisable(GL_LIGHTING);
    glColor3f(0, 0, 0);
    PlaneRandom(point, UF_PLANESAMPLENO);
    glBegin(GL_POINTS);
      for (i=0 ; i<UF_PLANESAMPLENO; i++)
        glVertex3f(point[i*2], point[i*2+1], 0.02);
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
  char spheredrawn=FALSE, planedrawn=FALSE;
  for (i=1 ; i<UF_MAXMETHOD ; i++)
    if (G_SphereBitmap[i])
    {
      if (i<=9)
      {
        if(!spheredrawn)
        {
          DrawSphere();
          spheredrawn = TRUE;
        }
        glClearColor(0.6, 0.6, 0.8, 0.0);
        glPushMatrix();
        glRotatef(-90, 1, 0, 0);
        glCallList(i); 
        glPopMatrix();
      }
      else 
      { 
        if (!planedrawn)
        {
          DrawPlane();
          planedrawn = TRUE;
        }
        glClearColor(0.0, 0.0, 0.0, 0.0);
        glPushMatrix();
        glTranslatef(-1, -1, 0);
        glScalef(2, 2, 1);        
        glCallList(i); 
        glPopMatrix();
      }
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
  else if (key >= 'a' && key <= 'z')
    idx = key - 'a' + 10;
  else if (key >= 'A' && key <= 'Z')
    idx = key - 'A' + 10;
  else
    return;
  G_SphereBitmap[idx] = !G_SphereBitmap[idx];
  switch(idx)
  {
    case ONCUBE:
      printf("Oncube set to %d\n", G_SphereBitmap[idx]); break;
    case SPHERESTRIPREG:
      printf("SphereStripReg set to %d\n", G_SphereBitmap[idx]); break;
    case SPHERESTRIPRND:
      printf("SphereStripRnd set to %d\n", G_SphereBitmap[idx]); break;
    case TWOPOLEREG:
      printf("TwoPoleReg set to %d\n", G_SphereBitmap[idx]); break;
    case TWOPOLERND:
      printf("TwoPoleRnd set to %d\n", G_SphereBitmap[idx]); break;
    case HAMMERSLEY:
      printf("SphereHammersley set to %d\n", G_SphereBitmap[idx]); break;
    case HAMMERSLEY2:
      printf("SphereHammersley2 (p=%d) set to %d\n", G_p1, G_SphereBitmap[idx]); break;
    case HALTON:
      printf("SphereHalton (2,%d) set to %d\n", G_p2, G_SphereBitmap[idx]); break;
    case HALTON2:
      printf("SphereHalton2  set to %d\n", G_SphereBitmap[idx]); break;
    case PHAMMERSLEY:
      printf("PlaneHammersley set to %d\n", G_SphereBitmap[idx]); break;
    case PHAMMERSLEY2:
      printf("PlaneHammersley2 set to %d\n", G_SphereBitmap[idx]); break;
    case PHALTON:
      printf("PlaneHalton set to %d\n", G_SphereBitmap[idx]); break;
    case PHALTON2:
      printf("PlaneHalton2 set to %d\n", G_SphereBitmap[idx]); break;
    case PRANDOM:
      printf("PlaneRandom set to %d\n", G_SphereBitmap[idx]); break;
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

    case MENU_RECORDTRANSFORM:
      G_RecordTransform=TRUE;
      break;

    case MENU_SAVESCREEN:
      SaveRGBImage("rgbimg.ppm");
      break;

    case MENU_RELOAD:
      InitModel();
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
  sprintf(title, "Light Field Viewer\n");
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
  glutAddMenuEntry("Record transform matrix", MENU_RECORDTRANSFORM);
  glutAddMenuEntry("Save current image", MENU_SAVESCREEN);
  glutAddMenuEntry("Reload object", MENU_RELOAD);
  glutAttachMenu(GLUT_RIGHT_BUTTON);

  InitModel();
  InitRenderer();
  glutMainLoop();
}
