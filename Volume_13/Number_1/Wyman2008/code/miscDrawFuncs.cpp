/*************************************************
** miscDrawFuncs.cpp                            **
** -----------                                  **
**                                              **
** Contains miscellaneous drawing routines      **
**    for things like drawing the timer,        **
**    displaying a texture as full screen,      **
**    drawing inset images, etc.  The only      **
**    function useful for caustics is:          **
**       SetRefractiveObjectShaderParameters()  **
**                                              **
** Chris Wyman (9/07/2006)                      **
*************************************************/

#include "glut_template.h"
#include "glm.h"
#include "trackball.h"
#include "materials.h"

/***************************************************
** Model globals                                  **
***************************************************/
extern GLuint sphereList;

extern int currentModel;

extern GLuint uvObj;  
extern int screen_width , screen_height;

extern int geomImageMapCreated;
extern GLuint geomImageTex;                     // the geometry image projected to a octahedron, unfolded to a square.
extern GLuint geomIndexMap;
extern GLuint objBack, objBackNorms;
extern GLuint objFront, objFrontNorms;
extern GLuint objDist, objDistBackFace;
extern GLuint objBackFace, objBackFaceNorms;
extern GLuint macbethColorTex, woodTex;

extern float offset;
extern unsigned char *uvMappingData;
extern int drawComplex;
extern GLfloat objScale;
extern GLfloat myNear, myFar;
extern int whichBackSide;

extern double initialMatrix[16];
extern GLfloat fishEyeProj[16];
extern GLuint *render_textures;

extern float time_angle;
extern int frame, clockPtr;
extern clock_t myClock[20];
extern GLfloat moveOthersX, moveOthersY, moveOthersZ;
extern GLint moveTmp;
extern GLfloat fisheyeOffset[3];

GLfloat boundingSphereNormalization = 8;
extern GLfloat normalizationFactor;

extern FrameBuffer *fb, *backgroundGeomFB, *backNormalsFB, *tempFB;
extern FrameBuffer *geomDepthFB, *causticFB, *pseudoPhotonMapFB;
extern GLuint tmpFBTex0, tmpFBTex1;

/* drawing parameters for refractive object's shaders */
extern GLfloat lookFrom[2][4], origLookFrom[2][4];
extern GLdouble lookAtMatrix[2][16], inverseLookAtMatrix[2][16];
extern int currentCGVertShader, currentCGFragShader;
extern float index2_1, index2_1sqr;
extern float time_angleEnv;
extern int usingCGShaders;

// some experimental variables to look at a mipmap-style heirarchy for LS caustic maps
extern GLuint causticMapMipmapLevels[16];
extern int numMipmapLevels;
extern FrameBuffer *causticMipmapsFB;

/***************************************************
** Draws the insets on right/left sides of screen **
***************************************************/
void DisplaySideImages( void )
{
    glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D( 0, 1, 0, 1 );
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glPushAttrib( GL_LIGHTING_BIT | GL_TEXTURE_BIT );
	glDisable(GL_LIGHTING);
	glActiveTexture( GL_TEXTURE0 );
	glEnable(GL_TEXTURE_2D);
	glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );

	/* right side display */
	glBindTexture(GL_TEXTURE_2D, TEXTURE_IMAGE_OF_BACKGROUND_GEOM );
	glColor3f(1,1,1);
	glBegin(GL_QUADS);
	glTexCoord2f( 0, 0 ); glVertex2f( 0.8, 0 );
	glTexCoord2f( 1, 0 ); glVertex2f( 1.0, 0 );
	glTexCoord2f( 1, 1 ); glVertex2f( 1.0, 0.2 );
	glTexCoord2f( 0, 1 ); glVertex2f( 0.8, 0.2 );
	glEnd();

	glBindTexture(GL_TEXTURE_2D, TEXTURE_BACKFACING_NORMALS );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE );
	glColor3f(1,1,1);
	glBegin(GL_QUADS);
	glTexCoord2f( 0, 0 ); glVertex2f( 0.8, 0.2 );
	glTexCoord2f( 1, 0 ); glVertex2f( 1.0, 0.2 );
	glTexCoord2f( 1, 1 ); glVertex2f( 1.0, 0.4 );
	glTexCoord2f( 0, 1 ); glVertex2f( 0.8, 0.4 );
	glEnd();

	glBindTexture(GL_TEXTURE_2D, TEXTURE_PHOTON_LOCATION_BUFFER );
	glColor3f(1,1,1);
	glBegin(GL_QUADS);
	glTexCoord2f( 0, 0 ); glVertex2f( 0.8, 0.4 );
	glTexCoord2f( 1, 0 ); glVertex2f( 1.0, 0.4 );
	glTexCoord2f( 1, 1 ); glVertex2f( 1.0, 0.6 );
	glTexCoord2f( 0, 1 ); glVertex2f( 0.8, 0.6 );
	glEnd();

	glBindTexture(GL_TEXTURE_2D, TEXTURE_BACKFACING_DEPTH );
	glColor3f(1,1,1);
	glBegin(GL_QUADS);
	glTexCoord2f( 0, 0 ); glVertex2f( 0.8, 0.6 );
	glTexCoord2f( 1, 0 ); glVertex2f( 1.0, 0.6 );
	glTexCoord2f( 1, 1 ); glVertex2f( 1.0, 0.8 );
	glTexCoord2f( 0, 1 ); glVertex2f( 0.8, 0.8 );
	glEnd();

	glBindTexture(GL_TEXTURE_2D, backgroundGeomFB->GetDepthTextureID() );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE );
	glColor3f(1,1,1);
	glBegin(GL_QUADS);
	glTexCoord2f( 0, 0 ); glVertex2f( 0.8, 0.8 );
	glTexCoord2f( 1, 0 ); glVertex2f( 1.0, 0.8 );
	glTexCoord2f( 1, 1 ); glVertex2f( 1.0, 1.0 );
	glTexCoord2f( 0, 1 ); glVertex2f( 0.8, 1.0 );
	glEnd();

	glDisable(GL_TEXTURE_2D);
	glPopAttrib( );
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
}



/***************************************************
** Select which texture to draw full-screen       **
***************************************************/
void DrawBigScreenTexture( int displayBig )
{
	if (displayBig == 1)
	  {
		  if (backgroundGeomFB) displayLargeTexture( TEXTURE_IMAGE_OF_BACKGROUND_GEOM );
	  }
    if (displayBig == 3)
	  {
		  if (backgroundGeomFB) displayLargeTexture( backgroundGeomFB->GetColorTextureID(1 ) );
	  }
    if (displayBig == 2)
	  {
		  if (backgroundGeomFB) displayLargeTexture( backgroundGeomFB->GetDepthTextureID() );
	  }
	if (displayBig == 4)
	  {
		  if (backNormalsFB) displayLargeTexture( TEXTURE_BACKFACING_NORMALS );
	  }
    if (displayBig == 5)
	  {
		  if (backNormalsFB) displayLargeTexture( TEXTURE_BACKFACING_DEPTH );
	  }
    if (displayBig == 6)
	  {
		  if (pseudoPhotonMapFB) displayLargeTexture( pseudoPhotonMapFB->GetColorTextureID( 0 ), 
													  GL_NEAREST, GL_NEAREST, 1 );
	  }
    if (displayBig == 7)
	  {
		  if (causticFB) displayLargeTexture( TEXTURE_PHOTON_LOCATION_BUFFER );
	  }
    if (displayBig == 8)
	  {
		  if (causticFB) displayLargeTexture( TEXTURE_PHOTON_INCIDENT_DIRECTION );
	  }
	if (displayBig == 9)
	  {
		  if (causticFB) displayLargeTexture( TEXTURE_COMPLETE_DEPTH_FROM_LIGHT );
	  }
    if (displayBig == 10)
	  {
		  if (tempFB) displayLargeTexture( tempFB->GetColorTextureID( 0 ) );
	  }
    if (displayBig == 11)
	  {
		  if (tempFB) displayLargeTexture( tempFB->GetColorTextureID( 1 ) );
	  }

}


void displayLargeTexture( GLuint tex, int minMode, int magMode, float color, GLenum textureFunc )
{
    glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D( 0, 1, 0, 1 );
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glPushAttrib( GL_LIGHTING_BIT | GL_TEXTURE_BIT | GL_ENABLE_BIT );
	glDisable(GL_LIGHTING);
	glEnable(GL_TEXTURE_2D);
	glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, textureFunc );

	glDisable( GL_DEPTH_TEST );
	glDepthMask( GL_FALSE );
	glBindTexture( GL_TEXTURE_2D, tex );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magMode ); 
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minMode ); 
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );
	glColor3f(color,color,color);
	glBegin(GL_QUADS);
	glTexCoord2f( 0, 0 ); glVertex2f( 0.0, 0.0 );
	glTexCoord2f( 1, 0 ); glVertex2f( 1, 0.0 );
	glTexCoord2f( 1, 1 ); glVertex2f( 1, 1 );
	glTexCoord2f( 0, 1 ); glVertex2f( 0.0, 1 );
	glEnd();
	glDepthMask( GL_TRUE );
	glEnable( GL_DEPTH_TEST );

	glDisable(GL_TEXTURE_2D);
	glBindTexture( GL_TEXTURE_2D, 0 );
	glPopAttrib( );
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
}

void displayLargeTextureWithDepth( GLuint tex, GLuint depth, int minMode, int magMode )
{
    glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D( 0, 1, 0, 1 );
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glPushAttrib( GL_LIGHTING_BIT | GL_TEXTURE_BIT | GL_ENABLE_BIT );
	glDisable(GL_LIGHTING);

	glDepthFunc( GL_ALWAYS );
	glDepthMask( GL_TRUE );
	glActiveTexture( GL_TEXTURE0 );
	glBindTexture( GL_TEXTURE_2D, tex );
	glEnable(GL_TEXTURE_2D);
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magMode ); 
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minMode ); 
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );
	glActiveTexture( GL_TEXTURE1 );
	glBindTexture( GL_TEXTURE_2D, depth );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magMode ); 
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minMode ); 
	glEnable(GL_TEXTURE_2D);

	BindCGProgram( CG_COPY_TEXTURE_AND_DEPTH_F );
	glBegin(GL_QUADS);
	glTexCoord2f( 0, 0 ); glVertex2f( 0.0, 0.0 );
	glTexCoord2f( 1, 0 ); glVertex2f( 1, 0.0 );
	glTexCoord2f( 1, 1 ); glVertex2f( 1, 1 );
	glTexCoord2f( 0, 1 ); glVertex2f( 0.0, 1 );
	glEnd();
	UnbindCGProgram( CG_COPY_TEXTURE_AND_DEPTH_F );
	glDepthFunc( GL_LEQUAL );

	glDisable(GL_TEXTURE_2D);
	glBindTexture( GL_TEXTURE_2D, 0 );
	glActiveTexture( GL_TEXTURE0 );
	glDisable(GL_TEXTURE_2D);
	glBindTexture( GL_TEXTURE_2D, 0 );
	glPopAttrib( );
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
}

void displayLargeTextureWhereNonblack( GLuint colorTex, GLuint maskTex, int minMode, int magMode, float color)
{
    glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D( 0, 1, 0, 1 );
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glPushAttrib( GL_LIGHTING_BIT | GL_TEXTURE_BIT );
	glDisable(GL_LIGHTING);
	glEnable(GL_TEXTURE_2D);
	glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE );

	glActiveTexture( GL_TEXTURE0 );
	glBindTexture( GL_TEXTURE_2D, colorTex );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magMode ); 
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minMode ); 
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );
	glActiveTexture( GL_TEXTURE1 );
	glBindTexture( GL_TEXTURE_2D, maskTex );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magMode ); 
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minMode ); 
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );
	
	//glEnable( GL_DEPTH_TEST );
	glDepthMask( GL_FALSE );
    BindCGProgram( CG_DRAW_TEXTURE_WITH_MASK_FRAGMENT );
	glColor3f(color,color,color);
	glBegin(GL_QUADS);
	glTexCoord2f( 0, 0 ); glVertex2f( 0.0, 0.0 );
	glTexCoord2f( 1, 0 ); glVertex2f( 1, 0.0 );
	glTexCoord2f( 1, 1 ); glVertex2f( 1, 1 );
	glTexCoord2f( 0, 1 ); glVertex2f( 0.0, 1 );
	glEnd();
	UnbindCGProgram( CG_DRAW_TEXTURE_WITH_MASK_FRAGMENT );
	glDepthMask( GL_TRUE );
	//glDisable( GL_DEPTH_TEST );

	glDisable(GL_TEXTURE_2D);
	glBindTexture( GL_TEXTURE_2D, 0 );
	glActiveTexture( GL_TEXTURE0 );
	glDisable(GL_TEXTURE_2D);
	glBindTexture( GL_TEXTURE_2D, 0 );
	glPopAttrib( );
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
}


void drawErrorMessage( void )
{
	glPushAttrib( GL_ALL_ATTRIB_BITS );
	glDisable(GL_LIGHTING);
	glDisable(GL_DEPTH_TEST);
	/* draw 2D text to the screen */
	glMatrixMode( GL_PROJECTION );
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D( 0.0, screen_width, 0.0, screen_height );
	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
	glLoadIdentity();
	glRasterPos2i(0,0);
	glColor3f(1,1,0);

	glPushMatrix();
	glRasterPos2i(0,0);
	glTranslatef(screen_width/2 - 200, screen_height-45, 0);
	glScalef( 0.2, 0.2, 0.2 );
	PrintString("ERROR RELOADING SHADERS!!");
	glPopMatrix();

	glPopMatrix();
	glMatrixMode( GL_PROJECTION );
	glPopMatrix();
	glMatrixMode( GL_MODELVIEW );

	glPopAttrib();
}


/***************************************************
** Use GLUT to print out a basic string           **
***************************************************/
void PrintString2(char *str)        
{                                     
  int len, i;                        
                                   
  len = (int) strlen(str);
  for(i=0; i<len; i++)
        glutStrokeCharacter(GLUT_STROKE_ROMAN, str[i]);
}

void PrintString(char *str, void *font)        
{                                     
  int len, i;                        
                                   
  len = (int) strlen(str);
  for(i=0; i<len; i++)
        glutBitmapCharacter(font, str[i]);
}

/***************************************************
** Draws frame rate in lower-left coner of screen **
***************************************************/
void displayTimer( float fps )
{
	char buf[1024];

	glPushAttrib( GL_ALL_ATTRIB_BITS );
	glDisable(GL_LIGHTING);
	glDisable(GL_DEPTH_TEST);
	/* draw 2D text to the screen */
	glMatrixMode( GL_PROJECTION );
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D( 0.0, screen_width, 0.0, screen_height );
	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
	glLoadIdentity();
	glRasterPos2i(0,0);
	glColor3f(1,1,1);

	glPushMatrix();
	glRasterPos2i(3,10);
	sprintf( buf, "%.2f fps", fps );
	PrintString( buf );
	glPopMatrix();

	glPopMatrix();
	glMatrixMode( GL_PROJECTION );
	glPopMatrix();
	glMatrixMode( GL_MODELVIEW );

	glPopAttrib();
}

/***************************************************
** Update the timer data                          **
***************************************************/
float UpdateFPS( void )
{
	clock_t oldClock;
	unsigned int diff;

	if (frame < 20)
	{
		myClock[frame++] = clock();
		return 0.0;
	}

	oldClock = myClock[clockPtr];
	myClock[clockPtr] = clock();
	diff = myClock[clockPtr] - oldClock;
	clockPtr = (clockPtr+1) % 20;

	return (20*CLOCKS_PER_SEC)/((float)diff);
}


/***************************************************
** set light material properties                  **
***************************************************/
void SetLightProperties( void )
{
	GLfloat lightAmb[4] = {0.2, 0.2, 0.2, 1};
	GLfloat lightDif[4] = {1, 1, 1, 1};
	GLfloat lightSpec[4] = {1, 1, 1, 1};
	glLightfv( GL_LIGHT0, GL_AMBIENT, lightAmb );
	glLightfv( GL_LIGHT0, GL_DIFFUSE, lightDif );
	glLightfv( GL_LIGHT0, GL_SPECULAR, lightSpec );
}

/****************************************************
** Set vertex & fragment program parameters        **
****************************************************/
void SetRefractiveObjectShaderParameters( FrameBuffer *currFB, int fromPOV )
{
	float tmpM[16]; 
	double tmpM2[16];

	SetCGParameter4f( CurrentCGFragmentShader(), "local0",  
								2*myFar*myNear*(myFar-myNear), 
								myFar*myFar - 2*myNear*myFar + myNear*myNear, 
								myFar*myFar - myNear*myNear, 
								myFar*myFar + 2*myNear*myFar + myNear*myNear );
	SetCGParameter4f( CurrentCGFragmentShader(), "local1", 
								2*myFar*myNear, 
								myFar - myNear, 
		    					myFar + myNear, 
								2*objScale*sqrt(3.0) );
	SetCGParameter4f( CurrentCGFragmentShader(), "local2", 
								1.0/index2_1, 
								1.0/index2_1sqr,
								1.0/currFB->GetWidth(), 
								1.0/currFB->GetHeight() );
	SetCGParameter4f( CurrentCGFragmentShader(), "local3", 
								index2_1, 
								index2_1sqr, 
								objScale, 
								sqrt( 1 - 1.0/index2_1sqr ) );
	SetCGParameter4f( CurrentCGFragmentShader(), "local4", 
								myFar*myNear, 
								myFar - myNear, 
								myFar, 
								2*objScale*sqrt(3.0) );
	SetCGParameter4f( CurrentCGFragmentShader(), "up", 0, 1, 0, 0 );
	SetCGParameter4f( CurrentCGFragmentShader(), "lookat", 0, 0, -1, 0 );
	SetCGParameterv( CurrentCGFragmentShader(), "normalizationFactor", 1, &normalizationFactor );

	matInverseTransposed( tmpM2, lookAtMatrix[ fromPOV ] );
    tmpM[0] = tmpM2[0] * (0) + tmpM2[4] * (1) + tmpM2[8] * (0);
	tmpM[1] = tmpM2[1] * (0) + tmpM2[5] * (1) + tmpM2[9] * (0);
	tmpM[2] = tmpM2[2] * (0) + tmpM2[6] * (1) + tmpM2[10] * (0);
	{
		float xformVertex[4];
		xformVertex[0] = lookAtMatrix[ fromPOV ][0] * (20) + lookAtMatrix[ fromPOV ][4] * (-3.5) + lookAtMatrix[ fromPOV ][8] * (10) + lookAtMatrix[ fromPOV ][12] * (1);
		xformVertex[1] = lookAtMatrix[ fromPOV ][1] * (20) + lookAtMatrix[ fromPOV ][5] * (-3.5) + lookAtMatrix[ fromPOV ][9] * (10) + lookAtMatrix[ fromPOV ][13] * (1);
		xformVertex[2] = lookAtMatrix[ fromPOV ][2] * (20) + lookAtMatrix[ fromPOV ][6] * (-3.5) + lookAtMatrix[ fromPOV ][10] * (10) + lookAtMatrix[ fromPOV ][14] * (1);
		xformVertex[3] = lookAtMatrix[ fromPOV ][3] * (20) + lookAtMatrix[ fromPOV ][7] * (-3.5) + lookAtMatrix[ fromPOV ][11] * (10) + lookAtMatrix[ fromPOV ][15] * (1);
		xformVertex[0] /= xformVertex[3];
		xformVertex[1] /= xformVertex[3];
		xformVertex[2] /= xformVertex[3];
		SetCGParameter4f( CurrentCGVertexShader(), "planeEq", tmpM[0], tmpM[1], tmpM[2],
			-(tmpM[0]*xformVertex[0])-(tmpM[1]*xformVertex[1])-(tmpM[2]*xformVertex[2])  );
	}

	/* matrix for environment map's rotation (so we can rotate refl/refr) */
	glMatrixMode( GL_MATRIX0_ARB );
	glLoadIdentity();
	glRotatef( time_angleEnv, 0, 1, 0 );
	MultiplyTrackballMatrix( ENVIRONMENT_TRACKBALL ); 
	if (fromPOV == EYE_POV)
		glMultMatrixd( inverseLookAtMatrix[EYE_POV] );
	else if (fromPOV == LIGHT_POV)
		glMultMatrixd( inverseLookAtMatrix[LIGHT_POV] );

	/* pass stuff to Scott's relection shader (if used) */
	glMatrixMode( GL_MATRIX1_ARB );
	glLoadMatrixd( inverseLookAtMatrix[LIGHT_POV] );
	SetCGParameter4f(CurrentCGFragmentShader(), "lightWSPos", lookFrom[LIGHT_POV][0],
						lookFrom[LIGHT_POV][1], lookFrom[LIGHT_POV][2], 0);

	/* put back in modelview for rendering */
	glMatrixMode( GL_MODELVIEW );
}

void displayLoadingMessage( char *inbuf )
{
	char buf[512];

	glPushAttrib( GL_ALL_ATTRIB_BITS );

	glDrawBuffer( GL_FRONT );

	glDisable(GL_LIGHTING);
	glDisable(GL_DEPTH_TEST);
	/* draw 2D text to the screen */
	glMatrixMode( GL_PROJECTION );
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D( 0.0, screen_width, 0.0, screen_height );
	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
	glLoadIdentity();

	glEnable( GL_BLEND );
	glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
	glColor4f( 0.5, 0.5, 0.9, 0.75 );
	glBegin(GL_QUADS);
	glTexCoord2f( 0, 0 ); glVertex2f( 0.0, 0.0 );
	glTexCoord2f( 1, 0 ); glVertex2f( screen_width, 0.0 );
	glTexCoord2f( 1, 1 ); glVertex2f( screen_width, screen_height );
	glTexCoord2f( 0, 1 ); glVertex2f( 0.0, screen_height );
	glEnd();
	glDisable( GL_BLEND );

	glColor3f(1,1,0);
	sprintf( buf, "Loading %s....", inbuf );
    glRasterPos2i(25, screen_height/2 - 15);
	PrintString(buf, GLUT_BITMAP_TIMES_ROMAN_24);

	glPopMatrix();
	glMatrixMode( GL_PROJECTION );
	glPopMatrix();
	glMatrixMode( GL_MODELVIEW );

	glPopAttrib();

	glFlush();
}


void drawHelpScreen( void )
{
#define HELP_LINES 27
#define LINE_OFFSET 18

	char helpText[HELP_LINES][128] = { 
		"[h] Toggle this help text",
		"[r] Reload shaders from files",
		"[q] Quit", 
		"[ ] Toggle insets (of intermediate steps)",
		"[x] Switch between caustic rendering methods (original, dynamic splats, multires splats)",

		"[v] Cycle through varied environment maps",
		"[c] Toggle between complex refractor & sphere",
		"[f] Toggle temporal filtering",
		"[S/s] Increase/decrease refractor size",
		"[+/-] Increase/decrease index of refraction",

		"[y] Toggle refractor rotation about Y-axis",
		"[w] Toggle view betwen light and eye viewpoints",
		"",
		"[ / and * ] Flip through various full-screen intermediate results",
		"",

		"[ [ and ] ] Zoom in or out from the eye's viewpoint",
		"[L. Mouse] Rotate refractor",
		"[C. Mouse] Rotate view",

		"Alt + [L. Mouse] Rotate environment",
		"Alt + [C. Mouse] Rotate light position",
		"[R. Mouse] Load new refractor models via the menu",
		"",
		"[m] Output scene data to file 'mostRecent.settings'",

		"[End] Capture screen to 'screenCapture.ppm'",
		"[F11] Start capturing a video (to 'myMovie0.avi')",
		"[F12] Stop capturing video",
		"",
	};
	int i;

	glPushAttrib( GL_ALL_ATTRIB_BITS );
	glDisable(GL_LIGHTING);
	glDisable(GL_DEPTH_TEST);
	/* draw 2D text to the screen */
	glMatrixMode( GL_PROJECTION );
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D( 0.0, screen_width, 0.0, screen_height );
	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
	glLoadIdentity();

	glEnable( GL_BLEND );
	glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
	glColor4f( 0.5, 0.5, 0.5, 0.5 );
	glBegin(GL_QUADS);
	glTexCoord2f( 0, 0 ); glVertex2f( 0.0, 0.0 );
	glTexCoord2f( 1, 0 ); glVertex2f( screen_width, 0.0 );
	glTexCoord2f( 1, 1 ); glVertex2f( screen_width, screen_height );
	glTexCoord2f( 0, 1 ); glVertex2f( 0.0, screen_height );
	glEnd();
	glDisable( GL_BLEND );
	
	// Draw black outline of text
	glColor3f(0,0,0);
	for (i=0; i<HELP_LINES; i++)
	{
		glRasterPos2i(4,screen_height-15-LINE_OFFSET*i);
		PrintString(helpText[i]);
		glRasterPos2i(2,screen_height-15-LINE_OFFSET*i);
		PrintString(helpText[i]);
		glRasterPos2i(3,screen_height-16-LINE_OFFSET*i);
		PrintString(helpText[i]);
		glRasterPos2i(3,screen_height-14-LINE_OFFSET*i);
		PrintString(helpText[i]);
	}

	// Draw white text inside of outline
	glColor3f(1,1,1);
    for (i=0; i<HELP_LINES; i++)
	{
		glRasterPos2i(3,screen_height-15-LINE_OFFSET*i);
		PrintString(helpText[i]);
	}

	glPopMatrix();
	glMatrixMode( GL_PROJECTION );
	glPopMatrix();
	glMatrixMode( GL_MODELVIEW );

	glPopAttrib();
}


/****************************************************
** Texture Enable/Disable Functions                **
****************************************************/
void EnableTextureUnit( GLenum texUnit, GLenum texTarget, GLuint textureID )
{
	glActiveTexture( texUnit );
	glBindTexture( texTarget, textureID );
	glEnable( texTarget );
}

void DisableTextureUnit( GLenum texUnit, GLenum texTarget )
{
	glActiveTexture( texUnit );
	glBindTexture( texTarget, 0 );
	glDisable( texTarget );
}