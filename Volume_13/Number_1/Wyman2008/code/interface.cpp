/********************************************
** interface.cpp                           **
** -------------                           **
**                                         **
** This file includes describes the user   **
**    interface routines for the demo.     **
**    Basically, anything that happens if  **
**    the user presses a key or mouse      **
**    button is (mostly) determined by     **
**    this file.  Not very well commented. **
**                                         **
** Chris Wyman (8/08/2006)                 **
********************************************/

#include "glut_template.h"
#include "glm.h"
#include "trackball.h"
#include "materials.h"
#include "MovieMaker.h"

/***************************************************
** Globals describing buffer sizes...             **
***************************************************/
extern int screenSize, mainbufferSize, causticMapSize;
extern int numberOfPhotons, refractionBufSize, backgroundBufSize;
extern unsigned char* screenCapture;  

/***************************************************
** Globals! (In other words, I'm a C programmer)  **
***************************************************/
extern int screen_width , screen_height;
extern int buttonDown;
extern GLfloat myNear, myFar;
extern int geomImageMapCreated;

extern FrameBuffer *fb, *backgroundGeomFB, *backNormalsFB, *causticFB;
extern FrameBuffer *pseudoPhotonMapFB, *geomDepthFB, *tempFB;
extern GLuint causticLocationBuffer, causticColorBuffer;

/***************************************************
** globals modifiable via user-input              **
***************************************************/
extern int currentEnvMap;
extern int drawComplex;
extern GLfloat objScale;
extern int currentModel;
extern float index2_1, index2_1sqr;
extern int drawSideImages;
extern int drawHelpMenu;
extern int shaderLoadError;
extern int whichBackSide;

extern int triangleDisplay;
extern int displayType;
extern float offset;
extern int displayBig, drawQuads;
int tmpUIVal=0, usePseudoPhotonMap=2;
extern int usingAdaptiveMultresolutionSplats;

extern float time_angle, time_index, time_angleEnv, fps;
int rotating=0;
int indexing=0, rotatingEnv=0, rotatingScreenShots=0, screenShot=0;
extern int frameCount, benchmarking, usingCGShaders;
extern clock_t startTime;
extern GLfloat tmpUIValue;
GLfloat moveOthersX=7, moveOthersY=1.4, moveOthersZ=-8;
GLint moveTmp=0;
int refractBackground = 1;

GLint displayDragon = 1, displayMacBeth = 0, displayTable = 0;
extern GLint tmpMagFilter, tmpMinFilter;
extern MovieMaker *makeMovie;
extern int makingMovie, needToRerenderCaustics;
int movieNum=0;
int isAltPressed=0, drawCausticsWithDepth=0;
extern int viewpoint;
float viewDistance=1.0;
int temporalFiltering = 0, whichFloorTex=1, lookAtZ=0, programStarting=1;
int useExperimentalApproach=0, useNearest=1;
extern int currentTemporalBufferPointer, drawRefractionOnly;
float gaussianSigma = 1.25, rangeSigma = 0.015;
float photonPointSize = 5;
extern float searchRadius;
extern GLuint currentObjectList, currentObjectNormalList;
int currentModelNum = 2;

/***************************************************
** Matrix globals                                 **
***************************************************/
extern GLfloat origLookFrom[2][4];
extern GLfloat lookFrom[2][4];
extern GLdouble lookAtMatrix[2][16];
extern GLdouble inverseLookAtMatrix[2][16];

/***************************************************
** idle function... what to do when bored         **
***************************************************/
void idle ( void )
{

  // Sometimes it appears the 1st frame of caustics aren't quite right.
  //    This gets rid of the garbage by rendering the initial frame at startup twice.  Kludgy.
  if (programStarting) { needToRerenderCaustics = 1; programStarting=0; }

  if (rotating)
  {
	time_angle += 0.5;
	if (time_angle >= 360) time_angle -= 360;
	needToRerenderCaustics = 1;
  }

  glutPostRedisplay();
}

/***************************************************
** deal with user resizing window                 **
***************************************************/
void reshape( int w, int h )
{
  if (!screen_width || !screen_height)
  {
	ResizeTrackballWindow( w, h );
	screen_width=w; screen_height=h;
	glViewport( 0, 0, w, h );
	glMatrixMode( GL_PROJECTION );
	glLoadIdentity();
	gluPerspective( 90, 1, myNear, myFar );
	glMatrixMode( GL_MODELVIEW );
	glutPostRedisplay();
  }
  else glutReshapeWindow( screen_width, screen_height );
}

char *GetWindowName( int intermediateStep )
{
	static char buf[10][128] = {
		"Reducing Noise in Image-Space Caustics With Variable-Sized Splatting (DEMO)",
		"(1) Eye-space background image without refractor",
		"(2) Eye-space background surface depth map",
		"(3) Light-space background surface depth map",
		"(4) Eye-space view of refractor's backfacing surface normals",
		"(5) Eye-space view of refractor's backfacing surface depth",
		"(6) Four frame caustic map (one frame's intensity per channel)",
		"(7) Current frame's photon buffer (i.e., the photon positions)",
		"(8) Surface normal at final photon positions (light-space)",
		"(9) Light-space depth map (a.k.a. the shadow map)",
	};
	return buf[MIN(9,MAX(0,intermediateStep))];
}


void keys(unsigned char key, int x, int y)
{
	switch ( key )
    {
    default:
      break;
    case 'T':
		photonPointSize += 2.0;
		printf("    (-) Photon Splat Size (for original approach): %f\n", photonPointSize );
		needToRerenderCaustics = 1;
		glutPostRedisplay();
		break;
	case 't':
		photonPointSize -= 2.0;
		if (photonPointSize < 1) photonPointSize = 1;
		printf("    (-) Photon Splat Size (for original approach): %f\n", photonPointSize );
		needToRerenderCaustics = 1;
		glutPostRedisplay();
		break;
	case 'X':
	case 'x':
		if (!useExperimentalApproach && !usingAdaptiveMultresolutionSplats)
		{
			useExperimentalApproach = !useExperimentalApproach;	
			printf("    (-) Using dynamically varying splat sizes...\n" );
		}
		else if (useExperimentalApproach && !usingAdaptiveMultresolutionSplats)
		{
			useExperimentalApproach = !useExperimentalApproach;	
            usingAdaptiveMultresolutionSplats = !usingAdaptiveMultresolutionSplats;
			printf("    (-) Using adaptive multiresolution splats (4 fixed radii splats)...\n" );
		}
		else if (!useExperimentalApproach && usingAdaptiveMultresolutionSplats)
		{
			usingAdaptiveMultresolutionSplats = !usingAdaptiveMultresolutionSplats;
			printf("    (-) Using original I3D 2006 work of Wyman and Davis...\n" );
		}
		needToRerenderCaustics = 1;
		glutPostRedisplay();
		break;
	case '1':
		displayDragon = !displayDragon;
		needToRerenderCaustics = 1;
		glutPostRedisplay();
		break;
	case '2':
		displayMacBeth = !displayMacBeth;
		needToRerenderCaustics = 1;
		glutPostRedisplay();
		break;
	case '3':
		displayTable = !displayTable;
		needToRerenderCaustics = 1;
		glutPostRedisplay();
		break;
	case '8':
		break;
	case '9':
		break;
	case '*':
		displayBig++;
		if (displayBig > 9) displayBig=9;
		glutSetWindowTitle( GetWindowName( displayBig ) );
		glutPostRedisplay();
		break;
	case '/':
		displayBig--;
		if (displayBig < 0) displayBig = 0;
		glutSetWindowTitle( GetWindowName( displayBig ) );
		glutPostRedisplay();
		break;
	case 'a':
		benchmarking = !benchmarking;
		if (benchmarking)
		{
			frameCount = 0;
			startTime = clock();
		}
		else
		{
			clock_t currTime = clock();
			unsigned int diff = currTime - startTime;
			printf("(+) Benchmark results: %f fps over %d frames\n", 
					(frameCount*CLOCKS_PER_SEC)/((float)diff),
					frameCount );
		}
		break;
	case 'v':
		currentEnvMap++;
		if (currentEnvMap >= NUM_ENVMAPS) currentEnvMap = 0;
		glutPostRedisplay();
		break;
	case 'V':
		currentEnvMap--;
		if (currentEnvMap < 0) currentEnvMap = NUM_ENVMAPS-1;
		glutPostRedisplay();
		break;
	case 's':
		objScale -= 0.2;
		printf("    (-) Object scale: %f\n", objScale);
		needToRerenderCaustics = 1;
		glutPostRedisplay();
		break;
	case 'S':
		objScale += 0.2;
		printf("    (-) Object scale: %f\n", objScale);
		needToRerenderCaustics = 1;
		glutPostRedisplay();
		break;
	case 'y':
		rotating = !rotating;
		glutPostRedisplay();
		break;

	case 'f':
	case 'F':
		temporalFiltering = ( temporalFiltering == 0 ? 3 : 0 );
		currentTemporalBufferPointer = (temporalFiltering == 0 ? 0 : 1);
		printf("    (-) Temporal filtering: %s\n", temporalFiltering ? "Enabled" : "Disabled" );
		glutPostRedisplay();
		break;

	case '{':
	case '[':
		viewDistance -= 0.05;
		if (viewDistance < 0.05) viewDistance = 0.05;
		printf("Viewing Distance Factor: %f\n", viewDistance);

		{
			GLfloat tmpMat[16];

			/* find the updated eye position */
			GetTrackBallMatrix( EYE_TRACKBALL, tmpMat );
			lookFrom[EYE_POV][0] = viewDistance*tmpMat[0]*origLookFrom[EYE_POV][0] + 
				viewDistance*tmpMat[4]*origLookFrom[EYE_POV][1] + 
				viewDistance*tmpMat[8]*origLookFrom[EYE_POV][2] + 
				tmpMat[12]*origLookFrom[EYE_POV][3];
			lookFrom[EYE_POV][1] = viewDistance*tmpMat[1]*origLookFrom[EYE_POV][0] + 
				viewDistance*tmpMat[5]*origLookFrom[EYE_POV][1] + 
				viewDistance*tmpMat[9]*origLookFrom[EYE_POV][2] + 
				tmpMat[13]*origLookFrom[EYE_POV][3];
			lookFrom[EYE_POV][2] = viewDistance*tmpMat[2]*origLookFrom[EYE_POV][0] + 
				viewDistance*tmpMat[6]*origLookFrom[EYE_POV][1] + 
				viewDistance*tmpMat[10]*origLookFrom[EYE_POV][2] + 
				tmpMat[14]*origLookFrom[EYE_POV][3];
			lookFrom[EYE_POV][3] = viewDistance*tmpMat[3]*origLookFrom[EYE_POV][0] + 
				viewDistance*tmpMat[7]*origLookFrom[EYE_POV][1] + 
				viewDistance*tmpMat[11]*origLookFrom[EYE_POV][2] + 
				tmpMat[15]*origLookFrom[EYE_POV][3];

			/* compute the new, updated gluLookAt() matrix for the eye */
			glMatrixMode( GL_MODELVIEW );
			glPushMatrix();
			glLoadIdentity();
			gluLookAt(lookFrom[EYE_POV][0], lookFrom[EYE_POV][1], lookFrom[EYE_POV][2],
	  				    0, 0, lookAtZ,  
						0,1,0);
			glGetDoublev( GL_MODELVIEW_MATRIX, lookAtMatrix[EYE_POV] );
			matInvertd( lookAtMatrix[EYE_POV], inverseLookAtMatrix[EYE_POV] );
			glPopMatrix();
		}

		glutPostRedisplay();
		break;
	case '}':
	case ']':
		viewDistance += 0.05;
		printf("Viewing Distance Factor: %f\n", viewDistance);

		{
			GLfloat tmpMat[16];

			/* find the updated eye position */
			GetTrackBallMatrix( EYE_TRACKBALL, tmpMat );
			lookFrom[EYE_POV][0] = viewDistance*tmpMat[0]*origLookFrom[EYE_POV][0] + 
				viewDistance*tmpMat[4]*origLookFrom[EYE_POV][1] + 
				viewDistance*tmpMat[8]*origLookFrom[EYE_POV][2] + 
				tmpMat[12]*origLookFrom[EYE_POV][3];
			lookFrom[EYE_POV][1] = viewDistance*tmpMat[1]*origLookFrom[EYE_POV][0] + 
				viewDistance*tmpMat[5]*origLookFrom[EYE_POV][1] + 
				viewDistance*tmpMat[9]*origLookFrom[EYE_POV][2] + 
				tmpMat[13]*origLookFrom[EYE_POV][3];
			lookFrom[EYE_POV][2] = viewDistance*tmpMat[2]*origLookFrom[EYE_POV][0] + 
				viewDistance*tmpMat[6]*origLookFrom[EYE_POV][1] + 
				viewDistance*tmpMat[10]*origLookFrom[EYE_POV][2] + 
				tmpMat[14]*origLookFrom[EYE_POV][3];
			lookFrom[EYE_POV][3] = viewDistance*tmpMat[3]*origLookFrom[EYE_POV][0] + 
				viewDistance*tmpMat[7]*origLookFrom[EYE_POV][1] + 
				viewDistance*tmpMat[11]*origLookFrom[EYE_POV][2] + 
				tmpMat[15]*origLookFrom[EYE_POV][3];

			/* compute the new, updated gluLookAt() matrix for the eye */
			glMatrixMode( GL_MODELVIEW );
			glPushMatrix();
			glLoadIdentity();
			gluLookAt(lookFrom[EYE_POV][0], lookFrom[EYE_POV][1], lookFrom[EYE_POV][2],
	  				    0, 0, lookAtZ,  
						0,1,0);
			glGetDoublev( GL_MODELVIEW_MATRIX, lookAtMatrix[EYE_POV] );
			matInvertd( lookAtMatrix[EYE_POV], inverseLookAtMatrix[EYE_POV] );
			glPopMatrix();
		}

		glutPostRedisplay();
		break;

	case '+':
	case '=':
		if (index2_1 >= 1.5) break;
		index2_1 += 0.01;
		index2_1sqr = index2_1 * index2_1;
		printf("Index2 / index1: %f\n", index2_1);
		needToRerenderCaustics = 1;
		glutPostRedisplay();
		break;
	case '-':
	case '_':
		if (index2_1 <= 0.51) break;
		index2_1 -= 0.01;
		index2_1sqr = index2_1 * index2_1;
		printf("Index2 / index1: %f\n", index2_1);
		needToRerenderCaustics = 1;
		glutPostRedisplay();
		break;

	case 'C':
	case 'c':
		drawComplex = 1-drawComplex;
		needToRerenderCaustics = 1;
		glutPostRedisplay();
		break;
	case ' ':
		drawSideImages = 1-drawSideImages;
		glutPostRedisplay();
		break;
	case 'H':
	case 'h':
		drawHelpMenu = 1-drawHelpMenu;
		glutPostRedisplay();
		break;
	case 'q':
	case 'Q':
		exit(0);
		break;
	case 'r':
	case 'R':
		shaderLoadError=ReloadCGShaders();
		needToRerenderCaustics = 1;
		glutPostRedisplay();
		break;
	case 'b':
	case 'B':
		whichBackSide = 1-whichBackSide;
		needToRerenderCaustics = 1;
		glutPostRedisplay();
		break;
	case '>':
	case '.':
		moveOthersZ -= 0.2;
		needToRerenderCaustics = 1;
		glutPostRedisplay();
		break;
	case '<':
	case ',':
		moveOthersZ += 0.2;
		needToRerenderCaustics = 1;
		glutPostRedisplay();
		break;

	case 'W':
	case 'w':
		viewpoint = !viewpoint;
		glutPostRedisplay();
		break;

	case 'M':
	case 'm':
		{
			FILE *f;
			GLfloat tmpMatrix[16];

			f = fopen( "mostRecent.settings", "w" );

			GetTrackBallMatrix( OBJECT_TRACKBALL, tmpMatrix );
			fprintf(f, "Current OBJ Matrix: %f %f %f %f \n", tmpMatrix[0], tmpMatrix[1], tmpMatrix[2], tmpMatrix[3] );
			fprintf(f, "Current OBJ Matrix: %f %f %f %f \n", tmpMatrix[4], tmpMatrix[5], tmpMatrix[6], tmpMatrix[7] );
			fprintf(f, "Current OBJ Matrix: %f %f %f %f \n", tmpMatrix[8], tmpMatrix[9], tmpMatrix[10], tmpMatrix[11] );
			fprintf(f, "Current OBJ Matrix: %f %f %f %f \n", tmpMatrix[12], tmpMatrix[13], tmpMatrix[14], tmpMatrix[15] );

			GetTrackBallMatrix( ENVIRONMENT_TRACKBALL, tmpMatrix );
			fprintf(f, "Current Env Matrix: %f %f %f %f \n", tmpMatrix[0], tmpMatrix[1], tmpMatrix[2], tmpMatrix[3] );
			fprintf(f, "Current Env Matrix: %f %f %f %f \n", tmpMatrix[4], tmpMatrix[5], tmpMatrix[6], tmpMatrix[7] );
			fprintf(f, "Current Env Matrix: %f %f %f %f \n", tmpMatrix[8], tmpMatrix[9], tmpMatrix[10], tmpMatrix[11] );
			fprintf(f, "Current Env Matrix: %f %f %f %f \n", tmpMatrix[12], tmpMatrix[13], tmpMatrix[14], tmpMatrix[15] );

			GetTrackBallMatrix( EYE_TRACKBALL, tmpMatrix );
			fprintf(f, "Current Eye Matrix: %f %f %f %f \n", tmpMatrix[0], tmpMatrix[1], tmpMatrix[2], tmpMatrix[3] );
			fprintf(f, "Current Eye Matrix: %f %f %f %f \n", tmpMatrix[4], tmpMatrix[5], tmpMatrix[6], tmpMatrix[7] );
			fprintf(f, "Current Eye Matrix: %f %f %f %f \n", tmpMatrix[8], tmpMatrix[9], tmpMatrix[10], tmpMatrix[11] );
			fprintf(f, "Current Eye Matrix: %f %f %f %f \n", tmpMatrix[12], tmpMatrix[13], tmpMatrix[14], tmpMatrix[15] );

			GetTrackBallMatrix( LIGHT_TRACKBALL, tmpMatrix );
			fprintf(f, "Current Light Matrix: %f %f %f %f \n", tmpMatrix[0], tmpMatrix[1], tmpMatrix[2], tmpMatrix[3] );
			fprintf(f, "Current Light Matrix: %f %f %f %f \n", tmpMatrix[4], tmpMatrix[5], tmpMatrix[6], tmpMatrix[7] );
			fprintf(f, "Current Light Matrix: %f %f %f %f \n", tmpMatrix[8], tmpMatrix[9], tmpMatrix[10], tmpMatrix[11] );
			fprintf(f, "Current Light Matrix: %f %f %f %f \n", tmpMatrix[12], tmpMatrix[13], tmpMatrix[14], tmpMatrix[15] );

            PrintInverseTrackballMatrix( LIGHT_TRACKBALL );

			fprintf(f, "Object Scale: %f\n", objScale );
			fprintf(f, "Index Ratio: %f\n", index2_1 ); 
			fprintf(f, "Object Number: %d\n", currentModelNum );
			fprintf(f, "Environment Number: %d\n", currentEnvMap );

			GetTrackBallMatrix( MISC_TRACKBALL, tmpMatrix );
			fprintf(f, "Current Misc Matrix: %f %f %f %f\n", tmpMatrix[0], tmpMatrix[1], tmpMatrix[2], tmpMatrix[3] );
			fprintf(f, "Current Misc Matrix: %f %f %f %f\n", tmpMatrix[4], tmpMatrix[5], tmpMatrix[6], tmpMatrix[7] );
			fprintf(f, "Current Misc Matrix: %f %f %f %f\n", tmpMatrix[8], tmpMatrix[9], tmpMatrix[10], tmpMatrix[11] );
			fprintf(f, "Current Misc Matrix: %f %f %f %f\n", tmpMatrix[12], tmpMatrix[13], tmpMatrix[14], tmpMatrix[15] );

			fprintf(f, "Current Misc Offset: %f %f %f\n", moveOthersX, moveOthersY, moveOthersZ );
			fprintf(f, "Current Zoom Factor: %f\n", viewDistance );
			fprintf(f, "Current Displayed Objects: %d %d %d\n", displayDragon, displayMacBeth, displayTable );
			fprintf(f, "Current Viewpoint: %d\n", viewpoint );

			fclose( f );

			printf("Settings output to \"mostRecent.settings\"...\n");
		}
		break;
	}
}

/***************************************************
** deal with key strokes                          **
***************************************************/
void special_keys(int key, int x, int y)
{
  switch ( key )
    {
    default:
      break;
	case GLUT_KEY_LEFT:
		moveOthersX -= 0.2;
		needToRerenderCaustics = 1;
		glutPostRedisplay();
		break;
	case GLUT_KEY_RIGHT:
		moveOthersX += 0.2;
		needToRerenderCaustics = 1;
		glutPostRedisplay();
		break;
	case GLUT_KEY_UP:
		moveOthersY += 0.2;
		needToRerenderCaustics = 1;
		glutPostRedisplay();
		break;
	case GLUT_KEY_DOWN:
		moveOthersY -= 0.2;
		needToRerenderCaustics = 1;
		glutPostRedisplay();
		break;

	case GLUT_KEY_F5:
		drawRefractionOnly = !drawRefractionOnly;
		break;

	case GLUT_KEY_F8:
		if (lookAtZ == 0)
			lookAtZ = -3;
		else
			lookAtZ=0;
		/* compute the new, updated gluLookAt() matrix for the eye */
		glMatrixMode( GL_MODELVIEW );
		glPushMatrix();
		glLoadIdentity();
		gluLookAt(lookFrom[EYE_POV][0], lookFrom[EYE_POV][1], lookFrom[EYE_POV][2],
	  				    0, 0, lookAtZ,  
						0,1,0);
		glGetDoublev( GL_MODELVIEW_MATRIX, lookAtMatrix[EYE_POV] );
		matInvertd( lookAtMatrix[EYE_POV], inverseLookAtMatrix[EYE_POV] );
		glPopMatrix();
		glutPostRedisplay();
		break;

	case GLUT_KEY_F11:
		{
			char buf[128];
			sprintf( buf, "myMovie%d.avi", movieNum );
			printf("    (-) Starting video capture to '%s'\n", buf);
			makeMovie->StartCapture( buf, min(30, (int)((2*fps)+0.5)) );
			makingMovie = 1;
			movieNum++;
		}
		break;	
	case GLUT_KEY_F12:
		printf("    (-) Done capturing video...\n");
		makingMovie = 0;
		makeMovie->EndCapture();
		break;
	

	case GLUT_KEY_PAGE_UP:
		{
			int xPos = glutGet( GLUT_WINDOW_X );
			int yPos = glutGet( GLUT_WINDOW_Y );
			glutPositionWindow( xPos, yPos-10 );
		}
		break;

	case GLUT_KEY_PAGE_DOWN:
		{
			int xPos = glutGet( GLUT_WINDOW_X );
			int yPos = glutGet( GLUT_WINDOW_Y );
			glutPositionWindow( xPos, yPos+10 );
		}
		break;

	 case GLUT_KEY_END:
		{
			screenShot=1;
			printf("    (-) Capturing screen...\n");
			glutPostRedisplay();
		}
		break;
	}
}

/***************************************************
** deal with mouse movement                       **
***************************************************/
void motion(int x, int y)
{ 
  if (buttonDown == -1) return;

  if (buttonDown == GLUT_LEFT_BUTTON && !isAltPressed)
  {
      UpdateTrackballOnMotion( OBJECT_TRACKBALL, x, y );
	  needToRerenderCaustics = 1;
  }
  if (buttonDown == GLUT_LEFT_BUTTON && isAltPressed)
      UpdateTrackballOnMotion( ENVIRONMENT_TRACKBALL, x, y );

  if (buttonDown == GLUT_MIDDLE_BUTTON && !isAltPressed)
  {
	  GLfloat tmpMat[16];

	  /* update the eye's trackball matrix */
	  UpdateTrackballOnMotion( EYE_TRACKBALL, x, y );

	  /* find the updated eye position */
	  GetTrackBallMatrix( EYE_TRACKBALL, tmpMat );
	  lookFrom[EYE_POV][0] = viewDistance*tmpMat[0]*origLookFrom[EYE_POV][0] + 
		  viewDistance*tmpMat[4]*origLookFrom[EYE_POV][1] + 
		  viewDistance*tmpMat[8]*origLookFrom[EYE_POV][2] + 
		  tmpMat[12]*origLookFrom[EYE_POV][3];
	  lookFrom[EYE_POV][1] = viewDistance*tmpMat[1]*origLookFrom[EYE_POV][0] + 
		  viewDistance*tmpMat[5]*origLookFrom[EYE_POV][1] + 
		  viewDistance*tmpMat[9]*origLookFrom[EYE_POV][2] + 
		  tmpMat[13]*origLookFrom[EYE_POV][3];
	  lookFrom[EYE_POV][2] = viewDistance*tmpMat[2]*origLookFrom[EYE_POV][0] + 
		  viewDistance*tmpMat[6]*origLookFrom[EYE_POV][1] + 
		  viewDistance*tmpMat[10]*origLookFrom[EYE_POV][2] + 
		  tmpMat[14]*origLookFrom[EYE_POV][3];
	  lookFrom[EYE_POV][3] = viewDistance*tmpMat[3]*origLookFrom[EYE_POV][0] + 
		  viewDistance*tmpMat[7]*origLookFrom[EYE_POV][1] + 
		  viewDistance*tmpMat[11]*origLookFrom[EYE_POV][2] + 
		  tmpMat[15]*origLookFrom[EYE_POV][3];

	  /* compute the new, updated gluLookAt() matrix for the eye */
	  glMatrixMode( GL_MODELVIEW );
	  glPushMatrix();
	  glLoadIdentity();
	  gluLookAt(lookFrom[EYE_POV][0], lookFrom[EYE_POV][1], lookFrom[EYE_POV][2],
	  		    0, 0, lookAtZ,  
		        0,1,0);
	  glGetDoublev( GL_MODELVIEW_MATRIX, lookAtMatrix[EYE_POV] );
	  matInvertd( lookAtMatrix[EYE_POV], inverseLookAtMatrix[EYE_POV] );
	  glPopMatrix();
  }
  if (buttonDown == GLUT_MIDDLE_BUTTON && isAltPressed)
  {
	  GLfloat tmpMat[16];

	  /* update the light's trackball matrix */
	  UpdateTrackballOnMotion( LIGHT_TRACKBALL, x, y );

	  /* find the updated light position */
	  GetTrackBallMatrix( LIGHT_TRACKBALL, tmpMat );
	  lookFrom[LIGHT_POV][0] = tmpMat[0]*origLookFrom[LIGHT_POV][0] + tmpMat[4]*origLookFrom[LIGHT_POV][1] + tmpMat[8]*origLookFrom[LIGHT_POV][2] + tmpMat[12]*origLookFrom[LIGHT_POV][3];
	  lookFrom[LIGHT_POV][1] = tmpMat[1]*origLookFrom[LIGHT_POV][0] + tmpMat[5]*origLookFrom[LIGHT_POV][1] + tmpMat[9]*origLookFrom[LIGHT_POV][2] + tmpMat[13]*origLookFrom[LIGHT_POV][3];
	  lookFrom[LIGHT_POV][2] = tmpMat[2]*origLookFrom[LIGHT_POV][0] + tmpMat[6]*origLookFrom[LIGHT_POV][1] + tmpMat[10]*origLookFrom[LIGHT_POV][2] + tmpMat[14]*origLookFrom[LIGHT_POV][3];
	  lookFrom[LIGHT_POV][3] = tmpMat[3]*origLookFrom[LIGHT_POV][0] + tmpMat[7]*origLookFrom[LIGHT_POV][1] + tmpMat[11]*origLookFrom[LIGHT_POV][2] + tmpMat[15]*origLookFrom[LIGHT_POV][3];

	  /* compute the new, updated gluLookAt() matrix for the light */
	  glMatrixMode( GL_MODELVIEW );
	  glPushMatrix();
	  glLoadIdentity();
	  gluLookAt(lookFrom[LIGHT_POV][0], lookFrom[LIGHT_POV][1], lookFrom[LIGHT_POV][2],
	  		    0, 0, 0,  
		        0,1,0);
	  glGetDoublev( GL_MODELVIEW_MATRIX, lookAtMatrix[LIGHT_POV] );
	  matInvertd( lookAtMatrix[LIGHT_POV], inverseLookAtMatrix[LIGHT_POV] );
	  glPopMatrix();
	  needToRerenderCaustics = 1;
  }

  glutPostRedisplay();
}

/***************************************************
** deal with mouse clicks                         **
***************************************************/
void button(int b, int st, int x, int y)
{
	isAltPressed = (glutGetModifiers() == GLUT_ACTIVE_ALT) ;

	if (b == GLUT_LEFT_BUTTON && st == GLUT_DOWN)
	{
		buttonDown = GLUT_LEFT_BUTTON;
		if (!isAltPressed)
			SetTrackballOnClick( OBJECT_TRACKBALL, x, y );
		else
			SetTrackballOnClick( ENVIRONMENT_TRACKBALL, x, y );
	}
	else if (b == GLUT_MIDDLE_BUTTON && st == GLUT_DOWN)
	{

		buttonDown = GLUT_MIDDLE_BUTTON;
		if (!isAltPressed)
			SetTrackballOnClick( EYE_TRACKBALL, x, y );
		else
			SetTrackballOnClick( LIGHT_TRACKBALL, x, y );
	}
	else if (b == GLUT_RIGHT_BUTTON && st == GLUT_DOWN)
	{
		buttonDown = GLUT_RIGHT_BUTTON;
		SetTrackballOnClick( MISC_TRACKBALL, x, y );
	}
	else if (st == GLUT_UP)
		buttonDown = -1;
}


void menu( int value )
{
	int newSize;

	switch (value) 
	{
		default:
			break;
		case 100:
			keys( 'h', 0, 0 );
			break;
		case 300:
			keys( '2', 0, 0 );
			break;
		case 301:
			keys( '1', 0, 0 );
			break;
		case 302:
			keys( '3', 0, 0 );
			break;
		case 500:
		case 501:
		case 502:
		case 503:
		case 504:
		case 505:
		case 506:
			newSize = 64*pow(2.0,value-500);
			numberOfPhotons = newSize;
			resizeFramebuffer( causticFB, numberOfPhotons );
			glBindBuffer( GL_PIXEL_PACK_BUFFER_EXT, causticLocationBuffer );
			glBufferData( GL_PIXEL_PACK_BUFFER_EXT, numberOfPhotons*numberOfPhotons*4*sizeof(float), NULL, GL_DYNAMIC_DRAW );
			glBindBuffer( GL_PIXEL_PACK_BUFFER_EXT, causticColorBuffer );
			glBufferData( GL_PIXEL_PACK_BUFFER_EXT, numberOfPhotons*numberOfPhotons*4*sizeof(float), NULL, GL_DYNAMIC_DRAW );
			needToRerenderCaustics = 1;
			glutPostRedisplay();
			break;
		case 520:
		case 521:
		case 522:
		case 523:
		case 524:
		case 525:
		case 526:
			newSize = 64*pow(2.0,value-520);
			causticMapSize = newSize;
			resizeFramebuffer( pseudoPhotonMapFB, causticMapSize );
			resizeFramebuffer( tempFB, causticMapSize );
			needToRerenderCaustics = 1;
			glutPostRedisplay();
			break;
		case 540:
		case 541:
		case 542:
			newSize = 256*pow(2.0,value-540);
			screenSize = newSize;
			backgroundBufSize = newSize;
			resizeFramebuffer( backgroundGeomFB, backgroundBufSize );
			screen_width = newSize;
			screen_height = newSize;
			glutReshapeWindow( screen_width, screen_height );
			if (screenCapture)
				free( screenCapture );
			screenCapture = (unsigned char *) malloc( sizeof(unsigned char) * screenSize * screenSize * 3 );
			ResizeTrackballWindow( screenSize, screenSize );
			needToRerenderCaustics = 1;
			glutPostRedisplay();
			break;
		case 560:
		case 561:
		case 562:
		case 563:
		case 564:
		case 565:
		case 566:
			newSize = 64*pow(2.0,value-560);
			mainbufferSize = newSize;
			resizeFramebuffer( fb, mainbufferSize );
			needToRerenderCaustics = 1;
			glutPostRedisplay();
			break;
		case 600: // original caustics mapping approach
			usingAdaptiveMultresolutionSplats = 0;
			useExperimentalApproach = 0;
			needToRerenderCaustics = 1;
			printf("    (-) Using original I3D 2006 work of Wyman and Davis...\n" );
			glutPostRedisplay();
			break;
		case 601: // multi-res approach
			useExperimentalApproach = 0;
			usingAdaptiveMultresolutionSplats = 1;
			needToRerenderCaustics = 1;
			printf("    (-) Using adaptive multiresolution splats (4 fixed radii splats)...\n" );
			glutPostRedisplay();
			break;
		case 602: // continuously varying splats
			useExperimentalApproach = 1;
			usingAdaptiveMultresolutionSplats = 0;
			needToRerenderCaustics = 1;
			printf("    (-) Using dynamically varying splat sizes...\n" );
			glutPostRedisplay();
			break;
		case 27: /* quit */
			exit(0);
			break;
	}
	if ( value >= 200 && value < 300 )
	{
		GLuint tmp1, tmp2;
		int loaded;

		displayLoadingMessage( GetModelIdentifier( value-200 ) );

		loaded = LoadRefractingModel( value-200, &tmp1, &tmp2 );
		if (!loaded)
		{
			printf("*** Error loading model: %s\n", GetModelIdentifier( value-200 ) );
			return;
		}

		glDeleteLists( currentObjectList, 1 );
		glDeleteLists( currentObjectNormalList, 1 );

		currentObjectList = tmp1;
		currentObjectNormalList = tmp2;

		printf("(+) Loaded model: %s\n", GetModelIdentifier( value-200 ) );
		needToRerenderCaustics = 1;
		currentModelNum = value-200;
		glutPostRedisplay();
	}
}



/***************************************************
** function to setup menu entries                 **
***************************************************/
void SetupMenu( void )
{
}