/*************************************************
** improvedCaustics.cpp                         **
** ---------------                              **
**                                              **
** This is the main entry point for the caustic **
**    demo, and includes the GLUT display()     **
**    callback (and lots of global variable     **
**    declarations!)                            **
**                                              **
** Chris Wyman (9/28/2000)                      **
*************************************************/

#include "glut_template.h"
#include "glm.h"
#include "trackball.h"
#include "materials.h"
#include "MovieMaker.h"

/***************************************************
** Globals describing buffer sizes...             **
***************************************************/
int screenSize = DEFAULT_SCREEN_SIZE;
int mainbufferSize = DEFAULT_MAINBUFFER_SIZE;
int causticMapSize = DEFAULT_CAUSTICMAP_SIZE;
int numberOfPhotons = DEFAULT_NUM_PHOTONS;
int refractionBufSize = DEFAULT_REFRACTION_BUFFER_SIZE;
int backgroundBufSize = DEFAULT_BACKGROUND_BUFFER_SIZE;

/***************************************************
** Globals! (In other words, I'm a C programmer)  **
***************************************************/
GLfloat origLookFrom[2][4] = {
	{0, 0, 5, 1},   // the original EYE's point of view
	{0, 0, 5, 1} }; // the original LIGHT's point of view
GLfloat lookFrom[2][4] = { 
	{0, 0, 5, 1},   // the current EYE's point of view
	{0, 0, 5, 1} }; // the current LIGHT's point of view

GLdouble lookAtMatrix[2][16];
GLdouble inverseLookAtMatrix[2][16];

int screen_width = 0, screen_height = 0;
int object=0, material=MAT_BRASS;
int buttonDown = -1, lightingEnabled = 1;
int showShadowMap = 1,useShadowMap = 1, benchmarking=0, frameCount=0;
clock_t startTime;
float rotate = 0, fps = 0;
int displayType = 0;
double initialMatrix[16];
GLfloat myNear = 2, myFar = 25;
int cullMode = GL_BACK;
extern int rotatingScreenShots, rotating, screenShot, tmpUIVal, usePseudoPhotonMap;
extern GLfloat moveOthersX, moveOthersY, moveOthersZ;
GLint tmpMagFilter = GL_NEAREST, tmpMinFilter = GL_NEAREST;
MovieMaker *makeMovie;
int makingMovie=0;
extern FrameBuffer *fb; 
GLuint fbo_ColorTex, fbo_Color2Tex, fbo_DepthTex, fbo_StencilTex, fbo_TestTex;
GLuint ffColor, ffColor2, ffColor3, ffDepth, bfColor, bfDepth;
GLuint distColor, fp32ColorTex, fp32DepthTex;
GLuint distDepth, ooColor, ooDepth, o1Color, o1Depth, o2Color, o2Depth;
GLuint geomDepth;
int needToRerenderCaustics=1, causticImagesRefreshed=0, drawRefractionOnly=0;
float searchRadius = 0.1;

/***************************************************
** OpenGL State Limits Information globals        **
***************************************************/
GLint maxTextureRes=64;                           // max texture resolution (must be >= 64, by spec)

/***************************************************
** Texture globals                                **
***************************************************/
GLuint objFront=0, objFrontNorms=0;        // front surfaces
GLuint objBack=0, objBackNorms=0;          // very back surfaces
GLuint objBackFace=0, objBackFaceNorms=0;  // back facing surfaces
GLuint objDist=0, objDistBackFace=0;       // distance from front to "back" surfaces
GLuint arcCosFresnelTex=0;                 // a 1D texture approximating the arccos function
GLuint uvObj=0;                            // a texture generating using obj uv coords
GLuint geomImageTex=0;                     // the geometry image projected to a octahedron, unfolded to a square.
GLuint geomIndexMap=0;                     // stores indices into geomImage for front/back faces
GLuint mainImg=0;   
GLuint macbethColorTex=0, woodTex=0;
GLuint causticLocationBuffer, causticColorBuffer;
extern GLuint causticPointBuffer;
GLubyte arcCosTexture[4096];                // Data for the arcCos 1d texture
GLuint shadowTextureNum=0;
GLfloat normalizationFactor = 20;

/***************************************************
** Environment map globals                        **
***************************************************/
GLuint hdrCubeMap[NUM_ENVMAPS];

/***************************************************
** Model globals                                  **
***************************************************/
GLuint  sphereList;

/***************************************************
** globals modifiable via user-input              **
***************************************************/
int currentEnvMap = 0;
int drawComplex = 1;
GLfloat objScale = 2;
int currentModel = 1;
float index2_1 = 1.2, index2_1sqr = 1.44;
int drawSideImages = 0;
int drawHelpMenu = 0;
int shaderLoadError = 0;
int whichBackSide = 0;
int drawQuads = 0;
int triangleDisplay=0;
float offset = 0.1;
int displayBig = 0;
int frame=0, clockPtr=0;
clock_t myClock[20];
float time_angle=0, time_index = 1.2, time_angleEnv=0;
unsigned char* screenCapture=0;  // initialized in makeLineTexture()
char screenCaptureFilename[1024];
int viewpoint = EYE_POV;
GLfloat fishEyeProj[16];
int usingCGShaders=1;
extern GLint moveTmp;
extern int refractBackground;
GLfloat fisheyeOffset[3] = {0,0,0};
GLenum origDrawBufs[1] = { GL_COLOR_ATTACHMENT0_EXT };
GLenum causticDrawBufs[2] = { ATTACH_PHOTON_LOCATION_BUFFER, ATTACH_PHOTON_INCIDENT_DIRECTION };
GLenum backgroundDrawBufs[2] = { ATTACH_IMAGE_OF_BACKGROUND_GEOM, ATTACH_BACKGROUND_SURFACE_NORMS };

// For some reason, caustic buffers need two passes at startup to remove trash.
//   I should really fix the problem instead of kluding it by making it render caustics 2x (oh, well!)
int programStartup=1;




/***************************************************
** display function -- what to draw               **
***************************************************/
void display ( void )
{
  /* Initialize our framebuffer objects */
  if (!fb) initFramebuffers();

  /* determing what are fps have been up till now */
  fps = UpdateFPS();

  /* benchmarking may be desired, if so we need to know # of frames done */
  if (benchmarking) frameCount++;

  /* some setup */
  glEnable( GL_NORMALIZE );
  glEnable( GL_DEPTH_TEST );
  glClearDepth( 1.0 );
  glClearColor( 0, 0, 0, 0 );
  glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

  /* Draw the scene we want to the framebuffer 'fb' */
  AllowUnclampedColors();  // This process may involve values outside [0..1]
  if (drawRefractionOnly)
	DrawRefractedScene( viewpoint );
  else 
    DrawCausticsSceneWithLightSpaceGatheredPhotons( viewpoint );
  ResumeClampingColors();  // OK, we can allow OpenGL to start clamping to [0..1] again.

  /* we only want to draw to the size of the visible screen, even if 'fb' is bigger! */
  glViewport( 0, 0, screenSize, screenSize );
  
  /* either display our final result, or, depending on user input, some intermediate step */
  if (displayBig > 0)
		DrawBigScreenTexture(displayBig);
  else
  {
	  displayLargeTexture( fb->GetColorTextureID( 0 ), GL_LINEAR, GL_LINEAR );
	
	  /* draw miscellaneous data onscreen */
	  glDisable( GL_DEPTH_TEST );
      if (drawSideImages && displayBig < 1)
		DisplaySideImages();
	  if (drawHelpMenu)
		drawHelpScreen();
      if (shaderLoadError)
		drawErrorMessage();
      if (!screenShot && !makingMovie)
		displayTimer( fps );
      glEnable( GL_DEPTH_TEST );
  }

  /* grab a screenshot if necessary */
  if (screenShot)
  {
    /* note data is stored in memory with y-inverted to how stored in PPM, so invert when writing */
	glReadPixels( 0, 0, screenSize, screenSize, GL_RGB, GL_UNSIGNED_BYTE, screenCapture );
	WritePPM( screenCaptureFilename, PPM_RAW, screenSize, -screenSize, screenCapture );    
	screenShot = 0;
  }

  /* flush GL commands & swap buffers */
  glFlush();
  glutSwapBuffers();

  /* if we're using the built-in movie capture, we need to add a frame */
  if (makingMovie) makeMovie->AddCurrentFrame();
}



/***************************************************
** Main program entry point.                      **
***************************************************/
int main(int argc, char* argv[])
{
	/* Do all the general purpose startup stuff...*/
  glutInit(&argc, argv);

  /* we got a RGBA buffer and we're double buffering! */
  glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_ALPHA );

  /* set some size for the window. */
  glutInitWindowSize( screenSize, screenSize );

  /* arbitrarily set an initial window position... */
  glutInitWindowPosition( 100, 100 );

  /* make the window.  give it a cool title */
  glutCreateWindow("Reducing Noise in Image-Space Caustics With Variable-Sized Splatting (DEMO)");

  /* load glew */
  glewInit();

  /* set the callback functions */
  glutDisplayFunc( display );
  glutReshapeFunc( reshape );
  glutIdleFunc( idle );
  glutMouseFunc( button );
  glutMotionFunc( motion );
  glutKeyboardFunc( keys );
  glutSpecialFunc( special_keys );

  /* Print out basic info about the demo on the command line */
  DemoInformationToStdout();
  
  AllocateTrackballs( TOTAL_TRACKBALLS );

  if( argc > 0 && argv[1] )
	  ParseGeometrySettingsFile( argv[1] );
  else
	  UseDefaultSettings();

  /* load models/textures/other data */
  printf("(+) Loading and allocating textures...\n" );
  SetupAndAllocateTextures();
  LoadModels();

  SetupModelMenu( );

  glEnable( GL_LIGHTING );
  glEnable( GL_LIGHT0 );

  initLookAtMatrices();

  glPushMatrix();
  glLoadIdentity();
  glRotatef( -90, 1, 0, 0 ); 
  glGetDoublev( GL_MODELVIEW_MATRIX, initialMatrix );
  glPopMatrix();

  printf("(+) Loading vertex and fragment shaders...\n" );
  InitCGPrograms();

  printf("(+) Starting Main Loop...\n" );
  makeMovie = new MovieMaker();

  glutMainLoop();

  return 0;
}







