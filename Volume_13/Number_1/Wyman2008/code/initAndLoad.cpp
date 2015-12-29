/* initAndLoad.cpp
** ---------------
**
** This file loads available models, sets up EXT_framebuffer_objects,
**   loads textures, loads command-line scene settings (or sets defaults)
**
** This file is not well documented!  
**
** -Chris Wyman (8/4/2006)
*/

#include "glut_template.h"
#include "glm.h"
#include "trackball.h"
#include "materials.h"

/***************************************************
** Globals describing buffer sizes...             **
***************************************************/
extern int screenSize, mainbufferSize, causticMapSize;
extern int numberOfPhotons, refractionBufSize, backgroundBufSize;

/***************************************************
** Framebuffer globals                            **
***************************************************/
FrameBuffer *fb=0, *backgroundGeomFB=0, *backNormalsFB=0;
FrameBuffer *causticFB=0, *pseudoPhotonMapFB=0, *tempFB=0;
extern GLuint fbo_ColorTex, fbo_Color2Tex, fbo_DepthTex, fbo_StencilTex, fbo_TestTex;
extern GLuint ffColor, ffColor2, ffColor3, ffColor3, ffDepth, bfColor, bfDepth;
extern GLuint distColor, fp32ColorTex, fp32DepthTex;
extern GLuint distDepth, ooColor, ooDepth, o1Color, o1Depth, o2Color, o2Depth;
extern GLuint geomDepth;
GLuint cfb_ColorTex0, cfb_ColorTex1, cfb_DepthTex, rockTex, bg_ExtraDepthTex, cfb_ColorTex2, ffColor4, cfb_ColorTex3;
GLuint tmpFBTex0, tmpFBTex1, tmpFBDepthStencil;
GLuint photonMap_ColorTex0, photonMap_DepthTex, photonMap_ColorTexFromLight;
GLuint photonMap_frame[4];
GLuint causticMapMipmapLevels[16];

/***************************************************
** Model data and related global                  **
***************************************************/
GLMmodel *groundModel, *dragonModel;
GLuint currentObjectList=0, currentObjectNormalList=0;
GLuint dragonList=0;
GLuint groundList=0;
extern GLuint sphereList;
extern int currentModelNum;

/***************************************************
** Globals for Object/Env's yype & position       **
***************************************************/
extern GLfloat objScale;
extern int currentModel;
extern float index2_1, index2_1sqr;
extern int currentEnvMap, lookAtZ;

/***************************************************
** Texture globals                                **
***************************************************/
extern GLuint objFront, objFrontNorms;            // front surfaces
extern GLuint objBack, objBackNorms;              // very back surfaces
extern GLuint objBackFace, objBackFaceNorms;      // back facing surfaces
extern GLuint objDist, objDistBackFace;           // distance from front to "back" surfaces
extern GLuint arcCosFresnelTex;                   // a 1D texture approximating the arccos function
extern GLuint mainImg;   
extern GLubyte arcCosTexture[4096];               // Data for the arcCos 1d texture
extern GLuint shadowTextureNum;
extern GLuint hdrCubeMap[NUM_ENVMAPS];
extern GLuint macbethColorTex, woodTex;
extern GLfloat moveOthersX, moveOthersY, moveOthersZ;
extern GLuint causticLocationBuffer, causticColorBuffer;
GLuint causticPointBuffer;
extern int displayDragon, displayMacBeth, displayTable, viewpoint;
extern float viewDistance;

/***************************************************
** Miscellaneous globals                          **
***************************************************/
extern unsigned char* screenCapture;  
extern char screenCaptureFilename[1024];


/***************************************************
** Matrix globals                                 **
***************************************************/
extern GLfloat origLookFrom[2][4];
extern GLfloat lookFrom[2][4];
extern GLdouble lookAtMatrix[2][16];
extern GLdouble inverseLookAtMatrix[2][16];



void initLookAtMatrices( void )
{
	GLfloat tmpMat[16];

	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
	glLoadIdentity();
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
	gluLookAt(lookFrom[EYE_POV][0], lookFrom[EYE_POV][1], lookFrom[EYE_POV][2],
			0, 0, lookAtZ, 
		    0,1,0);
	glGetDoublev( GL_MODELVIEW_MATRIX, lookAtMatrix[EYE_POV] );
	matInvertd( lookAtMatrix[EYE_POV], inverseLookAtMatrix[EYE_POV] );
	glPopMatrix();

	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
	glLoadIdentity();
	GetTrackBallMatrix( LIGHT_TRACKBALL, tmpMat );
	lookFrom[LIGHT_POV][0] = tmpMat[0]*origLookFrom[LIGHT_POV][0] + 
		tmpMat[4]*origLookFrom[LIGHT_POV][1] + 
		tmpMat[8]*origLookFrom[LIGHT_POV][2] + 
		tmpMat[12]*origLookFrom[LIGHT_POV][3];
	lookFrom[LIGHT_POV][1] = tmpMat[1]*origLookFrom[LIGHT_POV][0] + 
		tmpMat[5]*origLookFrom[LIGHT_POV][1] + 
		tmpMat[9]*origLookFrom[LIGHT_POV][2] + 
		tmpMat[13]*origLookFrom[LIGHT_POV][3];
	lookFrom[LIGHT_POV][2] = tmpMat[2]*origLookFrom[LIGHT_POV][0] + 
		tmpMat[6]*origLookFrom[LIGHT_POV][1] + 
		tmpMat[10]*origLookFrom[LIGHT_POV][2] + 
		tmpMat[14]*origLookFrom[LIGHT_POV][3];
	lookFrom[LIGHT_POV][3] = tmpMat[3]*origLookFrom[LIGHT_POV][0] + 
		tmpMat[7]*origLookFrom[LIGHT_POV][1] + 
		tmpMat[11]*origLookFrom[LIGHT_POV][2] + 
		tmpMat[15]*origLookFrom[LIGHT_POV][3];
	gluLookAt(lookFrom[LIGHT_POV][0], lookFrom[LIGHT_POV][1], lookFrom[LIGHT_POV][2],  
            0, 0, 0,
		    0,1,0);
	glGetDoublev( GL_MODELVIEW_MATRIX, lookAtMatrix[LIGHT_POV] );
	matInvertd( lookAtMatrix[LIGHT_POV], inverseLookAtMatrix[LIGHT_POV] );
	glPopMatrix();
}

float log2( float x )
{
	return log(x)/log(2.0);
}


int resizeFramebuffer( FrameBuffer *fb, int newSize )
{
	int i;
	GLint format;
	GLint maxColorBuffers;

	if (!fb) return 0;

	glGetIntegerv( GL_MAX_COLOR_ATTACHMENTS_EXT, &maxColorBuffers );

	for ( i = 0; i < maxColorBuffers; i++)
	{
		if (fb->GetColorTextureID( i ) > 0)
		{
			glBindTexture( GL_TEXTURE_2D, fb->GetColorTextureID( i ) );
			glGetTexLevelParameteriv( GL_TEXTURE_2D, 0, GL_TEXTURE_INTERNAL_FORMAT, &format );	
			glTexImage2D(GL_TEXTURE_2D, 0, format, newSize, newSize, 0, GL_RGBA, GL_FLOAT, NULL);
		}
	}

	if (fb->GetDepthTextureID() > 0)
		{
			glBindTexture( GL_TEXTURE_2D, fb->GetDepthTextureID() );
			glGetTexLevelParameteriv( GL_TEXTURE_2D, 0, GL_TEXTURE_INTERNAL_FORMAT, &format );	
			glTexImage2D(GL_TEXTURE_2D, 0, format, newSize, newSize, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
		}

	glBindTexture( GL_TEXTURE_2D, 0 );

	fb->SetSize( newSize, newSize );
	fb->CheckFramebufferStatus( 1 );

	return 1;
}

/***************************************************
** Routine to initialize framebuffers and related **
**     textures.                                  **
***************************************************/
void initFramebuffers( void )
{
  printf("(+) Initializing EXT_framebuffer_objects...\n");

  pseudoPhotonMapFB = new FrameBuffer( causticMapSize, causticMapSize, "pseudoPhotonMapFB" );
  glGenTextures(1, &photonMap_frame[0]);
  glBindTexture(GL_TEXTURE_2D, photonMap_frame[0]);
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F_ARB, 
               pseudoPhotonMapFB->GetWidth(), pseudoPhotonMapFB->GetHeight(), 
			   0, GL_RGBA, GL_FLOAT, NULL);
  pseudoPhotonMapFB->AttachColorTexture( photonMap_frame[0], 0 );
  pseudoPhotonMapFB->CheckFramebufferStatus( 1 );

  tempFB = new FrameBuffer( causticMapSize, causticMapSize, "Temporary Storage FB" );
  glGenTextures(1, &tmpFBTex0);
  glBindTexture(GL_TEXTURE_2D, tmpFBTex0);
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F_ARB, 
               tempFB->GetWidth(), tempFB->GetHeight(), 
			   0, GL_RGBA, GL_FLOAT, NULL);
  glGenTextures(1, &tmpFBTex1);
  glBindTexture(GL_TEXTURE_2D, tmpFBTex1);
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F_ARB, 
               tempFB->GetWidth(), tempFB->GetHeight(), 
			   0, GL_RGBA, GL_FLOAT, NULL);
  tempFB->AttachColorTexture( tmpFBTex0, 0 );
  tempFB->AttachColorTexture( tmpFBTex1, 1 );
  tempFB->CheckFramebufferStatus( 1 );
 
  fb = new FrameBuffer( mainbufferSize, mainbufferSize, "screenFB (fb)" );
  glGenTextures(1, &fbo_ColorTex);
  glBindTexture(GL_TEXTURE_2D, fbo_ColorTex);
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 
               fb->GetWidth(), fb->GetHeight(), 0, GL_RGBA, GL_FLOAT, NULL);
  glGenTextures(1, &fbo_DepthTex);
  glBindTexture(GL_TEXTURE_2D, fbo_DepthTex);
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri( GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_LUMINANCE );
  glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 
               fb->GetWidth(), fb->GetHeight(), 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
  fb->AttachColorTexture( fbo_ColorTex, 0 );
  fb->AttachDepthTexture( fbo_DepthTex );
  fb->CheckFramebufferStatus( 1 );

  causticFB = new FrameBuffer( numberOfPhotons, numberOfPhotons, "causticFB" );
  glGenTextures(1, &cfb_ColorTex0);
  glBindTexture(GL_TEXTURE_2D, cfb_ColorTex0);
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F_ARB, 
               causticFB->GetWidth(), causticFB->GetHeight(), 0, GL_RGBA, GL_FLOAT, NULL);
  glGenTextures(1, &cfb_ColorTex1);
  glBindTexture(GL_TEXTURE_2D, cfb_ColorTex1);
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F_ARB, 
               causticFB->GetWidth(), causticFB->GetHeight(), 0, GL_RGBA, GL_FLOAT, NULL);
  glGenTextures(1, &cfb_DepthTex);
  glBindTexture(GL_TEXTURE_2D, cfb_DepthTex);
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri( GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_LUMINANCE );
  glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 
               causticFB->GetWidth(), causticFB->GetHeight(), 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
  causticFB->AttachColorTexture( cfb_ColorTex0, 0 );
  causticFB->AttachColorTexture( cfb_ColorTex1, 1 );
  causticFB->AttachDepthTexture( cfb_DepthTex );
  causticFB->CheckFramebufferStatus( 1 );

  glGenBuffers( 1, &causticLocationBuffer );
  glBindBuffer( GL_PIXEL_PACK_BUFFER_EXT, causticLocationBuffer );
  glBufferData( GL_PIXEL_PACK_BUFFER_EXT, numberOfPhotons*numberOfPhotons*4*sizeof(float), NULL, GL_DYNAMIC_DRAW );

  glGenBuffers( 1, &causticColorBuffer );
  glBindBuffer( GL_PIXEL_PACK_BUFFER_EXT, causticColorBuffer );
  glBufferData( GL_PIXEL_PACK_BUFFER_EXT, numberOfPhotons*numberOfPhotons*4*sizeof(float), NULL, GL_DYNAMIC_DRAW );


  backgroundGeomFB = new FrameBuffer( backgroundBufSize, backgroundBufSize, "backgroundGeomFB" );
  glGenTextures(1, &ffColor);
  glBindTexture(GL_TEXTURE_2D, ffColor);
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F_ARB, 
               backgroundGeomFB->GetWidth(), backgroundGeomFB->GetHeight(), 
			   0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
  glGenTextures(1, &ffColor2);
  glBindTexture(GL_TEXTURE_2D, ffColor2);
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F_ARB, 
               backgroundGeomFB->GetWidth(), backgroundGeomFB->GetHeight(), 
			   0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);			
  glGenTextures(1, &ffDepth);
  glBindTexture(GL_TEXTURE_2D, ffDepth);
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri( GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_LUMINANCE );
  glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 
               backgroundGeomFB->GetWidth(), backgroundGeomFB->GetHeight(), 
			   0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
  backgroundGeomFB->AttachColorTexture( ffColor, 0 );
  backgroundGeomFB->AttachColorTexture( ffColor2, 1 );
  backgroundGeomFB->AttachDepthTexture( ffDepth );
  backgroundGeomFB->CheckFramebufferStatus( 1 );


  backNormalsFB = new FrameBuffer( refractionBufSize, refractionBufSize, "backNormalsFB" );
  glGenTextures(1, &bfColor);
  glBindTexture(GL_TEXTURE_2D, bfColor);
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 
               backNormalsFB->GetWidth(), backNormalsFB->GetHeight(), 
			   0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
  glGenTextures(1, &bfDepth);
  glBindTexture(GL_TEXTURE_2D, bfDepth);
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri( GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_LUMINANCE );
  glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 
               backNormalsFB->GetWidth(), backNormalsFB->GetHeight(), 
			   0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
  backNormalsFB->AttachColorTexture( bfColor, 0 );
  backNormalsFB->AttachDepthTexture( bfDepth );
  backNormalsFB->CheckFramebufferStatus( 1 );

}


/***************************************************
** Routine to load models & create display lists  **
***************************************************/
void LoadModels( void )
{
  int loaded;
  char filename[256];

  printf("(+) Checking model directory for available models...\n");
  CheckModelAvailability();

  /* load the refractor model & associated data */
  printf("(+) Loading default/current refractor model (%s)... ", GetModelIdentifier( currentModelNum )); fflush( stdout );
  loaded = LoadRefractingModel( currentModelNum, &currentObjectList, &currentObjectNormalList );
  if (!loaded)
  {
	  printf("\n*** Error loading model: %s\n", GetModelIdentifier( currentModelNum ) );
	  exit(0);
  }

  printf("\n(+) Loading background models... "); fflush( stdout );

  /* load the optional complex model in background */
  {
     sprintf( filename, "%s%s", MODELS_DIRECTORY, MODEL_BACKGROUND_OBJ );
     dragonModel = glmReadOBJ( filename );
	 glmUnitize( dragonModel );
     glmFacetNormals( dragonModel );
     glmVertexNormals( dragonModel, 90.0 );

	 dragonList = glGenLists(1);
     glNewList( dragonList, GL_COMPILE );
     glmDraw( dragonModel, GLM_SMOOTH );
     glEndList();
  }

  /* load the optional ground terrain model */
  {
     sprintf( filename, "%s%s", MODELS_DIRECTORY, MODEL_GROUND_PLANE );
     groundModel = glmReadOBJ( filename ); 
     glmUnitize( groundModel );
     glmFacetNormals( groundModel );
     glmVertexNormals( groundModel, 90.0 );

     groundList = glGenLists(1);
     glNewList( groundList, GL_COMPILE );
     glmDraw( groundModel, GLM_SMOOTH );
     glEndList();
  }

  /* create simple sphere model */
  sphereList = glGenLists(1);
  glNewList(sphereList, GL_COMPILE);
  CreateSphere( 0.5, 80 );
  glEndList(); 

  printf("\n");
}


/***************************************************
** Parse the file containg saved positions from a **
**   previous run of the program.                 **
***************************************************/
void ParseGeometrySettingsFile( char *filename )
{
	int tmp;
	GLfloat objMat[16], lightMat[16], eyeMat[16], envMat[16], miscMat[16];
	GLfloat tmpX, tmpY, tmpZ;
	FILE *fp;
	printf("(+) Loading command line geometry setting file '%s'...\n", filename );
	fp = fopen( filename, "r" );
	if (!fp) FatalError("Unable to open geometry file!");

	fscanf( fp, "Current OBJ Matrix: %f %f %f %f\n", &objMat[0], &objMat[1], &objMat[2], &objMat[3] );
	fscanf( fp, "Current OBJ Matrix: %f %f %f %f\n", &objMat[4], &objMat[5], &objMat[6], &objMat[7] );
	fscanf( fp, "Current OBJ Matrix: %f %f %f %f\n", &objMat[8], &objMat[9], &objMat[10], &objMat[11] );
	fscanf( fp, "Current OBJ Matrix: %f %f %f %f\n", &objMat[12], &objMat[13], &objMat[14], &objMat[15] );
	SetTrackballMatrixTo( OBJECT_TRACKBALL, objMat );

	fscanf( fp, "Current Env Matrix: %f %f %f %f\n", &envMat[0], &envMat[1], &envMat[2], &envMat[3] );
	fscanf( fp, "Current Env Matrix: %f %f %f %f\n", &envMat[4], &envMat[5], &envMat[6], &envMat[7] );
	fscanf( fp, "Current Env Matrix: %f %f %f %f\n", &envMat[8], &envMat[9], &envMat[10], &envMat[11] );
	fscanf( fp, "Current Env Matrix: %f %f %f %f\n", &envMat[12], &envMat[13], &envMat[14], &envMat[15] );
	SetTrackballMatrixTo( ENVIRONMENT_TRACKBALL, envMat );

	fscanf( fp, "Current Eye Matrix: %f %f %f %f\n", &eyeMat[0], &eyeMat[1], &eyeMat[2], &eyeMat[3] );
	fscanf( fp, "Current Eye Matrix: %f %f %f %f\n", &eyeMat[4], &eyeMat[5], &eyeMat[6], &eyeMat[7] );
	fscanf( fp, "Current Eye Matrix: %f %f %f %f\n", &eyeMat[8], &eyeMat[9], &eyeMat[10], &eyeMat[11] );
	fscanf( fp, "Current Eye Matrix: %f %f %f %f\n", &eyeMat[12], &eyeMat[13], &eyeMat[14], &eyeMat[15] );
	SetTrackballMatrixTo( EYE_TRACKBALL, eyeMat );

	fscanf( fp, "Current Light Matrix: %f %f %f %f\n", &lightMat[0], &lightMat[1], &lightMat[2], &lightMat[3] );
	fscanf( fp, "Current Light Matrix: %f %f %f %f\n", &lightMat[4], &lightMat[5], &lightMat[6], &lightMat[7] );
	fscanf( fp, "Current Light Matrix: %f %f %f %f\n", &lightMat[8], &lightMat[9], &lightMat[10], &lightMat[11] );
	fscanf( fp, "Current Light Matrix: %f %f %f %f\n", &lightMat[12], &lightMat[13], &lightMat[14], &lightMat[15] );
	SetTrackballMatrixTo( LIGHT_TRACKBALL, lightMat );

	fscanf( fp, "Object Scale: %f\n", &objScale ); 
	fscanf( fp, "Index Ratio: %f\n", &index2_1 ); 
	index2_1sqr = index2_1*index2_1;

	fscanf( fp, "Object Number: %d\n", &currentModelNum );    
	fscanf( fp, "Environment Number: %d\n", &currentEnvMap ); 

	fscanf( fp, "Current Misc Matrix: %f %f %f %f\n", &miscMat[0], &miscMat[1], &miscMat[2], &miscMat[3] );
	fscanf( fp, "Current Misc Matrix: %f %f %f %f\n", &miscMat[4], &miscMat[5], &miscMat[6], &miscMat[7] );
	fscanf( fp, "Current Misc Matrix: %f %f %f %f\n", &miscMat[8], &miscMat[9], &miscMat[10], &miscMat[11] );
	tmp = fscanf( fp, "Current Misc Matrix: %f %f %f %f\n", &miscMat[12], &miscMat[13], &miscMat[14], &miscMat[15] );
	if (tmp == 4) SetTrackballMatrixTo( MISC_TRACKBALL, miscMat );

	tmp = fscanf( fp, "Current Misc Offset: %f %f %f\n", &tmpX, &tmpY, &tmpZ );
	if (tmp == 3)
	{
		moveOthersX = tmpX;
		moveOthersY = tmpY;
		moveOthersZ = tmpZ;
	}

	fscanf( fp, "Current Zoom Factor: %f\n", &viewDistance );
	fscanf( fp, "Current Displayed Objects: %d %d %d\n", &displayDragon, &displayMacBeth, &displayTable );
	fscanf( fp, "Current Viewpoint: %d\n", &viewpoint );

	fclose( fp );
}

void UseDefaultSettings( void )
{
	GLfloat objMat[16];
	printf("(+) No command line settings file specified.  Using defaults...\n" );

	objMat[0] = -0.35;     objMat[1] = 0;     objMat[2] = -0.9367497; objMat[3] = 0;
	objMat[4] = 0.8939239; objMat[5] = -0.28; objMat[6] = -0.35;      objMat[7] = 0;
    objMat[8] = -0.28;     objMat[9] = -0.96; objMat[10] = 0;         objMat[11] = 0;
	objMat[12] = 0;        objMat[13] = 0;    objMat[14] = 0;         objMat[15] = 1;
	SetTrackballMatrixTo( OBJECT_TRACKBALL, objMat );

	objMat[0] = 0.83;       objMat[1] = 0.2471841;  objMat[2] = 0.5;          objMat[3] = 0;
	objMat[4] = -0.2471841; objMat[5] = 0.9369098;	objMat[6] = -0.2471841;   objMat[7] = 0;
    objMat[8] = -0.5;       objMat[9] = 0.2471841;	objMat[10] = 0.83;        objMat[11] = 0;
	objMat[12] = 0;         objMat[13] = 0;			objMat[14] = 0;           objMat[15] = 1;
	SetTrackballMatrixTo( EYE_TRACKBALL, objMat );


	objScale = 2.8;
	index2_1 = 1.2;
	index2_1sqr = index2_1*index2_1;

	currentEnvMap = 4; 

	moveOthersX = 0;
	moveOthersY = 0;
	moveOthersZ = -5;

	viewDistance = 1;
	displayDragon = 0;
	displayMacBeth = 1;
	displayTable = 1;
}

void OpenGLVersionInformation( void )
{
	int glError;
	int retData;

	printf("(+) GL Version         : %s\n", glGetString( GL_VERSION ) );
	printf("(+) GL Renderer        : %s\n", glGetString( GL_RENDERER ) );
	printf("(+) GL Vendor          : %s\n", glGetString( GL_VENDOR ) );
	glGetError();
	glGetString( GL_SHADING_LANGUAGE_VERSION_ARB );
	glError = glGetError();
	if (glError == GL_INVALID_ENUM && glewIsSupported( "GL_ARB_shading_language_100" ) )
		printf("(+) GL Shading Language: Supports only initial GL ARB shaders (rev. 1.051)\n");
	else
		printf("(+) GL Shading Language: %s\n", glGetString( GL_SHADING_LANGUAGE_VERSION_ARB ) );
	printf("(+) GLEW Version       : %s\n", glewGetString( GLEW_VERSION ) );
	printf("(+) OpenGL Implementation Data:\n");

	glGetIntegerv( GL_MAX_COLOR_ATTACHMENTS_EXT, &retData );
	printf("    (-) MAX_COLOR_ATTACHMENTS_EXT : %d\n", retData );
	glGetIntegerv( GL_MAX_RENDERBUFFER_SIZE_EXT, &retData );
	printf("    (-) MAX_RENDERBUFFER_SIZE_EXT : %d\n", retData );
	printf("\n");

}

void Unsupported( char *msg )
{
	printf("\n\nError!  %s\n", msg);
	exit(-1);
}

void  DemoInformationToStdout( void )
{
	printf("(+) Demo program for: \n");
	printf("     \"Reducing Noise in Image-Space Caustics With Variable-Sized Splatting\" \n");
	printf("                      Chris Wyman, University of Iowa\n");
	printf("                  Carsten Dachsbacher, University of Erlangen\n\n");
	printf("(+) Running: OpenGL v%s, GLEW v%s \n", glGetString( GL_VERSION ), glewGetString( GLEW_VERSION ) );
	printf("(+) Compiled against: CG v1.4.0, GLUT v3.7.6 \n");
	printf("(+) Running on a: %s %s\n", glGetString( GL_VENDOR ), glGetString( GL_RENDERER ));
	glGetError();
	
	/* check if the machine is equipped to run the demo */
	if (!GLEW_VERSION_2_0)
		Unsupported("This demo requires OpenGL 2.0 or higher!");
	if (!GLEW_EXT_framebuffer_object)
		Unsupported("This demo requires support for extension EXT_framebuffer_object!");
	if (!GLEW_ARB_texture_float)
		Unsupported("This demo requires support for extension ARB_texture_float!");
	if (!GLEW_ARB_color_buffer_float)
		Unsupported("This demo requires support for extension ARB_color_buffer_float!");

	printf("\n(+) Press 'h' to display on-screen help.\n\n");

}

void LoadHDRTextures( void )
{
	char buf[1024];
    glPixelStorei( GL_UNPACK_ALIGNMENT, 1 );

	{
		int tmpW, tmpH;		
		sprintf( buf, "%s%s", ENVIRONMENT_MAPS_DIRECTORY, "stpeters_cross.hdr" );
		hdrCubeMap[0] = LoadLightProbeCube( buf, &tmpW, &tmpH );
		sprintf( buf, "%s%s", ENVIRONMENT_MAPS_DIRECTORY, "uffizi_cross.hdr" );
		hdrCubeMap[1] = LoadLightProbeCube( buf, &tmpW, &tmpH );
		sprintf( buf, "%s%s", ENVIRONMENT_MAPS_DIRECTORY, "galileo_cross.hdr" );
		hdrCubeMap[2] = LoadLightProbeCube( buf, &tmpW, &tmpH );
		sprintf( buf, "%s%s", ENVIRONMENT_MAPS_DIRECTORY, "grace_cross2.hdr" );
		hdrCubeMap[3] = LoadLightProbeCube( buf, &tmpW, &tmpH );
		sprintf( buf, "%s%s", ENVIRONMENT_MAPS_DIRECTORY, "rnl_cross.hdr" );
		hdrCubeMap[4] = LoadLightProbeCube( buf, &tmpW, &tmpH );
		sprintf( buf, "%s%s", ENVIRONMENT_MAPS_DIRECTORY, "campus_cross.hdr" );
		hdrCubeMap[5] = LoadLightProbeCube( buf, &tmpW, &tmpH );
		sprintf( buf, "%s%s", ENVIRONMENT_MAPS_DIRECTORY, "kitchen_cross.hdr" );
		hdrCubeMap[6] = LoadLightProbeCube( buf, &tmpW, &tmpH );
		sprintf( buf, "%s%s", ENVIRONMENT_MAPS_DIRECTORY, "kitchen_cross2.hdr" );
		hdrCubeMap[7] = LoadLightProbeCube( buf, &tmpW, &tmpH );
		sprintf( buf, "%s%s", ENVIRONMENT_MAPS_DIRECTORY, "building_cross.hdr" );
		hdrCubeMap[8] = LoadLightProbeCube( buf, &tmpW, &tmpH );
		sprintf( buf, "%s%s", ENVIRONMENT_MAPS_DIRECTORY, "beach_cross.hdr" );
		hdrCubeMap[9] = LoadLightProbeCube( buf, &tmpW, &tmpH );
		sprintf( buf, "%s%s", ENVIRONMENT_MAPS_DIRECTORY, "nvlobby_new_cube_mipmap.dds" );
		hdrCubeMap[10] = LoadDDSCube( buf, &tmpW, &tmpH );
		sprintf( buf, "%s%s", ENVIRONMENT_MAPS_DIRECTORY, "black_cross.hdr" );
		hdrCubeMap[11] = LoadLightProbeCube( buf, &tmpW, &tmpH );
	}
}


void SetupAndAllocateTextures(void)
{
   GLfloat white[4] = {1.0, 1.0, 1.0, 1.0};
   char buf[1024];
   int i;

   for ( i=0; i < 1024; i++) 
   {
	   arcCosTexture[4*i+2]   = (int)((acos(2.0*(i/1023.0)-1)/M_PI)*255);    // acos
	   arcCosTexture[4*i+3] = (int)((( 2*asin( 2.0*(i/1023.0)-1 )+M_PI )/(2*M_PI))*255);
	   arcCosTexture[4*i+0] = MIN( 255, (int)(2*(F0 + (1.0-F0)*pow( 1-(i/1023.0), 5.0))*255) );     // fresnel reflect
	   arcCosTexture[4*i+1] = MAX( 0, (int)((1 - 2*(F0 + (1.0-F0)*pow( 1-(i/1023.0), 5.0)))*255) ); // fresnel refract
   }

   glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

   glGenTextures(1, &arcCosFresnelTex );
   glBindTexture( GL_TEXTURE_1D, arcCosFresnelTex );
   glTexImage1D( GL_TEXTURE_1D, 0, GL_RGBA, 1024, 0, GL_RGBA, GL_UNSIGNED_BYTE, arcCosTexture );
   glTexParameteri( GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
   glTexParameteri( GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
   glTexParameteri( GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );

   screenCapture = (unsigned char *) malloc( sizeof(unsigned char) * screenSize * screenSize * 3 );
   if (!screenCapture) FatalError( "Unable to initialize memory!" );
   sprintf( screenCaptureFilename, "screenCapture.ppm" );

   glGenTextures(1, &macbethColorTex );
   glBindTexture( GL_TEXTURE_2D, macbethColorTex );
   sprintf( buf, "%s%s", TEXTURES_DIRECTORY, "MacbethColorChecker_uncorrected.bmp" );
   {
	   int w, h;
	   unsigned char *macbeth = ReadBMP( buf, &w, &h ); 
	   gluBuild2DMipmaps( GL_TEXTURE_2D, GL_RGB, w, h, GL_RGB, GL_UNSIGNED_BYTE, macbeth );
   }
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR );  
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR_MIPMAP_LINEAR );

   glGenTextures(1, &rockTex );
   glBindTexture( GL_TEXTURE_2D, rockTex );
   sprintf( buf, "%s%s", TEXTURES_DIRECTORY, "rock01.ppm" );
   {
	   int w, h, mode;
	   unsigned char *wood = ReadPPM( buf, &mode, &w, &h ); 
	   gluBuild2DMipmaps( GL_TEXTURE_2D, GL_RGB, w, h, GL_RGB, GL_UNSIGNED_BYTE, wood );
   }
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR );  
   glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR_MIPMAP_LINEAR );

   LoadHDRTextures();
}


GLuint LoadTexture( char *fname, unsigned int mask )
{
	GLuint tex;
	int w, h, mode;
	unsigned char *data = ReadPPM( fname, &mode, &w, &h );

	glGenTextures(1, &tex );
    glBindTexture( GL_TEXTURE_2D, tex );
	if (mask & TEX_INTERNAL_RGBA)
		glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, data );
	else
	    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, data );

	
	if (mask & TEX_REPEAT_S)
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
	else if (mask & TEX_MIRROR_REPEAT_S)
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT );
	else if (mask & TEX_CLAMP_S)
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );
	else if (mask & TEX_CLAMP_TO_BORDER_S)
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER );
	else 
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );

	
	if (mask & TEX_REPEAT_T)
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
	else if (mask & TEX_MIRROR_REPEAT_T)
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT );
	else if (mask & TEX_CLAMP_T)
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );
	else if (mask & TEX_CLAMP_TO_BORDER_T)
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER );
	else 
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );

	
	if (mask & TEX_MIN_NEAREST) 
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	 else if (mask & TEX_MIN_NEAR_MIP_NEAR)
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST );
	else if (mask & TEX_MIN_NEAR_MIP_LINEAR)
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR );
	else if (mask & TEX_MIN_LINEAR_MIP_NEAR)
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST );
	else if (mask & TEX_MIN_LINEAR_MIP_LINEAR)
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR );
	else 
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    
	
	if (mask & TEX_MAG_NEAREST)
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
	else 
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );


	free( data );
	return tex;
}


/*
   Create a sphere with radius r, and precision n
   Draw a point for zero radius spheres
*/
void CreateSphere(double r,int n)
{
   int i,j;
   double theta1,theta2,theta3;
   double e[3],p[3];

   if (r < 0) return;
   if (n < 4) return;

   for (j=0;j<n/2;j++) {
      theta1 = j * 2 * M_PI / n - M_PI/2;
      theta2 = (j + 1) * 2 * M_PI / n - M_PI/2;

      glBegin(GL_TRIANGLE_STRIP);
      for (i=0;i<=n;i++) {
         theta3 = i * 2 * M_PI / n;

         e[0] = -cos(theta2) * cos(theta3);
         e[1] = -sin(theta2);
         e[2] = -cos(theta2) * sin(theta3);
         p[0] = r * e[0];
         p[1] = r * e[1];
         p[2] = r * e[2];

		 // store the sphere's curvature into the curvature texture
		 glMultiTexCoord4f( GL_TEXTURE7, 1.0/r, 1.0/r, 1.0/r, 1.0/(r*r) );	              
         glVertexAttrib4dARB(2,e[0],e[1],e[2],2*r);
		 glColor4d(0.5*e[0]+0.5,0.5*e[1]+0.5,0.5*e[2]+0.5, r/sqrt(3.0));
         glVertex3d(p[0],p[1],p[2]);

         e[0] = -cos(theta1) * cos(theta3);
         e[1] = -sin(theta1);
         e[2] = -cos(theta1) * sin(theta3);
         p[0] = r * e[0];
         p[1] = r * e[1];
         p[2] = r * e[2];

		 // store the sphere's curvature into the curvature texture
		 glMultiTexCoord4f( GL_TEXTURE7, 1.0/r, 1.0/r, 1.0/r, 1.0/(r*r) );
         glVertexAttrib4dARB(2,e[0],e[1],e[2],2*r);
         glColor4d(0.5*e[0]+0.5,0.5*e[1]+0.5,0.5*e[2]+0.5, r/sqrt(3.0));
         glVertex3d(p[0],p[1],p[2]);
      }
      glEnd();
   }

}