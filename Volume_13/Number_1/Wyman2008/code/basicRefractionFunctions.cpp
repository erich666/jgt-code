/********************************************
** basicRefractionFunctions.cpp            **
** ----------------------------            **
**                                         **
** This file performs basic image-space    **
**    refraction, ala the SIGGRAPH 2005    **
**    paper.                               **
**                                         **
** Chris Wyman (8/08/2006)                 **
********************************************/

#include "glut_template.h"
#include "glm.h"
#include "trackball.h"
#include "materials.h"
#include "MovieMaker.h"
#include <CG/cg.h>
#include <CG/cgGL.h>

typedef struct 
{
	CGcontext shaderContext;
	CGprofile vertProfile, fragProfile;
	CGprogram programs[NUM_CG_PROGRAMS];
	CGprofile programType[NUM_CG_PROGRAMS];
	char *programName[NUM_CG_PROGRAMS];
} CGData;


extern CGData cgData;
extern FrameBuffer *fb, *backgroundGeomFB, *backNormalsFB, *causticFB;
extern GLfloat myNear, myFar;
extern int whichBackSide, drawCausticsWithDepth;
extern GLfloat fisheyeOffset[3];
extern GLuint groundList, currentObjectList, currentObjectNormalList, dragonList;

/* info for drawing background geometry */
extern GLuint macbethColorTex, woodTex, arcCosFresnelTex, rockTex;
extern GLfloat objScale;
extern float time_angle, time_angleEnv;
extern GLint displayDragon, displayMacBeth, displayTable;

extern GLfloat normalizationFactor;
extern int currentModel;
extern int drawComplex;
extern double initialMatrix[16];
extern GLfloat moveOthersX, moveOthersY, moveOthersZ;
extern GLint moveTmp, tmpMagFilter, tmpMinFilter;
extern GLuint hdrCubeMap[NUM_ENVMAPS];
extern int currentEnvMap, displayType, material, usingCGShaders;
extern GLuint sphereList;
extern GLfloat lookFrom[2][4];
extern GLdouble lookAtMatrix[2][16], inverseLookAtMatrix[2][16];
extern GLenum origDrawBufs[1], causticDrawBufs[2], backgroundDrawBufs[2];
extern GLuint causticLocationBuffer, causticColorBuffer;
extern int drawQuads, needToRerenderCaustics, causticImagesRefreshed;
extern int tmpUIVal, usePseudoPhotonMap;
extern GLfloat offsetIndices[256];
extern GLuint causticPointBuffer;
GLfloat shadowMapBias = -0.05;
extern GLuint photonMap_frame[4];
extern int temporalFiltering, whichFloorTex;
int currentTemporalBufferPointer=0;
extern int displayBig;
extern int drawSideImages;
extern int drawHelpMenu;
extern int shaderLoadError;
extern int makingMovie;
extern float rotate, fps;
extern int screenShot; 
extern int viewpoint;
extern int refractBackground;

/* set up so we can do stuff in floating-point without worrying about */
/*    OpenGL mysteriously clamping color values to [0..1]             */
void AllowUnclampedColors( void )
{
	glClampColorARB( GL_CLAMP_VERTEX_COLOR_ARB, GL_FALSE );
    glClampColorARB( GL_CLAMP_FRAGMENT_COLOR_ARB, GL_FALSE );
    glClampColorARB( GL_CLAMP_READ_COLOR_ARB, GL_FALSE );
}

/* Set GL clamping state back to defaults */
/*    REMEMBER -- If you use/read this texture/fb *after* this point,      */
/*                you might get clamped values...  The documentation seems */
/*                to suggest otherwise, but experience shows it can occur. */
void ResumeClampingColors( void )
{
	glClampColorARB( GL_CLAMP_VERTEX_COLOR_ARB, GL_TRUE );
    glClampColorARB( GL_CLAMP_FRAGMENT_COLOR_ARB, GL_FIXED_ONLY_ARB );
    glClampColorARB( GL_CLAMP_READ_COLOR_ARB, GL_FIXED_ONLY_ARB );
}

/* setup the viewing matrix from the current POV */
void SetupFrameBufferMatrices( int fromPOV )
{
	
	glMatrixMode( GL_PROJECTION );
	glLoadIdentity();
	gluPerspective( 90, 1, myNear, myFar );
	glMatrixMode( GL_MODELVIEW );
	glLoadIdentity();
	glMultMatrixd( lookAtMatrix[ fromPOV ] ); 
}


/* Draw a refracted scene into the buffer "fb" */
void DrawRefractedScene( int fromPOV )
{
    /* create the buffers needed for refraction (depth images, normals, background geom)  */
    createRefractionBuffers( lookAtMatrix[ fromPOV ], ENABLE_JUST_COLOR, fromPOV );

    /* draw the main stuff -- either into back buffer or into pbuffer */
    fb->BindBuffer();
    glViewport( 0, 0, fb->GetWidth(), fb->GetHeight() );

	/* draw the simple refraction */
    DrawSimpleRefractedScene( fb, fromPOV );

    /* ok, done drawing to offscreen buffer, start writing to the main framebuffer again */
    fb->UnbindBuffer();
}



/* create all the necessary buffers (normal, bg geom, depth, etc) from a particular POV */
void createRefractionBuffers( GLdouble lookAtMatrix[16], int colorOrPhotons, int fromPOV )
{
	backNormalsFB->BindBuffer();
	if (!whichBackSide)
		createBackSideMaps( backNormalsFB, fromPOV );
	else
		createBackFacesMaps( backNormalsFB, fromPOV );
	backNormalsFB->UnbindBuffer();

	backgroundGeomFB->BindBuffer();
	createOtherObjectEyeTexture( backgroundGeomFB, colorOrPhotons, fromPOV );
    backgroundGeomFB->UnbindBuffer();
}



/****************************************************
** Draw refractive geometry from a particular POV  **
**    This draws no caustics, and performs no      **
**    computations that could be used for caustics **
****************************************************/
void DrawSimpleRefractedScene( FrameBuffer *currFB, int fromPOV )
{
  glDrawBuffers(1, origDrawBufs);
  glViewport( 0, 0, currFB->GetWidth(), currFB->GetHeight() );

  glMatrixMode( GL_PROJECTION );
  glLoadIdentity();
  gluPerspective( 90, 1, myNear, myFar );
  glMatrixMode( GL_MODELVIEW );

  glClearDepth( 1.0 );
  glClearColor( 1, 1, 1, 0 );
  glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
  glClearColor( 0, 0, 0, 0 );

  glEnable( GL_NORMALIZE );
  glEnable( GL_DEPTH_TEST );

  /* set the eye's lookat */
  glLoadIdentity();
  glMultMatrixd( lookAtMatrix[ fromPOV ] );

  // Draw the environment map
  DrawEnvironmentMap( fromPOV, ENABLE_JUST_COLOR );

  // Draw the refractive object
  {
	EnableTextureUnit( GL_TEXTURE0, GL_TEXTURE_CUBE_MAP, TEXTURE_ENVIRONMENT_MAP );
	EnableTextureUnit( GL_TEXTURE1, GL_TEXTURE_2D, backgroundGeomFB->GetDepthTextureID() );
	EnableTextureUnit( GL_TEXTURE3, GL_TEXTURE_2D,       TEXTURE_BACKFACING_NORMALS );
	EnableTextureUnit( GL_TEXTURE4, GL_TEXTURE_1D,       TEXTURE_PRECOMPUTED_ACOS_FRESNEL );
	EnableTextureUnit( GL_TEXTURE5, GL_TEXTURE_2D,       TEXTURE_BACKFACING_DEPTH );
	EnableTextureUnit( GL_TEXTURE6, GL_TEXTURE_2D,       TEXTURE_IMAGE_OF_BACKGROUND_GEOM );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, tmpMagFilter );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, tmpMinFilter );

	glPushMatrix();
	EnableReflectionRefractionShader( ENABLE_JUST_COLOR );
	SetRefractiveObjectShaderParameters( currFB, fromPOV );

	glEnable(GL_LIGHTING);
	glEnable( GL_CULL_FACE );
	glCullFace( GL_BACK );
	glRotatef( time_angle, 0, 1, 0 );
	MultiplyTrackballMatrix( OBJECT_TRACKBALL );
	glMultMatrixd( initialMatrix );
	glScalef( objScale, objScale, objScale );

	if (drawComplex)
		glCallList( currentObjectList );
	else
		glCallList( sphereList );

	glDisable(GL_LIGHTING);
	glPopMatrix();
	glDisable( GL_CULL_FACE );
    
	DisableReflectionRefractionShader( ENABLE_JUST_COLOR );

	DisableTextureUnit( GL_TEXTURE6, GL_TEXTURE_2D );
	DisableTextureUnit( GL_TEXTURE5, GL_TEXTURE_2D );
	DisableTextureUnit( GL_TEXTURE4, GL_TEXTURE_1D );
	DisableTextureUnit( GL_TEXTURE3, GL_TEXTURE_2D );
	DisableTextureUnit( GL_TEXTURE1, GL_TEXTURE_2D );
	DisableTextureUnit( GL_TEXTURE0, GL_TEXTURE_CUBE_MAP );
  }

  // Draw the non refractive objects in the scene
  DrawOtherSceneObjects( 1, ENABLE_JUST_COLOR );
}


/* draw the environment map in the background from a particular POV */
void DrawEnvironmentMap( int fromPOV, int colorOrPhoton )
{
	glDisable( GL_LIGHTING );
	glDepthMask( GL_FALSE );
	EnableTextureUnit( GL_TEXTURE0, GL_TEXTURE_CUBE_MAP, TEXTURE_ENVIRONMENT_MAP );

	if ( colorOrPhoton == ENABLE_JUST_COLOR )
	  BindCGPrograms( CG_INDEX_CUBE_MAP_V, CG_INDEX_CUBE_MAP_JUSTCOLOR_F );
	else if ( colorOrPhoton == ENABLE_JUST_PHOTONS )
	  BindCGPrograms( CG_INDEX_CUBE_MAP_V, CG_INDEX_CUBE_MAP_JUSTPHOTONS_F );
	else if ( colorOrPhoton == ENABLE_COLOR_AND_NORMALS )
	  BindCGPrograms( CG_INDEX_CUBE_MAP_V, CG_INDEX_CUBE_MAP_COLORANDNORMS_F );

	glMatrixMode( GL_MATRIX0_ARB );
	glPushMatrix();
	glLoadIdentity();
	if (fromPOV == EYE_POV)
		glMultMatrixd( lookAtMatrix[EYE_POV] );
	else if (fromPOV == LIGHT_POV)
		glMultMatrixd( lookAtMatrix[LIGHT_POV] );

	glMatrixMode( GL_MATRIX1_ARB );
	glPushMatrix();
	glLoadIdentity();
	glRotatef( time_angleEnv, 0, 1, 0 );
	MultiplyTrackballMatrix( ENVIRONMENT_TRACKBALL );

	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
	glLoadIdentity();
	glScalef( 15, 15, 15 );
	glutSolidSphere( 1, 20, 20 );

	if ( colorOrPhoton == ENABLE_JUST_COLOR )
	  UnbindCGPrograms( CG_INDEX_CUBE_MAP_V, CG_INDEX_CUBE_MAP_JUSTCOLOR_F );
	else if ( colorOrPhoton == ENABLE_JUST_PHOTONS )
	  UnbindCGPrograms( CG_INDEX_CUBE_MAP_V, CG_INDEX_CUBE_MAP_JUSTPHOTONS_F );
	else if ( colorOrPhoton == ENABLE_COLOR_AND_NORMALS )
	  UnbindCGPrograms( CG_INDEX_CUBE_MAP_V, CG_INDEX_CUBE_MAP_COLORANDNORMS_F );
	DisableTextureUnit( GL_TEXTURE0, GL_TEXTURE_CUBE_MAP );
	glDepthMask( GL_TRUE );

	glMatrixMode( GL_MATRIX0_ARB );
	glPopMatrix();
	glMatrixMode( GL_MATRIX1_ARB );
	glPopMatrix();
	glMatrixMode( GL_MODELVIEW );
	glPopMatrix();
}



/* draw the normals for the back side (i.e. the geometry on */
/*    the refractive object *FURTHEST* from the 'eye'       */
void createBackSideMaps( FrameBuffer *drawFB, int fromPOV )
{
	SetupFrameBufferMatrices( fromPOV );
	glViewport( 0, 0, drawFB->GetWidth(), drawFB->GetHeight() );

	/* setup for the far view */
	glClearDepth( 0 );
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

	/* draw object */
	glEnable( GL_DEPTH_TEST );
	glDepthFunc( GL_GREATER );
	glEnable( GL_CULL_FACE );
	glCullFace( GL_FRONT );
	glPushMatrix();
	glRotatef( time_angle, 0, 1, 0 );
	MultiplyTrackballMatrix( OBJECT_TRACKBALL );
	glMultMatrixd( initialMatrix );
	glScalef( objScale, objScale, objScale );
	BindCGProgram( CG_RENDER_BF_NORMALS_F );
	if (drawComplex)
		glCallList( currentObjectNormalList );
	else
		glCallList( sphereList );
	UnbindCGProgram( CG_RENDER_BF_NORMALS_F );
	glPopMatrix();
	glDepthFunc( GL_LESS );
	glCullFace( GL_BACK );
	glDisable( GL_CULL_FACE );

}

/* draw the normals for the back faces... Using normal depth */
/*    tests, this draws the geometry that is occluded once   */
/*    from the eye's POV (i.e., the geom that's 2nd closest) */
void createBackFacesMaps( FrameBuffer *drawFB, int fromPOV )
{
	SetupFrameBufferMatrices( fromPOV );
	glViewport( 0, 0, drawFB->GetWidth(), drawFB->GetHeight() );

	glClearDepth( 1 );

	/* setup for the far view */
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

	/* draw object */
	glEnable( GL_DEPTH_TEST );
	glDepthFunc( GL_LESS );
	glEnable( GL_CULL_FACE );
	glCullFace( GL_FRONT );
	glPushMatrix();
	glRotatef( time_angle, 0, 1, 0 );
	MultiplyTrackballMatrix( OBJECT_TRACKBALL );
	glMultMatrixd( initialMatrix );
	glScalef( objScale, objScale, objScale );
	BindCGProgram( CG_RENDER_BF_NORMALS_F );
	if (drawComplex)
		glCallList( currentObjectNormalList );
	else
		glCallList( sphereList );
	UnbindCGProgram( CG_RENDER_BF_NORMALS_F );
	glPopMatrix();
	glCullFace( GL_BACK );
	glDisable( GL_CULL_FACE );

}


/*********************************************************
** Draw other objects (besides refractive one) in scene **
*********************************************************/
void DrawOtherSceneObjects( int shaded, int colorOrPhoton )
{
  // For any sort of object...
  glPushMatrix();

  if ( colorOrPhoton == ENABLE_JUST_COLOR )
	  BindCGPrograms( CG_TEXTURE_OTHER_OBJS_V, CG_TEXTURE_OTHER_OBJS_JUSTCOLOR_F );
  else if ( colorOrPhoton == ENABLE_JUST_PHOTONS )
	  BindCGPrograms( CG_TEXTURE_OTHER_OBJS_V, CG_TEXTURE_OTHER_OBJS_JUSTPHOTONS_F );
  else if ( colorOrPhoton == ENABLE_COLOR_AND_NORMALS )
	  BindCGPrograms( CG_TEXTURE_OTHER_OBJS_V, CG_TEXTURE_OTHER_OBJS_COLORANDNORMS_F );

  // For the Macbeth color chart (or some other 2D texture)...
  if (displayMacBeth)
  {
	glPushMatrix();
	glTranslatef( 0, 2, -5 );
	glDisable( GL_LIGHTING );
	EnableTextureUnit( GL_TEXTURE0, GL_TEXTURE_2D, macbethColorTex );
	glBegin( GL_QUADS );
    if (shaded)
	    glColor4f(1,1,1,0);
	else
		glColor4f(0.2,0.2,0.2,0);
	glNormal3f( 0, 0, 1 );
	glTexCoord2f( 1, 0 ); glVertex3f( 4.365, -3, 0 );
	glTexCoord2f( 1, 1 ); glVertex3f( 4.365, 3, 0 );
	glTexCoord2f( 0, 1 ); glVertex3f( -4.365, 3, 0 );
	glTexCoord2f( 0, 0 ); glVertex3f( -4.365, -3, 0 );
	glEnd();
	DisableTextureUnit( GL_TEXTURE0, GL_TEXTURE_2D );
	glPopMatrix();
  }
  if (!displayTable)
  {
	GLfloat lightAmb[4] = {0.0, 0.0, 0.0, 1};
	GLfloat lightDif[4] = {1, 1, 1, 1};
	GLfloat lightSpec[4] = {1, 1, 1, 1};
	GLfloat tmp[4];
	glPushMatrix();
	glPushAttrib( GL_LIGHTING_BIT );
	glEnable( GL_LIGHTING );
	glLightfv( GL_LIGHT0, GL_AMBIENT, lightAmb );
	glLightfv( GL_LIGHT0, GL_DIFFUSE, lightDif );
	glLightfv( GL_LIGHT0, GL_SPECULAR, lightSpec );
	glLightfv( GL_LIGHT0, GL_POSITION, lookFrom[LIGHT_POV] );
	glGetLightfv( GL_LIGHT0, GL_POSITION, tmp );
	glEnable( GL_LIGHT0 );
	if (shaded)
	{
		GLfloat amb[4] = {0,0,0,1};
		SetCurrentMaterialToColor( GL_FRONT_AND_BACK, 0.2, 1.0, 0.0 );
		glMaterialfv( GL_FRONT_AND_BACK, GL_AMBIENT, amb ); 
	}
	else
	{
		GLfloat amb[4] = {0.2,0.2,0.2,1};
		SetCurrentMaterialToColor( GL_FRONT_AND_BACK, 0.0, 0.0, 0.0 );
		glMaterialfv( GL_FRONT_AND_BACK, GL_AMBIENT, amb ); 
	}
	glEnable( GL_CULL_FACE );
	glCullFace( GL_BACK );
	if ( colorOrPhoton == ENABLE_JUST_COLOR )
	  BindCGPrograms( CG_PHONG_SHADE_PLUS_VERTEX, CG_PHONG_SHADE_JUSTCOLOR_F );
	else if (colorOrPhoton == ENABLE_JUST_PHOTONS )
	  BindCGPrograms( CG_PHONG_SHADE_PLUS_VERTEX, CG_PHONG_SHADE_JUSTPHOTONS_F );
	else if ( colorOrPhoton == ENABLE_COLOR_AND_NORMALS )
	  BindCGPrograms( CG_PHONG_SHADE_PLUS_VERTEX, CG_PHONG_SHADE_COLORANDNORMALS_F );

	SetCGParameterv( CG_PHONG_SHADE_PLUS_VERTEX, "lightPos", 4, tmp );

	SetCurrentMaterial( GL_FRONT_AND_BACK, MAT_EMERALD );
	glTranslatef( 0, -5, 0 );
	glScalef( 20, 120, 20 );
	glCallList( groundList );
	glPopAttrib();

	if ( colorOrPhoton == ENABLE_JUST_COLOR )
	  UnbindCGPrograms( CG_PHONG_SHADE_PLUS_VERTEX, CG_PHONG_SHADE_JUSTCOLOR_F );
	else if (colorOrPhoton == ENABLE_JUST_PHOTONS )
	  UnbindCGPrograms( CG_PHONG_SHADE_PLUS_VERTEX, CG_PHONG_SHADE_JUSTPHOTONS_F );
	else if ( colorOrPhoton == ENABLE_COLOR_AND_NORMALS )
	  UnbindCGPrograms( CG_PHONG_SHADE_PLUS_VERTEX, CG_PHONG_SHADE_COLORANDNORMALS_F );

	glPopMatrix();
  }
  else
  {
	glPushMatrix();

	glDisable( GL_LIGHTING );
	if (whichFloorTex==0)
		EnableTextureUnit( GL_TEXTURE0, GL_TEXTURE_2D, woodTex );
	else
		EnableTextureUnit( GL_TEXTURE0, GL_TEXTURE_2D, rockTex );
	glBegin( GL_QUADS );
    if (shaded)
		glColor4f(1,1,1,0);
	else
		glColor4f(0.2,0.2,0.2,0);
	glNormal3f( 0, 1, 0 );
	glTexCoord2f( 1, 1 ); glVertex3f( 20, -3.5, 10 );
	glTexCoord2f( 1, 0 ); glVertex3f( 20, -3.5, -10 );
	glTexCoord2f( 0, 0 ); glVertex3f( -20, -3.5, -10 );
	glTexCoord2f( 0, 1 ); glVertex3f( -20, -3.5, 10 );
	glEnd();
	DisableTextureUnit( GL_TEXTURE0, GL_TEXTURE_2D );
	glPopMatrix();
  }

  if ( colorOrPhoton == ENABLE_JUST_COLOR )
	  UnbindCGPrograms( CG_TEXTURE_OTHER_OBJS_V, CG_TEXTURE_OTHER_OBJS_JUSTCOLOR_F );
  else if ( colorOrPhoton == ENABLE_JUST_PHOTONS )
	  UnbindCGPrograms( CG_TEXTURE_OTHER_OBJS_V, CG_TEXTURE_OTHER_OBJS_JUSTPHOTONS_F );
  else if ( colorOrPhoton == ENABLE_COLOR_AND_NORMALS )
	  UnbindCGPrograms( CG_TEXTURE_OTHER_OBJS_V, CG_TEXTURE_OTHER_OBJS_COLORANDNORMS_F );

  // For an OpenGL illuminated complex model...
  if (displayDragon)
  {
  GLfloat lightAmb[4] = {0.0, 0.0, 0.0, 1};
  GLfloat lightDif[4] = {1, 1, 1, 1};
  GLfloat lightSpec[4] = {1, 1, 1, 1};
  GLfloat tmp[4];
  glPushMatrix();
  glEnable( GL_LIGHTING );
  glLightfv( GL_LIGHT0, GL_AMBIENT, lightAmb );
  glLightfv( GL_LIGHT0, GL_DIFFUSE, lightDif );
  glLightfv( GL_LIGHT0, GL_SPECULAR, lightSpec );
  glLightfv( GL_LIGHT0, GL_POSITION, lookFrom[LIGHT_POV] );
  glGetLightfv( GL_LIGHT0, GL_POSITION, tmp );
  glTranslatef( moveOthersX-14*(moveTmp/360.0), moveOthersY, moveOthersZ );
  MultiplyTrackballMatrix( MISC_TRACKBALL );
  glScalef( 4, 4, 4 );
  glEnable( GL_LIGHT0 );
  if (shaded)
  {
    GLfloat amb[4] = {0,0,0,1};
	SetCurrentMaterialToColor( GL_FRONT_AND_BACK, 0.2, 1.0, 0.0 );
	glMaterialfv( GL_FRONT_AND_BACK, GL_AMBIENT, amb ); 
  }
  else
  {
    GLfloat amb[4] = {0.2,0.2,0.2,1};
	SetCurrentMaterialToColor( GL_FRONT_AND_BACK, 0.0, 0.0, 0.0 );
	glMaterialfv( GL_FRONT_AND_BACK, GL_AMBIENT, amb ); 
  }
  glEnable( GL_CULL_FACE );
  glCullFace( GL_BACK );
  if ( colorOrPhoton == ENABLE_JUST_COLOR )
	  BindCGPrograms( CG_PHONG_SHADE_PLUS_VERTEX, CG_PHONG_SHADE_JUSTCOLOR_F );
  else if (colorOrPhoton == ENABLE_JUST_PHOTONS )
	  BindCGPrograms( CG_PHONG_SHADE_PLUS_VERTEX, CG_PHONG_SHADE_JUSTPHOTONS_F );
  else if ( colorOrPhoton == ENABLE_COLOR_AND_NORMALS )
	  BindCGPrograms( CG_PHONG_SHADE_PLUS_VERTEX, CG_PHONG_SHADE_COLORANDNORMALS_F );

  SetCGParameterv( CG_PHONG_SHADE_PLUS_VERTEX, "lightPos", 4, tmp );
  glCallList( dragonList );
  
  if ( colorOrPhoton == ENABLE_JUST_COLOR )
	  UnbindCGPrograms( CG_PHONG_SHADE_PLUS_VERTEX, CG_PHONG_SHADE_JUSTCOLOR_F );
  else if (colorOrPhoton == ENABLE_JUST_PHOTONS )
	  UnbindCGPrograms( CG_PHONG_SHADE_PLUS_VERTEX, CG_PHONG_SHADE_JUSTPHOTONS_F );
  else if ( colorOrPhoton == ENABLE_COLOR_AND_NORMALS )
	  UnbindCGPrograms( CG_PHONG_SHADE_PLUS_VERTEX, CG_PHONG_SHADE_COLORANDNORMALS_F );
  
  glDisable( GL_CULL_FACE );
  glDisable( GL_LIGHT0 );
  glDisable( GL_LIGHTING );
  glPopMatrix();
  }

  // For any sort of object... 
  glPopMatrix();
}

/***************************************************
** Draw just the background geometry, as seen     **
**      from the current POV ('eye' position)     **
***************************************************/
void createOtherObjectEyeTexture( FrameBuffer *drawFB, int colorOrPhoton, int fromPOV ) 
{	
	if (colorOrPhoton == ENABLE_JUST_COLOR)
		glDrawBuffers( 1, backgroundDrawBufs );
	else if (colorOrPhoton == ENABLE_COLOR_AND_NORMALS)
		glDrawBuffers( 2, backgroundDrawBufs );

    SetupFrameBufferMatrices( fromPOV );
	glViewport( 0, 0, drawFB->GetWidth(), drawFB->GetHeight() );

	/* clear the view */
	glClearColor( 0.0, 0.0, 0.0, 1.0 );
	glClearDepth( 1.0 );
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

	/* set some necessary GL state. */
	glEnable( GL_DEPTH_TEST );
	glDepthFunc( GL_LESS );

	/* draw these nebulous "other" object */
	glEnable( GL_NORMALIZE );
	DrawOtherSceneObjects( 1, colorOrPhoton );
	glDisable( GL_NORMALIZE );

	glDrawBuffer( origDrawBufs[0] );
}

void EnableReflectionRefractionShader( int colorOrPhoton )
{
	if ( colorOrPhoton == ENABLE_JUST_COLOR )
	{
		if (refractBackground)
			BindCGPrograms( CG_REFRACT_OTHER_OBJS_V, CG_REFRACT_OTHER_OBJS_JUSTCOLOR_F );
		else
			BindCGPrograms( CG_REFRACT_VERT, CG_REFRACT_FRAG );
	}
	else
	{
		if (refractBackground)
			BindCGPrograms( CG_REFRACT_OTHER_OBJS_V, CG_REFRACT_OTHER_OBJS_JUSTPHOTON_F );
		else
			BindCGPrograms( CG_REFRACT_VERT, CG_REFRACT_FRAG );
	}
}

void DisableReflectionRefractionShader( int colorOrPhoton )
{
	if ( colorOrPhoton == ENABLE_JUST_COLOR )
	{
		if (refractBackground )
			UnbindCGPrograms( CG_REFRACT_OTHER_OBJS_V, CG_REFRACT_OTHER_OBJS_JUSTCOLOR_F );
		else
			UnbindCGPrograms( CG_REFRACT_VERT, CG_REFRACT_FRAG );
	}
	else
	{
		if (refractBackground)
			UnbindCGPrograms( CG_REFRACT_OTHER_OBJS_V, CG_REFRACT_OTHER_OBJS_JUSTPHOTON_F );
		else
			UnbindCGPrograms( CG_REFRACT_VERT, CG_REFRACT_FRAG );
	}
}


/***************************************************
** Draw refractive geometry from a particular POV **
**    but also store positions of final hitpoints **
**    into a secondary buffer that can be read    **
**    back later to draw caustics.                **
***************************************************/
void DrawSceneCausticsToBuffer( FrameBuffer *currFB, int fromPOV )
{
  glDrawBuffers(2, causticDrawBufs);
  glViewport( 0, 0, currFB->GetWidth(), currFB->GetHeight() );

  glMatrixMode( GL_PROJECTION );
  glLoadIdentity();
  gluPerspective( 90, 1, myNear, myFar );
  glMatrixMode( GL_MODELVIEW );

  glClearDepth( 1.0 );
  glClearColor( 1, 1, 1, 0 );
  glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
  glClearColor( 0, 0, 0, 0 );

  glEnable( GL_NORMALIZE );
  glEnable( GL_DEPTH_TEST );

  // set the eye's lookat 
  glLoadIdentity();
  glMultMatrixd( lookAtMatrix[ fromPOV ] );

  // Draw the environment map
  DrawEnvironmentMap( fromPOV, ENABLE_JUST_PHOTONS );

  // Draw the refractive object
  {
	EnableTextureUnit( GL_TEXTURE0, GL_TEXTURE_CUBE_MAP, TEXTURE_ENVIRONMENT_MAP );
	EnableTextureUnit( GL_TEXTURE3, GL_TEXTURE_2D,       TEXTURE_BACKFACING_NORMALS );
	EnableTextureUnit( GL_TEXTURE4, GL_TEXTURE_1D,       TEXTURE_PRECOMPUTED_ACOS_FRESNEL );
	EnableTextureUnit( GL_TEXTURE5, GL_TEXTURE_2D,       TEXTURE_BACKFACING_DEPTH );
	EnableTextureUnit( GL_TEXTURE6, GL_TEXTURE_2D,       TEXTURE_IMAGE_OF_BACKGROUND_GEOM );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, tmpMagFilter );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, tmpMinFilter );

	glPushMatrix();
	EnableReflectionRefractionShader( ENABLE_JUST_PHOTONS );
	SetRefractiveObjectShaderParameters( currFB, fromPOV );

	glEnable(GL_LIGHTING);
	glEnable( GL_CULL_FACE );
	glCullFace( GL_BACK );
	
	glRotatef( time_angle, 0, 1, 0 );
	MultiplyTrackballMatrix( OBJECT_TRACKBALL );
	glMultMatrixd( initialMatrix );
	glScalef( objScale, objScale, objScale );
	SetCurrentMaterial( GL_FRONT, material );

	if (drawComplex)
		glCallList( currentObjectList );
	else
		glCallList( sphereList );
	glDisable(GL_LIGHTING);
	glPopMatrix();
	glDisable( GL_CULL_FACE );
	
    DisableReflectionRefractionShader( ENABLE_JUST_PHOTONS );
	DisableTextureUnit( GL_TEXTURE6, GL_TEXTURE_2D );
	DisableTextureUnit( GL_TEXTURE5, GL_TEXTURE_2D );
	DisableTextureUnit( GL_TEXTURE4, GL_TEXTURE_1D );
	DisableTextureUnit( GL_TEXTURE3, GL_TEXTURE_2D );
	DisableTextureUnit( GL_TEXTURE0, GL_TEXTURE_CUBE_MAP );
  }

  // Draw the non refractive objects in the scene
  DrawOtherSceneObjects(1, ENABLE_JUST_PHOTONS);

  glDrawBuffers(1, origDrawBufs);
} 

void SetupOpenGLLight( void )
{
	GLfloat lightAmb[4] = {0.0, 0.0, 0.0, 1};
	GLfloat lightDif[4] = {1, 1, 1, 1};
	GLfloat lightSpec[4] = {1, 1, 1, 1};
	glEnable( GL_LIGHTING );
	glLightfv( GL_LIGHT0, GL_AMBIENT, lightAmb );
	glLightfv( GL_LIGHT0, GL_DIFFUSE, lightDif );
	glLightfv( GL_LIGHT0, GL_SPECULAR, lightSpec );
	glLightfv( GL_LIGHT0, GL_POSITION, lookFrom[LIGHT_POV] );
	glEnable( GL_LIGHT0 );
}