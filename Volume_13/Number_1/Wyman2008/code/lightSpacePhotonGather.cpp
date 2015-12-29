/********************************************
** lightSpacePhotonGather.cpp              **
** --------------------------              **
**                                         **
** This file includes the code for         **
**    computing the caustic contribution   **
**    from the scene.  The code is pretty  **
**    well commented, and function and     **
**    variable names are reasonably self-  **
**    descriptive.
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

/***************************************************
** Globals describing buffer sizes...             **
***************************************************/
extern int screenSize, mainbufferSize, causticMapSize;
extern int numberOfPhotons, refractionBufSize, backgroundBufSize;

extern CGData cgData;
extern FrameBuffer *fb, *backgroundGeomFB, *backNormalsFB, *causticFB;
extern FrameBuffer *pseudoPhotonMapFB, *geomDepthFB, *tempFB;
extern GLfloat myNear, myFar;
extern int whichBackSide, drawCausticsWithDepth;
extern GLfloat fisheyeOffset[3];
extern GLuint groundList;

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
extern GLuint sphereList, dragonList, currentObjectList, currentObjectNormalList;
extern GLfloat lookFrom[2][4];
extern GLdouble lookAtMatrix[2][16], inverseLookAtMatrix[2][16];
extern GLenum origDrawBufs[1], causticDrawBufs[2], backgroundDrawBufs[2];
extern GLuint causticLocationBuffer, causticColorBuffer;
extern int drawQuads, needToRerenderCaustics, causticImagesRefreshed;
extern int tmpUIVal, usePseudoPhotonMap;
extern GLfloat offsetIndices[256];
extern GLuint causticPointBuffer;
extern GLfloat shadowMapBias;
extern GLuint photonMap_frame[4];
extern int temporalFiltering, whichFloorTex;
extern int currentTemporalBufferPointer;
extern int displayBig;
extern int drawSideImages;
extern int drawHelpMenu;
extern int shaderLoadError;
extern int makingMovie;
extern float rotate, fps;
extern int screenShot; 
extern int viewpoint;
extern int refractBackground;
extern int useExperimentalApproach, useNearest;
extern float gaussianSigma, rangeSigma, photonPointSize;
extern GLuint tmpFBDepthStencil;
extern int programStartup;
int usingAdaptiveMultresolutionSplats=0;

// some experimental variables to look at a mipmap-style heirarchy for LS caustic maps
extern GLuint causticMapMipmapLevels[16];
extern int numMipmapLevels;
extern FrameBuffer *causticMipmapsFB;


/* This is called from display() to draw the scene into the framebuffer object "fb" */
/*   the fromPOV is either LIGHT_POV or EYE_POV, depending on where the viewer says */
/*   they want to observe from (i.e., by hitting the "w" key)                       */
/*   This is an alternate for DrawRefractedScene() described in the file            */
/*     "basicRefractionFunctions.cpp"                                               */
void DrawCausticsSceneWithLightSpaceGatheredPhotons( int fromPOV )
{
	/* there's no need to repeatedly render photons if the light isn't moving! */
	if (needToRerenderCaustics)
	{
		/* create the buffers needed for refraction from light (for caustics)           */
		/*    (Note this is a function shared among all caustic rendering paths)        */
		createRefractionBuffers( lookAtMatrix[ LIGHT_POV ], ENABLE_COLOR_AND_NORMALS, LIGHT_POV );

		/* we want to draw the caustics into the buffer made for them                   */
		/*    (So it'll be the right size...  might be a different res than the screen) */
		causticFB->BindBuffer();
		glViewport( 0, 0, causticFB->GetWidth(), causticFB->GetHeight() );

		/* draw the caustics positions & directions info to the framebuffer             */
		/*    (Again, this is a shared function in all caustic rendering paths)         */
		DrawSceneCausticsToBuffer( causticFB, LIGHT_POV );

		/* readback the refracted directions of the positions of the photons            */
		glBindBuffer( GL_PIXEL_PACK_BUFFER_EXT, causticColorBuffer );
		glReadBuffer( ATTACH_PHOTON_INCIDENT_DIRECTION );
		glReadPixels( 0, 0, numberOfPhotons, numberOfPhotons, GL_RGBA, GL_FLOAT, BUFFER_OFFSET(0) );

		/* readback the final resting positions of the caustics into "causticLocationBuffer" */
		glBindBuffer( GL_PIXEL_PACK_BUFFER_EXT, causticLocationBuffer );
		glReadBuffer( ATTACH_PHOTON_LOCATION_BUFFER );
		glReadPixels( 0, 0, numberOfPhotons, numberOfPhotons, GL_RGBA, GL_FLOAT, BUFFER_OFFSET(0) );
		glBindBuffer( GL_PIXEL_PACK_BUFFER_EXT, 0 );

		/* ok, done drawing into the offscreen caustic buffer */
		causticFB->UnbindBuffer();

		/* we just drew the caustics... no need to redraw yet! (Unless we're just       */
		/*   starting, when we should render twice to remove framebuffer garbage)       */
		if (!programStartup) 
			needToRerenderCaustics = 0; 
		else programStartup = 0;

		/* set flag to notify that caustic images have been redrawn! */
		causticImagesRefreshed = 1;

		/* since we're doing the gather from the light's POV, we need not do it every   */
		/*     frame, so we can do it inside the if, for a speed bump.                  */
		PerformLightSpaceGather( tempFB, pseudoPhotonMapFB );

		/* Now we've got our caustic map.  However, simply projecting this onto the     */
		/*     scene means any surface will get a caustic, whether shadowed (from the   */
		/*     refractor) or not!  By copying the depth buffer from the background      */
		/*     image we used above, we can sort this out to only cast caustics onto the */
		/*     "frontmost" surface behind the refractor.                                */
		backgroundGeomFB->BindBuffer();
		glViewport(0, 0, backgroundGeomFB->GetWidth(), backgroundGeomFB->GetHeight() );
		glDrawBuffer( GL_COLOR_ATTACHMENT1_EXT );
		glDisable( GL_DEPTH_TEST );
		glDepthMask( GL_FALSE );
		displayLargeTexture( backgroundGeomFB->GetDepthTextureID() );
		glDepthMask( GL_TRUE );
		glEnable( GL_DEPTH_TEST );
		glDrawBuffer( GL_COLOR_ATTACHMENT0_EXT );
		backgroundGeomFB->UnbindBuffer();

	} /* end if (needToRerenderCaustics)... */

	/* create the buffers needed for refraction from the "eye's" point of view  */
	CreateRefractionBuffersForLightSpacePhotonGather( lookAtMatrix[ fromPOV ], fromPOV );

    /* draw the main "buffer," this is a framebuffer object to facilitate easier */
	/*    use of supersampling by a simple change of the size of this buffer,    */
	/*    currently set up as a #define in the glut_template.h file.             */
    fb->BindBuffer();
	glViewport( 0, 0, fb->GetWidth(), fb->GetHeight() );

	/* draw the scene, with caustics on the background geometry, refracted thru the */
	/*    refractor                                                                 */
    FinalDrawWithAlreadyComputedLightSpaceGather( fb, fromPOV );

    /* ok, done drawing to offscreen buffer, start writing to the main framebuffer again */
    fb->UnbindBuffer();

	/* we've refreshed everything here, so unset this flag */
	causticImagesRefreshed = 0;
}


/* create all the necessary buffers (normal, bg geom, depth, etc) from a particular POV      */
void CreateRefractionBuffersForLightSpacePhotonGather( GLdouble lookAtMatrix[16], int fromPOV )
{
	/* Draw out background geometry from the current eye's POV */
	backgroundGeomFB->BindBuffer();
	glViewport( 0, 0, backgroundGeomFB->GetWidth(), backgroundGeomFB->GetHeight() );
	DrawShadowedBackgroundWithLightSpaceCaustics( backgroundGeomFB, fromPOV ); 
    backgroundGeomFB->UnbindBuffer();

	/* we need the backfacing normals & z-buffer for refraction! */
	backNormalsFB->BindBuffer();
	if (!whichBackSide)
		createBackSideMaps( backNormalsFB, fromPOV );
	else
		createBackFacesMaps( backNormalsFB, fromPOV );
	backNormalsFB->UnbindBuffer();
}



/***************************************************
** Draw just the background geometry, as seen     **
**      from the current POV ('eye' position)     **
***************************************************/
void DrawShadowedBackgroundWithLightSpaceCaustics( FrameBuffer *drawFB, int fromPOV ) 
{
	GLfloat shadowMapMatrix[16];
	GLfloat borderColor[4] = {1,1,1,1};
	float channelUse[4] = {1,0,0,0};

	/* we're going to include caustics, which are stored in a buffer containing */
	/*     photons from (potentially) multiple frames.  Select which channels   */
	/*     we'll be using this time                                             */
	if (temporalFiltering)
	{
		channelUse[0] = 1;
		channelUse[1] = (temporalFiltering > 0 ? 1 : 0);
		channelUse[2] = (temporalFiltering > 1 ? 1 : 0);
		channelUse[3] = (temporalFiltering > 2 ? 1 : 0);
	}
  else /* if there's no temporal filtering, just use photons in the red channel */
	{
		currentTemporalBufferPointer = 0;
		channelUse[0] = 1;
		channelUse[1] = channelUse[2] = channelUse[3] = 0;
	}

	/* setup the buffer for our draw */
    SetupFrameBufferMatrices( fromPOV );
	glClearColor( 0.0, 0.0, 0.0, 0.0 );
	glClearDepth( 1.0 );
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

	/* first draw the environment, this can be before the GL state calls,   */
	/*    because it sets its own state information.                        */                     
	DrawEnvironmentMap( fromPOV, ENABLE_JUST_COLOR );

	/* set some necessary GL state. */
	glEnable( GL_DEPTH_TEST );
	glDepthFunc( GL_LESS );
	glEnable( GL_NORMALIZE );

	/* a matrix going from light-space to eye-space */
	glMatrixMode(GL_MATRIX0_ARB);
	glPushMatrix();
	glLoadIdentity();
	glMultMatrixd( lookAtMatrix[fromPOV] );
	glMultMatrixd( inverseLookAtMatrix[LIGHT_POV] );

	/* compute the shadow map matrix.  See any number of good references,   */
	/*     e.g., http://developer.nvidia.com/object/hwshadowmap_paper.html  */
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glTranslatef(0.5, 0.5, 0.5+shadowMapBias );
	glScalef(0.5, 0.5, 0.5);
	gluPerspective(90, (float)causticFB->GetWidth()/(float)causticFB->GetHeight(), myNear, myFar);
	glMultMatrixd( lookAtMatrix[LIGHT_POV] );
	glGetFloatv(GL_TRANSPOSE_MODELVIEW_MATRIX, shadowMapMatrix);
	glPopMatrix();

	EnableTextureUnit( GL_TEXTURE3, GL_TEXTURE_2D, backgroundGeomFB->GetColorTextureID(1) ); 

	/* we've got this light-space caustic map.  We'll need to index into it */
	EnableTextureUnit( GL_TEXTURE2, GL_TEXTURE_2D, TEXTURE_LIGHTSPACE_CAUSTIC_MAP ); 
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

	/* since the colors of the background (particularly phong geometry) is based */
	/*    off incident light direction, let the shader know we'll want to take   */
	/*    the photon's directions into account.                                  */
	SetCGParameter1f( CG_LIGHTGATHER_SHADMAP_PHONG_F, "usePhotonDirection", 1 ); 

	/* setup the texture generation for shadow mapping */
	EnableTextureUnit( GL_TEXTURE0, GL_TEXTURE_2D, TEXTURE_COMPLETE_DEPTH_FROM_LIGHT );
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
	glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR);
	glTexGenfv(GL_S, GL_EYE_PLANE, &shadowMapMatrix[0]);
	glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR);
	glTexGenfv(GL_T, GL_EYE_PLANE, &shadowMapMatrix[4]);
	glTexGeni(GL_R, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR);
	glTexGenfv(GL_R, GL_EYE_PLANE, &shadowMapMatrix[8]);
	glTexGeni(GL_Q, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR);
	glTexGenfv(GL_Q, GL_EYE_PLANE, &shadowMapMatrix[12]);
	glEnable(GL_TEXTURE_GEN_S);
	glEnable(GL_TEXTURE_GEN_T);
	glEnable(GL_TEXTURE_GEN_R);
	glEnable(GL_TEXTURE_GEN_Q);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
	glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_ALPHA );
	glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

	/* we need a matrix from eye-space to light-space for shadow mapping */
	glMatrixMode( GL_TEXTURE );
	glPushMatrix();
	glLoadMatrixd( lookAtMatrix[LIGHT_POV] );
	glMultMatrixd( inverseLookAtMatrix[fromPOV] );  // == eye-space
	glMatrixMode( GL_MODELVIEW );

	/* bind the shader for the diffuse objects, setup parameters */
	BindCGPrograms( CG_LIGHTGATHER_SHADMAP_TEXTURES_V, CG_LIGHTGATHER_SHADMAP_TEXTURES_F );
	SetCGParameter1f( CG_LIGHTGATHER_SHADMAP_TEXTURES_F, "temporalFilterFactor", 1.0/(temporalFiltering+1) );
	SetCGParameter4f( CG_LIGHTGATHER_SHADMAP_TEXTURES_F, "temporalChannelUse", 
			channelUse[0], channelUse[1], channelUse[2], channelUse[3] );

	/* draw the MacBeth color chart, if desired */
	if (displayMacBeth)
	{
		glPushMatrix();
		glDisable( GL_LIGHTING );         // it's just a texture!
		EnableTextureUnit( GL_TEXTURE1, GL_TEXTURE_2D, macbethColorTex );
		glActiveTexture( GL_TEXTURE0 );   // I've found this necessary sometimes...
		glBegin( GL_QUADS );
		glColor4f(1,1,1,1);
		glNormal3f(0,0,1);
		glMultiTexCoord2f( GL_TEXTURE1, 1, 0 ); glVertex3f( 4.365, -1, -5 );
		glMultiTexCoord2f( GL_TEXTURE1, 1, 1 ); glVertex3f( 4.365, 5, -5 );
		glMultiTexCoord2f( GL_TEXTURE1, 0, 1 ); glVertex3f( -4.365, 5, -5 );
		glMultiTexCoord2f( GL_TEXTURE1, 0, 0 ); glVertex3f( -4.365, -1, -5 );
		glEnd();
		DisableTextureUnit( GL_TEXTURE1, GL_TEXTURE_2D );
		glActiveTexture( GL_TEXTURE0 );
		glPopMatrix();
	}

	/* display the ground plane (a.k.a. "table"), if desired */
	if (!displayTable)
	{
		UnbindCGPrograms( CG_LIGHTGATHER_SHADMAP_TEXTURES_V, CG_LIGHTGATHER_SHADMAP_TEXTURES_F );
		GLfloat amb[4] = {0,0,0,1};
		GLfloat tmp[4];
		SetupOpenGLLight();                            // Setup & enable lighting w/one light
		glGetLightfv( GL_LIGHT0, GL_POSITION, tmp );   // We need it's position.
		glPushMatrix();
		SetCurrentMaterialToColor( GL_FRONT_AND_BACK, 0.2, 1.0, 0.0 );
		glMaterialfv( GL_FRONT_AND_BACK, GL_AMBIENT, amb );  // Previous function sets ambient "wrong"
		glEnable( GL_CULL_FACE );                      // Helps with speed...  This model is big!
		glCullFace( GL_BACK );
		BindCGPrograms( CG_LIGHTGATHER_SHADMAP_PHONG_V, CG_LIGHTGATHER_SHADMAP_PHONG_F );
		SetCGParameterv( CG_LIGHTGATHER_SHADMAP_PHONG_V, "lightPos", 4, tmp );
		SetCGParameter1f( CG_LIGHTGATHER_SHADMAP_PHONG_F, "temporalFilterFactor", 1.0/(temporalFiltering+1) );
		SetCGParameter4f( CG_LIGHTGATHER_SHADMAP_PHONG_F, "temporalChannelUse", 
						channelUse[0], channelUse[1], channelUse[2], channelUse[3] );
		glTranslatef( 0, -5, 0 );
		glScalef( 20, 120, 20 );
		glCallList( groundList );                       
		UnbindCGPrograms( CG_LIGHTGATHER_SHADMAP_PHONG_V, CG_LIGHTGATHER_SHADMAP_PHONG_F );
		glDisable( GL_CULL_FACE );
		glDisable( GL_LIGHT0 );
		glDisable( GL_LIGHTING );
		glPopMatrix();
	}
	else
	{
		glPushMatrix();
		glDisable( GL_LIGHTING );
		EnableTextureUnit( GL_TEXTURE1, GL_TEXTURE_2D, whichFloorTex==0 ? woodTex : rockTex );
		glActiveTexture( GL_TEXTURE0 );
		glBegin( GL_QUADS );
		glColor4f(1,1,1,1);
		glNormal3f(0,1,0);
		glMultiTexCoord2f( GL_TEXTURE1, 1, 1 ); glVertex3f( 20, -3.5, 10 );
		glMultiTexCoord2f( GL_TEXTURE1, 1, 0 ); glVertex3f( 20, -3.5, -10 );
		glMultiTexCoord2f( GL_TEXTURE1, 0, 0 ); glVertex3f( -20, -3.5, -10 );
		glMultiTexCoord2f( GL_TEXTURE1, 0, 1 ); glVertex3f( -20, -3.5, 10 );
		glEnd();
		DisableTextureUnit( GL_TEXTURE1, GL_TEXTURE_2D );
		glActiveTexture( GL_TEXTURE0 );
		glPopMatrix();
		UnbindCGPrograms( CG_LIGHTGATHER_SHADMAP_TEXTURES_V, CG_LIGHTGATHER_SHADMAP_TEXTURES_F );
	}
	
	/* draw the Stanford dragon.  This has specular lighting */
	if (displayDragon)
	{
		GLfloat amb[4] = {0,0,0,1};
		GLfloat tmp[4];
		SetupOpenGLLight();                            // Setup & enable lighting w/one light
		glGetLightfv( GL_LIGHT0, GL_POSITION, tmp );   // We need it's position.
		glPushMatrix();
		glTranslatef( moveOthersX-14*(moveTmp/360.0), moveOthersY, moveOthersZ );
		MultiplyTrackballMatrix( MISC_TRACKBALL );     // Rotate object, if necessary
		glScalef( 4, 4, 4 );                           // Give it a decent scale
		SetCurrentMaterialToColor( GL_FRONT_AND_BACK, 0.2, 1.0, 0.0 );
		glMaterialfv( GL_FRONT_AND_BACK, GL_AMBIENT, amb );  // Previous function sets ambient "wrong"
		glEnable( GL_CULL_FACE );                      // Helps with speed...  This model is big!
		glCullFace( GL_BACK );
		BindCGPrograms( CG_LIGHTGATHER_SHADMAP_PHONG_V, CG_LIGHTGATHER_SHADMAP_PHONG_F );
		SetCGParameterv( CG_LIGHTGATHER_SHADMAP_PHONG_V, "lightPos", 4, tmp );
		SetCGParameter1f( CG_LIGHTGATHER_SHADMAP_PHONG_F, "temporalFilterFactor", 1.0/(temporalFiltering+1) );
		SetCGParameter4f( CG_LIGHTGATHER_SHADMAP_PHONG_F, "temporalChannelUse", 
						channelUse[0], channelUse[1], channelUse[2], channelUse[3] );
		glCallList( dragonList );                       // the dragon!
		UnbindCGPrograms( CG_LIGHTGATHER_SHADMAP_PHONG_V, CG_LIGHTGATHER_SHADMAP_PHONG_F );
		glDisable( GL_CULL_FACE );
		glDisable( GL_LIGHT0 );
		glDisable( GL_LIGHTING );
		glPopMatrix();
	}

	/* reset OpenGL state to what we expect */
	glDisable( GL_NORMALIZE );

	/* undo all the nasty matrix manipulations */
	glActiveTexture( GL_TEXTURE0 );
	glMatrixMode( GL_TEXTURE );
	glPopMatrix();
	glMatrixMode( GL_MATRIX0_ARB );
	glPopMatrix();
	glMatrixMode( GL_MODELVIEW );

	/* disable textures & texture generation for shadow mapping */
    glDisable(GL_TEXTURE_GEN_S);
	glDisable(GL_TEXTURE_GEN_T);
	glDisable(GL_TEXTURE_GEN_R);
	glDisable(GL_TEXTURE_GEN_Q);
	glDisable(GL_TEXTURE_2D);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
	glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_LUMINANCE );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    DisableTextureUnit( GL_TEXTURE0, GL_TEXTURE_2D );
	DisableTextureUnit( GL_TEXTURE2, GL_TEXTURE_2D );
	DisableTextureUnit( GL_TEXTURE3, GL_TEXTURE_2D );
	glActiveTexture( GL_TEXTURE0 );
	SetCGParameter1f( CG_LIGHTGATHER_SHADMAP_PHONG_F, "usePhotonDirection", 0 );
}



/* takes the background geometry textures already computed (which includes one with */
/*     accumulated photon contributions) and the buffers necessary for refraction   */
/*     and render the final scene!                                                  */
void FinalDrawWithAlreadyComputedLightSpaceGather( FrameBuffer *currFB, int fromPOV )
{
  /* setup viewing parameters */
  glViewport( 0, 0, currFB->GetWidth(), currFB->GetHeight() );
  glMatrixMode( GL_PROJECTION );
  glLoadIdentity();
  gluPerspective( 90, 1, myNear, myFar );
  glMatrixMode( GL_MODELVIEW );

  /* clear the buffer */
  glClearDepth( 1.0 );
  glClearColor( 1, 1, 1, 0 );
  glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
  glClearColor( 0, 0, 0, 0 );

  /* enable useful GL state */
  glEnable( GL_NORMALIZE );
  glEnable( GL_DEPTH_TEST );
  glEnable( GL_CULL_FACE );
  glCullFace( GL_BACK );

  /* set the eye's lookat */
  glLoadIdentity();
  glMultMatrixd( lookAtMatrix[ fromPOV ] );
  glPushMatrix();

  /* Draw the (background) non refractive objects in the scene */
  displayLargeTextureWithDepth( TEXTURE_IMAGE_OF_BACKGROUND_GEOM, backgroundGeomFB->GetDepthTextureID() );
  
  /* Go ahead and actually draw the refractive object */
  EnableTextureUnit( GL_TEXTURE0, GL_TEXTURE_CUBE_MAP, TEXTURE_ENVIRONMENT_MAP );
  EnableTextureUnit( GL_TEXTURE1, GL_TEXTURE_2D,       backgroundGeomFB->GetDepthTextureID() );
  EnableTextureUnit( GL_TEXTURE3, GL_TEXTURE_2D,       TEXTURE_BACKFACING_NORMALS );
  EnableTextureUnit( GL_TEXTURE4, GL_TEXTURE_1D,       TEXTURE_PRECOMPUTED_ACOS_FRESNEL );
  EnableTextureUnit( GL_TEXTURE5, GL_TEXTURE_2D,       TEXTURE_BACKFACING_DEPTH );
  EnableTextureUnit( GL_TEXTURE6, GL_TEXTURE_2D,       TEXTURE_IMAGE_OF_BACKGROUND_GEOM );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, tmpMagFilter );  /* in case we need to filter background */
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, tmpMinFilter );

  EnableReflectionRefractionShader( ENABLE_JUST_COLOR );
  SetRefractiveObjectShaderParameters( currFB, fromPOV );
	
  glRotatef( time_angle, 0, 1, 0 );
  MultiplyTrackballMatrix( OBJECT_TRACKBALL );
  glMultMatrixd( initialMatrix );
  glScalef( objScale, objScale, objScale );

  glEnable( GL_DEPTH_TEST );

  /* draw the current object (or a simple sphere) */
  if (drawComplex)
  	glCallList( currentObjectList );
  else
	glCallList( sphereList );

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



/* find caustic contribution using an image-space gather using a image from the light's POV */
void PerformLightSpaceGather( FrameBuffer *tmpFB, FrameBuffer *drawFB )
{
	float channelUse[4] = {1,0,0,0};
	GLenum drawBufs[2] = { GL_COLOR_ATTACHMENT1_EXT, GL_COLOR_ATTACHMENT0_EXT };

	/* If we're using multiple frames, set up the array we'll pass to the gather */
	/*     shader that selects which channels (of the photon texture) to use.    */
	if (temporalFiltering)
	{
		currentTemporalBufferPointer = (currentTemporalBufferPointer+1) % (temporalFiltering+1);
		channelUse[0] = 1;
		channelUse[1] = (temporalFiltering > 0 ? 1 : 0);
		channelUse[2] = (temporalFiltering > 1 ? 1 : 0);
		channelUse[3] = (temporalFiltering > 2 ? 1 : 0);
	}
	else /* If we're not filtering from multiple frames, photons are in the red channel */
	{
		currentTemporalBufferPointer = 0;
		channelUse[0] = 1;
		channelUse[1] = channelUse[2] = channelUse[3] = 0;
	}

	/* setup the texture we'll draw to */
    tmpFB->BindBuffer();
	glViewport( 0, 0, tmpFB->GetWidth(), tmpFB->GetHeight() );
	
	/* First, we're going to render all the photons into an light-space image to       */
	/*     use for the gather.  Basically a per-pixel count of photons.                */

	/* set the buffer to render to */
	glDrawBuffers( 2, drawBufs );
	SetupFrameBufferMatrices( LIGHT_POV );

	/* clear the view */
	glClearColor( 0.0, 0.0, 0.0, 0.0 );
	glClear( GL_COLOR_BUFFER_BIT );

	/* set some necessary GL state. */
	glDisable( GL_CULL_FACE );
	glDisable( GL_DEPTH_TEST );
	glDepthMask( GL_FALSE );

	/* setup the correct textures */ 
	/*   We need the surface normals of the background to compute the photon contributions */
    EnableTextureUnit( GL_TEXTURE0, GL_TEXTURE_2D, TEXTURE_BACKGROUND_SURFACE_NORMS ); 

	/* we're going to draw the photons into a buffer from the light's POV             */
	/*     the photon buffer vertex array already contains light-space coordinates!   */
    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();

	/* setup vertex arrays to draw photons */
	glBindBuffer( GL_ARRAY_BUFFER, causticLocationBuffer );  // store hit points of photon
	glVertexPointer( 4, GL_FLOAT, 0, BUFFER_OFFSET( 0 ) );
	glEnableClientState( GL_VERTEX_ARRAY );

	glBindBuffer( GL_ARRAY_BUFFER, causticColorBuffer );     // actually stores the incident direction!
	glTexCoordPointer( 4, GL_FLOAT, 0, BUFFER_OFFSET( 0 ) );
	glEnableClientState( GL_TEXTURE_COORD_ARRAY );

	/* bind and setup the relevant vertex and fragment shaders */
	if (!usingAdaptiveMultresolutionSplats)
	{
		BindCGPrograms( CG_LIGHT_GATHER_W_GAUSSIAN_SPLATS_V, CG_LIGHT_GATHER_W_GAUSSIAN_SPLATS_F );
		SetCGParameter1f( CG_LIGHT_GATHER_W_GAUSSIAN_SPLATS_F, "causticMapRes", drawFB->GetWidth() ); 
		SetCGParameter1f( CG_LIGHT_GATHER_W_GAUSSIAN_SPLATS_F, "photonResolutionMultiplier", 
		   (drawFB->GetWidth()/(float)numberOfPhotons)*(drawFB->GetWidth()/(float)numberOfPhotons) );
		SetCGParameter1f( CG_LIGHT_GATHER_W_GAUSSIAN_SPLATS_V, "constSplatSize", 
			useExperimentalApproach ? -1 : photonPointSize );
	}
	else
	{
		BindCGPrograms( CG_LIGHT_GATHER_W_GAUSSIAN_SPLATS_V, CG_LIGHT_GATHER_W_MULTIRES_SPLATS_F );
		SetCGParameter1f( CG_LIGHT_GATHER_W_MULTIRES_SPLATS_F, "causticMapRes", drawFB->GetWidth() ); 
		SetCGParameter1f( CG_LIGHT_GATHER_W_MULTIRES_SPLATS_F, "photonResolutionMultiplier", 
		   (drawFB->GetWidth()/(float)numberOfPhotons)*(drawFB->GetWidth()/(float)numberOfPhotons) );
		SetCGParameter1f( CG_LIGHT_GATHER_W_GAUSSIAN_SPLATS_V, "constSplatSize", 15 );
	}

	/* draw photons splats using point sprites */
	glEnable(GL_BLEND);
	glBlendFuncSeparate(GL_ONE, GL_ONE, GL_ONE, GL_ONE);  
	glBlendEquation(GL_FUNC_ADD);    
	glPointSize( photonPointSize );
	glEnable( GL_VERTEX_PROGRAM_POINT_SIZE_ARB );
    glDrawArrays( GL_POINTS, 0, numberOfPhotons*numberOfPhotons ); 
	glDisable( GL_VERTEX_PROGRAM_POINT_SIZE_ARB );

	glPointSize( 1.0 );
    glDisable(GL_BLEND);

	/* return state to more usual values, disable textures */
	if (!usingAdaptiveMultresolutionSplats)
		UnbindCGPrograms( CG_LIGHT_GATHER_W_GAUSSIAN_SPLATS_V, CG_LIGHT_GATHER_W_GAUSSIAN_SPLATS_F );
	else
		UnbindCGPrograms( CG_LIGHT_GATHER_W_GAUSSIAN_SPLATS_V, CG_LIGHT_GATHER_W_MULTIRES_SPLATS_F );
    glDisableClientState( GL_VERTEX_ARRAY );
	glDisableClientState( GL_TEXTURE_COORD_ARRAY );
    glBindBuffer( GL_ARRAY_BUFFER, 0 );
    DisableTextureUnit( GL_TEXTURE0, GL_TEXTURE_2D );

	glDrawBuffer( origDrawBufs[0] );
	tmpFB->UnbindBuffer();

	// Now copy results into texture containing (upto) 4 frames of data.  In the case of the
	//    multires splats, there's an interpolation pass here (i.e., more than a copy)
	drawFB->BindBuffer();

	glViewport( 0, 0, drawFB->GetWidth(), drawFB->GetHeight() );
	glDrawBuffer( ATTACH_LIGHTSPACE_CAUSTIC_MAP );

	// Only gonna draw the results into one channel, so we can blur it later over multiple frames
	glColorMask( 
		(currentTemporalBufferPointer==0? GL_TRUE : GL_FALSE),
		(currentTemporalBufferPointer==1? GL_TRUE : GL_FALSE),
		(currentTemporalBufferPointer==2? GL_TRUE : GL_FALSE),
		(currentTemporalBufferPointer==3? GL_TRUE : GL_FALSE)
	);

	// If we're using the adaptive splat sized based on 4 predetermined radii, we need another pass.
	//   Note, w/o this an additional pass isn't needed, but a copy currently occurs...  That slows 
	//   the other results!
	if (usingAdaptiveMultresolutionSplats)
	{
		
		// Do an image-space selection of which gaussian blur to use...  no fancy matrices needed 
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluOrtho2D( 0, 1, 0, 1 );
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		glClear( GL_COLOR_BUFFER_BIT );
		
		// setup the shader and textures, as necessry 
		BindCGPrograms( CG_COMBINE_MULTILEVEL_SPLATS_V, CG_COMBINE_MULTILEVEL_SPLATS_F );
		EnableTextureUnit( GL_TEXTURE0, GL_TEXTURE_2D,   tempFB->GetColorTextureID(0) ); 
			glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, useNearest ? GL_NEAREST : GL_LINEAR ); 
			glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, useNearest ? GL_NEAREST : GL_LINEAR );
		EnableTextureUnit( GL_TEXTURE1, GL_TEXTURE_2D,   tempFB->GetColorTextureID(1) );
			glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, useNearest ? GL_NEAREST : GL_LINEAR ); 
			glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, useNearest ? GL_NEAREST : GL_LINEAR ); 

		// draw the image-space quad 
		glColor3f(1,1,1);
		glBegin(GL_QUADS);
		glTexCoord2f( 0, 0 ); glVertex2f( 0.0, 0.0 );
		glTexCoord2f( 1, 0 ); glVertex2f( 1, 0.0 );
		glTexCoord2f( 1, 1 ); glVertex2f( 1, 1 );
		glTexCoord2f( 0, 1 ); glVertex2f( 0.0, 1 );
		glEnd();

		// unbind shaders, reset state, all that fun stuff 
		DisableTextureUnit( GL_TEXTURE1, GL_TEXTURE_2D );
		DisableTextureUnit( GL_TEXTURE0, GL_TEXTURE_2D );
		UnbindCGPrograms( CG_COMBINE_MULTILEVEL_SPLATS_V, CG_COMBINE_MULTILEVEL_SPLATS_F );

	}
	else
	{
		displayLargeTexture( tempFB->GetColorTextureID(1), GL_NEAREST, GL_NEAREST, 1.0, GL_REPLACE );
	}

	glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE );
	glDrawBuffer( origDrawBufs[0] );
	drawFB->UnbindBuffer();
}