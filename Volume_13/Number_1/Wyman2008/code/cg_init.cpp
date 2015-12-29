/************************************
** cg_init.cpp                     **
** ----------                      **
**                                 **
** Initializes and sets up various **
**   CG shader stuff for OpenGL... ** 
**                                 **
** Chris Wyman (1/6/2003)          **
************************************/

#include "glut_template.h"
#include <CG/cg.h>
#include <CG/cgGL.h>

/* initialize the program structures */
shader cgProgram[NUM_CG_PROGRAMS] = {
	/*  0 */ { "shaders/simpleRefraction/refractionVert.cg",						GL_VERTEX_PROGRAM_ARB,   0, 0, NULL },
	/*  1 */ { "shaders/simpleRefraction/refractionFrag.cg",						GL_FRAGMENT_PROGRAM_ARB, 0, 0, NULL },

	/*  2 */ { "shaders/refractionWBackground/refractionOtherObjsVert.cg",		GL_VERTEX_PROGRAM_ARB,   0, 0, NULL },

	/*  3 */ { "shaders/phongShadingPlusVert.cg",						GL_VERTEX_PROGRAM_ARB, 0, 0, NULL },

	/*  4 */ { "shaders/renderTexWithDepth.Fragment.cg",					GL_FRAGMENT_PROGRAM_ARB, 0, 0, NULL },

	/*  5 */ { "shaders/storePhotonsInLightBufferVert.cg",				GL_VERTEX_PROGRAM_ARB, 0, 0, NULL },
	/*  6 */ { "shaders/storePhotonsInLightBufferFrag.cg",				GL_FRAGMENT_PROGRAM_ARB, 0, 0, NULL },

	/*  7 */ { "shaders/pseudoPhotonMapGatherLightPOVVert.cg",			GL_VERTEX_PROGRAM_ARB, 0, 0, NULL },
	/*  8 */ { "shaders/pseudoPhotonMapGatherLightPOVFrag.cg",			GL_FRAGMENT_PROGRAM_ARB, 0, 0, NULL },

	/*  9 */ { "shaders/refractionWBackground/refractionOtherObjsFrag.justColor.cg",			GL_FRAGMENT_PROGRAM_ARB, 0, 0, NULL },
	/* 10 */ { "shaders/refractionWBackground/refractionOtherObjsFrag.justPhotonData.cg",	GL_FRAGMENT_PROGRAM_ARB, 0, 0, NULL },

    /* 11 */ { "shaders/phongShadingPlusFrag.justColor.cg",				GL_FRAGMENT_PROGRAM_ARB, 0, 0, NULL },
	/* 12 */ { "shaders/phongShadingPlusFrag.justPhotons.cg",			GL_FRAGMENT_PROGRAM_ARB, 0, 0, NULL },
	/* 13 */ { "shaders/phongShadingPlusFrag.colorAndNormals.cg",		GL_FRAGMENT_PROGRAM_ARB, 0, 0, NULL },

	/* 14 */ { "shaders/drawShadowMappedSurfacesWithTexture.lightGather.Vert.cg",	GL_VERTEX_PROGRAM_ARB, 0, 0, NULL },
	/* 15 */ { "shaders/drawShadowMappedSurfacesWithTexture.lightGather.Frag.cg",	GL_FRAGMENT_PROGRAM_ARB, 0, 0, NULL },
	/* 16 */ { "shaders/drawShadowMappedSurfacesWithPhong.lightGather.Vert.cg",		GL_VERTEX_PROGRAM_ARB, 0, 0, NULL },
	/* 17 */ { "shaders/drawShadowMappedSurfacesWithPhong.lightGather.Frag.cg",		GL_FRAGMENT_PROGRAM_ARB, 0, 0, NULL },

	/* 18 */ { "shaders/experimentalCausticMapConstruction.Vert.cg",		GL_VERTEX_PROGRAM_ARB, 0, 0, NULL },
	/* 19 */ { "shaders/experimentalCausticMapConstruction.Frag.cg",		GL_FRAGMENT_PROGRAM_ARB, 0, 0, NULL },

	/* 20 */ { "shaders/combineMipmapLevels.Vert.cg",					GL_VERTEX_PROGRAM_ARB, 0, 0, NULL },
	/* 21 */ { "shaders/combineMipmapLevels.Frag.cg",					GL_FRAGMENT_PROGRAM_ARB, 0, 0, NULL },

	/* 22 */ { "shaders/lightGatherWGaussianSplats.Vert.cg",				GL_VERTEX_PROGRAM_ARB, 0, 0, NULL },
	/* 23 */ { "shaders/lightGatherWGaussianSplats.Frag.cg",				GL_FRAGMENT_PROGRAM_ARB, 0, 0, NULL },

	/* 24 */ { "shaders/combineMultiLevelGaussianSplats.Vert.cg",		GL_VERTEX_PROGRAM_ARB, 0, 0, NULL },
	/* 25 */ { "shaders/combineMultiLevelGaussianSplats.Frag.cg",		GL_FRAGMENT_PROGRAM_ARB, 0, 0, NULL },

	/* 26 */ { "shaders/renderBFNormalsToImage.Frag.cg",					GL_FRAGMENT_PROGRAM_ARB, 0, 0, NULL },

	/* 27 */ { "shaders/indexCubeMap.Vert.cg",							GL_VERTEX_PROGRAM_ARB, 0, 0, NULL },
	/* 28 */ { "shaders/indexOtherObjectTex.Vert.cg",					GL_VERTEX_PROGRAM_ARB, 0, 0, NULL },
	/* 29 */ { "shaders/indexOtherObjTex.justPhotons.Frag.cg",			GL_FRAGMENT_PROGRAM_ARB, 0, 0, NULL },
	/* 30 */ { "shaders/indexOtherObjTex.justColor.Frag.cg",				GL_FRAGMENT_PROGRAM_ARB, 0, 0, NULL },
	/* 31 */ { "shaders/indexOtherObjTex.colorAndNormals.Frag.cg",		GL_FRAGMENT_PROGRAM_ARB, 0, 0, NULL },
	/* 32 */ { "shaders/indexCubeMap.justPhotons.Frag.cg",               GL_FRAGMENT_PROGRAM_ARB, 0, 0, NULL },
	/* 33 */ { "shaders/indexCubeMap.justColor.Frag.cg",                 GL_FRAGMENT_PROGRAM_ARB, 0, 0, NULL },
	/* 34 */ { "shaders/indexCubeMap.colorAndNormals.Frag.cg",           GL_FRAGMENT_PROGRAM_ARB, 0, 0, NULL },

	/* 35 */ { "shaders/copyTextureWithDepth.Frag.cg",                   GL_FRAGMENT_PROGRAM_ARB, 0, 0, NULL },

	/* 36 */ { "shaders/lightGatherWMultiresSplats.Frag.cg",             GL_FRAGMENT_PROGRAM_ARB, 0, 0, NULL },
};

typedef struct 
{
	CGcontext shaderContext;
	CGprofile vertProfile, fragProfile;
	CGprogram programs[NUM_CG_PROGRAMS];
	CGprofile programType[NUM_CG_PROGRAMS];
	char *programName[NUM_CG_PROGRAMS];
} CGData;

CGData cgData;
int boundVertex=-1;
int boundFragment=-1;

void EnableCGTexture( int progNum, GLuint texID, char *paramName )
{
	CGparameter paramPtr = cgGetNamedParameter( cgData.programs[progNum], paramName );
	cgGLSetTextureParameter( paramPtr, texID );
	cgGLEnableTextureParameter( paramPtr );
}

void DisableCGTexture( int progNum, char *paramName )
{
	CGparameter paramPtr = cgGetNamedParameter( cgData.programs[progNum], paramName );
	cgGLDisableTextureParameter( paramPtr );
}

int SetCGParameterv( int progNum, char *paramName, int arraySize, float *array )
{
	CGparameter paramPtr = cgGetNamedParameter( cgData.programs[progNum], paramName );
	if (paramPtr==0) return 0;
	switch( arraySize )
	{
	case 1:
		cgGLSetParameter1fv( paramPtr, array );
		break;
	case 2:
		cgGLSetParameter2fv( paramPtr, array );
		break;
	case 3:
		cgGLSetParameter3fv( paramPtr, array );
		break;
	case 4:
		cgGLSetParameter4fv( paramPtr, array );
		break;
	default:
		Error( "SetCGParameterv should have an arraySize in the range [1-4]!");
		return 0;
	}
	return 1;
}

int SetCGParameter4f( int progNum, char *paramName, float x, float y, float z, float w )
{
	float tmp[4] = {x, y, z, w};	
	return SetCGParameterv( progNum, paramName, 4, tmp );
}

int SetCGParameter3f( int progNum, char *paramName, float x, float y, float z )
{
	float tmp[3] = {x, y, z};	
	return SetCGParameterv( progNum, paramName, 3, tmp );
}

int SetCGParameter2f( int progNum, char *paramName, float x, float y )
{
	float tmp[2] = {x, y};	
	return SetCGParameterv( progNum, paramName, 2, tmp );
}

int SetCGParameter1f( int progNum, char *paramName, float x )
{
	float tmp[1] = {x};	
	return SetCGParameterv( progNum, paramName, 1, tmp );
}

/* call before using a program to set the GL state */
void BindCGProgram( int progNum )
{
	cgGLEnableProfile( cgData.programType[progNum] );
	cgGLBindProgram( cgData.programs[progNum] );
	if ( cgProgram[progNum].type == GL_VERTEX_PROGRAM_ARB )
		boundVertex = progNum;
	else
		boundFragment = progNum;
}

/* call before using a program to set the GL state */
void BindCGPrograms( int vertProgNum, int fragProgNum )
{
	BindCGProgram( vertProgNum );
	BindCGProgram( fragProgNum );
	if ( cgProgram[vertProgNum].type == GL_VERTEX_PROGRAM_ARB )
		boundVertex = vertProgNum;
	else
		boundFragment = vertProgNum;
	if ( cgProgram[fragProgNum].type == GL_VERTEX_PROGRAM_ARB )
		boundVertex = fragProgNum;
	else
		boundFragment = fragProgNum;
}

/* call after finishing with a program to unset GL state */
void UnbindCGProgram( int progNum )
{
	cgGLDisableProfile( cgData.programType[progNum] );
}

/* call before using a program to set the GL state */
void UnbindCGPrograms( int vertProgNum, int fragProgNum )
{
	UnbindCGProgram( vertProgNum );
	UnbindCGProgram( fragProgNum );
}

int CurrentCGFragmentShader( void )
{
	return boundFragment;
}

int CurrentCGVertexShader( void )
{
	return boundVertex;
}

/* double-check required extensions, load shaders, and double check that     */
/* they're valid and we have sufficient resources                            */
int ReloadCGShaders( void )
{
	int i;
	int readError = 0;

	for (i=0;i<NUM_CG_PROGRAMS;i++)
	{
		cgDestroyProgram( cgData.programs[i] );
		cgData.programs[i] = cgCreateProgramFromFile( cgData.shaderContext, CG_SOURCE, cgProgram[i].filename,
													  cgData.programType[i], "main", 0 );
		if (!cgData.programs[i])
		{
			CGerror  err = cgGetError();
			const char* errStr = cgGetErrorString( err );
			printf("CG Error: Error while compiling program #%d (%s)\n", i, cgData.programName[i] );
			readError++;
		}
		else
			cgGLLoadProgram( cgData.programs[i] );
	}

	printf("(+) Successfully reloaded %d CG shaders\n", i-readError );
	return (readError > 0 ? 1 : 0);
}

/* double-check required extensions, load shaders, and double check that     */
/* they're valid and we have sufficient resources                            */
void InitCGPrograms( void )
{
	int i;
	int readErrors=0;
	
	cgData.shaderContext = cgCreateContext();

	cgData.vertProfile = cgGLGetLatestProfile( CG_GL_VERTEX );
	cgData.fragProfile = cgGLGetLatestProfile( CG_GL_FRAGMENT );
	cgGLSetOptimalOptions( cgData.vertProfile );
	cgGLSetOptimalOptions( cgData.fragProfile );

	for (i=0;i<NUM_CG_PROGRAMS;i++)
	{
		cgData.programName[i] = strdup(cgProgram[i].filename);
		cgData.programType[i] = cgProgram[i].type == GL_VERTEX_PROGRAM_ARB ? cgData.vertProfile : cgData.fragProfile;
		cgData.programs[i] = cgCreateProgramFromFile( cgData.shaderContext, CG_SOURCE, cgProgram[i].filename,
													  cgProgram[i].type == GL_VERTEX_PROGRAM_ARB ? cgData.vertProfile : cgData.fragProfile, 
													  "main", 0 );
		if (!cgData.programs[i])
		{
			printf("%s (%s)\n", cgGetErrorString( cgGetError() ), cgData.programName[i] );
			readErrors++;
		}

		cgGLLoadProgram( cgData.programs[i] );
	}

	printf("     (-) Successfully read %d CG shaders\n", i-readErrors );
}
