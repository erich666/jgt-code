/*************************************************
** glut_template.h                              **
** ---------------                              **
**                                              **
** This is the main header for the demo, With   **
**    function prototypes, constants, etc.      **
**                                              **
** Chris Wyman (9/28/2000)                      **
*************************************************/

#ifndef GLUT_TEMPLATE_H
#define GLUT_TEMPLATE_H


// Directories that might be changed for different demo setups.
#define MODELS_DIRECTORY            "models\\"
#define TEXTURES_DIRECTORY          "textures\\"
#define SHADERS_DIRECTORY           "shaders\\"
#define ENVIRONMENT_MAPS_DIRECTORY  "textures\\"
#define NDIST_DIRECTORY             "otherData\\"
#define CURVATURE_DIRECTORY         "otherData\\"

// If additional models are added (in loadModel.cpp) this should be incremented
#define AVAILABLE_MODELS          11

/***************************************************************************************/
/*             BELOW THIS POINT SHOULD NOT GENERALLY NEED TO BE CHANGED                */
/*            (AT LEAST FOR BASIC RECOMPILING TO GET THE PROGRAM RUNNING)              */
/**************************************************************************************/

// Standard includes
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include "ppm.h"
#include "bmp.h"
#include "DotM.h"
#include "nv_dds.h"
#include "glm.h"
#include "framebufferObject.h"


/* The following definitions are to help with allocating textures and render targets to  */
/*     optimize for speed.  We'd really rather not have so many textures defined that we */
/*     thrash GPU memory or use excess render targets because not all output data is     */
/*     utilized.  This allows a quick glimpse of which framebuffer objects have targets  */
/*     in use.  Please note some are utilized in multiple passes.  Also, one or two      */
/*     have not had TEXTURE_* definitions made (there are still some ->GetColor calls    */
/*     and/or ->GetDepth calls in the code.                                              */
#define TEXTURE_ENVIRONMENT_MAP                   hdrCubeMap[currentEnvMap]
#define TEXTURE_PRECOMPUTED_ACOS_FRESNEL          arcCosFresnelTex
#define TEXTURE_GAUSSIAN_WEIGHTS                  samplingPatternTex
#define TEXTURE_BACKFACING_NORMALS                backNormalsFB->GetColorTextureID(0)
#define TEXTURE_BACKFACING_DEPTH                  backNormalsFB->GetDepthTextureID()

#define TEXTURE_PHOTON_LOCATION_BUFFER            causticFB->GetColorTextureID(0)
#define TEXTURE_PHOTON_INCIDENT_DIRECTION         causticFB->GetColorTextureID(1)
#define TEXTURE_COMPLETE_DEPTH_FROM_LIGHT         causticFB->GetDepthTextureID()
#define ATTACH_PHOTON_LOCATION_BUFFER             GL_COLOR_ATTACHMENT0_EXT
#define ATTACH_PHOTON_INCIDENT_DIRECTION          GL_COLOR_ATTACHMENT1_EXT

// Used inside ComputePseudoPhotonMapGatherFromLight().  Be aware currently drawFB == causticFB
#define TEXTURE_TEMPORARY_PHOTON_STORAGE          drawFB->GetColorTextureID(0)
#define ATTACH_TEMPORARY_PHOTON_STORAGE           GL_COLOR_ATTACHMENT0_EXT

// The blurred-in-image space caustic map.  Note this buffer serves different purposes in light & eye space gathers
#define TEXTURE_LIGHTSPACE_CAUSTIC_MAP            pseudoPhotonMapFB->GetColorTextureID(0)
#define ATTACH_LIGHTSPACE_CAUSTIC_MAP             GL_COLOR_ATTACHMENT0_EXT

#define TEXTURE_IMAGE_OF_BACKGROUND_GEOM          backgroundGeomFB->GetColorTextureID(0)
#define ATTACH_IMAGE_OF_BACKGROUND_GEOM           GL_COLOR_ATTACHMENT0_EXT
#define TEXTURE_BACKGROUND_SURFACE_NORMS          backgroundGeomFB->GetColorTextureID(1)
#define ATTACH_BACKGROUND_SURFACE_NORMS           GL_COLOR_ATTACHMENT1_EXT


// Extensions used by the program -- may not be up to date!
/*
#define EXTENSIONS_USED "ARB_draw_buffers EXT_framebuffer_object ARB_vertex_program ARB_fragment_program \
                         ARB_vertex_buffer_object ARB_pixel_buffer_object ARB_color_buffer_float ARB_texture_float \
                         ARB_shadow ARB_depth_texture NV_fragment_program2"
*/

// Used for vertex buffer object indexing.
#define BUFFER_OFFSET(i) ((char *)NULL + (i))

// Definitions that should really be in a useful #include file...
#ifndef M_PI
#define M_PI            3.14159265358979323846
#endif 
#ifndef M_SQRT2
#define M_SQRT2         1.41421356237309504880
#endif 
#ifndef MAX
#define MAX(x,y) ((x)>(y)?(x):(y))
#endif
#ifndef MIN
#define MIN(x,y) ((x)<(y)?(x):(y))
#endif

// GLUT callback function prototypes
void display ( void );
void reshape( int w, int h );
void idle ( void );
void motion(int x, int y);
void button(int b, int st, int x, int y);
void menu( int value );
void keys(unsigned char key, int x, int y);
void special_keys(int key, int x, int y);

// Prototypes for functions that perform major refractive and/or caustic rendering
void createDistanceMap( FrameBuffer *drawFB );
void createBackSideMaps( FrameBuffer *drawFB, int fromPOV );
void createBackFacesMaps( FrameBuffer *drawFB, int fromPOV );
void DrawOtherSceneObjects( int shaded, int colorOrPhoton );
void DrawShadowedBackgroundWithLightSpaceCaustics( FrameBuffer *drawFB, int fromPOV );
void createOtherObjectEyeTexture( FrameBuffer *drawFB, int colorOrPhoton, int fromPOV );
void createRefractionBuffers( GLdouble lookAtMatrix[16], int colorOrPhotons, int fromPOV );
void CreateRefractionBuffersForLightSpacePhotonGather( GLdouble lookAtMatrix[16], int fromPOV );
void PerformLightSpaceGather( FrameBuffer *tmpFB, FrameBuffer *drawFB );
void ComputePseudoPhotonMapGatherFromLight( FrameBuffer *drawFB );
void DrawRefractedScene( int fromPOV );
void DrawCausticsSceneWithLightSpaceGatheredPhotons( int fromPOV );
void DrawSimpleRefractedScene( FrameBuffer *currFB, int fromPOV );
void DrawSceneCausticsToBuffer( FrameBuffer *currFB, int fromPOV );
void FinalDrawWithAlreadyComputedLightSpaceGather( FrameBuffer *currFB, int fromPOV );

// prototypes for loading data
float ***loadHDR(char *filename, int *w, int *h);
float *loadHDRtoArray(char *filename, int *w, int *h);
GLuint LoadLightProbeCube( char *filename, int *mapW, int *mapH );
GLuint LoadDDSCube( char *filename, int *mapW, int *mapH );
GLuint LoadTexture( char *fname, unsigned int mask );
void LoadHDRTextures( void );
void ParseGeometrySettingsFile( char *filename );
void SetupAndAllocateTextures( void );
void LoadModels( void );
void CheckModelAvailability( void );

// Misc. auxilary functions, functions to hide extra detail, other init functions
void DemoInformationToStdout( void );
void SetLightProperties( void );
void initFramebuffers( void );
void initLookAtMatrices( void );
void SetupFrameBufferMatrices( int fromPOV );
void DrawEnvironmentMap( int fromPOV, int colorOrPhoton );
void SetRefractiveObjectShaderParameters( FrameBuffer *currFB, int fromPOV );
void EnableTextureUnit( GLenum texUnit, GLenum texTarget, GLuint textureID );
void DisableTextureUnit( GLenum texUnit, GLenum texTarget );
void EnableReflectionRefractionShader( int colorOrPhoton );
void DisableReflectionRefractionShader( int colorOrPhoton );
void AllowUnclampedColors( void );
void ResumeClampingColors( void );
void displayTimer( float fps );
float UpdateFPS( void );
void DrawBigScreenTexture( int displayBig );
void DisplaySideImages( void );
void SetLightProperties( void );
void SetupMenu( void );
void PrintString(char *str, void *font=GLUT_BITMAP_HELVETICA_12);
void drawHelpScreen( void );
void drawErrorMessage( void );
void displayLargeTexture( GLuint tex, int minMode=GL_LINEAR, int magMode=GL_LINEAR, float color=1, GLenum textureFunc=GL_MODULATE );
void displayLargeTextureWithDepth( GLuint tex, GLuint depth, int minMode=GL_LINEAR, int magMode=GL_LINEAR );
void displayLoadingMessage( char *inbuf );
void CreateSphere(double r,int n);
void SetupOpenGLLight( void );
void SetupModelMenu( void );
char *GetModelIdentifier( int i );
void UseDefaultSettings( void );
int resizeFramebuffer( FrameBuffer *fb, int newSize );

// Some simple math prototypes
int matInvertd(double src[16], double inverse[16]);
void matInverseTransposed( double res[16], double m[16] );


// Defines what we're currently rendering.  Various functions use different
//    shaders (which render different relevant values) depending on the 
//    use.  This allows a reduction in render targets (and increase in speed)
//    if not all data is needed.  Values don't matter, if they're unique.
#define ENABLE_JUST_COLOR          0
#define ENABLE_JUST_PHOTONS        1
#define ENABLE_COLOR_AND_NORMALS   2



/*******************************************************************************************/
/*  Parameters to pass to the "mask" parameter of LoadTexture()....                        */
/*******************************************************************************************/
#define TEX_DEFAULT                        (unsigned int)0
#define TEX_MAG_NEAREST                    (unsigned int)1
#define TEX_MAG_LINEAR                     (unsigned int)2
#define TEX_REPEAT_S                       (unsigned int)4
#define TEX_MIRROR_REPEAT_S                (unsigned int)8
#define TEX_CLAMP_S                       (unsigned int)16
#define TEX_CLAMP_TO_EDGE_S               (unsigned int)32
#define TEX_CLAMP_TO_BORDER_S             (unsigned int)64
#define TEX_REPEAT_T                     (unsigned int)128
#define TEX_MIRROR_REPEAT_T              (unsigned int)256
#define TEX_CLAMP_T                      (unsigned int)512
#define TEX_CLAMP_TO_EDGE_T             (unsigned int)1024
#define TEX_CLAMP_TO_BORDER_T           (unsigned int)2048
#define TEX_MIN_NEAREST                 (unsigned int)4096
#define TEX_MIN_LINEAR                  (unsigned int)8192
#define TEX_MIN_NEAR_MIP_NEAR          (unsigned int)16384
#define TEX_MIN_NEAR_MIP_LINEAR        (unsigned int)32768
#define TEX_MIN_LINEAR_MIP_NEAR        (unsigned int)65536
#define TEX_MIN_LINEAR_MIP_LINEAR     (unsigned int)131072
#define TEX_INTERNAL_RGB              (unsigned int)262144
#define TEX_INTERNAL_RGBA             (unsigned int)524288


/*******************************************************************************************/
/* Assign each shader (both vertex & fragment) a unique number (0 <= ID < NUM_CG_PROGRAMS) */
/*******************************************************************************************/
#define NUM_CG_PROGRAMS                       37
#define CG_REFRACT_VERT                        0 
#define CG_REFRACT_FRAG                        1 
#define CG_REFRACT_OTHER_OBJS_V                2 
#define CG_PHONG_SHADE_PLUS_VERTEX             3 
#define CG_DRAW_TEXTURE_WITH_MASK_FRAGMENT     4 
#define CG_STORE_PHOTONS_IN_LIGHT_BUFFER_V     5 
#define CG_STORE_PHOTONS_IN_LIGHT_BUFFER_F     6 
#define CG_PSEUDO_PHOTONMAP_GATHER_LIGHTPOV_V  7 
#define CG_PSEUDO_PHOTONMAP_GATHER_LIGHTPOV_F  8 
#define CG_REFRACT_OTHER_OBJS_JUSTCOLOR_F      9 
#define CG_REFRACT_OTHER_OBJS_JUSTPHOTON_F    10 
#define CG_PHONG_SHADE_JUSTCOLOR_F            11 
#define CG_PHONG_SHADE_JUSTPHOTONS_F          12 
#define CG_PHONG_SHADE_COLORANDNORMALS_F      13 
#define CG_LIGHTGATHER_SHADMAP_TEXTURES_V     14 
#define CG_LIGHTGATHER_SHADMAP_TEXTURES_F	  15 
#define CG_LIGHTGATHER_SHADMAP_PHONG_V        16 
#define CG_LIGHTGATHER_SHADMAP_PHONG_F		  17 
#define CG_EXPERIMENTAL_V                     18 
#define CG_EXPERIMENTAL_F                     19 
#define CG_EXPERIMENTAL2_V                    20 
#define CG_EXPERIMENTAL2_F                    21 
#define CG_LIGHT_GATHER_W_GAUSSIAN_SPLATS_V   22 
#define CG_LIGHT_GATHER_W_GAUSSIAN_SPLATS_F   23 
#define CG_COMBINE_MULTILEVEL_SPLATS_V        24 
#define CG_COMBINE_MULTILEVEL_SPLATS_F        25 
#define CG_RENDER_BF_NORMALS_F                26 

#define CG_INDEX_CUBE_MAP_V                   27
#define CG_TEXTURE_OTHER_OBJS_V               28
#define CG_TEXTURE_OTHER_OBJS_JUSTPHOTONS_F   29
#define CG_TEXTURE_OTHER_OBJS_JUSTCOLOR_F     30
#define CG_TEXTURE_OTHER_OBJS_COLORANDNORMS_F 31
#define CG_INDEX_CUBE_MAP_JUSTPHOTONS_F       32
#define CG_INDEX_CUBE_MAP_JUSTCOLOR_F         33
#define CG_INDEX_CUBE_MAP_COLORANDNORMS_F     34
#define CG_COPY_TEXTURE_AND_DEPTH_F           35
#define CG_LIGHT_GATHER_W_MULTIRES_SPLATS_F   36 




#define MODEL_GROUND_PLANE        "marsTerrain2.obj"
#define MODEL_BACKGROUND_OBJ      "dragon_250k.smf"

#define MODEL_TYPE_DOTM           0
#define MODEL_TYPE_OBJ            1
#define MODEL_TYPE_IFS            2


#define NUM_ENVMAPS               12

// definition for the fresnel material type
#define F0                   0.05

// trackball definitions
#define OBJECT_TRACKBALL       0
#define LIGHT_TRACKBALL        1
#define MISC_TRACKBALL         2
#define EYE_TRACKBALL          3
#define ENVIRONMENT_TRACKBALL  4
#define TOTAL_TRACKBALLS       5

// Values that are passed around defining which point of view 
//     we're currently rendering from.  
#define EYE_POV        0
#define LIGHT_POV      1

// Defines the resolution of the various buffers.  These are square buffers, 
//     and resolutions need to be powers of two.  To avoid square textures,
//     some code modifications are necessary.  They might not be easily done.
#define DEFAULT_SCREEN_SIZE               512
#define DEFAULT_MAINBUFFER_SIZE           1024
#define DEFAULT_BACKGROUND_BUFFER_SIZE    512
#define DEFAULT_REFRACTION_BUFFER_SIZE    512
#define DEFAULT_CAUSTICMAP_SIZE           512
#define DEFAULT_NUM_PHOTONS               512


/*******************************************************************************************/
/* A structure to keep relevant shader info together                                       */
/*******************************************************************************************/
typedef struct __shader {
  char filename[128];         /* Filename of file containing shader code                   */
  int type;                   /* GL_VERTEX_PROGRAM_ARB or GL_FRAGMENT_PROGRAM_ARB          */
  int isInitialized;          /* Have we initalized this program (to dbl check validity?)  */
  GLuint shaderNum;           /* A unique GL-assigned shader number                        */ 
  char *string;               /* A string containing ASCII code for the shader             */ 
} shader;


void EnableCGTexture( int progNum, GLuint texID, char *paramName );
void DisableCGTexture( int progNum, char *paramName );
int SetCGParameterv( int progNum, char *paramName, int arraySize, float *array );
int SetCGParameter4f( int progNum, char *paramName, float x, float y, float z, float w );
int SetCGParameter3f( int progNum, char *paramName, float x, float y, float z );
int SetCGParameter2f( int progNum, char *paramName, float x, float y );
int SetCGParameter1f( int progNum, char *paramName, float x );
void BindCGProgram( int progNum );
void BindCGPrograms( int vertProgNum, int fragProgNum );
void UnbindCGProgram( int progNum );
void UnbindCGPrograms( int vertProgNum, int fragProgNum );
int ReloadCGShaders( void );
void InitCGPrograms( void );
int CurrentCGFragmentShader( void );
int CurrentCGVertexShader( void );


// Basic structure to store curvatures of all models
typedef struct {
   int numVertices;
   float *curv;
} curvatures;

int LoadRefractingModel( int ID, GLuint *dispList, GLuint *normDispList );

#endif

