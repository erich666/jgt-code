/********************************************
** loadModel.cpp                           **
** -----------                             **
**                                         **
** Contains code and file information for  **
**    loading objects.  Adding additional  **
**    models to the demo is pretty simple  **
**    by following the instructions above  **
**    the 'availableModels' definition.    **
**                                         **
** Chris Wyman (9/07/2006)                 **
********************************************/


#include "glut_template.h"

typedef struct {
    /* text handle OpenGL program can use to identify this model to the user */
	char displayableIdentifier[128];

	/* basic model information */
	char modelFile[128];     // name of model file, without directory
    int modelType;           // to identify i/o function to use

	/* auxilary information necessary to use object as a refractor */
	char nDistFile[128];     // name of nDist file (with vertex distances along -N direction)
    char curvatureFile[128]; // name of file containing vertex curvature data

	/* a flag set by the program during initialization to determine if model is available */
	int filesAvailable;      // either 1 or 0, depending on if all files specified are available
} modelInfo;


/* Notes about adding additional models:
**     1) The 1st field 'displayableIdentifier' is optional, but without it the user
**        will have a difficult time selecting the model from the menu
**     2) The 2nd field 'modelFile' is required -- it is the object file.
**     3) The 3rd field 'modelType' is required, unless you change the LoadRefractingModel() 
**        procedure to recognize file type based upon the filename.
**     4) The 4th field 'nDistFile' is not required (leave as "" if no file exists).  This
**        file would contain the distance along the negative normal direction (to the other 
**        side of the model) for each vertex.  Without this file, the refraction code only
**        uses the distance along V to approximate the refracted distance.
**     5) The 5th field 'curvatureFile' is currently required to load a model.  This stores
**        curvature data at each vertex.
**     6) The final field should be set to 1.  This is a flag used by the program.
*/
modelInfo availableModels[AVAILABLE_MODELS] = 
{
	/*  0 */ { "Al Capone",			"al.obj",				MODEL_TYPE_OBJ,		"",						"alCurvature.txt",			1 },
	/*  1 */ { "Armadillo",			"armadillo.sphparam.m",	MODEL_TYPE_DOTM,	"armadillo_nDist.dat",	"armadilloCurvature.txt",	1 },
	/*  2 */ { "Beethoven",			"beethoven.ifs",		MODEL_TYPE_IFS,		"",						"beethovenCurvature.txt",	1 }, 
	/*  3 */ { "Cow",				"cow.sphparam.m",		MODEL_TYPE_DOTM,	"cow_nDist.dat",		"cowCurvature.txt",			1 },
	/*  4 */ { "F-16",				"f-16.obj",				MODEL_TYPE_OBJ,		"f-16_nDist.dat",		"f16Curvature.txt",			1 },
	/*  5 */ { "Gargoyle",			"gargoyle.sphparam.m",	MODEL_TYPE_DOTM,	"gargoyle_nDist.dat",	"gargoyleCurvature.txt",	1 },
	/*  6 */ { "Happy Buddha",		"buddha_50k.obj",		MODEL_TYPE_OBJ,		"buddha_nDist.dat",		"buddha50kCurvature.txt",	1 },
	/*  7 */ { "Horse",				"horse.sphparam.m",		MODEL_TYPE_DOTM,	"horse_nDist.dat",		"horseCurvature.txt",		1 },
	/*  8 */ { "Stanford Bunny",	"bunny.sphparam.m",		MODEL_TYPE_DOTM,	"bunny_nDist.dat",		"bunnyCurvature.txt",		1 },
	/*  9 */ { "Stanford Dragon",	"dragon_250k.smf",		MODEL_TYPE_OBJ,		"dragon_250k_nDist.dat","dragon250kCurvature.txt",	1 },
	/* 10 */ { "Venus Head",		"venus.sphparam.m",		MODEL_TYPE_DOTM,	"venus_nDist.dat",		"venusCurvature.txt",		1 },
};


/* internal file prototypes */
curvatures *LoadCurvature( char *file );
double *Load_NDist( char *file, int numVertices );


/* this checks the existance of model files to determine if they should be 
**    included in the menu
*/
void CheckModelAvailability( void )
{
	FILE *f;
	char buf[256];
	int i;

	for (i=0;i<AVAILABLE_MODELS;i++)
	{
		availableModels[i].filesAvailable = 1;
		if ( strcmp( availableModels[i].modelFile, "" ) )
		{
			sprintf( buf, "%s%s", MODELS_DIRECTORY,    availableModels[i].modelFile );
			f = fopen( buf, "r" );
			if (!f) availableModels[i].filesAvailable = 0;		
			else { fclose( f ); f = NULL; }
		}
		else
			availableModels[i].filesAvailable = 0;

		if ( strcmp( availableModels[i].curvatureFile, "" ) )
		{
			sprintf( buf, "%s%s",  CURVATURE_DIRECTORY, availableModels[i].curvatureFile );
			f = fopen( buf, "r" );
			if (!f) availableModels[i].filesAvailable = 0;		
			else { fclose( f ); f = NULL; }
		}
		else
			availableModels[i].filesAvailable = 0;
	}
}

int LoadRefractingModel( int ID, GLuint *dispList, GLuint *normDispList )
{
	GLMmodel *modelTmp1=0;
	mymodel *modelTmp2=0;
	curvatures *c=0;
	double *nDist=0;
	char modelFilename[256];
	char nDistFilename[256];
	char curvFilename[256];

	/* check that inputs were valid */
	if (ID >= AVAILABLE_MODELS || !dispList || !normDispList)
		return 0;
	if (!availableModels[ID].filesAvailable)
		return 0;

	/* determine locations of relevent files */
	sprintf( modelFilename, "%s%s", MODELS_DIRECTORY,    availableModels[ID].modelFile );
	sprintf( nDistFilename, "%s%s", NDIST_DIRECTORY,     availableModels[ID].nDistFile );
	sprintf( curvFilename, "%s%s",  CURVATURE_DIRECTORY, availableModels[ID].curvatureFile );

	/* now load the model, depending on it's file type */
	if ( availableModels[ID].modelType == MODEL_TYPE_OBJ ||
		 availableModels[ID].modelType == MODEL_TYPE_IFS )
	{
		/* load and normalize the object file */
		modelTmp1 = (availableModels[ID].modelType == MODEL_TYPE_OBJ ? 
			glmReadOBJ( modelFilename ) :
			glmReadIFS( modelFilename ) );
		glmUnitize( modelTmp1 );
		glmFacetNormals( modelTmp1 );
		glmVertexNormals( modelTmp1, 90.0 );

		/* load curvature data */
		c = LoadCurvature( curvFilename );
		if (!c)
		{		
			glmDelete( modelTmp1 );
			return 0;
		}

		/* load negative normal distances */
		nDist = Load_NDist( nDistFilename, modelTmp1->numvertices );
		
		/* create standard display list for rendering */
		*dispList = glGenLists(1);
		glNewList(*dispList, GL_COMPILE);
		glmDrawForRefraction( modelTmp1, nDist, c->curv );
		glEndList();

		/* create a display list containing the object's surface normal as its color */
		*normDispList = glGenLists(1);
		glNewList(*normDispList, GL_COMPILE);
		glmDrawNormalsForRefraction( modelTmp1, nDist, c->curv );
		glEndList();
	}
	else if ( availableModels[ID].modelType == MODEL_TYPE_DOTM )
	{
		/* load and normalize the object file */
		modelTmp2 = ReadDotM( modelFilename );
		UnitizeDotM( modelTmp2 );

		/* load curvature data */
		c = LoadCurvature( curvFilename );
		if (!c)
		{		
			FreeDotM(modelTmp2);
			return 0;
		}

		/* load negative normal distances */
		nDist = Load_NDist( nDistFilename, modelTmp2->numVertices );
		
		/* create standard display list for rendering */
		*dispList = CreateDotMList( modelTmp2, nDist, c->curv );

		/* create a display list containing the object's surface normal as its color */
		*normDispList = glGenLists(1);
		glNewList(*normDispList, GL_COMPILE);
		DrawDotMNormals( modelTmp2, nDist, c->curv );
		glEndList();
	}
	else return 0;

	/* perform some memory cleanup before returning */
	if (modelTmp1)
		glmDelete(modelTmp1);
	if (modelTmp2)
		FreeDotM(modelTmp2);
	free( nDist );
	free( c->curv );
	free( c );

	return 1;
}



double *Load_NDist( char *file, int numVertices )
{
	int i, num=-1, useDefault=0;
	FILE *f=0;
	double *ptr;

	if ( strcmp( file, "" ))
		f = fopen( file, "r" );
	if (!f)
		useDefault=1;
	else
		fscanf( f, "%d", &num );
	if (num != numVertices)
		useDefault=1;

	ptr = (double *)malloc( numVertices * sizeof( double ) );

	if (!useDefault)
	{
		for (i=0;i<num;i++)
			fscanf( f, "%lf", &(ptr[i]) );
		fclose( f );
	}
	else
		for (i=0;i<numVertices;i++) ptr[i] = 0;
    
	return ptr;
}


curvatures *LoadCurvature( char *file )
{
	int j, errCount=0;
	char tmp[1024];
	FILE *f;
	curvatures *c;

	errCount=0;
	f = fopen( file, "r" );
	if (!f) 
		return 0;
	
	c = (curvatures *)malloc( sizeof ( curvatures ) );
	if (!c)
		return 0;

	fscanf( f, "v: %d\n", &c->numVertices );
	fscanf( f, "Order: k1, k2, 0.5*(k1+k2), k1*k2\n" );  // Get the 2nd line of these files...

	c->curv = (float *) malloc ( 4 * c->numVertices * sizeof( float ) );
	if (!c->curv)
	{
		free( c );
		return 0;
	}

	for ( j = 0; j < c->numVertices; j++ )
	{
		if (4 != fscanf( f, "%f %f %f %f\n", &c->curv[4*j], &c->curv[4*j+1],
				         &c->curv[4*j+2], &c->curv[4*j+3] ) )
		{  // in this case, we ran into some values that fscaf() doesn't recognize as number (e.g., NaN)
			fgets( tmp, 1024, f );
			c->curv[4*j] = c->curv[4*j+1] = 
				c->curv[4*j+2] = c->curv[4*j+3] = 0;
			errCount++;
		}

	}

	fclose( f );
	return c;
}


char *GetModelIdentifier( int i )
{
	if (i >= AVAILABLE_MODELS) return 0;
	return availableModels[i].displayableIdentifier;
}

void SetupModelMenu( void )
{
	int i;

	GLint modeMenu = glutCreateMenu( menu );
	glutAddMenuEntry( "Original Caustics Mapping", 600 );
	glutAddMenuEntry( "Multi-Resolution Caustics", 601 );
	glutAddMenuEntry( "Lens-Based, Continuously Varying Caustics", 602 );

	GLint objMenu = glutCreateMenu( menu );

	for (i=0; i < AVAILABLE_MODELS; i++)
	{
		char buf[1024];
		if (availableModels[i].filesAvailable)
		{
			strncpy( buf, availableModels[i].displayableIdentifier, 1023 );
			glutAddMenuEntry( buf, 200+i );
		}
	}

	GLint bgMenu = glutCreateMenu( menu );
	glutAddMenuEntry( "Toggle display of MacBeth Color Chart", 300 );
	glutAddMenuEntry( "Toggle display of Opaque Dragon", 301 );
	glutAddMenuEntry( "Toggle display of ground (terrain or plane)", 302);

	GLint screenSizeMenu = glutCreateMenu( menu );
	glutAddMenuEntry( "256 x 256 window size", 540 );
	glutAddMenuEntry( "512 x 512 window size", 541 );
	glutAddMenuEntry( "1024 x 1024 window size", 542 );

	GLint phBufMenu = glutCreateMenu( menu );
	glutAddMenuEntry( "64 x 64 photon buffer", 500 );
	glutAddMenuEntry( "128 x 128 photon buffer", 501 );
	glutAddMenuEntry( "256 x 256 photon buffer", 502 );
	glutAddMenuEntry( "512 x 512 photon buffer", 503 );
	glutAddMenuEntry( "1024 x 1024 photon buffer", 504 );
	glutAddMenuEntry( "2048 x 2048 photon buffer", 505 );
	glutAddMenuEntry( "4096 x 4096 photon buffer", 506 );

	GLint causticMapMenu = glutCreateMenu( menu );
	glutAddMenuEntry( "64 x 64 caustic map", 520 );
	glutAddMenuEntry( "128 x 128 caustic map", 521 );
	glutAddMenuEntry( "256 x 256 caustic map", 522 );
	glutAddMenuEntry( "512 x 512 caustic map", 523 );
	glutAddMenuEntry( "1024 x 1024 caustic map", 524 );
	glutAddMenuEntry( "2048 x 2048 caustic map", 525 );
	glutAddMenuEntry( "4096 x 4096 caustic map", 526 );

	GLint mainBufferMenu = glutCreateMenu( menu );
	glutAddMenuEntry( "64 x 64 final render size", 560 );
	glutAddMenuEntry( "128 x 128 final render size", 561 );
	glutAddMenuEntry( "256 x 256 final render size", 562 );
	glutAddMenuEntry( "512 x 512 final render size", 563 );
	glutAddMenuEntry( "1024 x 1024 final render size", 564 );
	glutAddMenuEntry( "2048 x 2048 final render size", 565 );
	glutAddMenuEntry( "4096 x 4096 final render size", 566 );

	GLint resizeMenu = glutCreateMenu( menu );
	glutAddSubMenu( "Window Size", screenSizeMenu );
	glutAddSubMenu( "Final Render Size (for sub/supersampling)", mainBufferMenu );
	glutAddSubMenu( "Photon Buffer Size", phBufMenu );
	glutAddSubMenu( "Caustic Map Size", causticMapMenu );

	glutCreateMenu( menu );
	glutAddSubMenu( "Select caustic rendering method", modeMenu );
	glutAddMenuEntry( " ", 0 );
	glutAddSubMenu( "Load new refractor", objMenu );
	glutAddSubMenu( "Change background objects", bgMenu );
	glutAddSubMenu( "Change render-to-texture size", resizeMenu );
	glutAddMenuEntry( " ", 0 );
	glutAddMenuEntry( "Toggle onscreen help", 100 );
	glutAddMenuEntry( " ", 0 );
	glutAddMenuEntry( "Quit", 27 );
	glutAttachMenu( GLUT_RIGHT_BUTTON );
}