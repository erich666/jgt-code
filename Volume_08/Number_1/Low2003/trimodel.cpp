#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <assert.h>
#include <GL/glut.h>
#include "common.h"
#include "gltx.h"
#include "trimodel.h"

#define MAX_PATH_LEN	1023
#define MAX_FILE_LEN	255


// use OpenGL's default material
static const TM_Material DefaultMat = { { 0.2, 0.2, 0.2 },
										{ 0.8, 0.8, 0.8 },
										{ 0.0, 0.0, 0.0 },
										{ 0.0, 0.0, 0.0 }, 
										0.0, 0.0, NULL };



static void GetPathName( char *path, const char *path_file )
	// extract path name from a full-path filename
{
	int len = strlen( path_file );
	int i, j;

	// find the rightmost slash
	for ( i = len-1; i >= 0; i-- )
	{
		if ( path_file[i] == '/' || path_file[i] == '\\' ) break;
	}

	if ( i >= MAX_PATH_LEN ) error_exit( __FILE__, __LINE__, "Path name too long" );

	// if there is no slash
	if ( i < 0 ) 
	{
		path[0] = '\0';
		return;
	}

	// copy path name
	for( j = 0; j <= i; j++ ) path[j] = path_file[j];
	path[j] = '\0';
}



static void InitModel( TM_Model *model )
{
	model->numTris = 0;
	model->tri = NULL;
	model->numMats = 0;
	model->mat = NULL;
	model->numTexEnvs = 0;
	model->tex = NULL;
	model->numTexImages = 0;
	model->image = NULL;
}



static GLTXimage *ReadTexture( const char *path, const char *file )
{
	char pathfile[ MAX_PATH_LEN + MAX_FILE_LEN + 1 ];

	if ( strlen( path ) + strlen( file ) > MAX_PATH_LEN + MAX_FILE_LEN )
		error_exit( __FILE__, __LINE__, "Path name too long" );

	strcpy( pathfile, path );
	strcat( pathfile, file );

	GLTXimage *img = gltxReadRGB( pathfile );

	if ( img == NULL ) error_exit( __FILE__, __LINE__, "Cannot read image file \"%s\"", pathfile );
	if ( img->components != 3 ) 
		error_exit( __FILE__, __LINE__, "\"%s\" is not a 3-component RGB image", pathfile );

	return img;
}



static void CompBoundingBox( TM_Model *model )
{
	model->min_xyz[0] = FLT_MAX;
	model->min_xyz[1] = FLT_MAX;
	model->min_xyz[2] = FLT_MAX;
	model->max_xyz[0] = -FLT_MAX;
	model->max_xyz[1] = -FLT_MAX;
	model->max_xyz[2] = -FLT_MAX;

	for ( int i = 0; i < model->numTris; i++ )
	{
		TM_Triangle *tri = &model->tri[i];

		for ( int k = 0; k < 3; k++ )
		{
			if ( tri->v[k][0] < model->min_xyz[0] ) model->min_xyz[0] = tri->v[k][0];
			if ( tri->v[k][1] < model->min_xyz[1] ) model->min_xyz[1] = tri->v[k][1];
			if ( tri->v[k][2] < model->min_xyz[2] ) model->min_xyz[2] = tri->v[k][2];
			if ( tri->v[k][0] > model->max_xyz[0] ) model->max_xyz[0] = tri->v[k][0];
			if ( tri->v[k][1] > model->max_xyz[1] ) model->max_xyz[1] = tri->v[k][1];
			if ( tri->v[k][2] > model->max_xyz[2] ) model->max_xyz[2] = tri->v[k][2];
		}
	}

	model->dim_xyz[0] = model->max_xyz[0] - model->min_xyz[0];
	model->dim_xyz[1] = model->max_xyz[1] - model->min_xyz[1];
	model->dim_xyz[2] = model->max_xyz[2] - model->min_xyz[2];
	model->center[0] = 0.5 * (model->max_xyz[0] + model->min_xyz[0]);
	model->center[1] = 0.5 * (model->max_xyz[1] + model->min_xyz[1]);
	model->center[2] = 0.5 * (model->max_xyz[2] + model->min_xyz[2]);
}




TM_Model *TM_ReadModel( const char *pathfile, double scaleFact )
{
	char badfile[] = "Invalid input model file";
	char buf[ MAX_FILE_LEN + 1 ];
	char path[ MAX_PATH_LEN + 1 ];

	// allocate a TM_Model
	TM_Model *model = (TM_Model *) checked_malloc( sizeof(TM_Model) );
	InitModel( model );

	// open input file
	FILE *fp = fopen( pathfile, "r" );
	if ( fp == NULL ) 
		error_exit( __FILE__, __LINE__, "Cannot open input model file \"%s\"", pathfile );

	// read number of texture files
	if ( fscanf( fp, "%d", &model->numTexImages ) < 1 ) 
		error_exit( __FILE__, __LINE__, badfile );

	fprintf( stderr, "Number of texture image files = %d\n", model->numTexImages );

	// allocate array to store images
	model->image = (GLTXimage **) checked_malloc( sizeof(GLTXimage *) * model->numTexImages );

	int i, j;

	// read textures
	GetPathName( path, pathfile );  // assume image files in same directory as model file

	for ( i = 0; i < model->numTexImages; i++ )
	{
		// read texture filename
		if ( fscanf( fp, "%d \"%s", &j, buf ) < 2 ) error_exit( __FILE__, __LINE__, badfile );
		buf[ strlen( buf ) - 1 ] = '\0';

		fprintf( stderr, "Texture file %d = \"%s%s\"\n", j, path, buf );

		model->image[i] = ReadTexture( path, buf );
	}


	// read display list

	int triCount = 0;
	int matCount = 0;
	int texEnvCount = 0;

	TM_Triangle *trilist = NULL;
	TM_Material *matlist = NULL;
	TM_TexEnv   *texlist = NULL;

	while ( !feof( fp ) )
	{
		if ( fscanf( fp, "%s", buf ) < 1 ) break;

		switch( buf[0] )
		{
			case 'P':	// polygon/triangle
			{
				TM_Triangle *tri = (TM_Triangle *) checked_malloc( sizeof(TM_Triangle) );

				// add node to front of linked-list
				tri->next = trilist;
				trilist = tri;

				for ( int k = 0; k < 3; k++ )
				{
					int m = fscanf( fp, "%lf %lf %lf %f %f %f %f %f",
									&tri->v[k][0], &tri->v[k][1], &tri->v[k][2],
									&tri->n[k][0], &tri->n[k][1], &tri->n[k][2],
									&tri->t[k][0], &tri->t[k][1] );

					if ( m < 8 ) error_exit( __FILE__, __LINE__, badfile );

					// scale model
					tri->v[k][0] *= scaleFact;
					tri->v[k][1] *= scaleFact;
					tri->v[k][2] *= scaleFact;
				}

				// if no material has been read, then use the default material
				if ( matCount <= 0 )
				{
					TM_Material *mat = (TM_Material *) checked_malloc( sizeof(TM_Material) );

					*mat = DefaultMat;

					// add node to front of linked-list
					mat->next = matlist;
					matlist = mat;
								
					matCount++;	
				}

				tri->mat = matCount - 1;
				tri->tex = texEnvCount - 1;
				
				triCount++;
				break;
			}

			case 'M':	// material
			{
				TM_Material *mat = (TM_Material *) checked_malloc( sizeof(TM_Material) );

				// add node to front of linked-list
				mat->next = matlist;
				matlist = mat;

				int m = fscanf( fp, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f",
									&mat->am[0], &mat->am[1], &mat->am[2],
									&mat->di[0], &mat->di[1], &mat->di[2],
									&mat->sp[0], &mat->sp[1], &mat->sp[2],
									&mat->em[0], &mat->em[1], &mat->em[2],
									&mat->sh, &mat->tr );

				if ( m < 14 ) error_exit( __FILE__, __LINE__, badfile );

				matCount++;
				break;
			}

			case 'T':	// texture binding
			{
				TM_TexEnv *tex = (TM_TexEnv *) checked_malloc( sizeof(TM_TexEnv) );

				// add node to front of linked-list
				tex->next = texlist;
				texlist = tex;

				int m = fscanf( fp, "%d %d %d %d",
									&tex->texImage, &tex->wrapS, &tex->wrapT, &tex->model );
				if ( m < 4 ) error_exit( __FILE__, __LINE__, badfile );

				m = fscanf( fp, "%f %f %f",
								&tex->blend[0], &tex->blend[1],	&tex->blend[2] );
				if ( m < 3 ) error_exit( __FILE__, __LINE__, badfile );
				
				texEnvCount++;
				break;
			}

			default:
			{
				error_exit( __FILE__, __LINE__, badfile );
				break;
			}
		}
	}
	
	fclose( fp );

	fprintf( stderr, "Number of triangles read = %d\n", triCount );
	fprintf( stderr, "Number of materials read = %d\n", matCount );
	fprintf( stderr, "Number of texture environments read = %d\n", texEnvCount );

	// allocate memory for arrays in model
	model->tri = (TM_Triangle *) checked_malloc( sizeof(TM_Triangle) * triCount );
	model->mat = (TM_Material *) checked_malloc( sizeof(TM_Material) * matCount );
	model->tex = (TM_TexEnv *) checked_malloc( sizeof(TM_TexEnv) * texEnvCount );
	
	model->numTris = triCount;
	model->numMats = matCount;
	model->numTexEnvs = texEnvCount;

	// transfer items from linked-lists to arrays, and deallocate linked-lists
	for ( i = triCount - 1; i >= 0; i-- )
	{
		model->tri[i] = *trilist;
		free( trilist );
		trilist = model->tri[i].next;
		model->tri[i].next = NULL;
	}

	for ( i = matCount - 1; i >= 0; i-- )
	{
		model->mat[i] = *matlist;
		free( matlist );
		matlist = model->mat[i].next;
		model->mat[i].next = NULL;
	}

	for ( i = texEnvCount - 1; i >= 0; i-- )
	{
		model->tex[i] = *texlist;
		free( texlist );
		texlist = model->tex[i].next;
		model->tex[i].next = NULL;
	}

	CompBoundingBox( model );

	return model;
}



void TM_DeleteModel( TM_Model *model )
{
	for ( int i = 0; i < model->numTexImages; i++ )	gltxDelete( model->image[i] );
	free( model->tri );
	free( model->mat );
	free( model->tex );
	free( model->image );
	free( model );
}



TM_DList *TM_MakeOGLDList( const TM_Model *model, bool mipmap )
{
	TM_DList *dlist = (TM_DList *) checked_malloc( sizeof(TM_DList) );

	dlist->numTexIDs = model->numTexImages;
	dlist->oglTexID = (GLuint *) checked_malloc( sizeof(GLuint) * dlist->numTexIDs );

	int i;

	// create OpenGL texture objects

	glPixelStorei( GL_UNPACK_ALIGNMENT, 1 );

	for ( i = 0; i < dlist->numTexIDs; i++ )
	{
		glGenTextures( 1, &dlist->oglTexID[i] );
		if ( dlist->oglTexID[i] == 0 ) 
			error_exit( __FILE__, __LINE__, "Cannot create texture object" );

		glBindTexture( GL_TEXTURE_2D, dlist->oglTexID[i] );

		if ( mipmap )
		{
			int k = gluBuild2DMipmaps( GL_TEXTURE_2D, 3, model->image[i]->width, model->image[i]->height,
			  		                   GL_RGB, GL_UNSIGNED_BYTE, model->image[i]->data );
			if ( k != 0 ) error_exit( __FILE__, __LINE__, "Cannot create mipmap" );
		}
		else
			glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB, model->image[i]->width, model->image[i]->height,
					      0, GL_RGB, GL_UNSIGNED_BYTE, model->image[i]->data );
	}


	// create OpenGL display list

	dlist->oglDList = glGenLists( 1 );
	if ( dlist->oglDList == 0 ) 
		error_exit( __FILE__, __LINE__, "Cannot create display list" );

	glNewList( dlist->oglDList, GL_COMPILE );

	bool textured = true;
	bool glBeginEnded = true;

	int currTexEnv = -2;  // any dummy integer less than -1
	int currMat = -1;     // any dummy integer less than 0


	for ( i = 0; i < model->numTris; i++ )
	{
		TM_Triangle *tri = &model->tri[i];

		// set texture
		if ( tri->tex != currTexEnv )
		{
			if ( !glBeginEnded ) 
			{	
				glEnd();
				glBeginEnded = true;
			}

			if ( tri->tex < 0 || model->tex[ tri->tex ].texImage < 0 )
			{
				if ( textured )
				{
					glPushAttrib( GL_TEXTURE_BIT | GL_ENABLE_BIT );
					glDisable( GL_TEXTURE_2D );
					textured = false;
				}
			}
			else
			{
				TM_TexEnv *tex = &model->tex[ tri->tex ];

				if ( !textured )
				{
					glPopAttrib();
					textured = true;
				}

				if ( tex->model == TM_TEX_MODULATE )
					glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE );
				else if ( tex->model == TM_TEX_DECAL )
					glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL );
				else if ( tex->model == TM_TEX_BLEND )
				{
					float blend[4];
					copyArray3( blend, tex->blend );
					blend[3] = 1.0;

					glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_BLEND );
					glTexEnvfv( GL_TEXTURE_ENV, GL_TEXTURE_ENV_COLOR, blend );
				}

				glBindTexture( GL_TEXTURE_2D, dlist->oglTexID[ tex->texImage ] );

				if ( tex->wrapS == TM_TEX_REPEAT )
					glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
				else 
					glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );

				if ( tex->wrapT == TM_TEX_REPEAT )
					glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
				else 
					glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );

				glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

				if ( mipmap )
					glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, 
									 GL_LINEAR_MIPMAP_LINEAR );
				else
					glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
			}

			currTexEnv = tri->tex;
		}

		// set material
		if ( tri->mat != currMat )
		{
			if ( !glBeginEnded )
			{	
				glEnd();
				glBeginEnded = true;
			}

			assert( tri->mat >= 0 );

			TM_Material *mat = &model->mat[ tri->mat ];

			float am[4], di[4], sp[4], em[4];
			copyArray3( am, mat->am );
			copyArray3( di, mat->di );
			copyArray3( sp, mat->sp );
			copyArray3( em, mat->em );
			am[3] = 1.0F - mat->tr;
			di[3] = 1.0F - mat->tr;
			sp[3] = 1.0F - mat->tr;
			em[3] = 1.0F - mat->tr;

			glColor4fv( di );

			glMaterialfv( GL_FRONT_AND_BACK, GL_AMBIENT, am );
			glMaterialfv( GL_FRONT_AND_BACK, GL_DIFFUSE, di );
			glMaterialfv( GL_FRONT_AND_BACK, GL_SPECULAR, sp );
			glMaterialfv( GL_FRONT_AND_BACK, GL_EMISSION, em );
			glMaterialf( GL_FRONT_AND_BACK, GL_SHININESS, 128.0 * mat->sh );

			currMat = tri->mat;
		}

		// draw triangle
		if ( glBeginEnded )
		{
			glBegin( GL_TRIANGLES );
			glBeginEnded = false;
		}

		for ( int j = 0; j < 3; j++ )
		{
			if ( textured ) glTexCoord2fv( tri->t[j] );
			glNormal3fv( tri->n[j] );
			glVertex3dv( tri->v[j] );
		}
	}

	if ( !glBeginEnded )
	{	
		glEnd();
		glBeginEnded = true;
	}

	if ( !textured )
	{
		glPopAttrib();
		textured = true;
	}
	
	glEndList();

	return dlist;
}



void TM_DeleteOGLDList( TM_DList *dlist )
{
	glDeleteLists( dlist->oglDList, 1 );
	glDeleteTextures( dlist->numTexIDs, dlist->oglTexID );
	free( dlist->oglTexID );
	free ( dlist );
}



void TM_DrawModel( const TM_DList *dlist, bool texMap, bool lighting, 
				   bool wireframe, bool smooth, bool cullBackFaces )
{
	glPushAttrib( GL_ALL_ATTRIB_BITS );

	glDisable( GL_DITHER );

	if ( cullBackFaces ) 
		glEnable( GL_CULL_FACE ); 
	else 
		glDisable( GL_CULL_FACE );

	if ( smooth ) 
		glShadeModel( GL_SMOOTH ); 
	else 
		glShadeModel( GL_FLAT );

	if ( texMap )
		glEnable( GL_TEXTURE_2D );
	else
		glDisable( GL_TEXTURE_2D );

	if ( lighting )
		glEnable( GL_LIGHTING );
	else
		glDisable( GL_LIGHTING );

	if ( wireframe )
		glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
	else
		glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );

	glCallList( dlist->oglDList );

	glPopAttrib();
}

