#ifndef _TRIMODEL_H_
#define _TRIMODEL_H_

#include <GL/glut.h>
#include "gltx.h"

#define	TM_TEX_REPEAT   0
#define TM_TEX_CLAMP    1
#define TM_TEX_MODULATE 0
#define TM_TEX_DECAL    1
#define TM_TEX_BLEND    2


typedef struct TM_Material {
    float am[3];		// ambient RGB
    float di[3];		// diffuse RGB
    float sp[3];		// specular RGB
    float em[3];		// emission RGB
    float sh;			// shininess (0=rough, 1=shiny)
	float tr;			// transparency (0=opaque, 1=transparent)
	TM_Material *next;  // used only when in linked-list
}
TM_Material;


typedef struct TM_TexEnv {
	int   texImage;		// index to array of texture images (-1 = no texture)
	int	  wrapS;		// 0=REPEAT or 1=CLAMP
	int   wrapT;		// 0=REPEAT or 1=CLAMP
	int   model;		// 0=MODULATE, 1=DECAL or 2=BLEND
	float blend[3];		// Blend color RGB
	TM_TexEnv *next;	// used only when in linked-list
}
TM_TexEnv;


typedef struct TM_Triangle {
	double v[3][3];		// 3 3D vertex coordinates
	float  n[3][3];		// 3 vertex normals
	float  t[3][2];		// 3 texture coordinates
	int    mat;			// index to array of materials (>=0)
	int    tex;			// index to array of texture-environments (-1 = no texture)
	TM_Triangle *next;  // used only when in linked-list
}
TM_Triangle;


typedef struct TM_Model {
	int numTris;		// number of triangles
	TM_Triangle *tri;	// array of triangles
	int numMats;		// number of materials	
	TM_Material *mat;	// array of materials
	int numTexEnvs;		// number of texture-environments
	TM_TexEnv *tex;		// array of texture-environments
	int numTexImages;	// number of texture images
	GLTXimage **image;	// array of pointers to RGB image

	// some model's stats
	double min_xyz[3];	// corner of bounding box with minimum x, y, z
	double max_xyz[3];	// corner of bounding box with maximum x, y, z
	double dim_xyz[3];	// dimensions of bounding box in x, y, z
	double center[3];	// center of bounding box
}
TM_Model;


typedef struct TM_DList {
	// OpenGL display list
	int numTexIDs;		// number of OpenGL texture objects
	GLuint *oglTexID;	// array of OpenGL texture objects' IDs
	GLuint oglDList;	// OpenGL display list ID
}
TM_DList;


extern TM_Model *TM_ReadModel( const char *pathfile, double scaleFact );

extern void      TM_DeleteModel( TM_Model *model );

extern TM_DList *TM_MakeOGLDList( const TM_Model *model, bool mipmap );
	// Call this only after OpenGL has been initialized.

extern void      TM_DeleteOGLDList( TM_DList *dlist );
	// Call this only after OpenGL has been initialized.

extern void      TM_DrawModel( const TM_DList *dlist, bool texMap, bool lighting, 
						       bool wireframe, bool smooth, bool cullBackFaces );
	// Call this only after OpenGL has been initialized.

#endif
