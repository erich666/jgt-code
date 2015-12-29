/***************************************
** loadifs.c                          **
** -----------                        **
**                                    **
** Loads an IFS file.  These files    **
**   come from the Brown Mesh Set     **
**   and are used by the G3D library. **
** This code is for basic input ONLY. **
**                                    **
** Assumes a little-endian machine    **
**   32-bit floats and ints, to fix   **
**   these assumptions, fix the read  **
**   functions at the beginning of    **
**   the code.                        **
**                                    **
** Chris Wyman (6/30/2005)            **
***************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "glut_template.h"
#include "glm.h"

unsigned int readUInt32( FILE *f )
{
	char tmp[4];
	tmp[0] = fgetc( f );
	tmp[1] = fgetc( f );
	tmp[2] = fgetc( f );
	tmp[3] = fgetc( f );
	return *((unsigned int*)tmp);
}

int readInt32( FILE *f )
{
	unsigned int tmp = readUInt32( f );
	return *(int *)&tmp;
}

float readFloat32( FILE *f )
{
	unsigned int tmp = readUInt32( f );
	return *(float *)&tmp;
}


GLMmodel *glmReadIFS( char *filename )
{
  GLMmodel*    model;
  FILE*        file;
  unsigned int i;
  char         tmpStr[1024];
  unsigned int tmpStrLen;

  /* open the file */
  file = fopen(filename, "rb");
  if (!file)
  {
	  char buf[1024];
	  sprintf( buf, "Unable to open file '%s'\n", filename );
	  FatalError( buf );
  }

  /* read the header */
  tmpStrLen = readUInt32( file );
  fgets( tmpStr, tmpStrLen+1, file );
  if ( strcmp(tmpStr, "IFS") )
  {
      char buf[1024];
	  sprintf( buf, "'%s' does not appear to be an IFS file!\n", filename );
	  FatalError( buf );
  }
  if ( readFloat32( file ) != 1.0 )
  {
	  char buf[1024];
	  sprintf( buf, "'%s' appears to use a newer IFS file format!\n", filename );
	  Warning( buf );
  }

  /* read in the model name */
  tmpStrLen = readUInt32( file );
  fgets( tmpStr, tmpStrLen+1, file );

  /* allocate a new model */
  model = (GLMmodel*)malloc(sizeof(GLMmodel));  
  assert(model);

  /* make sure we initialize it as expected */
  model->pathname      = (char *)strdup(filename);
  model->mtllibname    = NULL;
  model->numvertices   = 0;
  model->vertices      = NULL;
  model->numnormals    = 0;
  model->normals       = NULL;
  model->numtexcoords  = 0;
  model->texcoords     = NULL;
  model->numfacetnorms = 0;
  model->facetnorms    = NULL;
  model->numtriangles  = 0;
  model->triangles     = NULL;
  model->nummaterials  = 0;
  model->materials     = NULL;
  model->numgroups     = 0;
  model->groups        = NULL;
  model->position[0]   = 0.0;
  model->position[1]   = 0.0;
  model->position[2]   = 0.0;
  
  /* make a default group for all triangles */
  glmAddGroup(model, tmpStr);
    
  /* read vertex header */
  tmpStrLen = readUInt32( file );
  fgets( tmpStr, tmpStrLen+1, file );
  model->numvertices = readUInt32( file );
  // Here tmpStr == "VERTICES" (or it should... could add corruption check here)

  /* allocate and read vertices */
  model->vertices = (GLfloat*)malloc(sizeof(GLfloat) * 3 * (model->numvertices + 1));
  for (i = 1; i <= model->numvertices; i++)
  {
     model->vertices[3*i+0] = readFloat32( file );
	 model->vertices[3*i+1] = readFloat32( file );
	 model->vertices[3*i+2] = readFloat32( file );
  }

  /* read triangle header */
  tmpStrLen = readUInt32( file );
  fgets( tmpStr, tmpStrLen+1, file );
  model->numtriangles = readUInt32( file );
  // Here tmpStr == "TRIANGLES" (or it should... could add corruption check here)

  /* allocate and read triangles */
  model->triangles = (GLMtriangle*)malloc(sizeof(GLMtriangle) *
        model->numtriangles);
  model->groups->triangles = (GLuint*)malloc(sizeof(GLuint) * 
	    model->numtriangles);
  for (i=0; i < model->numtriangles; i++)
  {
	  model->triangles[i].vindices[0] = readUInt32( file ) + 1;
	  model->triangles[i].vindices[1] = readUInt32( file ) + 1;
	  model->triangles[i].vindices[2] = readUInt32( file ) + 1;
	  model->groups->triangles[model->groups->numtriangles++] = i;
  }

  /* close the file */
  fclose(file);

  /* compute facet normals */
  glmFacetNormals( model );

  /* return the model */
  return model;
}
