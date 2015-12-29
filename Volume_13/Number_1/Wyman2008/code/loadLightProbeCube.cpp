/********************************************
** loadLightProbeCube.cpp                  **
** -----------                             **
**                                         **
** Code to load a light probe stored as a  **
**    cross into a cubemap.  Another func  **
**    in here loads a DDS cubemap, using   **
**    code borrowed from an nVidia Demo.   **
**                                         **
** Chris Wyman (9/07/2006)                 **
********************************************/

#include <stdio.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include "nv_dds.h"

#define MIN(x,y)    ((x)<(y)?(x):(y))

using namespace nv_dds;

float *loadHDRtoArray(char *filename, int *w, int *h);
void FatalHDRError( char *str, char *fname );

int IsPowerOf2( int val )
{
	if (val == 0 || val == 1) return 1;
	if (val % 2 != 0) return 0;
	return IsPowerOf2( val / 2 );
}

GLuint LoadLightProbeCube( char *filename, int *mapW, int *mapH )
{
	int width, height, i, j, count;
	int tmp;
	float *array = loadHDRtoArray( filename, &width, &height );
	float *posX, *negX, *posY, *negY, *posZ, *negZ;
	GLuint textureID;

	/* check to make sure the height is 4*2^n and the width is 3*2^n */
	tmp = height / 4;
	if (tmp*4 != height)
		FatalHDRError( "Error: Cube map file '%s' has incorrect format!\n", filename ); 
	if (tmp*3 != width)
		FatalHDRError( "Error: Cube map file '%s' has incorrect format!\n", filename ); 
	if (!IsPowerOf2( tmp ))
		FatalHDRError( "Error: Cube map faces in file '%s' are not powers of 2!\n", filename );

	/* pass out the width and height of our cubemap faces */
	*mapW = tmp;
	*mapH = tmp;

	/* allocate memory for our cube map faces */
	posX = (float *)malloc( 3 * tmp * tmp * sizeof( float ) );
	negX = (float *)malloc( 3 * tmp * tmp * sizeof( float ) );
	posY = (float *)malloc( 3 * tmp * tmp * sizeof( float ) );
	negY = (float *)malloc( 3 * tmp * tmp * sizeof( float ) );
	posZ = (float *)malloc( 3 * tmp * tmp * sizeof( float ) );
	negZ = (float *)malloc( 3 * tmp * tmp * sizeof( float ) );

	/* make sure the memory was available */
	if (!posX || !posY || !posZ || !negX || !negY || !negZ)
		FatalHDRError( "Error: Unable to allocate memory while loading '%s'!\n", filename ); 

	/* transfer cube faces to separate arrays */
	count = 0; 
	for (i=tmp;i<2*tmp;i++)
		for (j=0;j<tmp;j++)
		{
			posY[count++] = MIN( array[3* (j*width + i) + 0], 1);
			posY[count++] = MIN( array[3* (j*width + i) + 1], 1);
			posY[count++] = MIN( array[3* (j*width + i) + 2], 1);
		}

	count = 0;
	for (j=tmp;j<2*tmp;j++)
		for (i=tmp-1;i>=0;i--)
		{
			negZ[count++] = MIN( array[3* (j*width + i) + 0], 1);
			negZ[count++] = MIN( array[3* (j*width + i) + 1], 1);
			negZ[count++] = MIN( array[3* (j*width + i) + 2], 1);
		}

	count = 0;
	for (j=tmp;j<2*tmp;j++)
		for (i=2*tmp-1;i>=tmp;i--)
		{
			posX[count++] = MIN( array[3* (j*width + i) + 0], 1);
			posX[count++] = MIN( array[3* (j*width + i) + 1], 1);
			posX[count++] = MIN( array[3* (j*width + i) + 2], 1);		
		}

	count = 0;
	for (j=tmp;j<2*tmp;j++)
		for (i=3*tmp-1;i>=2*tmp;i--)
		{
			posZ[count++] = MIN( array[3* (j*width + i) + 0], 1);
			posZ[count++] = MIN( array[3* (j*width + i) + 1], 1);
			posZ[count++] = MIN( array[3* (j*width + i) + 2], 1);
		}

	count = 0;
	for (i=2*tmp-1;i>=tmp;i--)
		for (j=3*tmp-1;j>=2*tmp;j--)
		{
			negY[count++] = MIN( array[3* (j*width + i) + 0], 1);
			negY[count++] = MIN( array[3* (j*width + i) + 1], 1);
			negY[count++] = MIN( array[3* (j*width + i) + 2], 1);
		}

	count = 0;
	for (j=4*tmp-1;j>=3*tmp;j--)
		for (i=tmp;i<2*tmp;i++)
		{
			negX[count++] = MIN( array[3* (j*width + i) + 0], 1);
			negX[count++] = MIN( array[3* (j*width + i) + 1], 1);
			negX[count++] = MIN( array[3* (j*width + i) + 2], 1);
		}

	/* free memory from the HDR image now that we have the cube faces saved */
	free( array );

	/* setup the cube map texture */
	glPixelStorei( GL_UNPACK_ALIGNMENT, 1 );
	glGenTextures( 1, &textureID );
	glBindTexture( GL_TEXTURE_CUBE_MAP, textureID );
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, GL_RGBA, tmp, tmp, 0, GL_RGB, GL_FLOAT, posX );
	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 0, GL_RGBA, tmp, tmp, 0, GL_RGB, GL_FLOAT, posY );
	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 0, GL_RGBA, tmp, tmp, 0, GL_RGB, GL_FLOAT, posZ );
	glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 0, GL_RGBA, tmp, tmp, 0, GL_RGB, GL_FLOAT, negX );
	glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, GL_RGBA, tmp, tmp, 0, GL_RGB, GL_FLOAT, negY );
	glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, GL_RGBA, tmp, tmp, 0, GL_RGB, GL_FLOAT, negZ );

	/* return the ID */
	return textureID;
}



GLuint LoadDDSCube( char *filename, int *mapW, int *mapH )
{
	CDDSImage image;
	GLuint texobj;
	 
	if (!image.load(filename, false))
		FatalHDRError( "Error: Unable to load '%s'!\n", filename );

	glGenTextures(1, &texobj);
	glEnable(GL_TEXTURE_CUBE_MAP);
	glBindTexture(GL_TEXTURE_CUBE_MAP, texobj);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glDisable( GL_TEXTURE_CUBE_MAP );

	*mapW = image.get_width();
	*mapH = image.get_height();

	image.upload_textureCubemap();
	return texobj;
}