/********************************************
** loadHDR.cpp                             **
** -----------                             **
**                                         **
** Loads an high dynamic range image using **
**    commands described in rgbe.cpp.      **
**                                         **
** Chris Wyman (9/07/2006)                 **
********************************************/

#include <stdio.h>
#include <stdlib.h>
#include "gfx.h"



void FatalHDRError( char *str, char *fname )
{
  printf( str, fname );
  exit(-1);
}


float ***loadHDR(char *filename, int *w, int *h)
{
  FILE *f;
  float *tmpbuf;
  float ***image;
  int width, height, i, j, x, y;

  /* open the file */
  f = fopen( filename, "rb" );
  if (!f) FatalHDRError( "Unable to load image file \"%s\" in loadHDR()!\n", filename ); 

  /* read in header information */
  if (RGBE_ReadHeader( f, &width, &height, NULL ) != RGBE_RETURN_SUCCESS)
    FatalHDRError( "Unable to read HDR header in image \"%s\"!\n", filename );

  /* allocate memory, both temporary & final */
  tmpbuf = (float *)malloc( 3 * width * height * sizeof(float) );
  image = (float ***)malloc( width * sizeof( float ** ) );
  if (!tmpbuf || !image) FatalHDRError( "Unable to allocate memory for image \"%s\"!\n", filename );
  for (i=0;i<width;i++)
    {
      image[i] = (float **)malloc( height * sizeof( float * ) );
      if (!image[i]) FatalHDRError( "Unable to allocate memory for image \"%s\"!\n", filename );
      for (j=0;j<height;j++)
	{
	  image[i][j] = (float *)malloc( 3 * sizeof( float ) );
	  if (!image[i][j]) FatalHDRError( "Unable to allocate memory for image \"%s\"!\n", filename );
	}
    }

  /* read in the image data */
  if (RGBE_ReadPixels_RLE(f, tmpbuf, width, height) != RGBE_RETURN_SUCCESS)
    FatalHDRError( "Unable to read HDR data in image \"%s\"!\n", filename );
  
  /* clse the file */
  fclose(f);
  
  /* transfer data into the permanent array */
  for (y=0; y<height; y++)
    for (x=0; x<width; x++)
      {
	float *ptr = tmpbuf+3*(x+y*width);
	image[x][y][0] = *ptr;
	image[x][y][1] = *(ptr+1);
	image[x][y][2] = *(ptr+2);
      }

  /* free temporary memory & return */
  free( tmpbuf );
  *w = width;
  *h = height;
  return image;
}



float *loadHDRtoArray(char *filename, int *w, int *h)
{
  FILE *f;
  float *tmpbuf;
  int width, height;

  /* open the file */
  f = fopen( filename, "rb" );
  if (!f) FatalHDRError( "Unable to load image file \"%s\" in loadHDR()!\n", filename ); 

  /* read in header information */
  if (RGBE_ReadHeader( f, &width, &height, NULL ) != RGBE_RETURN_SUCCESS)
    FatalHDRError( "Unable to read HDR header in image \"%s\"!\n", filename );

  /* allocate memory, both temporary & final */
  tmpbuf = (float *)malloc( 3 * width * height * sizeof(float) );
  if (!tmpbuf ) FatalHDRError( "Unable to allocate memory for image \"%s\"!\n", filename );

  /* read in the image data */
  if (RGBE_ReadPixels_RLE(f, tmpbuf, width, height) != RGBE_RETURN_SUCCESS)
    FatalHDRError( "Unable to read HDR data in image \"%s\"!\n", filename );
  
  /* clse the file */
  fclose(f);
 
  *w = width;
  *h = height;
  return tmpbuf;
}
