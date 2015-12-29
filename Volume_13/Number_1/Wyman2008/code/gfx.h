/**********************************
** ppm.h                         **
** -----                         **
**                               **
** Header required for the PPM   **
**   I/O files (input_ppm.c and  **
**   write_ppm.c)                **
**                               **
** Chris Wyman (9/28/2000)       **
**********************************/

#ifdef __cplusplus
extern "C"
{
#endif

#ifndef __PPM_H__
#define __PPM_H__

/* some constants we use.  Use same val.s as GL to be consistant */
#ifndef __gl_h_
#define GL_UNSIGNED_BYTE                    0x1401
#define GL_FLOAT                            0x1406
#define GL_RGB                              0x1907
#define GL_RGBA                             0x1908
#endif

#define __PPM_TEXCOLOR(t,i,j) (((t)->texImage)+4*((i)+(j)*((t)->img_width)))

/* types of images we can read/write...
**
** NOTE: the PBMs are written in a very
**       primitive way, so if the input 
**       isn't a B&W image, the result
**       will be *very* distorted
*/
#define PPM_ASCII    3
#define PPM_RAW      6
#define PGM_ASCII    2
#define PGM_RAW      5
#define PBM_ASCII    1
#define PBM_RAW      4

#define RGBE_IMAGE   10

/* info about types of images */
#define PPM_MAX      7


/* define return codes */
#define GFXIO_OK            0
#define GFXIO_OPENERROR     1
#define GFXIO_BADFILE       2
#define GFXIO_UNSUPPORTED   3

/* image types */
#define GFXIO_UBYTE    GL_UNSIGNED_BYTE
#define GFXIO_FLOAT    GL_FLOAT     

/* image formats */
#define GFXIO_RGB      GL_RGB
#define GFXIO_RGBA     GL_RGBA


/* 
** 'texture' structure, what images are read into
** or written from
*/
typedef struct {
  int img_width, img_height;
  char *texImage;
  float *texImagef;
  int type;
  int format;
} texture;


/* function prototypes to read into texture structures */
texture *ReadPPMTexture( char * );
texture *ReadRGBETexture( char *infilename );

/* write directly from a stream of either chars or floats */
int WriteImage( int mode, char *f, char *ptr, int width, int height );
int WriteImageFloat( int mode, char *f, float *ptr, int width, int height );
int WriteTextureToImage( int mode, char *f, texture *T );
int WriteGammaPGMFloat( char *f, float *ptr, int width, int height, double gamma );
int WriteGammaImageFloat( int mode, char *f, float *ptr, int width, int height, double gamma );
int WriteRGBEImage( char *f, float *ptr, int width, int height );

/* misc, error functions */
void FatalError( char *msg );
void Error( char *msg );
void Warning( char *msg );

/* conversion functions */
char *ConvertUByteRGBAtoRGB( char *s, int pix );
char *ConvertUByteRGBtoRGBA( char *s, int pix );
float *ConvertFloatRGBAtoRGB( float *s, int pix );
float *ConvertFloatRGBtoRGBA( float *s, int pix );
char *ConvertFLOATtoUBYTE( float *s, int len );
float *ConvertUBTYEtoFLOAT( char *s, int len );

texture *ConvertTextureToRGB( texture *t );
texture *ConvertTextureToRGBA( texture *t );
texture *ConvertTextureToFloat( texture *t );
texture *ConvertTextureToUByte( texture *t );

/* 
** a function to determine if a particular
** integer number corresponds to a type of
** image we can write.
*/
int IsValidMode( int mode );

/* determints if a mode is a RAW or ASCII mode */
int ASCIIMode( int mode );
int RawMode( int mode );

#endif








/********************************************************************
** RGBE header...  from www.graphics.cornell.edu/~bjw/rgbe/rgbe.h  **
********************************************************************/

#ifndef _H_RGBE
#define _H_RGBE
/* THIS CODE CARRIES NO GUARANTEE OF USABILITY OR FITNESS FOR ANY PURPOSE.
 * WHILE THE AUTHORS HAVE TRIED TO ENSURE THE PROGRAM WORKS CORRECTLY,
 * IT IS STRICTLY USE AT YOUR OWN RISK.  */

/* utility for reading and writing Ward's rgbe image format.
   See rgbe.txt file for more details.
*/

#include <stdio.h>

typedef struct {
  int valid;            /* indicate which fields are valid */
  char programtype[16]; /* listed at beginning of file to identify it 
                         * after "#?".  defaults to "RGBE" */ 
  float gamma;          /* image has already been gamma corrected with 
                         * given gamma.  defaults to 1.0 (no correction) */
  float exposure;       /* a value of 1.0 in an image corresponds to
                         * <exposure> watts/steradian/m^2. 
                         * defaults to 1.0 */
} rgbe_header_info;

/* flags indicating which fields in an rgbe_header_info are valid */
#define RGBE_VALID_PROGRAMTYPE 0x01
#define RGBE_VALID_GAMMA       0x02
#define RGBE_VALID_EXPOSURE    0x04

/* return codes for rgbe routines */
#define RGBE_RETURN_SUCCESS 0
#define RGBE_RETURN_FAILURE -1

/* read or write headers */
/* you may set rgbe_header_info to null if you want to */
int RGBE_WriteHeader(FILE *fp, int width, int height, rgbe_header_info *info);
int RGBE_ReadHeader(FILE *fp, int *width, int *height, rgbe_header_info *info);

/* read or write pixels */
/* can read or write pixels in chunks of any size including single pixels*/
int RGBE_WritePixels(FILE *fp, float *data, int numpixels);
int RGBE_ReadPixels(FILE *fp, float *data, int numpixels);

/* read or write run length encoded files */
/* must be called to read or write whole scanlines */
int RGBE_WritePixels_RLE(FILE *fp, float *data, int scanline_width,
                         int num_scanlines);
int RGBE_ReadPixels_RLE(FILE *fp, float *data, int scanline_width,
                        int num_scanlines);

#endif /* _H_RGBE */




#ifdef __cplusplus
}
#endif
