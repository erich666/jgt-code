/*
 *
 * RayTrace Software Package, release 1.0.3,  July 2003.
 *
 * Author: Samuel R. Buss
 *
 * Software accompanying the book
 *		3D Computer Graphics: A Mathematical Introduction with OpenGL,
 *		by S. Buss, Cambridge University Press, 2003.
 *
 * Software is "as-is" and carries no warranty.  It may be used without
 *   restriction, but if you modify it, please change the filenames to
 *   prevent confusion between different versions.  Please acknowledge
 *   all use of the software in any publications or products based on it.
 *
 * Bug reports: Sam Buss, sbuss@ucsd.edu.
 * Web page: http://math.ucsd.edu/~sbuss/MathCG
 *
 */

#ifndef RGBIMAGE_H
#define RGBIMAGE_H

#include <stdio.h>
#include <assert.h>

// Comment in the next line to turn off the routines that use OpenGL
// #define RGBIMAGE_DONT_USE_OPENGL

class RgbImage
{
public:
	RgbImage();
	RgbImage( const char* filename );
	RgbImage( int numRows, int numCols );	// Initialize a blank bitmap of this size.
	~RgbImage();

	bool LoadBmpFile( const char *filename );		// Loads the bitmap from the specified file
	bool WriteBmpFile( const char* filename );		// Write the bitmap to the specified file
#ifndef RGBIMAGE_DONT_USE_OPENGL
	bool LoadFromOpenglBuffer();					// Load the bitmap from the current OpenGL buffer
#endif

	long GetNumRows() const { return NumRows; }
	long GetNumCols() const { return NumCols; }
	// Rows are word aligned
	long GetNumBytesPerRow() const { return ((3*NumCols+3)>>2)<<2; }	
	const void* ImageData() const { return (void*)ImagePtr; }

	const unsigned char* GetRgbPixel( long row, long col ) const;
	unsigned char* GetRgbPixel( long row, long col );
	void GetRgbPixel( long row, long col, float* red, float* green, float* blue ) const;
	void GetRgbPixel( long row, long col, double* red, double* green, double* blue ) const;

	void SetRgbPixelf( long row, long col, double red, double green, double blue );
	void SetRgbPixelc( long row, long col, 
					   unsigned char red, unsigned char green, unsigned char blue );

	// Error reporting. (errors also print message to stderr)
	int GetErrorCode() const { return ErrorCode; }
	enum {
		NoError = 0,
		OpenError = 1,			// Unable to open file for reading
		FileFormatError = 2,	// Not recognized as a 24 bit BMP file
		MemoryError = 3,		// Unable to allocate memory for image data
		ReadError = 4,			// End of file reached prematurely
		WriteError = 5			// Unable to write out data (or no date to write out)
	};
	bool ImageLoaded() const { return (ImagePtr!=0); }  // Is an image loaded?

	void Reset();			// Frees image data memory

private:
	unsigned char* ImagePtr;	// array of pixel values (integers range 0 to 255)
	long NumRows;				// number of rows in image
	long NumCols;				// number of columns in image
	int ErrorCode;				// error code

	static short readShort( FILE* infile );
	static long readLong( FILE* infile );
	static void skipChars( FILE* infile, int numChars );
	static void RgbImage::writeLong( long data, FILE* outfile );
	static void RgbImage::writeShort( short data, FILE* outfile );
	
	static unsigned char doubleToUnsignedChar( double x );

};

inline RgbImage::RgbImage()
{ 
	NumRows = 0;
	NumCols = 0;
	ImagePtr = 0;
	ErrorCode = 0;
}

inline RgbImage::RgbImage( const char* filename )
{
	NumRows = 0;
	NumCols = 0;
	ImagePtr = 0;
	ErrorCode = 0;
	LoadBmpFile( filename );
}

inline RgbImage::~RgbImage()
{ 
	delete[] ImagePtr;
}

// Returned value points to three "unsigned char" values for R,G,B
inline const unsigned char* RgbImage::GetRgbPixel( long row, long col ) const
{
	assert ( row<NumRows && col<NumCols );
	const unsigned char* ret = ImagePtr;
	long i = row*GetNumBytesPerRow() + 3*col;
	ret += i;
	return ret;
}

inline unsigned char* RgbImage::GetRgbPixel( long row, long col ) 
{
	assert ( row<NumRows && col<NumCols );
	unsigned char* ret = ImagePtr;
	long i = row*GetNumBytesPerRow() + 3*col;
	ret += i;
	return ret;
}

inline void RgbImage::GetRgbPixel( long row, long col, float* red, float* green, float* blue ) const
{
	assert ( row<NumRows && col<NumCols );
	const unsigned char* thePixel = GetRgbPixel( row, col );
	const float f = 1.0f/255.0f;
	*red = f*(float)(*(thePixel++));
	*green = f*(float)(*(thePixel++));
	*blue = f*(float)(*thePixel);
}

inline void RgbImage::GetRgbPixel( long row, long col, double* red, double* green, double* blue ) const
{
	assert ( row<NumRows && col<NumCols );
	const unsigned char* thePixel = GetRgbPixel( row, col );
	const double f = 1.0/255.0;
	*red = f*(double)(*(thePixel++));
	*green = f*(double)(*(thePixel++));
	*blue = f*(double)(*thePixel);
}

inline void RgbImage::Reset()
{
	NumRows = 0;
	NumCols = 0;
	delete[] ImagePtr;
	ImagePtr = 0;
	ErrorCode = 0;
}


#endif // RGBIMAGE_H
