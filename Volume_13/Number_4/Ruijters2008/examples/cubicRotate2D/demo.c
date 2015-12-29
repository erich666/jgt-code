/*****************************************************************************
 *	Date: January 3, 2006
 *  Date: September 2, 2008: adapted the original program of Philippe Thevenaz
 *        to demonstrate CUDA based cubic B-spline interpolation.
 *----------------------------------------------------------------------------
 *	This C program is based on the following paper:
 *		P. Thevenaz, T. Blu, M. Unser, "Interpolation Revisited,"
 *		IEEE Transactions on Medical Imaging,
 *		vol. 19, no. 7, pp. 739-758, July 2000.
 *----------------------------------------------------------------------------
 *	EPFL/STI/IOA/LIB/BM.4.137
 *	Philippe Thevenaz
 *	Station 17
 *	CH-1015 Lausanne VD
 *----------------------------------------------------------------------------
 *	phone (CET):	+41(21)693.51.61
 *	fax:			+41(21)693.37.01
 *	RFC-822:		philippe.thevenaz@epfl.ch
 *	X-400:			/C=ch/A=400net/P=switch/O=epfl/S=thevenaz/G=philippe/
 *	URL:			http://bigwww.epfl.ch/
 *----------------------------------------------------------------------------
 *	This file is best viewed with 4-space tabs (the bars below should be aligned)
 *	|	|	|	|	|	|	|	|	|	|	|	|	|	|	|	|	|	|	|
 *  |...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|
 ****************************************************************************/

/*****************************************************************************
 *	System includes
 ****************************************************************************/
#include	<math.h>
#include	<stdio.h>
#include	<stdlib.h>

/*****************************************************************************
 *	Other includes
 ****************************************************************************/
#include	"io.h"

typedef unsigned int uint;
extern float* CopyVolumeHostToDevice(float* host, uint width, uint height, uint depth);
extern void CopyVolumeDeviceToHost(float* host, float* device, uint width, uint height, uint depth);
extern void CubicBSplinePrefilter2DTimer(float* image, uint width, uint height);
extern float* interpolate(uint width, uint height, double angle, double xShift, double yShift, double xOrigin, double yOrigin, int masking);
extern void initTexture(float* bsplineCoeffs, uint width, uint height);
 
/*****************************************************************************
 *	Defines
 ****************************************************************************/
#define		PI	((double)3.14159265358979323846264338327950288419716939937510)

/*****************************************************************************
 *	Definition of extern procedures
 ****************************************************************************/
/*--------------------------------------------------------------------------*/
extern int		main
				(
					void
				)

{ /* begin main */

	float   *bsplineCoeffs, *cudaOutput;
	float	*ImageRasterArray, *OutputImage;
	double	xOrigin, yOrigin;
	double	Angle, xShift, yShift;
	long	Width, Height;
	long	SplineDegree;
	int		Masking;
	int		Error;

	/* access data samples */
	Error = ReadByteImageRawData(&ImageRasterArray, &Width, &Height);
	if (Error) {
		printf("Failure to import image data\n");
		return(1);
	}

	/* ask for transformation parameters */
	RigidBody(&Angle, &xShift, &yShift, &xOrigin, &yOrigin, &SplineDegree, &Masking);

	/* allocate output image */
	OutputImage = (float *)malloc((size_t)(Width * Height * (long)sizeof(float)));
	if (OutputImage == (float *)NULL) {
		free(ImageRasterArray);
		printf("Allocation of output image failed\n");
		return(1);
	}

	/* convert between a representation based on image samples */
	/* and a representation based on image B-spline coefficients */
	bsplineCoeffs = CopyVolumeHostToDevice(ImageRasterArray, Width, Height, 1);
	CubicBSplinePrefilter2DTimer(bsplineCoeffs, Width, Height);
	initTexture(bsplineCoeffs, Width, Height);

	/* Call the CUDA kernel */
	cudaOutput = interpolate(Width, Height, Angle, xShift, yShift, xOrigin, yOrigin, Masking);
	CopyVolumeDeviceToHost(OutputImage, cudaOutput, Width, Height, 1);

	/* save output */
	Error = WriteByteImageRawData(OutputImage, Width, Height);
	if (Error) {
		free(OutputImage);
		free(ImageRasterArray);
		printf("Failure to export image data\n");
		return(1);
	}

	free(OutputImage);
	free(ImageRasterArray);
	printf("Done\n");
	return(0);
} /* end main */
