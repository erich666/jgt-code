/*****************************************************************************
 *	Date: January 3, 2006
 *  Date: September 2, 2008: commented out the question regarding the B-spline
 *        degree, since the CI CUDA code only implements the 3rd order.
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

#pragma warning (disable: 4996)

/*****************************************************************************
 *	System includes
 ****************************************************************************/
#include	<math.h>
#include	<stdlib.h>
#include	<stdio.h>
#include	<string.h>

/*****************************************************************************
 *	Other includes
 ****************************************************************************/
#include	"io.h"
 
/*****************************************************************************
 *	Definition of extern procedures
 ****************************************************************************/
/*--------------------------------------------------------------------------*/
extern int		ReadByteImageRawData
				(
					float	**Image,	/* output image data */
					long	*Width,		/* output image width */
					long	*Height		/* output image height */
				)

{ /* begin ReadByteImageRawData */

	char	Filename[256];
	FILE	*f = (FILE *)NULL;
	float	*p;
	unsigned char
			*Line;
	long	x, y;
	int		Error;

	/* interactivity */
	do {
		printf("Give the input image file name (enter ? to cancel):\n");
		printf("--> ");
		scanf("%255s", Filename);
		f = fopen(Filename, "rb");
	} while (strcmp(Filename, "?") && (f == (FILE *)NULL));
	if (!strcmp(Filename, "?")) {
		if (f != (FILE *)NULL) {
			fclose(f);
			printf("Sorry: reserved file name\n");
		}
		printf("Cancel\n");
		return(1);
	}
	do {
		printf("Give the input image width:\n");
		printf("--> ");
		scanf("%ld", Width);

		printf("Give the input image height:\n");
		printf("--> ");
		scanf("%ld", Height);
	} while ((*Width < 1L) || (*Height < 1L));

	/* allocation of workspace */
	*Image = (float *)malloc((size_t)(*Width * *Height * (long)sizeof(float)));
	if (*Image == (float *)NULL) {
		fclose(f);
		printf("Allocation of input image failed\n");
		return(1);
	}
	Line = (unsigned char *)malloc((size_t)(*Width * (long)sizeof(unsigned char)));
	if (Line == (unsigned char *)NULL) {
		free(*Image);
		*Image = (float *)NULL;
		fclose(f);
		printf("Allocation of buffer failed\n");
		return(1);
	}

	/* perform reading in raster fashion */
	p = *Image;
	for (y = 0L; y < *Height; y++) {
		Error = (*Width != (long)fread(Line, sizeof(unsigned char), (size_t)*Width, f));
		if (Error) {
			free(Line);
			free(*Image);
			*Image = (float *)NULL;
			fclose(f);
			printf("File access failed\n");
			return(1);
		}
		for (x = 0L; x < *Width; x++) {
			*p++ = (float)Line[x];
		}
	}

	free(Line);
	fclose(f);
	return(0);
} /* end ReadByteImageRawData */

/*--------------------------------------------------------------------------*/
extern void		RigidBody
				(
					double	*Angle,		/* output image rotation angle in degrees */
					double	*xShift,	/* output image horizontal shift */
					double	*yShift,	/* output image vertical shift */
					double	*xOrigin,	/* output origin of the x-axis */
					double	*yOrigin,	/* output origin of the y-axis */
					long	*Degree,	/* output degree of the B-spline model */
					int		*Masking	/* whether or not to mask the image */
				)

{ /* RigidBody */

	printf("Give the image origin of the x-axis:\n");
	printf("--> ");
	scanf("%lf", xOrigin);

	printf("Give the image origin of the y-axis:\n");
	printf("--> ");
	scanf("%lf", yOrigin);

	printf("Give the counter-clockwise image rotation angle in degrees:\n");
	printf("--> ");
	scanf("%lf", Angle);

	printf("Give the image horizontal translation in pixels:\n");
	printf("--> ");
	scanf("%lf", xShift);

	printf("Give the image vertical translation in pixels:\n");
	printf("--> ");
	scanf("%lf", yShift);
/* The CI CUDA code only implements the 3rd order B-spline
	do {
		printf("Give the degree of the B-spline model [2, 3, 4, 5, 6, 7, 8, 9]:\n");
		printf("--> ");
		scanf("%ld", Degree);
	} while ((*Degree < 2L) || (9L < *Degree));
*/
	do {
		printf("Do you want to mask out the irrelevant part of the image (0: no; 1: yes)?\n");
		printf("--> ");
		scanf("%ld", Masking);
	} while ((*Masking != 0) && (*Masking != 1));
} /* RigidBody */

/*--------------------------------------------------------------------------*/
extern int		WriteByteImageRawData
				(
					float	*Image,		/* input image data */
					long	Width,		/* input image width */
					long	Height		/* input image height */
				)

{ /* begin WriteByteImageRawData */

	char	Filename[256];
	FILE	*f = (FILE *)NULL;
	float	*p;
	unsigned char
			*Line;
	long	rounded;
	long	x, y;
	int		Error;

	/* interactivity */
	printf("Give the output image file name (enter ? to cancel):\n");
	printf("--> ");
	scanf("%255s", Filename);
	if (!strcmp(Filename, "?")) {
		printf("Cancel\n");
		return(1);
	}
	f = fopen(Filename, "wb");
	if (f == (FILE *)NULL) {
		printf("Failed to open the output file\n");
		return(1);
	}

	/* allocation of workspace */
	Line = (unsigned char *)malloc((size_t)(Width * (long)sizeof(unsigned char)));
	if (Line == (unsigned char *)NULL) {
		fclose(f);
		printf("Allocation of buffer failed\n");
		return(1);
	}

	/* perform writing in raster fashion */
	p = Image;
	for (y = 0L; y < Height; y++) {
		for (x = 0L; x < Width; x++) {
			rounded = (long)floor((double)*p++ + 0.5);
			Line[x] = (rounded < 0L) ? ((unsigned char)0)
				: ((255L < rounded) ? ((unsigned char)255) : ((unsigned char)rounded));
		}
		Error = (Width != (long)fwrite(Line, sizeof(unsigned char), (size_t)Width, f));
		if (Error) {
			free(Line);
			fclose(f);
			printf("File access failed\n");
			return(1);
		}
	}

	free(Line);
	fclose(f);
	return(0);
} /* end WriteByteImageRawData */
