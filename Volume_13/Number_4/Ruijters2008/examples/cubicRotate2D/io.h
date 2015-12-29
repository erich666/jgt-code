/*****************************************************************************
 *	Date: January 3, 2006
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

/*--------------------------------------------------------------------------*/
extern int		ReadByteImageRawData
				(
					float	**Image,	/* output image data */
					long	*Width,		/* output image width */
					long	*Height		/* output image height */
				);

/*--------------------------------------------------------------------------*/
extern void		RigidBody
				(
					double	*Angle,		/* output image rotation */
					double	*xShift,	/* output image horizontal shift */
					double	*yShift,	/* output image vertical shift */
					double	*xOrigin,	/* output origin of the x-axis */
					double	*yOrigin,	/* output origin of the y-axis */
					long	*Degree,	/* output degree of the B-spline model */
					int		*Masking	/* whether or not to mask the image */
				);

/*--------------------------------------------------------------------------*/
extern int		WriteByteImageRawData
				(
					float	*Image,		/* input image data */
					long	Width,		/* input image width */
					long	Height		/* input image height */
				);
