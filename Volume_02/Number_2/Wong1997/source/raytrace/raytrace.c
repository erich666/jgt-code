/*
 * raytrace.c
 *
 * Copyright (C) 1989, 1991, Craig E. Kolb
 * All rights reserved.
 *
 * This software may be freely copied, modified, and redistributed
 * provided that this copyright notice is preserved on all copies.
 *
 * You may not distribute this software, in whole or in part, as part of
 * any commercial product without the express consent of the authors.
 *
 * There is no warranty or other guarantee of fitness of this software
 * for any purpose.  It is provided solely "as is".
 *
 * $Id: raytrace.c,v 4.0.1.1 92/01/10 17:13:02 cek Exp Locker: cek $
 *
 * $Log:	raytrace.c,v $
 * Revision 4.0.1.1  92/01/10  17:13:02  cek
 * patch3: Made status report print actual scanline number.
 * 
 * Revision 4.0  91/07/17  14:50:49  kolb
 * Initial version.
 * 
 */

#include "rayshade.h"
#include "libsurf/atmosphere.h"
#include "libsurf/surface.h"
#include "libcommon/sampling.h"
#include "options.h"
#include "stats.h"
#include "raytrace.h"
#include "viewing.h"

#define UNSAMPLED	-1
#define SUPERSAMPLED	-2

typedef struct {
	Pixel	*pix;	/* Pixel values */
	int	*samp;	/* Sample number */
} Scanline;

static int		*SampleNumbers;
static void	RaytraceInit();

static Ray	TopRay;				/* Top-level ray. */
Float		SampleTime();

Pixel		WhitePix = {1., 1., 1., 1.},
		BlackPix = {0., 0., 0., 0.};

/*
 * "Dither matrices" used to encode the 'number' of a ray that samples a
 * particular portion of a pixel.  Hand-coding is ugly, but...
 */
static int OneSample[1] = 	{0};
static int TwoSamples[4] =	{0, 2,
				 3, 1};
static int ThreeSamples[9] =	{0, 2, 7,
				 6, 5, 1,
				 3, 8, 4};
static int FourSamples[16] =	{ 0,  8,  2, 10,
				 12,  4, 14,  6,
				  3, 11,  1,  9,
				 15,  7, 13,  5};
static int FiveSamples[25] =	{ 0,  8, 23, 17,  2,
				 19, 12,  4, 20, 15,
				  3, 21, 16,  9,  6,
				 14, 10, 24,  1, 13,
				 22,  7, 18, 11,  5};
static int SixSamples[36] =	{ 6, 32,  3, 34, 35,  1,
				  7, 11, 27, 28,  8, 30,
				 24, 14, 16, 15, 23, 19,
				 13, 20, 22, 21, 17, 18,
				 25, 29, 10,  9, 26, 12,
				 36,  5, 33,  4,  2, 31};
static int SevenSamples[49] =	{22, 47, 16, 41, 10, 35,  4,
				  5, 23, 48, 17, 42, 11, 29,
				 30,  6, 24, 49, 18, 36, 12,
				 13, 31,  7, 25, 43, 19, 37,
				 38, 14, 32,  1, 26, 44, 20,
				 21, 39,  8, 33,  2, 27, 45,
				 46, 15, 40,  9, 34,  3, 28};
static int EightSamples[64] =	{ 8, 58, 59,  5,  4, 62, 63,  1,
				 49, 15, 14, 52, 53, 11, 10, 56,
				 41, 23, 22, 44, 45, 19, 18, 48,
				 32, 34, 35, 29, 28, 38, 39, 25,
				 40, 26, 27, 37, 36, 30, 31, 33,
				 17, 47, 46, 20, 21, 43, 42, 24,
				  9, 55, 54, 12, 13, 51, 50, 16,
				 64,  2,  3, 61, 60,  6,  7, 57};

void	AdaptiveRefineScanline(), FullySamplePixel(), FullySampleScanline(),
	SingleSampleScanline();
static int	ExcessiveContrast();
static Scanline scan0, scan1, scan2;


void
raytrace(argc, argv)
int argc;
char **argv;
{
	int y, *tmpsamp;
	Pixel *tmppix;
	Float usertime, systime, lasttime;

	/*
	 * If this is the first frame,
	 * allocate scanlines, etc.
	 */
	if (Options.framenum == Options.startframe)
		RaytraceInit();
	/*
	 * The top-level ray TopRay always has as its origin the
	 * eye position and as its medium NULL, indicating that it
	 * is passing through a medium with index of refraction
	 * equal to DefIndex.
	 */
	TopRay.pos = Camera.pos;
	TopRay.media = (Medium *)0;
	TopRay.depth = 0;

	/*
	 * Always fully sample the bottom and top rows and the left
	 * and right column of pixels.  This minimizes artifacts that
	 * may arise when piecing together images.
	 */
	FullySampleScanline(0, &scan0);

	SingleSampleScanline(1, &scan1);
	FullySamplePixel(0, 1, &scan1.pix[0], &scan1.samp[0]);
	FullySamplePixel(Screen.xsize -1, 1, &scan1.pix[Screen.xsize -1],
		&scan1.samp[Screen.xsize -1]);

	lasttime = 0;
	for (y = 1; y < Screen.ysize; y++) {
		SingleSampleScanline(y+1, &scan2);
		FullySamplePixel(0, y+1, &scan2.pix[0], &scan2.samp[0]);
		FullySamplePixel(Screen.xsize -1, y+1,
			&scan2.pix[Screen.xsize -1],
			&scan2.samp[Screen.xsize -1]);

		if (Sampling.sidesamples > 1)
			AdaptiveRefineScanline(y,&scan0,&scan1,&scan2);

		PictureWriteLine(scan0.pix);

		tmppix = scan0.pix;
		tmpsamp = scan0.samp;
		scan0.pix = scan1.pix;
		scan0.samp = scan1.samp;
		scan1.pix = scan2.pix;
		scan1.samp = scan2.samp;
		scan2.pix = tmppix;
		scan2.samp = tmpsamp;

		if ((y+Screen.miny-1) % Options.report_freq == 0) {
			fprintf(Stats.fstats,"Finished line %d (%lu rays",
						y+Screen.miny-1,
						Stats.EyeRays);
			if (Options.verbose) {
				/*
				 * Report total CPU and split times.
				 */
				RSGetCpuTime(&usertime, &systime);
				fprintf(Stats.fstats,", %2.2f sec,",
						usertime+systime);
				fprintf(Stats.fstats," %2.2f split",
						usertime+systime-lasttime);
				lasttime = usertime+systime;
			}
			fprintf(Stats.fstats,")\n");
			(void)fflush(Stats.fstats);
		}

	}
	/*
	 * Supersample last scanline.
	 */
	for (y = 1; y < Screen.xsize -1; y++) {
		if (scan0.samp[y] != SUPERSAMPLED)
			FullySamplePixel(y, Screen.ysize -1,
				&scan0.pix[y],
				&scan0.samp[y]);
	}
	PictureWriteLine(scan0.pix);
}

void
SingleSampleScanline(line, data)
int line;
Scanline *data;
{
	Float upos, vpos, yp;
	int x, usamp, vsamp;
	Pixel tmp;

	yp = line + Screen.miny - 0.5*Sampling.filterwidth;
	for (x = 0; x < Screen.xsize; x++) {
		/*
		 * Pick a sample number...
		 */
		data->samp[x] = nrand() * Sampling.totsamples;
		/*
		 * Take sample corresponding to sample #.
		 */
		usamp = data->samp[x] % Sampling.sidesamples;
		vsamp = data->samp[x] / Sampling.sidesamples;

		vpos = yp + vsamp * Sampling.filterdelta;
		upos = x + Screen.minx - 0.5*Sampling.filterwidth +
				usamp*Sampling.filterdelta;
		if (Options.jitter) {
			vpos += nrand()*Sampling.filterdelta;
			upos += nrand()*Sampling.filterdelta;
		}
		TopRay.time = SampleTime(SampleNumbers[data->samp[x]]);
		SampleScreen(upos, vpos, &TopRay,
			&data->pix[x], SampleNumbers[data->samp[x]]);
		if (Options.samplemap)
			data->pix[x].alpha = 0;
	}
}

void
FullySampleScanline(line, data)
int line;
Scanline *data;
{
	int x;

	for (x = 0; x < Screen.xsize; x++) {
		data->samp[x] = UNSAMPLED;
		FullySamplePixel(x, line, &data->pix[x], &data->samp[x]);
	}
}

void
FullySamplePixel(xp, yp, pix, prevsamp)
int xp, yp;
Pixel *pix;
int *prevsamp;
{
	Float upos, vpos, u, v;
	int x, y, sampnum;
	Pixel ctmp;

	if (*prevsamp == SUPERSAMPLED)
		return;	/* already done */

	Stats.SuperSampled++;
	if (*prevsamp == UNSAMPLED) {
		/*
		 * No previous sample; initialize to black.
		 */
		pix->r = pix->g = pix->b = pix->alpha = 0.;
	} else {
		if (Sampling.sidesamples == 1) {
			*prevsamp = SUPERSAMPLED;
			return;
		}
		x = *prevsamp % Sampling.sidesamples;
		y = *prevsamp / Sampling.sidesamples;
		pix->r *= Sampling.filter[x][y];
		pix->g *= Sampling.filter[x][y];
		pix->b *= Sampling.filter[x][y];
		pix->alpha *= Sampling.filter[x][y];
	}

	sampnum = 0;
	xp += Screen.minx;
	vpos = Screen.miny + yp - 0.5*Sampling.filterwidth;
	for (y = 0; y < Sampling.sidesamples; y++,
	     vpos += Sampling.filterdelta) {
		upos = xp - 0.5*Sampling.filterwidth;
		for (x = 0; x < Sampling.sidesamples; x++,
		     upos += Sampling.filterdelta) {
			if (sampnum != *prevsamp) {
				if (Options.jitter) {
					u = upos + nrand()*Sampling.filterdelta;
					v = vpos + nrand()*Sampling.filterdelta;
				} else {
					u = upos;
					v = vpos;
				}
				TopRay.time = SampleTime(SampleNumbers[sampnum]);
				SampleScreen(u, v, &TopRay, &ctmp,
					SampleNumbers[sampnum]);
				pix->r += ctmp.r*Sampling.filter[x][y];
				pix->g += ctmp.g*Sampling.filter[x][y];
				pix->b += ctmp.b*Sampling.filter[x][y];
				pix->alpha += ctmp.alpha*Sampling.filter[x][y];
			}
			if (++sampnum == Sampling.totsamples)
				sampnum = 0;
		}
	}

	if (Options.samplemap)
		pix->alpha = 255;

	*prevsamp = SUPERSAMPLED;
}

void
AdaptiveRefineScanline(y, scan0, scan1, scan2)
int y;
Scanline *scan0, *scan1, *scan2;
{
	int x, done;

	/*
	 * Walk down scan1, looking at 4-neighbors for excessive contrast.
	 * If found, supersample *all* neighbors not already supersampled.
	 * The process is repeated until either there are no
	 * high-contrast regions or all such regions are already supersampled.
	 */

	do {
		done = TRUE;
		for (x = 1; x < Screen.xsize -1; x++) {
			/*
		 	 * Find min and max RGB for area we care about
			 */
			if (ExcessiveContrast(x, scan0->pix, scan1->pix,
			    scan2->pix)) {
				if (scan1->samp[x-1] != SUPERSAMPLED) {
					done = FALSE;
					FullySamplePixel(x-1, y,
						&scan1->pix[x-1],
						&scan1->samp[x-1]);
				}
				if (scan0->samp[x] != SUPERSAMPLED) {
					done = FALSE;
					FullySamplePixel(x, y-1,
						&scan0->pix[x],
						&scan0->samp[x]);
				}
				if (scan1->samp[x+1] != SUPERSAMPLED) {
					done = FALSE;
					FullySamplePixel(x+1, y,
						&scan1->pix[x+1],
						&scan1->samp[x+1]);
				}
				if (scan2->samp[x] != SUPERSAMPLED) {
					done = FALSE;
					FullySamplePixel(x, y+1,
						&scan2->pix[x],
						&scan2->samp[x]);
				}
				if (scan1->samp[x] != SUPERSAMPLED) {
					done = FALSE;
					FullySamplePixel(x, y,
						&scan1->pix[x],
						&scan1->samp[x]);
				}
			}
		}
	} while (!done);
}

static int
ExcessiveContrast(x, pix0, pix1, pix2)
int x;
Pixel *pix0, *pix1, *pix2;
{
	Float mini, maxi, sum, diff;

	maxi = max(pix0[x].r, pix1[x-1].r);
	if (pix1[x].r > maxi) maxi = pix1[x].r;
	if (pix1[x+1].r > maxi) maxi = pix1[x+1].r;
	if (pix2[x].r > maxi) maxi = pix2[x].r;

	mini = min(pix0[x].r, pix1[x-1].r);
	if (pix1[x].r < mini) mini = pix1[x].r;
	if (pix1[x+1].r < mini) mini = pix1[x+1].r;
	if (pix2[x].r < mini) mini = pix2[x].r;

	diff = maxi - mini;
	sum = maxi + mini;
	if (sum > EPSILON && diff/sum > Options.contrast.r)
		return TRUE;

	maxi = max(pix0[x].g, pix1[x-1].g);
	if (pix1[x].g > maxi) maxi = pix1[x].g;
	if (pix1[x+1].g > maxi) maxi = pix1[x+1].g;
	if (pix2[x].g > maxi) maxi = pix2[x].g;

	mini = min(pix0[x].g, pix1[x-1].g);
	if (pix1[x].g < mini) mini = pix1[x].g;
	if (pix1[x+1].g < mini) mini = pix1[x+1].g;
	if (pix2[x].g < mini) mini = pix2[x].g;

	diff = maxi - mini;
	sum = maxi + mini;

	if (sum > EPSILON && diff/sum > Options.contrast.g)
		return TRUE;

	maxi = max(pix0[x].b, pix1[x-1].b);
	if (pix1[x].b > maxi) maxi = pix1[x].b;
	if (pix1[x+1].b > maxi) maxi = pix1[x+1].b;
	if (pix2[x].b > maxi) maxi = pix2[x].b;

	mini = min(pix0[x].b, pix1[x-1].b);
	if (pix1[x].b < mini) mini = pix1[x].b;
	if (pix1[x+1].b < mini) mini = pix1[x+1].b;
	if (pix2[x].b < mini) mini = pix2[x].b;

	diff = maxi - mini;
	sum = maxi + mini;
	if (sum > EPSILON && diff/sum > Options.contrast.b)
		return TRUE;

	return FALSE;
}

Float
SampleTime(sampnum)
int sampnum;
{
	Float window, jitter = 0.0, res;

	if (Options.shutterspeed <= 0.)
		return Options.framestart;
	if (Options.jitter)
		jitter = nrand();
	window = Options.shutterspeed / Sampling.totsamples;
	res = Options.framestart + window * (sampnum + jitter);
	TimeSet(res);
	return res;
}

static void
RaytraceInit()
{

	switch (Sampling.sidesamples) {
		case 1:
			SampleNumbers = OneSample;
			break;
		case 2:
			SampleNumbers = TwoSamples;
			break;
		case 3:
			SampleNumbers = ThreeSamples;
			break;
		case 4:
			SampleNumbers = FourSamples;
			break;
		case 5:
			SampleNumbers = FiveSamples;
			break;
		case 6:
			SampleNumbers = SixSamples;
			break;
		case 7:
			SampleNumbers = SevenSamples;
			break;
		case 8:
			SampleNumbers = EightSamples;
			break;
		default:
			RLerror(RL_PANIC,
				"Sorry, %d rays/pixel not supported.\n",
					Sampling.totsamples);
	}

	/*
 	 * Allocate pixel arrays and arrays to store sampling info.
 	 */
	scan0.pix = (Pixel *)Malloc(Screen.xsize * sizeof(Pixel));
	scan1.pix = (Pixel *)Malloc(Screen.xsize * sizeof(Pixel));
	scan2.pix = (Pixel *)Malloc(Screen.xsize * sizeof(Pixel));

	scan0.samp = (int *)Malloc(Screen.xsize * sizeof(int));
	scan1.samp = (int *)Malloc(Screen.xsize * sizeof(int));
	scan2.samp = (int *)Malloc(Screen.xsize * sizeof(int));
}
