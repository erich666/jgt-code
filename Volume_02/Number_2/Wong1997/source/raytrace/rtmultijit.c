/*
 * rtmultijit.c
 * 
 * This code is written to test the effect of multijitter sampling in 
 * ray tracing.
 * For simplicity, the number of cell along x and y are hardcoded by
 * macros CELLX and CELLY. To change the value, you have to recompile
 * the program. This code is designed to substitute for the original 
 * file raytrace.c
 * Function "raytrace()" is expected to be called.
 * 
 */

#include <values.h>
#include "rayshade.h"
#include "libsurf/atmosphere.h"
#include "libsurf/surface.h"
#include "libcommon/sampling.h"
#include "options.h"
#include "stats.h"
#include "raytrace.h"
#include "viewing.h"

#define CELLX 4
#define CELLY 4

#define RAN_DOUBLE(l, h)    (((double) random()/0x80000000U)*((h) - (l)) + (l))
#define RAN_INT(l, h)	    ((int) (RAN_DOUBLE(0, (h)-(l)+1) + (l)))


Pixel	*pix;	/* Pixel Buffer */
static Ray	TopRay;				/* Top-level ray. */
Pixel		WhitePix = {1., 1., 1., 1.},
		BlackPix = {0., 0., 0., 0.};

/* Function Prototypes */
Float SampleTime(int sampnum);
void RaytraceInit();


/* Implementation */
/* 
 * Modified from the C code from the article "Multi-Jittered Sampling"
 * by Kenneth Chiu, Peter Shirley, and Changyaw Wang,
 * (chiuk@cs.indiana.edu, shirley@iuvax.cs.indiana.edu,
 * and wangc@iuvax.cs.indiana.edu)
 * in "Graphics Gems IV", Academic Press, 1994
 *
 * MultiJitter() takes an array of Point2's and the dimension, and fills the
 * the array with the generated sample points.
 *
 *    p[] must have length m*n.
 *    m is the number of columns of cells.
 *    n is the number of rows of cells.
 */
void multijitter(int m, int n, Float *p) 
{
  double subcell_width;
  double t;
  int k;
  int i, j;

  subcell_width = 1.0/(m*n);

  /* Initialize points to the "canonical" multi-jittered pattern. */
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      p[(i*n + j)<<1] = i*n*subcell_width + j*subcell_width
	     	     + RAN_DOUBLE(0, subcell_width);
      p[((i*n + j)<<1)+1] = j*m*subcell_width + i*subcell_width
	             + RAN_DOUBLE(0, subcell_width);
    }
  }

  /* Shuffle x coordinates within each column of cells. */
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      k = RAN_INT(j, n - 1);
      t = p[(i*n + j)<<1];
      p[(i*n + j)<<1] = p[(i*n + k)<<1];
      p[(i*n + k)<<1] = t;
    }
  }

  /* Shuffle y coordinates within each row of cells. */
  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      k = RAN_INT(j, m - 1);
      t = p[((j*n + i)<<1)+1];
      p[((j*n + i)<<1)+1] = p[((k*n + i)<<1)+1];
      p[((k*n + i)<<1)+1] = t;
    }
  }
}




void raytrace(int argc, char **argv)
{
  int x, y, *tmpsamp, i, sampleno=CELLX*CELLY;
  Pixel *tmppix;
  Float usertime, systime, lasttime, upos, vpos;
  Float samples[CELLX*CELLY*2];
  Pixel acc, tmp;

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

  lasttime = 0;
  for (y=0; y < Screen.ysize; y++) {

	for (x = 0; x < Screen.xsize; x++) {
		multijitter(CELLX, CELLY, samples);
		acc.r = acc.g = acc.b = acc.alpha = 0;
		for (i=0 ; i<sampleno ; i++) {
	   		vpos = y + Screen.miny + samples[(i<<1)+1];
			upos = x + Screen.minx + samples[i<<1];
			TopRay.time = SampleTime(0);
			SampleScreen(upos, vpos, &TopRay, &tmp, 0);
			acc.r += tmp.r;
			acc.g += tmp.g;
			acc.b += tmp.b;
		}
		pix[x].r = acc.r/sampleno; /* box filter, or average */
		pix[x].g = acc.g/sampleno; 
		pix[x].b = acc.b/sampleno; 
		pix[x].alpha = 0;
	}
  
  	PictureWriteLine(pix); /* write the scanline */

	if ((y+Screen.miny-1) % Options.report_freq == 0) {
		fprintf(Stats.fstats,"Finished line %d (%lu rays",
					y+Screen.miny-1,
					Stats.EyeRays);
		if (Options.verbose) {
			/*
			* Report total CPU and split times.
			*/
			RSGetCpuTime(&usertime, &systime);
			fprintf(Stats.fstats,", %2.2f sec,", usertime+systime);
			fprintf(Stats.fstats," %2.2f split", usertime+systime-lasttime);
			lasttime = usertime+systime;
		}
	}
	fprintf(Stats.fstats,")\n");
	(void)fflush(Stats.fstats);
  }
}



Float SampleTime(int sampnum)
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



void RaytraceInit()
{
  pix = (Pixel*)Malloc(Screen.xsize*sizeof(Pixel));
  if (pix==NULL)
  {
    fprintf(stderr, "No enough memory for pix\n");
    exit(1);
  }
}
