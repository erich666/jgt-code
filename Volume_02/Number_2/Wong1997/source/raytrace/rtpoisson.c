/*
 * rtpoisson.c
 * 
 * This code is written to test the effect of poisson disk sampling in 
 * ray tracing.
 * For simplicity, the number of sample per pixel is hardcoded by
 * a macro SAMPLE_NUM and D (minimum distance between samples).
 * When D = 0, it is actually random sampling
 * To change the value, you have to recompile the program
 * This code is designed to substitute for the original file raytrace.c
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

#define SAMPLE_NUM 16
#define D          0.2

Pixel	*pix;	/* Pixel Buffer */
static Ray	TopRay;				/* Top-level ray. */
Pixel		WhitePix = {1., 1., 1., 1.},
		BlackPix = {0., 0., 0., 0.};

/* Function Prototypes */
Float SampleTime(int sampnum);
void RaytraceInit();


/* Implementation */
void poisson(int sampleno, Float dist, Float *sample)
{
  static int counter=0;
  int i, count;
  Float x, y, dist2, t1, t2;

  dist2=dist*dist; /* compare dist square, instead of dist for efficien */
  for (count=0 ; count<sampleno ; count++)
    while (TRUE)
    {
      /* gen sample */
      x = random()/(double)MAXLONG;
      y = random()/(double)MAXLONG;
      for (i=0 ; i<count ; i++)
      {
        t1 = sample[i<<1]-x;
        t2 = sample[(i<<1)+1]-y;
        if (dist2 > t1*t1 + t2*t2)
          break; /* too close, fail */
      }
      if (i<count) 
        continue; /* test fail, regenerate */
      sample[count<<1] = x;
      sample[(count<<1)+1] = y;
      break; /* pass the test and go on to next sample */
    }

  if (counter%5000==0)
  {
    FILE *fptr=NULL;
    Float scale=200.0, radius=5.0;
    counter = 0; /* reset counter to prevent overflow */
    /* Print the sample pattern to ps file */
    if ((fptr=fopen("poisson.ps","wt"))==NULL)
    {
      fprintf(stderr, "Cannot open poisson.ps\n");
      exit(1);
    }
    fprintf(fptr, "%%!\n100 100 translate\ngsave\n");
    fprintf(fptr, "1 setlinewidth\nnewpath 0 0 moveto %f 0 lineto %f %f lineto 0 %f lineto 0 0 lineto stroke  closepath\n",
    	    scale, scale, scale, scale);
    for (i=0 ; i<sampleno ; i++)
      fprintf(fptr, "newpath %f %f %f 0 360 arc stroke closepath\n",
              sample[i<<1]*scale, sample[(i<<1)+1]*scale, radius);
    fprintf(fptr, "grestore\nshowpage\n");
    fclose(fptr);                                 
  }      
  counter++;
}



void raytrace(int argc, char **argv)
{
  int x, y, *tmpsamp, i;
  Pixel *tmppix;
  Float usertime, systime, lasttime, upos, vpos;
  Float samples[SAMPLE_NUM*2];
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
		poisson(SAMPLE_NUM, D, samples);
		acc.r = acc.g = acc.b = acc.alpha = 0;
		for (i=0 ; i<SAMPLE_NUM ; i++) {
	   		vpos = y + Screen.miny + samples[(i<<1)+1];
			upos = x + Screen.minx + samples[i<<1];
			TopRay.time = SampleTime(0);
			SampleScreen(upos, vpos, &TopRay, &tmp, 0);
			acc.r += tmp.r;
			acc.g += tmp.g;
			acc.b += tmp.b;
		}
		pix[x].r = acc.r/SAMPLE_NUM; /* box filter, or average */
		pix[x].g = acc.g/SAMPLE_NUM; 
		pix[x].b = acc.b/SAMPLE_NUM; 
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
