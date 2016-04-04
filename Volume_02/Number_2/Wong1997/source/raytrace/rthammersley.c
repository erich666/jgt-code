/*
 * rthammersley.c
 * 
 * This code is written to test the effect of sampling with Hammersley
 * point set in ray tracing.
 * For simplicity, the average number of sample per pixel is defined by
 * macros AVE_SAMPLE_NUM. The basis is defined by macro P1.
 * To change the value, you have to recompile
 * the program. This code is designed to substitute for the original 
 * file raytrace.c
 *
 * The approach we used is very simple. We tile 1 hammersley point set 
 * pattern on the frame buffer. Hence, we can only specify average 
 * number of sample per pixel. 
 * Function "raytrace()" is expected to be called.
 * 
 */
#include <memory.h>
#include <values.h>
#include "rayshade.h"
#include "libsurf/atmosphere.h"
#include "libsurf/surface.h"
#include "libcommon/sampling.h"
#include "options.h"
#include "stats.h"
#include "raytrace.h"
#include "viewing.h"

#define AVE_SAMPLE_NUM 16
#define P1             2

Pixel	*pix;	 /* Pixel Buffer */
char *samplecnt; /* Number of samples in the each pixel */
static Ray	TopRay;				/* Top-level ray. */
Pixel		WhitePix = {1., 1., 1., 1.},
		BlackPix = {0., 0., 0., 0.};

/* Function Prototypes */
Float SampleTime(int sampnum);
void RaytraceInit();


/* Implementation */
void raytrace(int argc, char **argv)
{
  int x, y, *tmpsamp, i;
  Pixel *tmppix;
  Float usertime, systime, lasttime, upos, vpos;
  Pixel acc, tmp;
  Float a, p, ip, u, v, *minptr;
  int k, kk, pos, n, maxdim, mindim;

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

  memset(samplecnt, 0, Screen.xsize*Screen.ysize);
  lasttime = 0;
  n = AVE_SAMPLE_NUM*Screen.xsize*Screen.ysize;
  maxdim = (Screen.xsize>Screen.ysize)? Screen.xsize : Screen.ysize;
  mindim = (Screen.xsize>Screen.ysize)? Screen.ysize : Screen.xsize;
  minptr = (Screen.xsize>Screen.ysize)? &vpos : &upos;
  for (k=0 ; k<n ;)
  {
    upos = 0;
    ip = 1.0/P1;                           /* recipical of p1 */
    for (p=ip, kk=k ; kk ; p*=ip, kk/=P1)  /* kk = (int)(kk/p1) */
      if ((a = kk % P1))
        upos += a * p;
    upos *= maxdim; 
    vpos = maxdim * (k + 0.5) / n;
    if (*minptr > mindim) /* The sample is outside screen, but inside 1:1 square */
      continue;
    k++;                                   

    x = (int)upos; 
    y = (int)vpos;
    if (x==Screen.xsize) x--;
    if (y==Screen.ysize) y--;
    pos = y*Screen.xsize + x;
    if (!samplecnt[pos]) /* Pixel no yet init */
      pix[pos] = BlackPix; 
    TopRay.time = SampleTime(0);
    SampleScreen(upos, vpos, &TopRay, &tmp, 0);
    pix[pos].r += tmp.r;
    pix[pos].g += tmp.g;
    pix[pos].b += tmp.b;
    samplecnt[pos]++;
    if (!(k%10000))
      fprintf(stderr, "%d out of %d samples done\n", k, n);
  }  
   
  n = Screen.xsize * Screen.ysize;
  for (pos=0 ; pos<n ; pos++)
  { 
      pix[pos].r /= samplecnt[pos]; /* box filter, or average */
      pix[pos].g /= samplecnt[pos]; 
      pix[pos].b /= samplecnt[pos]; 
  }
  for (y=0 ; y<Screen.ysize ; y++)
    PictureWriteLine(&(pix[y*Screen.xsize])); /* write the scanline */
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
  pix = (Pixel*)Malloc(Screen.ysize*Screen.xsize*sizeof(Pixel));
  samplecnt = (char*)Malloc(Screen.ysize*Screen.xsize);
  if (pix==NULL || samplecnt==NULL)
  {
    fprintf(stderr, "No enough memory for pix and samplecnt\n");
    exit(1);
  }
}
