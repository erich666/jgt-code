/*
 * sampexp.c
 *
 * Copyright (C) 1995 Tien-tsin Wong
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
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <values.h>
#include "libcommon/common.h"
#include "options.h"
#include "stats.h"
#include "picture.h"
#include "viewing.h"
#include "libobj/geom.h"
#include "libobj/triangle.h"
#include "libobj/list.h"
#include "libobj/csg.h"
#include "libobj/grid.h"
#include "libsurf/atmosphere.h"
#include "poly.h"
#include "udpoint.h"


#define GENRANDOMDIR SphereHammersley /* define the function name which gen random vector */
/* #define RANDOMPOSSAMPLE */ /* Sample exposure on random position */
#define FIXEDPOSSAMPLE        /* Sample exposure on fixed grid by scan convert the texture space */
#define TRUE	1
#define FALSE 	0

char *listName="list";
char *gridName="grid";
char *csgName="csg";
char *triName="triangle";
char *worldName="World";
Pixel *textmem;  /* 1D texture memory */
char keepwork=TRUE;
List *masterlist_G;
Triangle *tri_G;
FILE  *fileptr=NULL;


/************************* Function Prototypes ***************************/
void SampleExposure(Geom *obj);
void InitTextureSamp(List *masterlist);
void SaveTextureSamp(List *masterlist);
void SampleTriangleExp(Triangle *tri, List *masterlist);
Vector GenRandomPos(Triangle *tri, Vec2d *uv);

#ifdef FIXEDPOSSAMPLE
static void TakeSampleProc(int x, int y, Poly_vert *point);
#endif

/**************************** Implementation *****************************/
int main(int argc, char **argv)
{
  Float utime, stime, lasttime;
  int i;
  extern Geom *World;

  RSInitialize(argc, argv);

  /*
   * Start the first frame.
   */
  RSStartFrame(Options.startframe);
  /*
   * Print more information than we'll ever need to know...
   */
  if (Options.verbose) 
  {
    /* World object info. */
    AggregatePrintInfo(World, Stats.fstats);
    /* Print info about rendering options and the like. */
    RSOptionsList();
  }
  /*
   * Print preprocessing time.
   */
  RSGetCpuTime(&utime, &stime);
  fprintf(Stats.fstats,"Preprocessing time:\t");
  fprintf(Stats.fstats,"%2.2fu  %2.2fs\n", utime, stime);
  fprintf(Stats.fstats,"Starting sampling exposure.\n");
  (void)fflush(Stats.fstats);
  lasttime = utime+stime;
  SampleExposure(World);
  return 0;
}


void SampleExposure(Geom *obj)
{
  Geom *objlist;
  static List *masterlistobj=NULL;

  if (obj==NULL)
    return;
  fprintf(Stats.fstats,"test triangle %s %s\n", obj->name, (*obj->methods->name)());
  /* only triangle can be sampled */
  if (masterlistobj!=NULL && strncmp((*obj->methods->name)(),triName,10)==0)
  {
    SampleTriangleExp((Triangle*)(obj->obj), masterlistobj);
  fprintf(Stats.fstats,"test triangle again %s %s\n", obj->name, (*obj->methods->name)());
  }
  else if (strncmp((*obj->methods->name)(),listName,10)==0)
  {
  fprintf(Stats.fstats,"test list %s %s\n", obj->name, (*obj->methods->name)());
    /*
     * test whether this object is an augmented list object need sampling.
     * Test also whether this is the first augmented list encounter.
     * If all passed, let obj be the masterlistobj.
     */
    if (masterlistobj==NULL && ((List*)(obj->obj))->resx>0)
    {
      masterlistobj = (List*)obj->obj;
      InitTextureSamp(masterlistobj);
    }
    /* Recursively search the elements in two object list */
    for (objlist=((List*)(obj->obj))->unbounded ; objlist ; objlist=objlist->next)
      SampleExposure(objlist);
    for (objlist=((List*)(obj->obj))->list ; objlist ; objlist=objlist->next)
      SampleExposure(objlist);
    if ((List*)(obj->obj)==masterlistobj)
    {
      SaveTextureSamp(masterlistobj);
      masterlistobj = NULL;
    }
  }

  else if (strncmp((*obj->methods->name)(),csgName,10)==0)
  {
  fprintf(Stats.fstats,"test csg %s %s \n", obj->name, (*obj->methods->name)());
    /* Recursively search the two elementary objects */
    SampleExposure(((Csg*)(obj->obj))->obj1);
    SampleExposure(((Csg*)(obj->obj))->obj2);
  }

  else if (strncmp((*obj->methods->name)(),gridName,10)==0)
  {
  fprintf(Stats.fstats,"test grid  %s %s\n", obj->name, (*obj->methods->name)());
    /* Recursively search the elements in two object list */
    for (objlist=((Grid*)(obj->obj))->unbounded ; objlist ; objlist=objlist->next)
      SampleExposure(objlist);
    for (objlist=((Grid*)(obj->obj))->objects ; objlist ; objlist=objlist->next)
      SampleExposure(objlist);
  }
  fprintf(Stats.fstats,"Going home %s %s\n", obj->name, (*obj->methods->name)());
}



void InitTextureSamp(List *masterlist)
{
  long i, pixelno;
  Pixel black;
  
  if (!masterlist->gentext)
  {
    fprintf(Stats.fstats,"init samp\n");  
    pixelno=masterlist->resx*masterlist->resy;
    textmem = (Pixel*)Malloc(pixelno*sizeof(Pixel));
    if (textmem == NULL)
    {
      fprintf(Stats.fstats,"No enough memory for exposure sampling");
      keepwork = FALSE;
    }
    /* init all pixel to black */
    black.r = black.g = black.b = 0;
    black.alpha = 0;
    for (i=0 ; i<pixelno ; i++)
      textmem[i] = black;
    fprintf(Stats.fstats,"END init samp\n");  
  }
  else
  {
    fileptr = fopen (masterlist->filename, "wt");
    fprintf(fileptr, "%d %d 0 0 0 0\n", masterlist->resx, masterlist->resy);
  }
}



void SaveTextureSamp(List *masterlist)
{
  if (masterlist->gentext)
  {
    fclose(fileptr);
    fileptr = NULL;
    return;
  }
  
  if (keepwork)
  {
    char *nullmsg[2]={"sampexp",NULL};
    int i;
    long ii, size;
  fprintf(Stats.fstats,"save texture\n");    
    size = (long)masterlist->resx*(long)masterlist->resy;
    for (ii=0 ; ii<size ; ii++)
    {
      if (textmem[ii].alpha>1)
        textmem[ii].r /= textmem[ii].alpha;  /* average the exposure value */
      textmem[ii].r *= masterlist->expo_scale;
      if (textmem[ii].r>1)
        textmem[ii].r = 1;
      if (textmem[ii].r<0)
        textmem[ii].r = 0;
      textmem[ii].g = textmem[ii].b = textmem[ii].r;
      textmem[ii].alpha = (Float)1;
    }
    Options.imgname = masterlist->filename;
    Options.framenum = Options.startframe;
    Options.appending = FALSE;
    Screen.xsize = masterlist->resx;
    Screen.ysize = masterlist->resy;    
    Screen.maxx = masterlist->resx;
    Screen.maxy = masterlist->resy;
    Screen.minx = 0;
    Screen.miny = 0;
    Options.alpha = FALSE;
    PictureStart(nullmsg);
    for (i=masterlist->resy-1 ; i>=0 ; i--)
      PictureWriteLine(&(textmem[i*masterlist->resx]));
    PictureFrameEnd();
    PictureEnd();
    free(textmem);
  }
  else
    keepwork = TRUE;
}



#ifdef RANDOMPOSSAMPLE /* Sample exposure at random postion on the triangle */
/* Algorithm 1:
 * Start sampling the surface exposure of the triangle
 * Calculate the area of triangle
 * Find the number of samples needed in this triangle
 * For each sample
 *     Find a random point on the triangle
 *     Fire a random ray to sample exposure
 *     Record the value in the exposure map with box filter applied
 */
/* Assume the textmem is opened for writing */
void SampleTriangleExp(Triangle *tri, List *masterlist)
{
  Ray sampleray;
  HitList hitlist;
  Float area, dist;
  Vector r1, r2, r3;
  Vec2d uv;
  long posno, i, j, location;

  fprintf(Stats.fstats,"sample triangle\n");
  if (!keepwork || tri->uv==NULL)
    return;
    
  /*
   * Calculate the area of triangle 
   * Area = 0.5*|N.{summation P_{k} x P_{k+1}}|
   * Reference: Ronald N. Goldman, "Area of Planar Polygons and Volume of 
   *            Polyhedra", Graphics Gems II, pp.170-171.
   */
  VecCross(&tri->p[0], &tri->p[1], &r1);
  VecCross(&tri->p[1], &tri->p[2], &r2);
  VecCross(&tri->p[2], &tri->p[0], &r3);
  VecAdd(r1, r2, &r1);
  VecAdd(r1, r3, &r1);
  area = 0.5 * fabs(dotp(&tri->nrm,&r1));

  /* the no of samples is area weighted */
  posno = (long)ceil(area*masterlist->posno);

  /* set the unused element of sampleray, hitlist */
  sampleray.depth  = 0;
  sampleray.sample = 1;
  sampleray.time   = 0;
  sampleray.media  = (Medium*)NULL;
  hitlist.nodes = 0;

  fprintf(Stats.fstats, "posno = %li, sampleperpos=%li\n", posno, masterlist->sampleperpos);
  /* We totally perform posno*sampleperpos sampling */
  for (i=0 ; i<posno ; i++)
  {
    sampleray.pos = GenRandomPos(tri, &uv);
    location = (long)floor(uv.v*masterlist->resy)*masterlist->resx 
             + (long)floor(uv.u*masterlist->resx);
    SphereHammersley(1);  // reset counter k in routine to start new sample sequences
    for (j=0 ; j<masterlist->sampleperpos ; j++)
    {
      sampleray.dir = SphereHammersley(masterlist->sampleperpos);
      if (dotp(&(sampleray.dir), &tri->nrm) < 0)
      {
        sampleray.dir.x = -sampleray.dir.x;  /* reverse the direction */
        sampleray.dir.y = -sampleray.dir.y;
        sampleray.dir.z = -sampleray.dir.z;
      }
      dist = FAR_AWAY;
      if (TraceRay(&sampleray, &hitlist, EPSILON, &dist)) /* hit something */
      {
        if (!masterlist->gentext)
          textmem[location].r += masterlist->expo_dist_h/(masterlist->expo_dist_h+dist);
        else
          fprintf(fileptr, "%d %d %f %f %f %f\n", (int)uv.u*masterlist->resx,
           (int)uv.v*masterlist->resy, dist, sampleray.dir.x, sampleray.dir.y, sampleray.dir.z);
      }
      else
        if (masterlist->gentext)
          fprintf(fileptr, "%d %d -1 0 0 0\n", (int)uv.u*masterlist->resx, (int)uv.v*masterlist->resy);
    }
    if (!masterlist->gentext)
      textmem[location].alpha+=masterlist->sampleperpos;  /* treat alpha as a sample counter */
  }
}
#endif





#ifdef FIXEDPOSSAMPLE /* Sample on the grid in texture space, => may have distortion */
/* Algorithm 2:
 * Start sampling the surface exposure of the triangle
 * For each pixel in the texture space of the triangle
 *     Fire fixed number (sampleperpos) of random rays to sample exposure
 *     Record the value in the exposure map with box filter applied
 */
void SampleTriangleExp(Triangle *tri, List *masterlist)
{
  int i;
  Poly p;
  Window win = {0, 0, masterlist->resx-1, masterlist->resy-1}; /* screen clipping window */

  fprintf(Stats.fstats,"sample triangle\n");
  if (!keepwork || tri->uv==NULL)
    return;


  p.n = 3;
  for (i=0; i<p.n; i++)
  {
    p.vert[i].sx = tri->uv[i].u*masterlist->resx; /* set screen space x,y,z */
    p.vert[i].sy = tri->uv[i].v*masterlist->resy;
    p.vert[i].sz = 0;
    p.vert[i].x  = tri->p[i].x;
    p.vert[i].y  = tri->p[i].y;
    p.vert[i].z  = tri->p[i].z;
    fprintf(Stats.fstats,"%f %f %f  %f %f %f\n", p.vert[i].sx, p.vert[i].sy,
    	    p.vert[i].sz, p.vert[i].x, p.vert[i].y, p.vert[i].z);
  }
  /* interpolate sx, sy, sz, x, y, z in poly_scan */
  p.mask = POLY_MASK(sx) | POLY_MASK(sy) | POLY_MASK(sz) |
           POLY_MASK(x)  | POLY_MASK(y)  | POLY_MASK(z);
  masterlist_G = masterlist;
  tri_G = tri;
  poly_scan(&p, &win, TakeSampleProc);     /* scan convert! */
}


/* called at each pixel by poly_scan */
static void TakeSampleProc(int x, int y, Poly_vert *point)
{
  static Ray sampleray;
  static HitList hitlist;
  static char first=TRUE;
  Float dist;
  long location;
  int j;

  /* set the unused element of sampleray, hitlist */
  if (first)
  {
    sampleray.depth  = 0;
    sampleray.sample = 1;
    sampleray.time   = 0;
    sampleray.media  = (Medium*)NULL;
    hitlist.nodes = 0;
    first = FALSE;
  }

  sampleray.pos.x = point->x;
  sampleray.pos.y = point->y;
  sampleray.pos.z = point->z;
  location = (long)y*masterlist_G->resx + (long)x;
  SphereHammersley(1);  /* reset counter k in routine to start new sample sequences */
  for (j=0 ; j<masterlist_G->sampleperpos ; j++)
  {
    sampleray.dir = SphereHammersley(masterlist_G->sampleperpos);
    if (dotp(&(sampleray.dir), &tri_G->nrm) < 0)
    {
      sampleray.dir.x = -sampleray.dir.x;  /* reverse the direction */
      sampleray.dir.y = -sampleray.dir.y;
      sampleray.dir.z = -sampleray.dir.z;
    }
    dist = FAR_AWAY;
    if (TraceRay(&sampleray, &hitlist, EPSILON, &dist)) 
    {
      if (!masterlist_G->gentext)
        textmem[location].r += masterlist_G->expo_dist_h/(masterlist_G->expo_dist_h+fabs(dist));
      else
        fprintf(fileptr, "%d %d %f %f %f %f\n", x, y, dist, sampleray.dir.x,
        	sampleray.dir.y, sampleray.dir.z);
    }
    else
      if (masterlist_G->gentext)
        fprintf(fileptr, "%d %d -1 0 0 0\n", x, y);
  }
  if (!masterlist_G->gentext)
    textmem[location].alpha+=masterlist_G->sampleperpos;  /* treat alpha as a sample counter */
}
#endif



Vector GenRandomPos(Triangle *tri, Vec2d *uv)
{
  Float b[3], sum, d;
  int i;
  Vector pos, *tri_p;
  Vec2d *tri_uv;

  while (TRUE)
  {
    for (i=0 ; i<3 ; i++)
      b[i] = (Float)random()/(Float)MAXLONG;
    if ((sum=b[0]+b[1]+b[2])<=1)
      break;
    if (3-sum<=1)
    {
      for (i=0 ; i<3 ; i++)
        b[i] = (Float)1-b[i];
      break;
    }
  }
  /* b is the barycentric coordinate */
  tri_p  = tri->p;
  tri_uv = tri->uv;
  pos.x = b[0]*tri_p[0].x + b[1]*tri_p[1].x + b[2]*tri_p[2].x;
  pos.y = b[0]*tri_p[0].y + b[1]*tri_p[1].y + b[2]*tri_p[2].y;
  pos.z = b[0]*tri_p[0].z + b[1]*tri_p[1].z + b[2]*tri_p[2].z;
  /* normalize b before weighted sum the uv coordinate */
  d = b[0]+b[1]+b[2];
  b[0] /= d;
  b[1] /= d;
  b[2] /= d;
  uv->u = b[0]*(tri_uv[0].u) + b[1]*(tri_uv[1].u) + b[2]*(tri_uv[2].u);
  uv->v = b[0]*(tri_uv[0].v) + b[1]*(tri_uv[1].v) + b[2]*(tri_uv[2].v);
  return pos;
}

