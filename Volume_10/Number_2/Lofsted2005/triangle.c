#include "triangle.h"
#include "math.h"

void printTri_big(Triangle_big *t)
{

  printf("V0: %f,%f,%f, V1: %f,%f,%f, V2: %f,%f,%f\n", t-> v0[0], t -> v0[1], t -> v0[2], t-> v1[0], t -> v1[1], t -> v1[2], t-> v2[0], t -> v2[1], t -> v2[2]);
  printf("Plane: %f,%f,%f,%f, i0: %d, i1: %d, i2: %d\n", t ->  normal[0], t ->  normal[1], t -> normal[2], t ->  d, t ->   i0, t ->  i1, t ->  i2);
  printf("Plucker a: %f,%f,%f, %f,%f,%f\n", t ->  a[0], t ->  a[1], t ->  a[2], t ->  a[3], t ->  a[4], t->  a[5]);
  printf("Plucker b: %f,%f,%f, %f,%f,%f\n", t ->  b[0], t ->  b[1], t ->  b[2], t ->  b[3], t ->  b[4], t->  b[5]);
  printf("Plucker c: %f,%f,%f, %f,%f,%f\n", t ->  c[0], t ->  c[1], t ->  c[2], t ->  c[3], t ->  c[4], t->  c[5]);
 
}

void printRay_big(Ray_big *t)
{

  printf("o: %f,%f,%f, d: %f,%f,%f, e: %f,%f,%f\n", t-> orig[0], t -> orig[1], t -> orig[2], t-> dir[0], t -> dir[1], t ->dir[2], t-> end[0], t -> end[1], t -> end[2]);
 
}


Plane * mkPlane(float v0[3], float v1[3], float v2[3])
{
  Plane *p = (Plane *)malloc(sizeof(Plane)); 
  float tmp1[3], tmp2[3];
  SUB(tmp1, v1, v0);
  SUB(tmp2, v2, v0);
  CROSS(p->normal, tmp1, tmp2);
  NORMALIZE(p->normal);
  
  p -> d = -(p->normal[0] * v0[0]) -(p->normal[1] * v0[1]) - (p->normal[2] * v0[2]);
  if(fabs(p->normal[0]) > fabs(p->normal[1])){
    if(fabs(p->normal[0]) > fabs(p->normal[2])){
      p -> i0 = 0;
      p -> i1 = 1;
      p -> i2 = 2;
    }
    else{
      p -> i0 = 2;
      p -> i1 = 0;
      p -> i2 = 1;
    }
  } 
  else{
    if(fabs(p->normal[1]) > fabs(p->normal[2])){
      p -> i0 = 1;
      p -> i1 = 0;
      p -> i2 = 2;
    }
    else{
      p -> i0 = 2;
      p -> i1 = 0;
      p -> i2 = 1;
    }
  }
  

  return p;
}

void mkPlucker(float v0[3], float v1[3], float v2[3], Plucker_coords *p)
{
  
 
  p -> a0 = v0[0] * v1[1] - v1[0] * v0[1];
  p -> a1 = v0[0] * v1[2] - v1[0] * v0[2];
  p -> a2 = v0[0] - v1[0];
  p -> a3 = v0[1] * v1[2] - v1[1] * v0[2];
  p -> a4 = v0[2] - v1[2];
  p -> a5 = v1[1] - v0[1];

  p -> b0 = v1[0] * v2[1] - v2[0] * v1[1];
  p -> b1 = v1[0] * v2[2] - v2[0] * v1[2];
  p -> b2 = v1[0] - v2[0];
  p -> b3 = v1[1] * v2[2] - v2[1] * v1[2];
  p -> b4 = v1[2] - v2[2];
  p -> b5 = v2[1] - v1[1];

  p -> c0 = v2[0] * v0[1] - v0[0] * v2[1];
  p -> c1 = v2[0] * v0[2] - v0[0] * v2[2];
  p -> c2 = v2[0] - v0[0];
  p -> c3 = v2[1] * v0[2] - v0[1] * v2[2];
  p -> c4 = v2[2] - v0[2];
  p -> c5 = v0[1] - v2[1];
    
  p -> v0[0] = v0[0];
  p -> v0[1] = v0[1];
  p -> v0[2] = v0[2];


}


void mkTriangle_small(float v0[3], float v1[3], float v2[3], Triangle_small *t)
{
    
  t -> v0[0] = v0[0];
  t -> v0[1] = v0[1];
  t -> v0[2] = v0[2];
  t -> v1[0] = v1[0];
  t -> v1[1] = v1[1];
  t -> v1[2] = v1[2];
  t -> v2[0] = v2[0];
  t -> v2[1] = v2[1];
  t -> v2[2] = v2[2];
 
}

Triangle_big* mkTriangle_big(float v0[3], float v1[3], float v2[3])
{
  float tmp1[3], tmp2[3];

  Triangle_big *t = (Triangle_big *) malloc(sizeof(Triangle_big));
  
  t -> v0[0] = v0[0];
  t -> v0[1] = v0[1];
  t -> v0[2] = v0[2];
  t -> v1[0] = v1[0];
  t -> v1[1] = v1[1];
  t -> v1[2] = v1[2];
  t -> v2[0] = v2[0];
  t -> v2[1] = v2[1];
  t -> v2[2] = v2[2];
 
  
  SUB(tmp1, v1, v0);
  SUB(tmp2, v2, v0);
  CROSS(t->normal, tmp1, tmp2);
  NORMALIZE(t->normal);
  
  
  t -> d = -(t->normal[0] * v0[0]) -(t->normal[1] * v0[1]) - (t->normal[2] * v0[2]);
  if(fabs(t->normal[0]) > fabs(t->normal[1])){
    if(fabs(t->normal[0]) > fabs(t->normal[2])){
      t -> i0 = 0;
      t -> i1 = 1;
      t -> i2 = 2;
    }
    else{
      t -> i0 = 2;
      t -> i1 = 0;
      t -> i2 = 1;
    }
  } 
  else{
    if(fabs(t->normal[1]) > fabs(t->normal[2])){
      t -> i0 = 1;
      t -> i1 = 0;
      t -> i2 = 2;
    }
    else{
      t -> i0 = 2;
      t -> i1 = 0;
      t -> i2 = 1;
    }
  }

  t -> a[0] = v0[0] * v1[1] - v1[0] * v0[1];
  t -> a[1] = v0[0] * v1[2] - v1[0] * v0[2];
  t -> a[2] = v0[0] - v1[0];
  t -> a[3] = v0[1] * v1[2] - v1[1] * v0[2];
  t -> a[4] = v0[2] - v1[2];
  t -> a[5] = v1[1] - v0[1];

  t -> b[0] = v1[0] * v2[1] - v2[0] * v1[1];
  t -> b[1] = v1[0] * v2[2] - v2[0] * v1[2];
  t -> b[2] = v1[0] - v2[0];
  t -> b[3] = v1[1] * v2[2] - v2[1] * v1[2];
  t -> b[4] = v1[2] - v2[2];
  t -> b[5] = v2[1] - v1[1];

  t -> c[0] = v2[0] * v0[1] - v0[0] * v2[1];
  t -> c[1] = v2[0] * v0[2] - v0[0] * v2[2];
  t -> c[2] = v2[0] - v0[0];
  t -> c[3] = v2[1] * v0[2] - v0[1] * v2[2];
  t -> c[4] = v2[2] - v0[2];
  t -> c[5] = v0[1] - v2[1];


  return t;
}


void mkTriangle_plane(float v0[3], float v1[3], float v2[3], Triangle_plane *t)
{
  float tmp1[3], tmp2[3];
  
  t -> v0[0] = v0[0];
  t -> v0[1] = v0[1];
  t -> v0[2] = v0[2];
  t -> v1[0] = v1[0];
  t -> v1[1] = v1[1];
  t -> v1[2] = v1[2];
  t -> v2[0] = v2[0];
  t -> v2[1] = v2[1];
  t -> v2[2] = v2[2];
 
  
  SUB(tmp1, v1, v0);
  SUB(tmp2, v2, v0);
  CROSS(t->normal, tmp1, tmp2);
  NORMALIZE(t->normal);
  
  
  t -> d = -(t->normal[0] * v0[0]) -(t->normal[1] * v0[1]) - (t->normal[2] * v0[2]);
  if(fabs(t->normal[0]) > fabs(t->normal[1])){
    if(fabs(t->normal[0]) > fabs(t->normal[2])){
      t -> i0 = 0;
      t -> i1 = 1;
      t -> i2 = 2;
    }
    else{
      t -> i0 = 2;
      t -> i1 = 0;
      t -> i2 = 1;
    }
  } 
  else{
    if(fabs(t->normal[1]) > fabs(t->normal[2])){
      t -> i0 = 1;
      t -> i1 = 0;
      t -> i2 = 2;
    }
    else{
      t -> i0 = 2;
      t -> i1 = 0;
      t -> i2 = 1;
    }
  }
   t->inv_n = 1.f/t->normal[t->i0];

}




void mkTriangle_half(float v0[3], float v1[3], float v2[3], Triangle_Halfplane *t)
{
  float tmp1[3], tmp2[3];
int X;
  int Y;
  /*
  t -> v0[0] = v0[0];
  t -> v0[1] = v0[1];
  t -> v0[2] = v0[2];
  t -> v1[0] = v1[0];
  t -> v1[1] = v1[1];
  t -> v1[2] = v1[2];
  t -> v2[0] = v2[0];
  t -> v2[1] = v2[1];
  t -> v2[2] = v2[2];
  */
  
  SUB(tmp1, v1, v0);
  SUB(tmp2, v2, v0);
  CROSS(t->normal, tmp1, tmp2);
  NORMALIZE(t->normal);
  
  
  t -> d = -(t->normal[0] * v0[0]) -(t->normal[1] * v0[1]) - (t->normal[2] * v0[2]);
  if(fabs(t->normal[0]) > fabs(t->normal[1])){
    if(fabs(t->normal[0]) > fabs(t->normal[2])){
      t -> i0 = 0;
      t -> i1 = 1;
      t -> i2 = 2;
    }
    else{
      t -> i0 = 2;
      t -> i1 = 0;
      t -> i2 = 1;
    }
  } 
  else{
    if(fabs(t->normal[1]) > fabs(t->normal[2])){
      t -> i0 = 1;
      t -> i1 = 0;
      t -> i2 = 2;
    }
    else{
      t -> i0 = 2;
      t -> i1 = 0;
      t -> i2 = 1;
    }
  }
  X = t->i1;
  Y = t->i2;

  t->n0x = -(v0[Y]- v1[Y]);
  t->n0y = v0[X] - v1[X];
  t->c0 = - v1[X]*t->n0x - v1[Y]*t->n0y;

  t->n1x = -(v1[Y] - v2[Y]);
  t->n1y = v1[X] - v2[X];
  t->c1 = - v2[X]*t->n1x - v2[Y]*t->n1y;

  t->n2x = -(v2[Y] - v0[Y]);
  t->n2y =  v2[X] - v0[X];
  t->c2 = - v0[X]*t->n2x - v0[Y]*t->n2y;

  t->inv_n = 1.f/t->normal[t->i0];
}

void mkTriangle_inv(float v0[3], float v1[3], float v2[3], Triangle_inv *t)
{
  
  float det, inv_det;
  float p1[3], p2[3], normal[3],tmp[3];
 
  SUB(p1, v0, v1);
  SUB(p2, v0, v2);
  CROSS(normal,p1,p2);

  CROSS(tmp,p2,normal);
  
  det = -DOT(p1,tmp);
  
  inv_det = 1.f/det;


  t->bb0[0] = inv_det * (p2[1]*normal[2] - p2[2]* normal[1]);
  t->bb1[0] = -inv_det * (p1[1]*normal[2] - p1[2]* normal[1]);
  t->bb2[0] = inv_det * (p1[1]*p2[2] - p1[2]* p2[1]);
  
  t->bb0[1] = -inv_det * (p2[0]*normal[2] - p2[2]* normal[0]);
  t->bb1[1] = inv_det * (p1[0]*normal[2] - p1[2]* normal[0]);
  t->bb2[1] = -inv_det * (p1[0]*p2[2] - p1[2]* p2[0]);
 
  t->bb0[2] = inv_det * (p2[0]*normal[1] - p2[1]* normal[0]);
  t->bb1[2] = -inv_det * (p1[0]* normal[1] - p1[1]*normal[0]);
  t->bb2[2] = inv_det * (p1[0]*p2[1] -p1[1]*p2[0]);

  t -> v0[0] = v0[0];
  t -> v0[1] = v0[1];
  t -> v0[2] = v0[2];

}


void mkRay_big(float orig[3], float dir[3], float end[3], Ray_big *r)
{

  
  r -> orig[0] = orig[0];
  r -> orig[1] = orig[1];
  r -> orig[2] = orig[2];

  r -> dir[0] = dir[0];
  r -> dir[1] = dir[1];
  r -> dir[2] = dir[2];

  r -> end[0] = end[0];
  r -> end[1] = end[1];
  r -> end[2] = end[2];
  
}

Intersection_big * mkIntersection_big(float *t, float *u, float *v, float point[3])
{
  Intersection_big *i = (Intersection_big *) malloc(sizeof(Intersection_big));
  i -> t = t;
  i -> u = u;
  i -> v = v;
  /*  i -> point[0] = point[0];
  i -> point[1] = point[1];
  i -> point[2] = point[2]; */
  return i;
}




/*
  This is the crossings algorithm with different args
*/
int test_hit(float org[3], float dir[3], float vert0[3],float vert1[3], float vert2[3])
{
  float plane[4];
  float vv0[3];
  float vv1[3];
  float cross[3];
  int i1 = 0;
  int i2 = 0;
  float t;
  float p[3];
  float u0[2];
  float u1[2];
  float u2[2];
  float vd;
  float vo ;
  int nc = 0;
  int sh;
  int nsh;
  vv0[0] = vert1[0] - vert0[0];
  vv0[1] = vert1[1] - vert0[1];
  vv0[2] = vert1[2] - vert0[2];
  vv1[0] = vert2[0] - vert0[0];
  vv1[1] = vert2[1] - vert0[1];
  vv1[2] = vert2[2] - vert0[2];
  CROSS(cross, vv0, vv1);
  /*NORMALIZE(cross);*/
  plane[0] = cross[0];
  plane[1] = cross[1];
  plane[2] = cross[2];
  plane[3] =  - cross[0]* vert0[0] - cross[1]* vert0[1] - cross[2]*vert0[2];

  
  if(fabs(plane[0]) > fabs(plane[1])){
    if(fabs(plane[0]) > fabs(plane[2])){
      
      i1 = 1;
      i2 = 2;
    }
    else{
      
      i1 = 0;
      i2 = 1;
    }
  } 
  else{
    if(fabs(plane[1]) > fabs(plane[2])){
      
      i1 = 0;
      i2 = 2;
    }
    else{
      
      i1 = 0;
      i2 = 1;
    }
  }

  vd = plane[0]*dir[0] + plane[1]*dir[1] + plane[2]*dir[2];
  
  if(vd == 0)
    return 0;
  vo = -(plane[0]*org[0] + plane[1]*org[1] + plane[2]*org[2] + plane[3]);
  t = vo/vd;
  
  if(t < 0)
    return 0;
  p[0] = org[0] + t*dir[0];
  p[1] = org[1] + t*dir[1];
  p[2] = org[2] + t*dir[2];
  
  
  
  u0[0] = vert0[i1] - p[i1];
  u0[1] = vert0[i2] - p[i2];
  u1[0] = vert1[i1] - p[i1];
  u1[1] = vert1[i2] - p[i2];
  u2[0] = vert2[i1] - p[i1];
  u2[1] = vert2[i2] - p[i2];
  
  
  if(u0[1] < 0)
    sh = -1;
  else
    sh = 1;
  if(u1[1] < 0)
    nsh = -1;
  else
    nsh = 1;
  if(sh != nsh){
    if(u0[0] > 0 && u1[0] > 0){
      nc++;
    }
    else{
      if(u0[0] > 0 || u1[0] > 0){
	if((u0[0] - u0[1]*(u1[0] - u0[0])/(u1[1] - u0[1])) > 0)
	  nc++;
      }
    }
  }
  sh = nsh;
  if(u2[1] < 0)
    nsh = -1;
  else
    nsh = 1;
  if(sh != nsh){
    if(u1[0] > 0 && u2[0] > 0){
      nc++;
    }
    else{
      if(u1[0] > 0 || u2[0] > 0){
	if((u1[0] - u1[1]*(u2[0] - u1[0])/(u2[1] - u1[1])) > 0)
	  nc++;
      }
    }
  }
  sh = nsh;
  if(u0[1] < 0)
    nsh = -1;
  else
    nsh = 1;
  if(sh != nsh){
    if(u2[0] > 0 && u0[0] > 0){
      nc++;
    }
    else{
      if(u2[0] > 0 || u0[0] > 0){
	if((u2[0] - u2[1]*(u0[0] - u2[0])/(u0[1] - u2[1])) > 0)
	  nc++;
      }
    }
  }
  if(nc % 2 == 0)
    return 0;
  
  return 1;
}





