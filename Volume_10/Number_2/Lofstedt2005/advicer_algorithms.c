/* 
   advicer_algorithms.h
   
   all function with _bary in the name will calculate barycentric coordinates and t-vale
   all algorithmas _t int the name only have to calculate t-value,
   otherwise no supplementary information is calculated.
   
   
*/

#include "advicer_algorithms.h"

/**************************************************************************

All algorithms below must calculate barycentric coordinates and t-value.

**************************************************************************/



/* 
   MT0 Möller-Trumbore barycentric coordinates and t-value calculation, small triangle struct
   
*/

int intersect_triangle_small_bary(Ray_big *r, Triangle_small *t, Intersection_big *p)
{
  float edge1[3], edge2[3], tvec[3], pvec[3], qvec[3];
  float det,inv_det;
  
  
  /* find vectors for two edges sharing vert0 */
  SUB(edge1, t -> v1, t -> v0);
  SUB(edge2, t -> v2, t -> v0);
  
  /* begin calculating determinant - also used to calculate U parameter */
  CROSS(pvec, r -> dir, edge2);
  
  /* if determinant is near zero, ray lies in plane of triangle */
  det = DOT(edge1, pvec);
  
  if (det > -EPSILON && det < EPSILON)
    return 0;
  inv_det = 1.0f / det;
  
  /* calculate distance from vert0 to ray origin */
  SUB(tvec, r -> orig, t -> v0);
  
  /* calculate U parameter and test bounds */
  *(p -> u) = DOT(tvec, pvec) * inv_det;
  if (*(p->u) < 0.0 || *(p->u) > 1.0)
    return 0;
  
  /* prepare to test V parameter */
  CROSS(qvec, tvec, edge1);
  
  /* calculate V parameter and test bounds */
  *(p->v) = DOT(r->dir, qvec) * inv_det;
  if (*(p->v) < 0.0 || *(p->u) + *(p->v) > 1.0)
    return 0;
  
  /* calculate t, ray intersects triangle */
  *(p->t) = DOT(edge2, qvec) * inv_det;
  return 1;
}

/* MT1 Möller-Trumbore barycentric coordinates and t-value calculation, small triangle struct*/
/* code rewritten to do tests on the sign of the determinant */
/* the division is at the end in the code                    */
int intersect_triangle1_small_bary(Ray_big *r, Triangle_small *t, Intersection_big *p)
{
  float edge1[3], edge2[3], tvec[3], pvec[3], qvec[3];
  float det,inv_det;
  
  /* find vectors for two edges sharing vert0 */
  SUB(edge1, t ->v1, t ->v0);
  SUB(edge2, t ->v2, t ->v0);
  
  /* begin calculating determinant - also used to calculate U parameter */
  CROSS(pvec, r ->dir, edge2);
  
  /* if determinant is near zero, ray lies in plane of triangle */
  det = DOT(edge1, pvec);
  
  if (det > EPSILON)
    {
      /* calculate distance from vert0 to ray origin */
      SUB(tvec, r->orig, t ->v0);
      
      /* calculate U parameter and test bounds */
      *(p -> u) = DOT(tvec, pvec);
      if (*(p->u) < 0.0 || *(p->u) > det)
	return 0;
      
      /* prepare to testV parameter */
      CROSS(qvec, tvec, edge1);
      
      /* calculate V parameter and test bounds */
      *(p->v) = DOT(r->dir, qvec);
      if (*(p->v) < 0.0 || *(p->u) + *(p->v) > det)
	return 0;
      
    }
  else if(det < -EPSILON)
    {
      /* calculate distance from vert0 to ray origin */
      SUB(tvec, r->orig, t->v0);
      
      /* calculate U parameter and test bounds */
      *(p->u) = DOT(tvec, pvec);
      if (*(p->u) > 0.0 || *(p->u) < det)
	return 0;
      
      /* prepare to test V parameter */
      CROSS(qvec, tvec, edge1);
      
      /* calculate V parameter and test bounds */
      *(p->v) = DOT(r->dir, qvec) ;
      if (*(p->v) > 0.0 || *(p->u) + *(p->v) < det)
	return 0;
    }
  else return 0;  /* ray is parallell to the plane of the triangle */
  
  
  inv_det = 1.0f / det;
  
  /* calculate t, ray intersects triangle */
  *(p->t) = DOT(edge2, qvec) * inv_det;
  /* för att kolla om vi är bakom ray origin*/
  
  *(p->u) *= inv_det;
  *(p->v) *= inv_det;
  
  return 1;
}

/*MT2 Möller-Trumbore barycentric coordinates and t-value calculation, small triangle struct*/
/* code rewritten to do tests on the sign of the determinant */
/* the division is before the test of the sign of the det    */
int intersect_triangle2_small_bary(Ray_big *r, Triangle_small *t, Intersection_big *p)
{
  float edge1[3], edge2[3], tvec[3], pvec[3], qvec[3];
  float det,inv_det;
  
  /* find vectors for two edges sharing vert0 */
  SUB(edge1, t->v1, t ->v0);
  SUB(edge2, t->v2, t->v0);
  
  /* begin calculating determinant - also used to calculate U parameter */
  CROSS(pvec, r->dir, edge2);
  
  /* if determinant is near zero, ray lies in plane of triangle */
  det = DOT(edge1, pvec);
  if(det == 0)
    return 0;
  /* calculate distance from vert0 to ray origin */
  SUB(tvec, r->orig, t->v0);
  inv_det = 1.0f / det;
  
  if (det > EPSILON)
    {
      /* calculate U parameter and test bounds */
      *(p->u) = DOT(tvec, pvec);
      if (*(p->u) < 0.0 || *(p->u) > det)
	return 0;
      
      /* prepare to test V parameter */
      CROSS(qvec, tvec, edge1);
      
      /* calculate V parameter an test bounds */
      *(p->v) = DOT(r->dir, qvec);
      if (*(p->v) < 0.0 || *(p->u) + *(p->v) > det)
	return 0;
      
    }
  else if(det < -EPSILON)
    {
      /* calculate U parameter and test bounds */
      *(p->u) = DOT(tvec, pvec);
      if (*(p->u) > 0.0 || *(p->u) < det)
	return 0;
      
      /* prepare to test V parameter */
      CROSS(qvec, tvec, edge1);
      
      /* calculate V parameter and test bounds */
      *(p->v) = DOT(r->dir, qvec) ;
      if (*(p->v) > 0.0 || *(p->u) + *(p->v) < det)
	return 0;
    }
  else return 0;  /* ray is parallell to the plane of the triangle */
  
  /* calculate t, ray intersects triangle */
  *(p->t) = DOT(edge2, qvec) * inv_det;
  *(p->u) *= inv_det;
  *(p->v) *= inv_det;
  
  return 1;
}

/* MT3 Möller-Trumbore barycentric coordinates and t-value calculation, small triangle struct*/
/* code rewritten to do tests on the sign of the determinant */
/* the division is before the test of the sign of the det    */
/* and one CROSS has been moved out from the if-else if-else */
int intersect_triangle3_small_bary(Ray_big *r, Triangle_small *t, Intersection_big *p)
{
  float edge1[3], edge2[3], tvec[3], pvec[3], qvec[3];
  float det,inv_det;
  
  /* find vectors for two edges sharing vert0 */
  SUB(edge1, t->v1, t->v0);
  SUB(edge2, t->v2, t->v0);
  
  /* begin calculating determinant - also used to calculate U parameter */
  CROSS(pvec, r->dir, edge2);
  
  /* if determinant is near zero, ray lies in plane of triangle */
  det = DOT(edge1, pvec);
  if (det == 0)
    return 0;
  /* calculate distance from vert0 to ray origin */
  SUB(tvec, r->orig, t->v0);
  inv_det = 1.0f / det;
  
  CROSS(qvec, tvec, edge1);
  
  if (det > EPSILON)
    {
      *(p->u) = DOT(tvec, pvec);
      if (*(p->u) < 0.0 || *(p->u) > det)
	return 0;
      
      /* calculate V parameter and test bounds */
      *(p->v) = DOT(r->dir, qvec);
      if (*(p->v) < 0.0 || *(p->u) + *(p->v) > det)
	return 0;
      
    }
  else if(det < -EPSILON)
    {
      /* calculate U parameter and test bounds */
      *(p->u) = DOT(tvec, pvec);
      if (*(p->u) > 0.0 || *(p->u) < det)
	return 0;
      
      /* calculate V parameter and test bounds */
      *(p->v) = DOT(r->dir, qvec) ;
      if (*(p->v) > 0.0 || *(p->u) + *(p->v) < det)
	return 0;
    }
  else return 0;  /* ray is parallell to the plane of the triangle */
  
  *(p->t) = DOT(edge2, qvec) * inv_det;
  *(p->u) *= inv_det;
  *(p->v) *= inv_det;
   return 1;
}



/* OR - O'Rourke */

int orourke_small_bary(Ray_big *r, Triangle_small *t, Intersection_big *p){

  float vo, vd; 
  float tmp1[3], tmp2[3], normal[3],d;
  float vol0, vol1, vol2;
  float ax,ay,az,bx,by,bz,cx,cy,cz,dx,dy,dz,div;
  


  ax = r->orig[0] - r -> end[0];
  ay = r->orig[1] - r -> end[1];
  az = r->orig[2] - r -> end[2];
  bx = t->v0[0] - r -> end[0];
  by = t->v0[1] - r -> end[1];
  bz = t->v0[2] - r -> end[2];
  cx = t->v1[0] - r -> end[0];
  cy = t->v1[1] - r -> end[1];
  cz = t->v1[2] - r -> end[2];

  dx = t->v2[0] - r -> end[0];
  dy = t->v2[1] - r -> end[1];
  dz = t->v2[2] - r -> end[2];
  
  
  vol0 = (ax * (by*cz - bz*cy) + ay * (bz*cx - bx*cz) + az *(bx*cy - by*cx));
  vol1 = (ax * (cy*dz - cz*dy) + ay * (cz*dx - cx*dz) + az *(cx*dy - cy*dx));
  
  
  if(vol0*vol1<0)
    return 0;


  vol2 = (ax * (dy*bz - dz*by) + ay * (dz*bx - dx*bz) + az *(dx*by - dy*bx));
  
  if(vol0*vol2<0)
    return 0;
  
  
  /* Wont need tests below if we return on different signs*/
  /* Same sign: ray intersects interior of triangle*/
  //if((vol0 > 0 && vol1 > 0 && vol2 > 0) || (vol0 < 0 && vol1 < 0 && vol2 < 0)){
  
  tmp1[0] = t->v0[0] - t-> v1[0];
  tmp1[1] = t->v0[1] - t-> v1[1];
  tmp1[2] = t->v0[2] - t-> v1[2];
  tmp2[0] = t->v0[0] - t-> v2[0];
  tmp2[1] = t->v0[1] - t-> v2[1];
  tmp2[2] = t->v0[2] - t-> v2[2];
  
  CROSS(normal, tmp1, tmp2);
  d = -(DOT(normal, t-> v0));
  
  vd = normal[0]*r->dir[0] + normal[1]*r->dir[1] + normal[2]*r->dir[2];
  if(vd == 0)
      return 0;
  vo = -(normal[0]*r->orig[0] + normal[1]*r->orig[1] + normal[2]*r->orig[2] + d);
  *(p->t) = vo/vd;
  div = 1.f/(normal[0]*ax + normal[1]*ay + normal[2]*az);
  *(p->u) = vol1*div;
  *(p->v) = vol0*div;
  //printf("vol: %f, %f, %f div: %f n: %f, %f, %f\n", vol0, vol1, vol2, div, normal[0], normal[1], normal[2]);
  return 1;
 
}

/*ORC - Optimization of O'Rourke, small triangle struct. Only works on ccw ray-triangle relationship*/
int orourke_small_baryCCW(Ray_big *r, Triangle_small *t, Intersection_big *p){

  float vo, vd; 
  float tmp1[3], tmp2[3], normal[3],d;
  float vol0, vol1, vol2, div;
  float ax,ay,az,bx,by,bz,cx,cy,cz,dx,dy,dz;
  

  ax = r->orig[0] - r -> end[0];
  ay = r->orig[1] - r -> end[1];
  az = r->orig[2] - r -> end[2];
  bx = t->v0[0] - r -> end[0];
  by = t->v0[1] - r -> end[1];
  bz = t->v0[2] - r -> end[2];
  cx = t->v1[0] - r -> end[0];
  cy = t->v1[1] - r -> end[1];
  cz = t->v1[2] - r -> end[2];


  vol0 = (ax * (by*cz - bz*cy) + ay * (bz*cx - bx*cz) + az *(bx*cy - by*cx));
  
  if(vol0 < 0)
    return 0;
  dx = t->v2[0] - r -> end[0];
  dy = t->v2[1] - r -> end[1];
  dz = t->v2[2] - r -> end[2];
  
  vol1 = (ax * (cy*dz - cz*dy) + ay * (cz*dx - cx*dz) + az *(cx*dy - cy*dx));

  if(vol1<0)
    return 0;
  vol2 = (ax * (dy*bz - dz*by) + ay * (dz*bx - dx*bz) + az *(dx*by - dy*bx));
  
  if(vol2<0)
    return 0;
    tmp1[0] = t->v0[0] - t-> v1[0];
    tmp1[1] = t->v0[1] - t-> v1[1];
    tmp1[2] = t->v0[2] - t-> v1[2];
    tmp2[0] = t->v0[0] - t-> v2[0];
    tmp2[1] = t->v0[1] - t-> v2[1];
    tmp2[2] = t->v0[2] - t-> v2[2];
    
    CROSS(normal, tmp1, tmp2);
    d = -(DOT(normal, t-> v0));
    
    vd = normal[0]*r->dir[0] + normal[1]*r->dir[1] + normal[2]*r->dir[2];
    vo = -(normal[0]*r->orig[0] + normal[1]*r->orig[1] + normal[2]*r->orig[2] + d);
    *(p->t) = vo/vd;
    div = 1.f/(normal[0]*ax + normal[1]*ay + normal[2]*az);
    *(p->u) = vol1*div;
    *(p->v) = vol0*div;
   
    return 1;

}

/* MA - Mahovsky with barycentric coordinates anf t-value, small triangle struct, i.e plucker coordinates are calculated on the fly,only work on ccw ray-triangle relationship */

int plucker_mahovsky_small_bary(Ray_big *r, Triangle_small *t, Intersection_big *p)
{
  float a0,a1,a2,a3,a4,a5,b0,b1,b2,b3,b4,b5,c0,c1,c3;
  float r0,r1,r2,r3,r4,r5;
  float A,B, bb0, bb1;
  float tmp1[3], normal[3], vd, vo,div1,div;
  
  /* Plucker för linjen*/
  r0 = r->orig[0] * r->end[1] - r->orig[1] * r->end[0];
  r1 = r->orig[0] * r->end[2] - r->orig[2] * r->end[0];
  r2 = r->orig[0] - r->end[0];
  r3 = r->orig[1] * r->end[2] - r->orig[2] * r->end[1];
  r4 = r->orig[2] - r->end[2];
  r5 = r->end[1] - r->orig[1];
  
      /*Plucker koefficienter för triangel sidorna*/
  a0 = t->v0[0] * t->v1[1] - t->v1[0] * t->v0[1];
  a1 = t->v0[0] * t->v1[2] - t->v1[0] * t->v0[2];
  a2 = t->v0[0] - t->v1[0];
  a3 = t->v0[1] * t->v1[2] - t->v1[1] * t->v0[2];
  a4 = t->v0[2] - t->v1[2];
  a5 = t->v1[1] - t->v0[1];

  A = r0*a4 + r1*a5 +r3*a2;
  bb0 = A + r2*a3 + r4*a0 + r5*a1;
  if( bb0 < 0.)
    return 0;

  b0 = t->v1[0] * t->v2[1] - t->v2[0] * t->v1[1];
  b1 = t->v1[0] * t->v2[2] - t->v2[0] * t->v1[2];
  b2 = t->v1[0] - t->v2[0];
  b3 = t->v1[1] * t->v2[2] - t->v2[1] * t->v1[2];
  b4 = t->v1[2] - t->v2[2];
  b5 = t->v2[1] - t->v1[1];
  
 
  B = r0*b4 + r1*b5 + r3*b2;
  bb1 = B + r2*b3 + r4*b0 +r5*b1;
  if(bb1<0.)
    return 0;

  c0 = t->v2[0] * t->v0[1] - t->v0[0] * t->v2[1];
  c1 = t->v2[0] * t->v0[2] - t->v0[0] * t->v2[2];
  c3 = t->v2[1] * t->v0[2] - t->v0[1] * t->v2[2];
  
  if(r2*c3 + r4*c0 + r5*c1 - A - B  < 0.)
    return 0;

 
      /*  Räkna ut avståndet*/
 
  normal[0] = a3 +b3 +c3;
  normal[1] =  a1 +b1 +c1;
  normal[2] = a0 +b0 +c0;
  
  SUB(tmp1, t->v0, r->orig);

  vo = DOT(tmp1, normal);
  vd = DOT(r->dir, normal);
  if(vd == 0)
      return 0;
  *(p->t) = vo/vd; 
  
  div = (normal[0]*r2+normal[1]*r5+normal[2]*r4);
  if(div == 0)
    return 0;
  div1 = 1.f/div;
  *(p->v) = bb0*div1;
  *(p->u) = bb1*div1;
 
  return 1;
}


/*
  PU - plucker coordinates algorithms that works with ccw ray-triangle relationship, 
*/
int plucker_small_bary(Ray_big *r, Triangle_small *t, Intersection_big *p)
{
  float a0,a1,a2,a3,a4,a5,b0,b1,b2,b3,b4,b5,c0,c1,c3;
  float r0,r1,r2,r3,r4,r5;
  float A,B;
  float a,b,c;	
   float tmp1[3], normal[3],vd, vo,div1,div;
 
  /* Plucker för linjen*/
  r0 = r->orig[0] * r->end[1] - r->orig[1] * r->end[0];
  r1 = r->orig[0] * r->end[2] - r->orig[2] * r->end[0];
  r2 = r->orig[0] - r->end[0];
  r3 = r->orig[1] * r->end[2] - r->orig[2] * r->end[1];
  r4 = r->orig[2] - r->end[2];
  r5 = r->end[1] - r->orig[1];

    /*Plucker koefficienter för triangel sidorna*/
  a0 = t->v0[0] * t->v1[1] - t->v1[0] * t->v0[1];
  a1 = t->v0[0] * t->v1[2] - t->v1[0] * t->v0[2];
  a2 = t->v0[0] - t->v1[0];
  a3 = t->v0[1] * t->v1[2] - t->v1[1] * t->v0[2];
  a4 = t->v0[2] - t->v1[2];
  a5 = t->v1[1] - t->v0[1];
  
  b0 = t->v1[0] * t->v2[1] - t->v2[0] * t->v1[1];
  b1 = t->v1[0] * t->v2[2] - t->v2[0] * t->v1[2];
  b2 = t->v1[0] - t->v2[0];
  b3 = t->v1[1] * t->v2[2] - t->v2[1] * t->v1[2];
  b4 = t->v1[2] - t->v2[2];
  b5 = t->v2[1] - t->v1[1];
  
  A = r0*a4 + r1*a5 +r3*a2;
  B = r0*b4 + r1*b5 + r3*b2;
  
  a = A + r2*a3 + r4*a0 + r5*a1;
  b = B + r2*b3 + r4*b0 +r5*b1;

  if(a* b < 0)
    return 0;
  
  c0 = t->v2[0] * t->v0[1] - t->v0[0] * t->v2[1];
  c1 = t->v2[0] * t->v0[2] - t->v0[0] * t->v2[2];
  c3 = t->v2[1] * t->v0[2] - t->v0[1] * t->v2[2];
  c = r2*c3 + r4*c0 + r5*c1 - A - B;
  
  if(c*a < 0)
  return 0;
 
  normal[0] = a3 +b3 +c3;
  normal[1] =  a1 +b1 +c1;
  normal[2] = a0 +b0 +c0;

  div = (normal[0]*r2+normal[1]*r5+normal[2]*r4);
  if(div == 0)
    return 0;
  div1 = 1.f/div;
  *(p->v) = b*div1;
  *(p->u) = a*div1;


  SUB(tmp1, t->v0, r->orig);

  vo = DOT(tmp1, normal);
  vd = DOT(r->dir, normal);
  if(vd == 0)
      return 0;
  *(p->t) = vo/vd;   
  return 1;
}

/* CH1p - Chirkov algorithm with triangle struct with precalculated plane. Other branching structure that original Chirkov code  */

int chirkov_other_bary(Ray_big *r, Triangle_plane *t, Intersection_big *p)
{
  float signSrc = t->normal[0]*r->orig[0] + t->normal[1]*r->orig[1] + t->normal[2]*r->orig[2] + t->d;
  float signDst = t->normal[0]*r->end[0] + t->normal[1]*r->end[1] + t->normal[2]*r->end[2] + t->d;
  float ay ;
  float az  ;
  float by  ;
  float bz  ;
  float dely  ;
  float delz ;
  float basey ;
  float basez ;
 
  float d ;
  float cy;
  float cz;
  int i1 ;
  int i2;
  float adelxbase, bary2,div, bary3, div1;
  
  if(signSrc*signDst >= 0)
    return 0;
  
  i1 = t ->  i1;
  i2 = t ->  i2;
  
  d = signSrc - signDst;

  ay =	t->v1[i1] - t->v0[i1];
  az =	t->v1[i2] - t->v0[i2];
  by =	t->v2[i1] - t->v0[i1];
  bz =	t->v2[i2] - t->v0[i2];
  
  dely = r->end[i1] - r->orig[i1] ;
  delz = r->end[i2] - r->orig[i2] ;
  
  basey = r->orig[i1] - t->v0[i1];
  basez = r->orig[i2] - t->v0[i2];

  adelxbase = signSrc*(ay *delz - az *dely) + d*(ay*basez - az*basey);
  bary2 =  (signSrc*(dely*bz - delz*by)	+ d*(basey*bz -	basez*by));
  if(adelxbase * bary2 >= 0.0){
    bary3;
    cy = t->v2[i1] -	t->v1[i1];
    cz = t->v2[i2] -	t->v1[i2];
    basey = r->orig[i1] - t->v1[i1];
    basez = r->orig[i2] - t->v1[i2];
    bary3 = signSrc*(dely*cz - delz*cy) + d*(basey*cz - basez*cy);
    if(adelxbase * bary3 < 0.0){
      div1 = 1.f/d;
      
      *(p->t) = -signSrc*div1;
      div = t->inv_n;
      *(p->u) = fabs(bary2*div);
      *(p->v) = fabs(adelxbase*div);
      
      return 1;
    }
   
  }


  return 0;
}



/*

CH3p - Chirkov with optimization on double calculation in original Chirkov code

*/

int chirkov3_other_bary(Ray_big *r, Triangle_plane *t, Intersection_big *p)
{
  float ax;
  float ay ;
  float az;
  float bx;
  float by;
  float bz;
  float delx;
  float dely;
  float delz;
  float basex;
  float basey;
  float basez;
  float cx;
  float cy;
  float cz;
  float d,bary1,bary2,div,bary3,div1;
  float signSrc = t->normal[0]*r->orig[0] + t->normal[1]*r->orig[1] + t->normal[2]*r->orig[2] + t->d;
  float signDst = t->normal[0]*r->end[0] + t->normal[1]*r->end[1] + t->normal[2]*r->end[2] + t->d;

  if(signSrc*signDst>=0.0)	
    return 0;
  
  d = signSrc - signDst;
  
  if(t->i0==0)
    {
      ay = t->v1[1] - t->v0[1];
      az = t->v1[2] - t->v0[2];
      by = t->v2[1] - t->v0[1];
      bz = t->v2[2] - t->v0[2];
      dely = r->end[1] - r->orig[1];
      delz = r->end[2] - r->orig[2];
      basey = r->orig[1] - t->v0[1];
      basez = r->orig[2] - t->v0[2];

      
      bary1 = ( signSrc*(dely*bz - delz*by) + d*(basey*bz - basez*by));
      bary3 = signSrc*(ay *delz - az *dely) + d*(ay*basez - az*basey);
   
      bary1 = ( signSrc*(dely*bz - delz*by) + d*(basey*bz - basez*by));
      if( (bary3) * bary1 >=0.0)
	{
	  cy = t->v2[1] - t->v1[1];
	  cz = t->v2[2] - t->v1[2];
	  basey = r->orig[1] - t->v1[1];
	  basez = r->orig[2] - t->v1[2];
	  bary2 =  (signSrc*(dely*cz - delz*cy) + d*(basey*cz - basez*cy));
	  if( (bary3) * bary2 <0.0){	
	    div1 = 1.f/d;
	    
	    *(p->t) = -signSrc*div1;
	    div = t->inv_n;
	    *(p->u) = fabs(bary2*div);
	    *(p->v) = fabs(bary1*div);  
	    
	    return 1;
	  }
	}
    }
  else
    if(t->i0==1)
      {
	ax = t->v1[0] - t->v0[0];
	az = t->v1[2] - t->v0[2];
	bx = t->v2[0] - t->v0[0];
	bz = t->v2[2] - t->v0[2];
	delx = r->end[0] - r->orig[0];
	delz = r->end[2] - r->orig[2];
	basex = r->orig[0] - t->v0[0];
	basez = r->orig[2] - t->v0[2];

	bary3 = signSrc*(az *delx - ax *delz) + d*(az*basex - ax*basez);
	bary1 = ( signSrc*(delz*bx - delx*bz) + d*(basez*bx - basex*bz));
	if( bary3 * bary1 >=0.0)
	  {
	    cx = t->v2[0] - t->v1[0];
	    cz = t->v2[2] - t->v1[2];
	    basex = r->orig[0] - t->v1[0];
	    basez = r->orig[2] - t->v1[2];
	    bary2 = ( signSrc*(delz*cx - delx*cz) + d*(basez*cx - basex*cz));
	    if( bary3 * bary2 <0.0){	
	      div1 = 1.f/d;
	      
	      *(p->t) = -signSrc*div1;
	      div = t->inv_n;
	      *(p->u) = fabs(bary2*div);
	      *(p->v) = fabs(bary1*div);
	      
	      return 1;
	    }
	  }
      }
    else
      {
	ax = t->v1[0] - t->v0[0];
	ay = t->v1[1] - t->v0[1];
	bx = t->v2[0] - t->v0[0];
	by = t->v2[1] - t->v0[1];
	delx = r->end[0] - r->orig[0];
	dely = r->end[1] - r->orig[1];
	basex = r->orig[0] - t->v0[0];
	basey = r->orig[1] - t->v0[1];
      
	bary3 = signSrc*(ax *dely - ay *delx) + d*(ax*basey - ay*basex);
	bary1 = ( signSrc*(delx*by - dely*bx) + d*(basex*by - basey*bx));
	if( bary3 * bary1 >=0.0)
	  {
	    cx = t->v2[0] - t->v1[0];
	    cy = t->v2[1] - t->v1[1];
	    basex = r->orig[0] - t->v1[0];
	    basey = r->orig[1] - t->v1[1];
	    bary2 = ( signSrc*(delx*cy - dely*cx) + d*(basex*cy - basey*cx));
	    if( bary3 *  bary2 <0.0){	
	      div1 = 1.f/d;
	      
	      *(p->t) = -signSrc*div1;
	      div = t->inv_n;
	      *(p->u) = fabs(bary2*div);
	      *(p->v) = fabs(bary1*div);
	      return 1;
	    }
	  }
      }
  return 0;
}

/*
  CH2p - Original Chirkov code
*/

int chirkov2_other_bary(Ray_big *r, Triangle_plane *t, Intersection_big *p)
{
  float ax;
  float ay ;
  float az;
  float bx;
  float by;
  float bz;
  float delx;
  float dely;
  float delz;
  float basex;
  float basey;
  float basez;
  float adelx;
  float adely;
  float adelz;
  float cx;
  float cy;
  float cz;
  float d,bary1,bary2,div1,div;
  float signSrc = t->normal[0]*r->orig[0] + t->normal[1]*r->orig[1] + t->normal[2]*r->orig[2] + t->d;
  float signDst = t->normal[0]*r->end[0] + t->normal[1]*r->end[1] + t->normal[2]*r->end[2] + t->d;

  if(signSrc*signDst>=0.0)	
    return 0;
  
  d = signSrc - signDst;
  
  if(t->i0==0)
    {
      ay = t->v1[1] - t->v0[1];
      az = t->v1[2] - t->v0[2];
      by = t->v2[1] - t->v0[1];
      bz = t->v2[2] - t->v0[2];
      dely = r->end[1] - r->orig[1];
      delz = r->end[2] - r->orig[2];
      basey = r->orig[1] - t->v0[1];
      basez = r->orig[2] - t->v0[2];
      
      adelx = signSrc*(ay * delz - az * dely);
      bary1 = ( signSrc*(dely*bz - delz*by) + d*(basey*bz - basez*by));
      if( (adelx + d*(ay*basez - az*basey)) * bary1 >=0.0)
	{
	  cy = t->v2[1] - t->v1[1];
	  cz = t->v2[2] - t->v1[2];
	  basey = r->orig[1] - t->v1[1];
	  basez = r->orig[2] - t->v1[2];
	  bary2 =  (signSrc*(dely*cz - delz*cy) + d*(basey*cz - basez*cy));
	  if( (adelx + d*(ay*basez - az*basey)) * bary2 <0.0){	
	    div1 = 1.f/d;
	    
	    *(p->t) = -signSrc*div1;
	    div = t->inv_n;
	    *(p->u) = fabs(bary2*div);
	    *(p->v) = fabs(bary1*div);  
	    
	    return 1;
	  }
	}
    }
  else
    if(t->i0==1)
      {
	ax = t->v1[0] - t->v0[0];
	az = t->v1[2] - t->v0[2];
	bx = t->v2[0] - t->v0[0];
	bz = t->v2[2] - t->v0[2];
	delx = r->end[0] - r->orig[0];
	delz = r->end[2] - r->orig[2];
	basex = r->orig[0] - t->v0[0];
	basez = r->orig[2] - t->v0[2];
	adely = signSrc*(az * delx - ax * delz);
	bary1 = ( signSrc*(delz*bx - delx*bz) + d*(basez*bx - basex*bz));
	if( (adely + d*(az*basex - ax*basez)) * bary1 >=0.0)
	  {
	    cx = t->v2[0] - t->v1[0];
	    cz = t->v2[2] - t->v1[2];
	    basex = r->orig[0] - t->v1[0];
	    basez = r->orig[2] - t->v1[2];
	    bary2 = ( signSrc*(delz*cx - delx*cz) + d*(basez*cx - basex*cz));
	    if( (adely + d*(az*basex - ax*basez)) * bary2 <0.0){	
	      div1 = 1.f/d;
	      
	      *(p->t) = -signSrc*div1;
	      div = t->inv_n;
	      *(p->u) = fabs(bary2*div);
	      *(p->v) = fabs(bary1*div);
	      
	      return 1;
	    }
	  }
      }
    else
      {
	ax = t->v1[0] - t->v0[0];
	ay = t->v1[1] - t->v0[1];
	bx = t->v2[0] - t->v0[0];
	by = t->v2[1] - t->v0[1];
	delx = r->end[0] - r->orig[0];
	dely = r->end[1] - r->orig[1];
	basex = r->orig[0] - t->v0[0];
	basey = r->orig[1] - t->v0[1];
	adelz = signSrc*(ax * dely - ay * delx);
	bary1 = ( signSrc*(delx*by - dely*bx) + d*(basex*by - basey*bx));
	if( (adelz + d*(ax*basey - ay*basex)) * bary1 >=0.0)
	  {
	    cx = t->v2[0] - t->v1[0];
	    cy = t->v2[1] - t->v1[1];
	    basex = r->orig[0] - t->v1[0];
	    basey = r->orig[1] - t->v1[1];
	    bary2 = ( signSrc*(delx*cy - dely*cx) + d*(basex*cy - basey*cx));
	    if( (adelz + d*(ax*basey - ay*basex)) *  bary2 <0.0){	
	      div1 = 1.f/d;
	      
	      *(p->t) = -signSrc*div1;
	      div = t->inv_n;
	      *(p->u) = fabs(bary2*div);
	      *(p->v) = fabs(bary1*div);
	      return 1;
	    }
	  }
      }
  return 0;
}




/*
  MApl - Plucker rtnv15n1 Mahovsky with precalculated plucker coordinates, only works on ccw ray-triangle relationship
*/

int plucker_mahovsky_other_bary(Ray_big *r, Plucker_coords *t, Intersection_big *p)
{

  float r0,r1,r2,r3,r4,r5;
  float A,B,bb0,bb1;
  float tmp1[3],  normal[3], vd, vo,div1;
  /* Plucker för linjen*/
  r0 = r->orig[0] * r->end[1] - r->orig[1] * r->end[0];
  r1 = r->orig[0] * r->end[2] - r->orig[2] * r->end[0];
  r2 = r->orig[0] - r->end[0];
  r3 = r->orig[1] * r->end[2] - r->orig[2] * r->end[1];
  r4 = r->orig[2] - r->end[2];
  r5 = r->end[1] - r->orig[1];
  
  
  A = r0*t->a4 + r1*t->a5 +r3*t->a2;
  bb0 = A+ r2*t->a3 +r4*t->a0+r5*t->a1; 
  if( bb0< 0)
    return 0;
  
  B = r0*t->b4 + r1*t->b5 + r3*t->b2;
  bb1 = B + r2*t->b3 + r4*t->b0 +r5*t->b1;
  if( bb1 < 0)
    return 0;

  
  if(r2*t->c3 + r4*t->c0 + r5*t->c1 - A - B < 0)
    return 0;
  
    /*  Räkna ut avståndet*/
  
  normal[0] = t->a3 + t->b3 + t->c3;
  normal[2] = t->a0 + t->b0 + t->c0;
  normal[1] =  t->a1 + t->b1 + t->c1;
  
  
  SUB(tmp1, t->v0, r->orig);
  
  vo = DOT(tmp1, normal);
  vd = DOT(r->dir, normal);
  if(vd == 0)
      return 0;
  *(p->t) = vo/vd;
  div1 = 1.f/(normal[0]*r2+normal[1]*r5+normal[2]*r4);
  *(p->v) = bb0*div1;
  *(p->u) = bb1*div1;


  return 1;
}


/*
  PUpl - Plucker coordinates version that handles non ccw ray triangle relationship
*/
int plucker_other_bary(Ray_big *r, Plucker_coords *t, Intersection_big *p)
{
  
  float r0,r1,r2,r3,r4,r5;
  float A,B;
  int a,b,c;	
  float tmp1[3], normal[3], vd, vo, div1;
  
  /* Plucker för linjen*/
  r0 = r->orig[0] * r->end[1] - r->orig[1] * r->end[0];
  r1 = r->orig[0] * r->end[2] - r->orig[2] * r->end[0];
  r2 = r->orig[0] - r->end[0];
  r3 = r->orig[1] * r->end[2] - r->orig[2] * r->end[1];
  r4 = r->orig[2] - r->end[2];
  r5 = r->end[1] - r->orig[1];
  
  
  A = r0*t->a4 + r1*t->a5 +r3*t->a2;
  B = r0*t->b4 + r1*t->b5 + r3*t->b2;
  a = A + r2*t->a3 +r4*t->a0+r5*t->a1 >=0.;
  b = B + r2*t->b3 + r4*t->b0 +r5*t->b1 >= 0.;
  
  if(a != b)
    return 0;
  
  c = r2*t->c3 + r4*t->c0 + r5*t->c1 - A - B >= 0.;
  
  if(c != a)
    return 0;

    /*  Räkna ut avståndet*/
  
  normal[0] = t->a3 + t->b3 + t->c3;
  normal[2] = t->a0 + t->b0 + t->c0;
  normal[1] = t->a1 + t->b1 + t->c1;
  
  div1 = 1.f/(normal[0]*r2+normal[1]*r5+normal[2]*r4);
  *(p->v) = b*div1;
  *(p->u) = a*div1;
  
  SUB(tmp1, t->v0, r->orig);
  
  vo = DOT(tmp1, normal);
  
  vd = DOT(r->dir, normal);
  if(vd == 0)
      return 0;
  *(p->t) = vo/vd; 

  return 1;
}

/* ORp orourke with precalculated plane equation*/
int orourke_other_bary(Ray_big *r, Triangle_plane *t, Intersection_big *p){
  
  float vo, vd; 
  float vol0, vol1, vol2; 
  float ax,ay,az,bx,by,bz,cx,cy,cz,dx,dy,dz,div;
 

  ax = r->orig[0] - r -> end[0];
  ay = r->orig[1] - r -> end[1];
  az = r->orig[2] - r -> end[2];
  bx = t->v0[0] - r -> end[0];
  by = t->v0[1] - r -> end[1];
  bz = t->v0[2] - r -> end[2];
  cx = t->v1[0] - r -> end[0];
  cy = t->v1[1] - r -> end[1];
  cz = t->v1[2] - r -> end[2];
 dx = t->v2[0] - r -> end[0];
  dy = t->v2[1] - r -> end[1];
  dz = t->v2[2] - r -> end[2];

  

  vol0 = (ax * (by*cz - bz*cy) + ay * (bz*cx - bx*cz) + az *(bx*cy - by*cx));
  vol1 = (ax * (cy*dz - cz*dy) + ay * (cz*dx - cx*dz) + az *(cx*dy - cy*dx));
 
  
  if(vol0*vol1<0)
    return 0;
 

  vol2 = (ax * (dy*bz - dz*by) + ay * (dz*bx - dx*bz) + az *(dx*by - dy*bx));
  
  if(vol0*vol2<0)
    return 0;
  vd = t->normal[0]*r->dir[0] + t->normal[1]*r->dir[1] + t->normal[2]*r->dir[2];
  if(vd == 0)
      return 0;
  vo = -(t->normal[0]*r->orig[0] + t->normal[1]*r->orig[1] + t->normal[2]*r->orig[2] + t->d);
  *(p->t) = vo/vd;
  div = 1.f/(t->normal[0]*ax + t->normal[1]*ay + t->normal[2]*az);
  *(p->u) = vol1*div;
  *(p->v) = vol0*div;
  return 1;
 
}


/* ORCp - Optimization of ORourke only handles ccw ray-traingle relationship*/
int orourke_other_baryCCW(Ray_big *r, Triangle_plane *t, Intersection_big *p){

  float vo, vd, div; 
  float vol0, vol1, vol2;
  float ax,ay,az,bx,by,bz,cx,cy,cz,dx,dy,dz;
  

  ax = r->orig[0] - r -> end[0];
  ay = r->orig[1] - r -> end[1];
  az = r->orig[2] - r -> end[2];
  bx = t->v0[0] - r -> end[0];
  by = t->v0[1] - r -> end[1];
  bz = t->v0[2] - r -> end[2];
  cx = t->v1[0] - r -> end[0];
  cy = t->v1[1] - r -> end[1];
  cz = t->v1[2] - r -> end[2];


  vol0 = (ax * (by*cz - bz*cy) + ay * (bz*cx - bx*cz) + az *(bx*cy - by*cx));
  
  if(vol0 < 0)
    return 0;
  dx = t->v2[0] - r -> end[0];
  dy = t->v2[1] - r -> end[1];
  dz = t->v2[2] - r -> end[2];
  
  vol1 = (ax * (cy*dz - cz*dy) + ay * (cz*dx - cx*dz) + az *(cx*dy - cy*dx));

  if(vol1<0)
    return 0;
  vol2 = (ax * (dy*bz - dz*by) + ay * (dz*bx - dx*bz) + az *(dx*by - dy*bx));
  
  if(vol2<0)
    return 0;
    
    
  vd = t->normal[0]*r->dir[0] + t->normal[1]*r->dir[1] + t->normal[2]*r->dir[2];
  if(vd==0)
    return 0;
  vo = -(t->normal[0]*r->orig[0] + t->normal[1]*r->orig[1] + t->normal[2]*r->orig[2] + t->d);
  *(p->t) = vo/vd;
  div = 1.f/(t->normal[0]*ax + t->normal[1]*ay + t->normal[2]*az);
  *(p->u) = vol1*div;
  *(p->v) = vol0*div;
   
    return 1;

}


 
 /* 
    HFp - Adaptaion of Half plane GReen rtn januari 93 Haines GG IV
  
*/
int halfplane_other_bary(Ray_big *r, Triangle_plane *t, Intersection_big *p)
{
  
  float n0x,n0y,c0,n1x,n1y,c1;
  int X = t->i1;
  int Y = t->i2;
  float p0, p1;
  float tt,vd,vo;
  float sign1,sign2;
  
  vd = DOT(t->normal, r->dir);
  if(vd == 0)
      return 0;
  vo = -(DOT(t->normal, r->orig) + t->d);
  
  if(vd*vo <= 0)
    {
      return 0;
  }
  tt = vo/vd;
  
  p0 = r->orig[X] + tt*r->dir[X];
  p1 = r->orig[Y] + tt*r->dir[Y];
 
  /*calculate halfplane equation for first line f(p) = n*p + c, where n is perpendicular to line though V0 and V1, prepDot*/
  n0x = -(t->v0[Y]- t->v1[Y]);
  n0y = t->v0[X] - t->v1[X];
  c0 = - t->v1[X]*n0x - t->v1[Y]*n0y;
  
  n1x = -(t->v1[Y] - t->v2[Y]);
  n1y = t->v1[X] - t->v2[X];
  c1 = - t->v2[X]*n1x - t->v2[Y]*n1y;
  
  sign1 = p0*n0x + p1*n0y + c0 ;
  sign2 = p0*n1x + p1*n1y + c1 ;
  
  if(sign1*sign2>0 )
    {
      n1x = -(t->v2[Y] - t->v0[Y]);
      n1y = t->v2[X] - t->v0[X];
      c1 = - t->v0[X]*n1x - t->v0[Y]*n1y;
      sign2 = p0*n1x + p1*n1y + c1;
      if(sign1*sign2>0){
	*(p->t) = tt;
     *(p->u)=sign1*t->inv_n;
	*(p->v)=sign2*t->inv_n; 
	return 1;
      }
    }  
  return 0;
}




/* HF2h - Halfplane with scaled t-value to delay division */
int halfplane2_other_bary(Ray_big *r, Triangle_plane *t, Intersection_big *p)
{
  float n0x,n0y,c0,n1x,n1y,c1;
  int X = t->i1;
  int Y = t->i2;
  float vd,vo;
  float sign1,sign2;

  vd = DOT(t->normal, r->dir);
  vo = - DOT(t->normal, r->orig) - t->d;
  
  if(vd*vo <= 0)
    return 0;
  
  n0x = -(t->v0[Y]- t->v1[Y]);
  n0y = t->v0[X] - t->v1[X];
  c0 = - t->v1[X]*n0x - t->v1[Y]*n0y;
  
  n1x = -(t->v1[Y] - t->v2[Y]);
  n1y = t->v1[X] - t->v2[X];
  c1 = - t->v2[X]*n1x - t->v2[Y]*n1y;

  sign1 = vd*(n0x*r->orig[X] + n0y*r->orig[Y]) + vo*(n0y * r->dir[Y] + n0x * r->dir[X]) + c0*vd; 
  sign2 = vd*(n1x*r->orig[X] + n1y*r->orig[Y]) + vo*(n1y * r->dir[Y] + n1x * r->dir[X]) + c1*vd;


  if(sign1*sign2>0)
    {
      n1x = -(t->v2[Y] - t->v0[Y]);
      n1y = t->v2[X] - t->v0[X];
      c1 = - t->v0[X]*n1x - t->v0[Y]*n1y;
      
      sign2 = vd*(n1x*r->orig[X] + n1y*r->orig[Y]) + vo*(n1y * r->dir[Y] + n1x * r->dir[X]) + c1*vd;
      if(sign1*sign2>0){
	float div = 1/vd;
	*(p->t) = vo*div;
	*(p->u)=div*sign1*t->inv_n;
	*(p->v)=div*sign2*t->inv_n;

	return 1;
      }
      
    }  
  return 0;
}


/*
  A2Dp - Area2D with precalculated plane
*/
int area2D_other_bary(Ray_big *r, Triangle_plane *t, Intersection_big *p)
{
   
  float vd,vo,tt,point[2];
 
  float v0x,v0y,v1x,v1y;
  float area1,area2,area3;
  int X = t-> i1;
  int Y = t-> i2;
 
  vd = DOT(t->normal, r->dir);
  if(vd == 0)
      return 0;
  vo = -(DOT(t->normal, r->orig) + t->d);
  
  if(vd*vo <= 0)
    {
      
      return 0;
  }
  tt = vo/vd;
 
  point[0] = r->orig[X] + tt*r->dir[X];
  point[1] = r->orig[Y] + tt*r->dir[Y];

  v0x = t->v1[X] - t->v0[X];
  v0y = t->v1[Y] - t->v0[Y];

  v1x = point[0] - t->v0[X];
  v1y = point[1] - t->v0[Y];

  area1 = v0x*v1y - v1x*v0y;

  v0x = t->v2[X] - t->v1[X];
  v0y = t->v2[Y] - t->v1[Y];

  v1x = point[0] - t->v1[X];
  v1y = point[1] - t->v1[Y];

  area2 = v0x*v1y - v1x*v0y;
  

  if(area1*area2 > 0.){
    v0x = t->v0[X] - t->v2[X];
    v0y = t->v0[Y] - t->v2[Y];
    
    v1x = point[0] - t->v2[X];
    v1y = point[1] - t->v2[Y];
    
    area3 = v0x*v1y - v1x*v0y;
    if(area1*area3 > 0.){
     
      
      *(p->t)=tt;
      *(p->u)=fabs(area1*t->inv_n);
      *(p->v)=fabs(area2*t->inv_n);
      return 1;
    }
  }
  return 0;
  
}



/* ARi - This is arenberg running on precomputed inverse. NOTE, the normal is == -bb2*/

int arenberg_other_bary_pre(Ray_big *r, Triangle_inv *t, Intersection_big *p)
{
  float num, den, tt,a,b;
  float trans1[3],point[3];
  
  den = DOT(r ->dir, t->bb2);
 
  if(den == 0.){
   return 0;
  }
  
  SUB(trans1, t -> v0, r -> orig);
  num = DOT(trans1, t->bb2);

  tt = num/den;
  
  if(tt <=0.){
    return 0;
  }
  
   
  SUB(trans1, r -> orig, t -> v0);
  point[0] = (tt * r -> dir[0]) + trans1[0];
  point[1] = (tt * r -> dir[1]) + trans1[1];
  point[2] = (tt * r -> dir[2]) + trans1[2];

  a = DOT(point,t->bb0);
  b = DOT(point,t->bb1);
 
  if( a < 0.0 || b < 0.0 || a + b > 1.0){
   return 0;
  }

 
  *(p->t) = tt;
  *(p->u) = a;
  *(p->v) = b;
  return 1;
}

/* AR2i - Optimized version of Arenberg */
int arenberg_other_bary_pre2(Ray_big *r, Triangle_inv *t, Intersection_big *p)
{
  float num, den, a, b, div;
  float trans1[3];

  den = DOT(r ->dir, t->bb2);

  if(den == 0.)
   return 0;
  
  SUB(trans1, t -> v0, r -> orig);
  num = DOT(trans1, t->bb2);

  if(den*num <= 0){
    return 0;
  }
  
  SUB(trans1, r -> orig, t -> v0);

  a = DOT(trans1, t->bb0)*den + num*DOT(r->dir,t->bb0);

  if(a <0)
    return 0;

  b = DOT(trans1, t->bb1)*den + num*DOT(r->dir,t->bb1);
  
  
  if(b <0)
    return 0;
  
  if(a + b > den)
    return 0;
 
  div = 1.f/den;
  *(p->t) = num*div;
  *(p->u) = a*div;
  *(p->v) = b*div;
  return 1;
}


 /*
   HFn - Adaptaion of Half plane GReen rtn januari 93 Haines GG IV
   with precalculated halfplane equations
*/
int halfplane_other_bary_pre(Ray_big *r, Triangle_Halfplane *t, Intersection_big *p)
{
 
  int X = t->i1;
  int Y = t->i2;
  float p0, p1;
  float tt,vd,vo;
  float sign1,sign2;
  
  vd = DOT(t->normal, r->dir);
  if(vd == 0)
      return 0;
  vo = -(DOT(t->normal, r->orig) + t->d);
  
  if(vd*vo <= 0)
    {
      
      return 0;
  }
  tt = vo/vd;
  
  p0 = r->orig[X] + tt*r->dir[X];
  p1 = r->orig[Y] + tt*r->dir[Y];
  
  sign1 = p0*t->n0x + p1*t->n0y + t->c0 ;
  sign2 = p0*t->n1x + p1*t->n1y + t->c1 ;
  
  if(sign1*sign2>0 )
    {
      sign2 = p0*t->n2x + p1*t->n2y + t->c2;
      if(sign1*sign2>0){
	*(p->t) = tt;
	*(p->u)=sign1*t->inv_n;
	*(p->v)=sign2*t->inv_n;
	return 1;
      }
    }  
  return 0;
}



/* HF2h - with scaled t-value to delay division */
int halfplane_other_bary_pre2(Ray_big *r, Triangle_Halfplane *t, Intersection_big *p)
{
 
  int X,Y;
  float sign1,sign2, sign3, div;
  float vd,vo;

  vd = DOT(t->normal, r->dir);
 
  vo = - DOT(t->normal, r->orig) - t->d;
  
  if(vd*vo <= 0)
    return 0;
 
  X = t->i1;
  Y = t->i2;

  sign1 = vd*(t->n0x*r->orig[X] + t->n0y*r->orig[Y]) + vo*(t->n0y * r->dir[Y] + t->n0x * r->dir[X]) + t->c0*vd; 
  sign2 = vd*(t->n1x*r->orig[X] + t->n1y*r->orig[Y]) + vo*(t->n1y * r->dir[Y] + t->n1x * r->dir[X]) + t->c1*vd;
 
  if(sign1*sign2<=0)
    return 0;
  
  sign3 = vd*(t->n2x*r->orig[X] + t->n2y*r->orig[Y]) + vo*(t->n2y * r->dir[Y] + t->n2x * r->dir[X]) + t->c2*vd;
  if(sign1*sign3>0){
    div = 1/vd;
    *(p->t) = vo*div;
    *(p->u)= div*sign1*t->inv_n;
    *(p->v)=div*sign2*t->inv_n;
    return 1;
  }
  
  
  return 0;
}

/************************************************************************************

 The version of the algorithms below only have to calculate the t-value. 
When possible algorithms that use barycentric coordinates to determine if there is a hit 
have be optimized to work on scled values and therefore have to do less divisions 

 ***********************************************************************************/


/*MT0*/
int intersect_triangle_small_t(Ray_big *r, Triangle_small *t, Intersection_big *p)
{
  float edge1[3], edge2[3], tvec[3], pvec[3], qvec[3];
  float det,inv_det;
  
  
  /* find vectors for two edges sharing vert0 */
  SUB(edge1, t -> v1, t -> v0);
  SUB(edge2, t -> v2, t -> v0);
  
  /* begin calculating determinant - also used to calculate U parameter */
  CROSS(pvec, r -> dir, edge2);
  
  /* if determinant is near zero, ray lies in plane of triangle */
  det = DOT(edge1, pvec);
  
  if (det > -EPSILON && det < EPSILON){
    return 0;
  }
  inv_det = 1.0f / det;
  
  /* calculate distance from vert0 to ray origin */
  SUB(tvec, r -> orig, t -> v0);
  
  /* calculate U parameter and test bounds */
  *(p -> u) = DOT(tvec, pvec) * inv_det;
  if (*(p->u) < 0.0 || *(p->u) > 1.0)
    return 0;
  
  /* prepare to test V parameter */
  CROSS(qvec, tvec, edge1);
  
  /* calculate V parameter and test bounds */
  *(p->v) = DOT(r->dir, qvec) * inv_det;
  if (*(p->v) < 0.0 || *(p->u) + *(p->v) > 1.0)
    return 0;
  
  /* calculate t, ray intersects triangle */
  *(p->t) = DOT(edge2, qvec) * inv_det;
  return 1;
}

/* MT1*/
/* code rewritten to do tests on the sign of the determinant */
/* the division is at the end in the code                    */
int intersect_triangle1_small_t(Ray_big *r, Triangle_small *t, Intersection_big *p)
{
  float edge1[3], edge2[3], tvec[3], pvec[3], qvec[3];
  float det,inv_det;
  
  /* find vectors for two edges sharing vert0 */
  SUB(edge1, t ->v1, t ->v0);
  SUB(edge2, t ->v2, t->v0);
  
  /* begin calculating determinant - also used to calculate U parameter */
  CROSS(pvec, r ->dir, edge2);
  
  /* if determinant is near zero, ray lies in plane of triangle */
  det = DOT(edge1, pvec);
  
  if (det > EPSILON)
    {
      /* calculate distance from vert0 to ray origin */
      SUB(tvec, r->orig, t ->v0);
      
      /* calculate U parameter and test bounds */
      *(p -> u) = DOT(tvec, pvec);
      if (*(p->u) < 0.0 || *(p->u) > det)
	return 0;
      
      /* prepare to testV parameter */
      CROSS(qvec, tvec, edge1);
      
      /* calculate V parameter and test bounds */
      *(p->v) = DOT(r->dir, qvec);
      if (*(p->v) < 0.0 || *(p->u) + *(p->v) > det)
	return 0;
      
    }
  else if(det < -EPSILON)
    {
      /* calculate distance from vert0 to ray origin */
      SUB(tvec, r->orig, t->v0);
      
      /* calculate U parameter and test bounds */
      *(p->u) = DOT(tvec, pvec);
      /*      printf("*u=%f\n",(float)*u); */
      /*      printf("det=%f\n",det); */
      if (*(p->u) > 0.0 || *(p->u) < det)
	return 0;
      
      /* prepare to test V parameter */
      CROSS(qvec, tvec, edge1);
      
      /* calculate V parameter and test bounds */
      *(p->v) = DOT(r->dir, qvec) ;
      if (*(p->v) > 0.0 || *(p->u) + *(p->v) < det)
	return 0;
    }
  else return 0;  /* ray is parallell to the plane of the triangle */
  
  
  inv_det = 1.0f / det;
  
  /* calculate t, ray intersects triangle */
  *(p->t) = DOT(edge2, qvec) * inv_det;
  return 1;
}

/* MT2 */
/* code rewritten to do tests on the sign of the determinant */
/* the division is before the test of the sign of the det    */
int intersect_triangle2_small_t(Ray_big *r, Triangle_small *t, Intersection_big *p)
{
  float edge1[3], edge2[3], tvec[3], pvec[3], qvec[3];
  float det,inv_det;
  
  /* find vectors for two edges sharing vert0 */
  SUB(edge1, t->v1, t ->v0);
  SUB(edge2, t->v2, t->v0);
  
  /* begin calculating determinant - also used to calculate U parameter */
  CROSS(pvec, r->dir, edge2);
  
  /* if determinant is near zero, ray lies in plane of triangle */
  det = DOT(edge1, pvec);
  
  /* calculate distance from vert0 to ray origin */
  SUB(tvec, r->orig, t->v0);
  if(det != 0.)
    inv_det = 1.0f / det;
  else
    return 0;
  if (det > EPSILON)
    {
      /* calculate U parameter and test bounds */
      *(p->u) = DOT(tvec, pvec);
      if (*(p->u) < 0.0 || *(p->u) > det)
	return 0;
      
      /* prepare to test V parameter */
      CROSS(qvec, tvec, edge1);
      
      /* calculate V parameter an test bounds */
      *(p->v) = DOT(r->dir, qvec);
      if (*(p->v) < 0.0 || *(p->u) + *(p->v) > det)
	return 0;
      
    }
  else if(det < -EPSILON)
    {
      /* calculate U parameter and test bounds */
      *(p->u) = DOT(tvec, pvec);
      if (*(p->u) > 0.0 || *(p->u) < det)
	return 0;
      
      /* prepare to test V parameter */
      CROSS(qvec, tvec, edge1);
      
      /* calculate V parameter and test bounds */
      *(p->v) = DOT(r->dir, qvec) ;
      if (*(p->v) > 0.0 || *(p->u) + *(p->v) < det)
	return 0;
    }
  else return 0;  /* ray is parallell to the plane of the triangle */
  *(p->t) = DOT(edge2, qvec) * inv_det;
  
  return 1;
}

/* MT3 */
/* code rewritten to do tests on the sign of the determinant */
/* the division is before the test of the sign of the det    */
/* and one CROSS has been moved out from the if-else if-else */
int intersect_triangle3_small_t(Ray_big *r, Triangle_small *t, Intersection_big *p)
{
  float edge1[3], edge2[3], tvec[3], pvec[3], qvec[3];
  float det,inv_det;
  
  /* find vectors for two edges sharing vert0 */
  SUB(edge1, t->v1, t->v0);
  SUB(edge2, t->v2, t->v0);
  
  /* begin calculating determinant - also used to calculate U parameter */
  CROSS(pvec, r->dir, edge2);
  
  /* if determinant is near zero, ray lies in plane of triangle */
  det = DOT(edge1, pvec);
  
  /* calculate distance from vert0 to ray origin */
  SUB(tvec, r->orig, t->v0);
  if(det != 0)
  inv_det = 1.0f / det;
  else
    return 0;
  CROSS(qvec, tvec, edge1);
  
  if (det > EPSILON)
    {
      *(p->u) = DOT(tvec, pvec);
      if (*(p->u) < 0.0 || *(p->u) > det)
	return 0;
      
      /* calculate V parameter and test bounds */
      *(p->v) = DOT(r->dir, qvec);
      if (*(p->v) < 0.0 || *(p->u) + *(p->v) > det)
	return 0;
      
    }
  else if(det < -EPSILON)
    {
      /* calculate U parameter and test bounds */
      *(p->u) = DOT(tvec, pvec);
      if (*(p->u) > 0.0 || *(p->u) < det)
	return 0;
      
      /* calculate V parameter and test bounds */
      *(p->v) = DOT(r->dir, qvec) ;
      if (*(p->v) > 0.0 || *(p->u) + *(p->v) < det)
	return 0;
    }
  else return 0;  /* ray is parallell to the plane of the triangle */
  
  *(p->t) = DOT(edge2, qvec) * inv_det;
   return 1;
}


/* OR  */
int orourke_small_t(Ray_big *r, Triangle_small *t, Intersection_big *p){

  float vo, vd; 
  float tmp1[3], tmp2[3], normal[3],d;
  float vol0, vol1, vol2;
  float ax,ay,az,bx,by,bz,cx,cy,cz,dx,dy,dz;
  

  ax = r->orig[0] - r -> end[0];
  ay = r->orig[1] - r -> end[1];
  az = r->orig[2] - r -> end[2];
  bx = t->v0[0] - r -> end[0];
  by = t->v0[1] - r -> end[1];
  bz = t->v0[2] - r -> end[2];
  cx = t->v1[0] - r -> end[0];
  cy = t->v1[1] - r -> end[1];
  cz = t->v1[2] - r -> end[2];

  dx = t->v2[0] - r -> end[0];
  dy = t->v2[1] - r -> end[1];
  dz = t->v2[2] - r -> end[2];
  vol0 = (ax * (by*cz - bz*cy) + ay * (bz*cx - bx*cz) + az *(bx*cy - by*cx));
  vol1 = (ax * (cy*dz - cz*dy) + ay * (cz*dx - cx*dz) + az *(cx*dy - cy*dx));

  if(vol0*vol1<0)
    return 0;



  vol2 = (ax * (dy*bz - dz*by) + ay * (dz*bx - dx*bz) + az *(dx*by - dy*bx));
  
  if(vol0*vol2<0)
    return 0;
    tmp1[0] = t->v0[0] - t-> v1[0];
    tmp1[1] = t->v0[1] - t-> v1[1];
    tmp1[2] = t->v0[2] - t-> v1[2];
    tmp2[0] = t->v0[0] - t-> v2[0];
    tmp2[1] = t->v0[1] - t-> v2[1];
    tmp2[2] = t->v0[2] - t-> v2[2];
    
    CROSS(normal, tmp1, tmp2);
    d = -(DOT(normal, t-> v0));
    
    vd = normal[0]*r->dir[0] + normal[1]*r->dir[1] + normal[2]*r->dir[2];
    vo = -(normal[0]*r->orig[0] + normal[1]*r->orig[1] + normal[2]*r->orig[2] + d);
    *(p->t) = vo/vd;
    
    
    return 1;

}

/* ORC */
int orourke_small_tCCW(Ray_big *r, Triangle_small *t, Intersection_big *p){

  float vo, vd; 
  float tmp1[3], tmp2[3], normal[3],d;
  float vol0, vol1, vol2;
  float ax,ay,az,bx,by,bz,cx,cy,cz,dx,dy,dz;
  

  ax = r->orig[0] - r -> end[0];
  ay = r->orig[1] - r -> end[1];
  az = r->orig[2] - r -> end[2];
  bx = t->v0[0] - r -> end[0];
  by = t->v0[1] - r -> end[1];
  bz = t->v0[2] - r -> end[2];
  cx = t->v1[0] - r -> end[0];
  cy = t->v1[1] - r -> end[1];
  cz = t->v1[2] - r -> end[2];


  vol0 = (ax * (by*cz - bz*cy) + ay * (bz*cx - bx*cz) + az *(bx*cy - by*cx));
  
  if(vol0 < 0)
    return 0;
  dx = t->v2[0] - r -> end[0];
  dy = t->v2[1] - r -> end[1];
  dz = t->v2[2] - r -> end[2];
  
  vol1 = (ax * (cy*dz - cz*dy) + ay * (cz*dx - cx*dz) + az *(cx*dy - cy*dx));

  if(vol1<0)
    return 0;
  vol2 = (ax * (dy*bz - dz*by) + ay * (dz*bx - dx*bz) + az *(dx*by - dy*bx));
  
  if(vol2<0)
    return 0;
    tmp1[0] = t->v0[0] - t-> v1[0];
    tmp1[1] = t->v0[1] - t-> v1[1];
    tmp1[2] = t->v0[2] - t-> v1[2];
    tmp2[0] = t->v0[0] - t-> v2[0];
    tmp2[1] = t->v0[1] - t-> v2[1];
    tmp2[2] = t->v0[2] - t-> v2[2];
    
    CROSS(normal, tmp1, tmp2);
    d = -(DOT(normal, t-> v0));
    
    vd = normal[0]*r->dir[0] + normal[1]*r->dir[1] + normal[2]*r->dir[2];
    vo = -(normal[0]*r->orig[0] + normal[1]*r->orig[1] + normal[2]*r->orig[2] + d);
    *(p->t) = vo/vd;
    
   
    return 1;

}

/* MA */
int plucker_mahovsky_small_t(Ray_big *r, Triangle_small *t, Intersection_big *p)
{
  float a0,a1,a2,a3,a4,a5,b0,b1,b2,b3,b4,b5,c0,c1,c3;
  float r0,r1,r2,r3,r4,r5;
  float A,B;
  float tmp1[3], normal[3], vd, vo;
  
  /* Plucker för linjen*/
  r0 = r->orig[0] * r->end[1] - r->orig[1] * r->end[0];
  r1 = r->orig[0] * r->end[2] - r->orig[2] * r->end[0];
  r2 = r->orig[0] - r->end[0];
  r3 = r->orig[1] * r->end[2] - r->orig[2] * r->end[1];
  r4 = r->orig[2] - r->end[2];
  r5 = r->end[1] - r->orig[1];
  
      /*Plucker koefficienter för triangel sidorna*/
  a0 = t->v0[0] * t->v1[1] - t->v1[0] * t->v0[1];
  a1 = t->v0[0] * t->v1[2] - t->v1[0] * t->v0[2];
  a2 = t->v0[0] - t->v1[0];
  a3 = t->v0[1] * t->v1[2] - t->v1[1] * t->v0[2];
  a4 = t->v0[2] - t->v1[2];
  a5 = t->v1[1] - t->v0[1];

  A = r0*a4 + r1*a5 +r3*a2;

  if(A + r2*a3 + r4*a0 + r5*a1 < 0.)
    return 0;

  b0 = t->v1[0] * t->v2[1] - t->v2[0] * t->v1[1];
  b1 = t->v1[0] * t->v2[2] - t->v2[0] * t->v1[2];
  b2 = t->v1[0] - t->v2[0];
  b3 = t->v1[1] * t->v2[2] - t->v2[1] * t->v1[2];
  b4 = t->v1[2] - t->v2[2];
  b5 = t->v2[1] - t->v1[1];
  
 
  B = r0*b4 + r1*b5 + r3*b2;
  
  if(B + r2*b3 + r4*b0 +r5*b1<0.)
    return 0;

  c0 = t->v2[0] * t->v0[1] - t->v0[0] * t->v2[1];
  c1 = t->v2[0] * t->v0[2] - t->v0[0] * t->v2[2];
  c3 = t->v2[1] * t->v0[2] - t->v0[1] * t->v2[2];
  
  if(r2*c3 + r4*c0 + r5*c1 - A - B  < 0.)
    return 0;

  normal[0] = a3 +b3 +c3;
  normal[1] =  a1 +b1 +c1;
  normal[2] = a0 +b0 +c0;
  
  
  SUB(tmp1, t->v0, r->orig);

  vo = DOT(tmp1, normal);
  vd = DOT(r->dir, normal);
  *(p->t) = vo/vd;
  
  
  return 1;
}



/*
  PU
*/
int plucker_small_t(Ray_big *r, Triangle_small *t, Intersection_big *p)
{
  float a0,a1,a2,a3,a4,a5,b0,b1,b2,b3,b4,b5,c0,c1,c3;
  float r0,r1,r2,r3,r4,r5;
  float A,B;
  int a,b,c;	
   float tmp1[3], normal[3],vd, vo;
 
  /* Plucker för linjen*/
  r0 = r->orig[0] * r->end[1] - r->orig[1] * r->end[0];
  r1 = r->orig[0] * r->end[2] - r->orig[2] * r->end[0];
  r2 = r->orig[0] - r->end[0];
  r3 = r->orig[1] * r->end[2] - r->orig[2] * r->end[1];
  r4 = r->orig[2] - r->end[2];
  r5 = r->end[1] - r->orig[1];

    /*Plucker koefficienter för triangel sidorna*/
  a0 = t->v0[0] * t->v1[1] - t->v1[0] * t->v0[1];
  a1 = t->v0[0] * t->v1[2] - t->v1[0] * t->v0[2];
  a2 = t->v0[0] - t->v1[0];
  a3 = t->v0[1] * t->v1[2] - t->v1[1] * t->v0[2];
  a4 = t->v0[2] - t->v1[2];
  a5 = t->v1[1] - t->v0[1];
  
  b0 = t->v1[0] * t->v2[1] - t->v2[0] * t->v1[1];
  b1 = t->v1[0] * t->v2[2] - t->v2[0] * t->v1[2];
  b2 = t->v1[0] - t->v2[0];
  b3 = t->v1[1] * t->v2[2] - t->v2[1] * t->v1[2];
  b4 = t->v1[2] - t->v2[2];
  b5 = t->v2[1] - t->v1[1];
  
  A = r0*a4 + r1*a5 +r3*a2;
  B = r0*b4 + r1*b5 + r3*b2;
  
  a = (A + r2*a3 + r4*a0 + r5*a1 > 0.f);
  b = (B + r2*b3 + r4*b0 +r5*b1 > 0.f);

  if(a != b)
    return 0;
  
  c0 = t->v2[0] * t->v0[1] - t->v0[0] * t->v2[1];
  c1 = t->v2[0] * t->v0[2] - t->v0[0] * t->v2[2];
  c3 = t->v2[1] * t->v0[2] - t->v0[1] * t->v2[2];
  c = r2*c3 + r4*c0 + r5*c1 - A - B  > 0.;
  
  if(c != a)
    return 0;

     /*  Räkna ut avståndet*/
  
  normal[0] = a3 +b3 +c3;
  normal[1] =  a1 +b1 +c1;
  normal[2] = a0 +b0 +c0;
  
  vd = DOT(r->dir, normal);
  if(vd == 0)
    return 0;
  SUB(tmp1, t->v0, r->orig);
  
  vo = DOT(tmp1, normal);
  
  *(p->t) = vo/vd; // OBS inte normaliserad normal !!!!!!! 
  
  return 1;
}

/* CH1p */
int chirkov_other_t(Ray_big *r, Triangle_plane *t, Intersection_big *p)
{
  float signSrc;
  float signDst;
  float ay ;
  float az  ;
  float by  ;
  float bz  ;
  float dely  ;
  float delz ;
  float basey ;
  float basez ;
  float d ;
  float cy;
  float cz;
  int i1 ;
  int i2;
  float adelxbase;

  signSrc = t->normal[0]*r->orig[0] + t->normal[1]*r->orig[1] + t->normal[2]*r->orig[2] + t->d;
  signDst = t->normal[0]*r->end[0] + t->normal[1]*r->end[1] + t->normal[2]*r->end[2] + t->d;
  
  if(signSrc*signDst >= 0.f)
    return 0;

  i1 = t ->  i1;
  i2 = t ->  i2;
  
  d = signSrc - signDst;

  ay =	t->v1[i1] - t->v0[i1];
  az =	t->v1[i2] - t->v0[i2];
  by =	t->v2[i1] - t->v0[i1];
  bz =	t->v2[i2] - t->v0[i2];
  
  dely = r->end[i1] - r->orig[i1] ;
  delz = r->end[i2] - r->orig[i2] ;
  
  basey = r->orig[i1] - t->v0[i1];
  basez = r->orig[i2] - t->v0[i2];

  adelxbase = signSrc*(ay *delz - az *dely) + d*(ay*basez - az*basey);
  
  if(adelxbase * (signSrc*(dely*bz - delz*by)	+ d*(basey*bz -	basez*by)) >= 0.0f){
    
    
    cy = t ->v2[i1] -	t->v1[i1];
    cz = t ->v2[i2] -	t->v1[i2];
    basey = r->orig[i1] - t->v1[i1];
    basez = r->orig[i2] - t->v1[i2];
    if(adelxbase * (signSrc*(dely*cz - delz*cy) + d*(basey*cz - basez*cy)) < 0.0f){
     
      *(p->t) = signSrc/d;
      
      return 1;
    }
  }
  return 0;
}



/*

CH2p

*/

int chirkov2_other_t(Ray_big *r, Triangle_plane *t, Intersection_big *p)
{
  float ax;
  float ay ;
  float az;
  float bx;
  float by;
  float bz;
  float delx;
  float dely;
  float delz;
  float basex;
  float basey;
  float basez;
  float adelx;
  float adely;
  float adelz;
  float cx;
  float cy;
  float cz;
  float d;
  float signSrc = t->normal[0]*r->orig[0] + t->normal[1]*r->orig[1] + t->normal[2]*r->orig[2] + t->d;
  float signDst = t->normal[0]*r->end[0] + t->normal[1]*r->end[1] + t->normal[2]*r->end[2] + t->d;

  if(signSrc*signDst>=0.0)	
    return 0;
  
  d = signSrc - signDst;
  
  if(t->i0==0)
    {
      ay = t->v1[1] - t->v0[1];
      az = t->v1[2] - t->v0[2];
      by = t->v2[1] - t->v0[1];
      bz = t->v2[2] - t->v0[2];
      dely = r->end[1] - r->orig[1];
      delz = r->end[2] - r->orig[2];
      basey = r->orig[1] - t->v0[1];
      basez = r->orig[2] - t->v0[2];
      
      adelx = signSrc*(ay * delz - az * dely);
      if( (adelx + d*(ay*basez - az*basey)) * ( signSrc*(dely*bz - delz*by) + d*(basey*bz - basez*by)) >=0.0)
	{
	  cy = t->v2[1] - t->v1[1];
	  cz = t->v2[2] - t->v1[2];
	  basey = r->orig[1] - t->v1[1];
	  basez = r->orig[2] - t->v1[2];
	  if( (adelx + d*(ay*basez - az*basey)) * ( signSrc*(dely*cz - delz*cy) + d*(basey*cz - basez*cy)) <0.0){	
	    *(p->t) = signSrc/d;
	      
	    
	    return 1;
	  }
	}
    }
  else
    if(t->i0==1)
      {
	ax = t->v1[0] - t->v0[0];
	az = t->v1[2] - t->v0[2];
	bx = t->v2[0] - t->v0[0];
	bz = t->v2[2] - t->v0[2];
	delx = r->end[0] - r->orig[0];
	delz = r->end[2] - r->orig[2];
	basex = r->orig[0] - t->v0[0];
	basez = r->orig[2] - t->v0[2];
	adely = signSrc*(az * delx - ax * delz);
	if( (adely + d*(az*basex - ax*basez)) * ( signSrc*(delz*bx - delx*bz) + d*(basez*bx - basex*bz)) >=0.0)
	  {
	    cx = t->v2[0] - t->v1[0];
	    cz = t->v2[2] - t->v1[2];
	    basex = r->orig[0] - t->v1[0];
	    basez = r->orig[2] - t->v1[2];
	    if( (adely + d*(az*basex - ax*basez)) * ( signSrc*(delz*cx - delx*cz) + d*(basez*cx - basex*cz)) <0.0){	
		      *(p->t) = signSrc/d;
	   
	      return 1;
	    }
	  }
      }
    else
      {
	ax = t->v1[0] - t->v0[0];
	ay = t->v1[1] - t->v0[1];
	bx = t->v2[0] - t->v0[0];
	by = t->v2[1] - t->v0[1];
	delx = r->end[0] - r->orig[0];
	dely = r->end[1] - r->orig[1];
	basex = r->orig[0] - t->v0[0];
	basey = r->orig[1] - t->v0[1];
	adelz = signSrc*(ax * dely - ay * delx);
	
	if( (adelz + d*(ax*basey - ay*basex)) * ( signSrc*(delx*by - dely*bx) + d*(basex*by - basey*bx)) >=0.0)
	  {
	    cx = t->v2[0] - t->v1[0];
	    cy = t->v2[1] - t->v1[1];
	    basex = r->orig[0] - t->v1[0];
	    basey = r->orig[1] - t->v1[1];
	    if( (adelz + d*(ax*basey - ay*basex)) * ( signSrc*(delx*cy - dely*cx) + d*(basex*cy - basey*cx)) <0.0){	
	      *(p->t) = signSrc/d;
	      return 1;
	    }
	  }
      }
  return 0;
}




/*
  MApl 
*/

int plucker_mahovsky_other_t(Ray_big *r, Plucker_coords *t, Intersection_big *p)
{

  float r0,r1,r2,r3,r4,r5;
  float A,B;
  float tmp1[3], normal[3], vd, vo;
  /* Plucker för linjen*/
  r0 = r->orig[0] * r->end[1] - r->orig[1] * r->end[0];
  r1 = r->orig[0] * r->end[2] - r->orig[2] * r->end[0];
  r2 = r->orig[0] - r->end[0];
  r3 = r->orig[1] * r->end[2] - r->orig[2] * r->end[1];
  r4 = r->orig[2] - r->end[2];
  r5 = r->end[1] - r->orig[1];
  
  
  A = r0*t->a4 + r1*t->a5 +r3*t->a2;
  if(A+ r2*t->a3 +r4*t->a0+r5*t->a1 < 0)
    return 0;
  
  B = r0*t->b4 + r1*t->b5 + r3*t->b2;
  if(B + r2*t->b3 + r4*t->b0 +r5*t->b1 < 0)
    return 0;

  
  if(r2*t->c3 + r4*t->c0 + r5*t->c1 - A - B < 0)
    return 0;
  
    /*  Räkna ut avståndet*/
  
  normal[0] = t->a3 +t->b3 +t->c3;
  normal[1] =  t->a1 +t->b1 +t->c1;
  normal[2] = t->a0 +t->b0 +t->c0;
  
  
  SUB(tmp1, t->v0, r->orig);
  
  vo = DOT(tmp1, normal);
  vd = DOT(r->dir, normal);
  if(vd != 0.)
  *(p->t) = vo/vd;  
  else
    return 0;

  return 1;
}


/*
  PUpl
*/
int plucker_other_t(Ray_big *r, Plucker_coords *t, Intersection_big *p)
{
  
  float r0,r1,r2,r3,r4,r5;
  float A,B;
  int a,b,c;	
  float tmp1[3], normal[3], vd, vo;
  
  /* Plucker för linjen*/
  r0 = r->orig[0] * r->end[1] - r->orig[1] * r->end[0];
  r1 = r->orig[0] * r->end[2] - r->orig[2] * r->end[0];
  r2 = r->orig[0] - r->end[0];
  r3 = r->orig[1] * r->end[2] - r->orig[2] * r->end[1];
  r4 = r->orig[2] - r->end[2];
  r5 = r->end[1] - r->orig[1];
  
  
  A = r0*t->a4 + r1*t->a5 +r3*t->a2;
  B = r0*t->b4 + r1*t->b5 + r3*t->b2;
  a = A + r2*t->a3 +r4*t->a0+r5*t->a1 >=0.;
  b = B + r2*t->b3 + r4*t->b0 +r5*t->b1 >= 0.;
  
  if(a != b)
    return 0;
  
  c = r2*t->c3 + r4*t->c0 + r5*t->c1 - A - B >= 0.;
  
  if(c != a)
    return 0;

  normal[0] = t->a3 +t->b3 +t->c3;
  normal[1] =  t->a1 +t->b1 +t->c1;
  normal[2] = t->a0 +t->b0 +t->c0;
  
  
  SUB(tmp1, t->v0, r->orig);
  
  vo = DOT(tmp1, normal);
  vd = DOT(r->dir, normal);
  if(vd != 0.)
  *(p->t) = vo/vd;  
  else
    return 0;
  
  return 1;
}

/* ORp */
int orourke_other_t(Ray_big *r, Triangle_plane *t, Intersection_big *p){
  
  float vo, vd; 
  float vol0, vol1, vol2; 
  float ax,ay,az,bx,by,bz,cx,cy,cz,dx,dy,dz;
 

  ax = r->orig[0] - r -> end[0];
  ay = r->orig[1] - r -> end[1];
  az = r->orig[2] - r -> end[2];
  bx = t->v0[0] - r -> end[0];
  by = t->v0[1] - r -> end[1];
  bz = t->v0[2] - r -> end[2];
  cx = t->v1[0] - r -> end[0];
  cy = t->v1[1] - r -> end[1];
  cz = t->v1[2] - r -> end[2];

 dx = t->v2[0] - r -> end[0];
  dy = t->v2[1] - r -> end[1];
  dz = t->v2[2] - r -> end[2];
  

  vol0 = (ax * (by*cz - bz*cy) + ay * (bz*cx - bx*cz) + az *(bx*cy - by*cx));
  vol1 = (ax * (cy*dz - cz*dy) + ay * (cz*dx - cx*dz) + az *(cx*dy - cy*dx));
 
  
  if(vol0*vol1<0)
    return 0;

 
  vol2 = (ax * (dy*bz - dz*by) + ay * (dz*bx - dx*bz) + az *(dx*by - dy*bx));
  
  if(vol0*vol2<0)
    return 0;
  
 
  vd = t->normal[0]*r->dir[0] + t->normal[1]*r->dir[1] + t->normal[2]*r->dir[2];
  vo = -(t->normal[0]*r->orig[0] + t->normal[1]*r->orig[1] + t->normal[2]*r->orig[2] + t->d);
  if(vd!=0.)
    *(p->t) = vo/vd;
  return 1;
 
}

/*ORCp*/
int orourke_other_tCCW(Ray_big *r, Triangle_plane *t, Intersection_big *p){

  float vo, vd; 
  float vol0, vol1, vol2;
  float ax,ay,az,bx,by,bz,cx,cy,cz,dx,dy,dz;
  

  ax = r->orig[0] - r -> end[0];
  ay = r->orig[1] - r -> end[1];
  az = r->orig[2] - r -> end[2];
  bx = t->v0[0] - r -> end[0];
  by = t->v0[1] - r -> end[1];
  bz = t->v0[2] - r -> end[2];
  cx = t->v1[0] - r -> end[0];
  cy = t->v1[1] - r -> end[1];
  cz = t->v1[2] - r -> end[2];


  vol0 = (ax * (by*cz - bz*cy) + ay * (bz*cx - bx*cz) + az *(bx*cy - by*cx));
  
  if(vol0 < 0)
    return 0;
  dx = t->v2[0] - r -> end[0];
  dy = t->v2[1] - r -> end[1];
  dz = t->v2[2] - r -> end[2];
  
  vol1 = (ax * (cy*dz - cz*dy) + ay * (cz*dx - cx*dz) + az *(cx*dy - cy*dx));

  if(vol1<0)
    return 0;
  vol2 = (ax * (dy*bz - dz*by) + ay * (dz*bx - dx*bz) + az *(dx*by - dy*bx));
  
  if(vol2<0)
    return 0;
    
    
  vd = t->normal[0]*r->dir[0] + t->normal[1]*r->dir[1] + t->normal[2]*r->dir[2];
  vo = -(t->normal[0]*r->orig[0] + t->normal[1]*r->orig[1] + t->normal[2]*r->orig[2] + t->d);
  *(p->t) = vo/vd;
  
   
    return 1;

}


 
 /*
   HFp
*/
int halfplane_other_t(Ray_big *r, Triangle_plane *t, Intersection_big *p)
{
  
  float n0x,n0y,c0,n1x,n1y,c1;
  int X = t->i1;
  int Y = t->i2;
  float p0, p1;
  float tt,vd,vo;
  int sign1,sign2;
  
  vd = DOT(t->normal, r->dir);
  vo = -(DOT(t->normal, r->orig) + t->d);
  
  

  if(vd*vo <= 0)
    {
      
      return 0;
  }
  tt = vo/vd;
  
  p0 = r->orig[X] + tt*r->dir[X];
  p1 = r->orig[Y] + tt*r->dir[Y];
 
  n0x = -(t->v0[Y]- t->v1[Y]);
  n0y = t->v0[X] - t->v1[X];
  c0 = - t->v1[X]*n0x - t->v1[Y]*n0y;
  
  n1x = -(t->v1[Y] - t->v2[Y]);
  n1y = t->v1[X] - t->v2[X];
  c1 = - t->v2[X]*n1x - t->v2[Y]*n1y;
  
  sign1 = p0*n0x + p1*n0y + c0 < 0.;
  sign2 = p0*n1x + p1*n1y + c1 < 0.;
  
  if(sign1 == sign2 )
    {
      n1x = -(t->v2[Y] - t->v0[Y]);
      n1y = t->v2[X] - t->v0[X];
      c1 = - t->v0[X]*n1x - t->v0[Y]*n1y;
      sign2 = p0*n1x + p1*n1y + c1 < 0.;
      if(sign1 == sign2){
	(*p->t) = tt;
	return 1;
      }
    }  
  return 0;
}

/* A2Dp*/
int area2D_other_t(Ray_big *r, Triangle_plane *t, Intersection_big *p)
{
   
  float vd,vo,tt,point[2];
 
  float v0x,v0y,v1x,v1y;
  float area1,area2,area3;
  int X = t-> i1;
  int Y = t-> i2;
 
  vd = DOT(t->normal, r->dir);
  vo = -(DOT(t->normal, r->orig) + t->d);
  
  if(vd*vo <= 0)
    {
      
      return 0;
  }
  tt = vo/vd;
 
  point[0] = r->orig[X] + tt*r->dir[X];
  point[1] = r->orig[Y] + tt*r->dir[Y];

  v0x = t->v1[X] - t->v0[X];
  v0y = t->v1[Y] - t->v0[Y];

  v1x = point[0] - t->v0[X];
  v1y = point[1] - t->v0[Y];

  area1 = v0x*v1y - v1x*v0y;

  v0x = t->v2[X] - t->v1[X];
  v0y = t->v2[Y] - t->v1[Y];

  v1x = point[0] - t->v1[X];
  v1y = point[1] - t->v1[Y];

  area2 = v0x*v1y - v1x*v0y;
  

  if(area1*area2 > 0.){
    v0x = t->v0[X] - t->v2[X];
    v0y = t->v0[Y] - t->v2[Y];
    
    v1x = point[0] - t->v2[X];
    v1y = point[1] - t->v2[Y];
    
    area3 = v0x*v1y - v1x*v0y;
    if(area1*area3 > 0.){
     
      
      return 1;
    }
  }
  return 0;
  
}


/*CH3p*/
int chirkov3_other_t(Ray_big *r, Triangle_plane *t, Intersection_big *p)
{
  float ax;
  float ay ;
  float az;
  float bx;
  float by;
  float bz;
  float delx;
  float dely;
  float delz;
  float basex;
  float basey;
  float basez;
  float cx;
  float cy;
  float cz;
  float d,bary1,bary2,div,bary3;
  float signSrc = t->normal[0]*r->orig[0] + t->normal[1]*r->orig[1] + t->normal[2]*r->orig[2] + t->d;
  float signDst = t->normal[0]*r->end[0] + t->normal[1]*r->end[1] + t->normal[2]*r->end[2] + t->d;

  if(signSrc*signDst>=0.0)	
    return 0;
  
  d = signSrc - signDst;
  
  if(t->i0==0)
    {
      ay = t->v1[1] - t->v0[1];
      az = t->v1[2] - t->v0[2];
      by = t->v2[1] - t->v0[1];
      bz = t->v2[2] - t->v0[2];
      dely = r->end[1] - r->orig[1];
      delz = r->end[2] - r->orig[2];
      basey = r->orig[1] - t->v0[1];
      basez = r->orig[2] - t->v0[2];

      
      bary1 = ( signSrc*(dely*bz - delz*by) + d*(basey*bz - basez*by));
      bary3 = signSrc*(ay *delz - az *dely) + d*(ay*basez - az*basey);
   
      bary1 = ( signSrc*(dely*bz - delz*by) + d*(basey*bz - basez*by));
      if( (bary3) * bary1 >=0.0)
	{
	  cy = t->v2[1] - t->v1[1];
	  cz = t->v2[2] - t->v1[2];
	  basey = r->orig[1] - t->v1[1];
	  basez = r->orig[2] - t->v1[2];
	  bary2 =  (signSrc*(dely*cz - delz*cy) + d*(basey*cz - basez*cy));
	  if( (bary3) * bary2 <0.0){	
	    div = 1.f/d;
	    
	    *(p->t) = -signSrc*div;
	    return 1;
	  }
	}
    }
  else
    if(t->i0==1)
      {
	ax = t->v1[0] - t->v0[0];
	az = t->v1[2] - t->v0[2];
	bx = t->v2[0] - t->v0[0];
	bz = t->v2[2] - t->v0[2];
	delx = r->end[0] - r->orig[0];
	delz = r->end[2] - r->orig[2];
	basex = r->orig[0] - t->v0[0];
	basez = r->orig[2] - t->v0[2];

	bary3 = signSrc*(az *delx - ax *delz) + d*(az*basex - ax*basez);
	bary1 = ( signSrc*(delz*bx - delx*bz) + d*(basez*bx - basex*bz));
	if( bary3 * bary1 >=0.0)
	  {
	    cx = t->v2[0] - t->v1[0];
	    cz = t->v2[2] - t->v1[2];
	    basex = r->orig[0] - t->v1[0];
	    basez = r->orig[2] - t->v1[2];
	    bary2 = ( signSrc*(delz*cx - delx*cz) + d*(basez*cx - basex*cz));
	    if( bary3 * bary2 <0.0){	
	      div = 1.f/d;
	      
	      *(p->t) = -signSrc*div;
	      return 1;
	    }
	  }
      }
    else
      {
	ax = t->v1[0] - t->v0[0];
	ay = t->v1[1] - t->v0[1];
	bx = t->v2[0] - t->v0[0];
	by = t->v2[1] - t->v0[1];
	delx = r->end[0] - r->orig[0];
	dely = r->end[1] - r->orig[1];
	basex = r->orig[0] - t->v0[0];
	basey = r->orig[1] - t->v0[1];
      
	bary3 = signSrc*(ax *dely - ay *delx) + d*(ax*basey - ay*basex);
	bary1 = ( signSrc*(delx*by - dely*bx) + d*(basex*by - basey*bx));
	if( bary3 * bary1 >=0.0)
	  {
	    cx = t->v2[0] - t->v1[0];
	    cy = t->v2[1] - t->v1[1];
	    basex = r->orig[0] - t->v1[0];
	    basey = r->orig[1] - t->v1[1];
	    bary2 = ( signSrc*(delx*cy - dely*cx) + d*(basex*cy - basey*cx));
	    if( bary3 *  bary2 <0.0){	
	      div = 1.f/d;
	      
	      *(p->t) = -signSrc*div;
	      
	      return 1;
	    }
	  }
      }
  return 0;
}

/*HF2h*/
int halfplane_other_t_pre2(Ray_big *r, Triangle_Halfplane *t, Intersection_big *p)
{
 
  int X,Y;
  float sign1,sign2,sign3;
  float vd,vo;

  vd = DOT(t->normal, r->dir);
 
  vo = - DOT(t->normal, r->orig) - t->d;
  
  if(vd*vo <= 0)
    return 0;
 
  X = t->i1;
  Y = t->i2;

  sign1 = vd*(t->n0x*r->orig[X] + t->n0y*r->orig[Y]) + vo*(t->n0y * r->dir[Y] + t->n0x * r->dir[X]) + t->c0*vd; 
  sign2 = vd*(t->n1x*r->orig[X] + t->n1y*r->orig[Y]) + vo*(t->n1y * r->dir[Y] + t->n1x * r->dir[X]) + t->c1*vd;
 
  if(sign1*sign2<=0)
    return 0;
  
  sign3 = vd*(t->n2x*r->orig[X] + t->n2y*r->orig[Y]) + vo*(t->n2y * r->dir[Y] + t->n2x * r->dir[X]) + t->c2*vd;
  if(sign1*sign3>0){
    *(p->t) = vo/vd;
    return 1;
  }
  
      
  return 0;
}


 /*
   HFh
*/
int halfplane_other_t_pre(Ray_big *r, Triangle_Halfplane *t, Intersection_big *p)
{
 
  int X = t->i1;
  int Y = t->i2;
  float p0, p1;
  float tt,vd,vo;
  float sign1,sign2;
  
  vd = DOT(t->normal, r->dir);
  if(vd == 0)
      return 0;
  vo = -(DOT(t->normal, r->orig) + t->d);
  
  if(vd*vo <= 0)
    return 0;
 
  tt = vo/vd;
  
  p0 = r->orig[X] + tt*r->dir[X];
  p1 = r->orig[Y] + tt*r->dir[Y];
  
  sign1 = p0*t->n0x + p1*t->n0y + t->c0 ;
  sign2 = p0*t->n1x + p1*t->n1y + t->c1 ;
  
  if(sign1*sign2>0 )
    {
      sign2 = p0*t->n2x + p1*t->n2y + t->c2;
      if(sign1*sign2>0){
	*(p->t) = tt;
	return 1;
      }
    }  
  return 0;
}

int arenberg_other_t_pre2(Ray_big *r, Triangle_inv *t, Intersection_big *p)
{
  float num, den, a2, b2;
  float trans1[3];

  den = DOT(r ->dir, t->bb2);

  if(den == 0.)
   return 0;
  
  SUB(trans1, t -> v0, r -> orig);
  num = DOT(trans1, t->bb2);

  if(den*num <= 0){
    return 0;
  }
  
  SUB(trans1, r -> orig, t -> v0);

  a2 = DOT(trans1, t->bb0)*den + num*DOT(r->dir,t->bb0);

  if(a2 <0)
    return 0;

  b2 = DOT(trans1, t->bb1)*den + num*DOT(r->dir,t->bb1);
  
  
  if(b2 <0)
    return 0;
  
  if(a2 + b2 > den)
    return 0;
  *(p->t) = num/den;
  
  return 1;
}



/* ARi*/

int arenberg_other_t_pre(Ray_big *r, Triangle_inv *t, Intersection_big *p)
{
  float num, den, tt,a,b;
  float trans1[3],point[3];
  
  den = DOT(r ->dir, t->bb2);
 
  if(den == 0.){
   return 0;
  }
  
  SUB(trans1, t -> v0, r -> orig);
  num = DOT(trans1, t->bb2);

  tt = num/den;
  
  if(tt <=0.){
    return 0;
  }
  
   
  SUB(trans1, r -> orig, t -> v0);
  point[0] = (tt * r -> dir[0]) + trans1[0];
  point[1] = (tt * r -> dir[1]) + trans1[1];
  point[2] = (tt * r -> dir[2]) + trans1[2];

  a = DOT(point,t->bb0);
  b = DOT(point,t->bb1);
 
  if( a < 0.0 || b < 0.0 || a + b > 1.0){
   return 0;
  }

 
  *(p->t) = tt;
  return 1;
}



/* HF2p */
int halfplane2_other_t(Ray_big *r, Triangle_plane *t, Intersection_big *p)
{
 
  float n0x,n0y,c0,n1x,n1y,c1;
  int X = t->i1;
  int Y = t->i2;
  float vd,vo;
  float sign1,sign2;

  vd = DOT(t->normal, r->dir);
  vo = - DOT(t->normal, r->orig) - t->d;
  
  if(vd*vo <= 0)
    return 0;
  
  n0x = -(t->v0[Y]- t->v1[Y]);
  n0y = t->v0[X] - t->v1[X];
  c0 = - t->v1[X]*n0x - t->v1[Y]*n0y;
  
  n1x = -(t->v1[Y] - t->v2[Y]);
  n1y = t->v1[X] - t->v2[X];
  c1 = - t->v2[X]*n1x - t->v2[Y]*n1y;

  sign1 = vd*(n0x*r->orig[X] + n0y*r->orig[Y]) + vo*(n0y * r->dir[Y] + n0x * r->dir[X]) + c0*vd; 
  sign2 = vd*(n1x*r->orig[X] + n1y*r->orig[Y]) + vo*(n1y * r->dir[Y] + n1x * r->dir[X]) + c1*vd;


  if(sign1*sign2>0)
    {
      n1x = -(t->v2[Y] - t->v0[Y]);
      n1y = t->v2[X] - t->v0[X];
      c1 = - t->v0[X]*n1x - t->v0[Y]*n1y;
      
      sign2 = vd*(n1x*r->orig[X] + n1y*r->orig[Y]) + vo*(n1y * r->dir[Y] + n1x * r->dir[X]) + c1*vd;
      if(sign1*sign2>0){
	*(p->t) = vo/vd;
	return 1;
      }
      
    }  
  return 0;
}
/*************************************************************************************

The version of the algorithms below don't have to calculate any supplementary information.
When possible they have been optimized to avoid divisions stemming from t-value and 
barycentric coordinates calculations 

 *************************************************************************************/

/* MT0 */
int intersect_triangle_small(Ray_big *r, Triangle_small *t, Intersection_big *p)
{
  float edge1[3], edge2[3], tvec[3], pvec[3], qvec[3];
  float det,inv_det;
  

  /* find vectors for two edges sharing vert0 */
  SUB(edge1, t -> v1, t -> v0);
  SUB(edge2, t -> v2, t -> v0);
  
  /* begin calculating determinant - also used to calculate U parameter */
  CROSS(pvec, r -> dir, edge2);
  
  /* if determinant is near zero, ray lies in plane of triangle */
  det = DOT(edge1, pvec);
  
  if (det > -EPSILON && det < EPSILON)
    return 0;
  inv_det = 1.0f / det;
  
  /* calculate distance from vert0 to ray origin */
  SUB(tvec, r -> orig, t -> v0);
  
  /* calculate U parameter and test bounds */
  *(p -> u) = DOT(tvec, pvec) * inv_det;
  if (*(p->u) < 0.0 || *(p->u) > 1.0)
    return 0;
  
  /* prepare to test V parameter */
  CROSS(qvec, tvec, edge1);
  
  /* calculate V parameter and test bounds */
  *(p->v) = DOT(r->dir, qvec) * inv_det;
  if (*(p->v) < 0.0 || *(p->u) + *(p->v) > 1.0)
    return 0;
  return 1;
}

/* MT1 */
/* code rewritten to do tests on the sign of the determinant */
/* the division is at the end in the code                    */
int intersect_triangle1_small(Ray_big *r, Triangle_small *t, Intersection_big *p)
{
  float edge1[3], edge2[3], tvec[3], pvec[3], qvec[3];
  float det;
  
  /* find vectors for two edges sharing vert0 */
  SUB(edge1, t ->v1, t ->v0);
  SUB(edge2, t ->v2, t->v0);
  
  /* begin calculating determinant - also used to calculate U parameter */
  CROSS(pvec, r ->dir, edge2);
  
  /* if determinant is near zero, ray lies in plane of triangle */
  det = DOT(edge1, pvec);
  
  if (det > EPSILON)
    {
      /* calculate distance from vert0 to ray origin */
      SUB(tvec, r->orig, t ->v0);
      
      /* calculate U parameter and test bounds */
      *(p -> u) = DOT(tvec, pvec);
      if (*(p->u) < 0.0 || *(p->u) > det)
	return 0;
      
      /* prepare to testV parameter */
      CROSS(qvec, tvec, edge1);
      
      /* calculate V parameter and test bounds */
      *(p->v) = DOT(r->dir, qvec);
      if (*(p->v) < 0.0 || *(p->u) + *(p->v) > det)
	return 0;
      
    }
  else if(det < -EPSILON)
    {
      /* calculate distance from vert0 to ray origin */
      SUB(tvec, r->orig, t->v0);
      
      /* calculate U parameter and test bounds */
      *(p->u) = DOT(tvec, pvec);
      if (*(p->u) > 0.0 || *(p->u) < det)
	return 0;
      
      /* prepare to test V parameter */
      CROSS(qvec, tvec, edge1);
      
      /* calculate V parameter and test bounds */
      *(p->v) = DOT(r->dir, qvec) ;
      if (*(p->v) > 0.0 || *(p->u) + *(p->v) < det)
	return 0;
    }
  else return 0;  /* ray is parallell to the plane of the triangle */
  
  return 1;
}

/* MT2 */
/* code rewritten to do tests on the sign of the determinant */
/* the division is before the test of the sign of the det    */
int intersect_triangle2_small(Ray_big *r, Triangle_small *t, Intersection_big *p)
{
  float edge1[3], edge2[3], tvec[3], pvec[3], qvec[3];
  float det,inv_det;
  
  /* find vectors for two edges sharing vert0 */
  SUB(edge1, t->v1, t ->v0);
  SUB(edge2, t->v2, t->v0);
  
  /* begin calculating determinant - also used to calculate U parameter */
  CROSS(pvec, r->dir, edge2);
  
  /* if determinant is near zero, ray lies in plane of triangle */
  det = DOT(edge1, pvec);
  if (det == 0.)
    return 0;
  /* calculate distance from vert0 to ray origin */
  SUB(tvec, r->orig, t->v0);
  inv_det = 1.0f / det;
  
  if (det > EPSILON)
    {
      /* calculate U parameter and test bounds */
      *(p->u) = DOT(tvec, pvec);
      if (*(p->u) < 0.0 || *(p->u) > det)
	return 0;
      
      /* prepare to test V parameter */
      CROSS(qvec, tvec, edge1);
      
      /* calculate V parameter an test bounds */
      *(p->v) = DOT(r->dir, qvec);
      if (*(p->v) < 0.0 || *(p->u) + *(p->v) > det)
	return 0;
      
    }
  else if(det < -EPSILON)
    {
      /* calculate U parameter and test bounds */
      *(p->u) = DOT(tvec, pvec);
      if (*(p->u) > 0.0 || *(p->u) < det)
	return 0;
      
      /* prepare to test V parameter */
      CROSS(qvec, tvec, edge1);
      
      /* calculate V parameter and test bounds */
      *(p->v) = DOT(r->dir, qvec) ;
      if (*(p->v) > 0.0 || *(p->u) + *(p->v) < det)
	return 0;
    }
  else return 0;  /* ray is parallell to the plane of the triangle */
  
  
  return 1;
}

/* MT3 */
/* code rewritten to do tests on the sign of the determinant */
/* the division is before the test of the sign of the det    */
/* and one CROSS has been moved out from the if-else if-else */
int intersect_triangle3_small(Ray_big *r, Triangle_small *t, Intersection_big *p)
{
  float edge1[3], edge2[3], tvec[3], pvec[3], qvec[3];
  float det,inv_det;
  
  /* find vectors for two edges sharing vert0 */
  SUB(edge1, t->v1, t->v0);
  SUB(edge2, t->v2, t->v0);
  
  /* begin calculating determinant - also used to calculate U parameter */
  CROSS(pvec, r->dir, edge2);
  
  /* if determinant is near zero, ray lies in plane of triangle */
  det = DOT(edge1, pvec);
  if(det == 0)
    return 0;
  /* calculate distance from vert0 to ray origin */
  SUB(tvec, r->orig, t->v0);
  inv_det = 1.0f / det;
  
  CROSS(qvec, tvec, edge1);
  
  if (det > EPSILON)
    {
      *(p->u) = DOT(tvec, pvec);
      if (*(p->u) < 0.0 || *(p->u) > det)
	return 0;
      
      /* calculate V parameter and test bounds */
      *(p->v) = DOT(r->dir, qvec);
      if (*(p->v) < 0.0 || *(p->u) + *(p->v) > det)
	return 0;
      
    }
  else if(det < -EPSILON)
    {
      /* calculate U parameter and test bounds */
      *(p->u) = DOT(tvec, pvec);
      if (*(p->u) > 0.0 || *(p->u) < det)
	return 0;
      
      /* calculate V parameter and test bounds */
      *(p->v) = DOT(r->dir, qvec) ;
      if (*(p->v) > 0.0 || *(p->u) + *(p->v) < det)
	return 0;
    }
  else return 0;  /* ray is parallell to the plane of the triangle */
  
   return 1;
}



/* OR */
int orourke_small(Ray_big *r, Triangle_small *t, Intersection_big *p){
  
  float vol0, vol1, vol2;
  float ax,ay,az,bx,by,bz,cx,cy,cz,dx,dy,dz;
  
  
  ax = r->orig[0] - r -> end[0];
  ay = r->orig[1] - r -> end[1];
  az = r->orig[2] - r -> end[2];
  bx = t->v0[0] - r -> end[0];
  by = t->v0[1] - r -> end[1];
  bz = t->v0[2] - r -> end[2];
  cx = t->v1[0] - r -> end[0];
  cy = t->v1[1] - r -> end[1];
  cz = t->v1[2] - r -> end[2];
  
  dx = t->v2[0] - r -> end[0];
  dy = t->v2[1] - r -> end[1];
  dz = t->v2[2] - r -> end[2];
  
  vol0 = (ax * (by*cz - bz*cy) + ay * (bz*cx - bx*cz) + az *(bx*cy - by*cx));
  vol1 = (ax * (cy*dz - cz*dy) + ay * (cz*dx - cx*dz) + az *(cx*dy - cy*dx));

  /* if different signs return 0 */
  if(vol0*vol1<0)
    return 0;

  

  vol2 = (ax * (dy*bz - dz*by) + ay * (dz*bx - dx*bz) + az *(dx*by - dy*bx));
  
  if(vol0*vol2<0)
    return 0;
    
  return 1;
}

/* ORC */
int orourke_smallCCW(Ray_big *r, Triangle_small *t, Intersection_big *p){

  float vol0, vol1, vol2;
  float ax,ay,az,bx,by,bz,cx,cy,cz,dx,dy,dz;
  

  ax = r->orig[0] - r -> end[0];
  ay = r->orig[1] - r -> end[1];
  az = r->orig[2] - r -> end[2];
  bx = t->v0[0] - r -> end[0];
  by = t->v0[1] - r -> end[1];
  bz = t->v0[2] - r -> end[2];
  cx = t->v1[0] - r -> end[0];
  cy = t->v1[1] - r -> end[1];
  cz = t->v1[2] - r -> end[2];


  vol0 = (ax * (by*cz - bz*cy) + ay * (bz*cx - bx*cz) + az *(bx*cy - by*cx));
  
  if(vol0 < 0)
    return 0;
  dx = t->v2[0] - r -> end[0];
  dy = t->v2[1] - r -> end[1];
  dz = t->v2[2] - r -> end[2];
  
  vol1 = (ax * (cy*dz - cz*dy) + ay * (cz*dx - cx*dz) + az *(cx*dy - cy*dx));

  if(vol1<0)
    return 0;
  vol2 = (ax * (dy*bz - dz*by) + ay * (dz*bx - dx*bz) + az *(dx*by - dy*bx));
  
  if(vol2<0)
    return 0;
  return 1;
  
}


/* MA */
int plucker_mahovsky_small(Ray_big *r, Triangle_small *t, Intersection_big *p)
{
  float a0,a1,a2,a3,a4,a5,b0,b1,b2,b3,b4,b5,c0,c1,c3;
  float r0,r1,r2,r3,r4,r5;
  float A,B;

  /* Plucker för linjen*/
  r0 = r->orig[0] * r->end[1] - r->orig[1] * r->end[0];
  r1 = r->orig[0] * r->end[2] - r->orig[2] * r->end[0];
  r2 = r->orig[0] - r->end[0];
  r3 = r->orig[1] * r->end[2] - r->orig[2] * r->end[1];
  r4 = r->orig[2] - r->end[2];
  r5 = r->end[1] - r->orig[1];
  
      /*Plucker koefficienter för triangel sidorna*/
  a0 = t->v0[0] * t->v1[1] - t->v1[0] * t->v0[1];
  a1 = t->v0[0] * t->v1[2] - t->v1[0] * t->v0[2];
  a2 = t->v0[0] - t->v1[0];
  a3 = t->v0[1] * t->v1[2] - t->v1[1] * t->v0[2];
  a4 = t->v0[2] - t->v1[2];
  a5 = t->v1[1] - t->v0[1];

  A = r0*a4 + r1*a5 +r3*a2;

  if(A + r2*a3 + r4*a0 + r5*a1 <0.)
    return 0;

  b0 = t->v1[0] * t->v2[1] - t->v2[0] * t->v1[1];
  b1 = t->v1[0] * t->v2[2] - t->v2[0] * t->v1[2];
  b2 = t->v1[0] - t->v2[0];
  b3 = t->v1[1] * t->v2[2] - t->v2[1] * t->v1[2];
  b4 = t->v1[2] - t->v2[2];
  b5 = t->v2[1] - t->v1[1];
  
 
  B = r0*b4 + r1*b5 + r3*b2;
  
  if(B + r2*b3 + r4*b0 +r5*b1<0.)
    return 0;

  c0 = t->v2[0] * t->v0[1] - t->v0[0] * t->v2[1];
  c1 = t->v2[0] * t->v0[2] - t->v0[0] * t->v2[2];
  c3 = t->v2[1] * t->v0[2] - t->v0[1] * t->v2[2];
  
  if(r2*c3 + r4*c0 + r5*c1 - A - B  < 0.)
    return 0;

 
  return 1;
}

/* MA2 */
int plucker_mahovsky_small2(Ray_big *r, Triangle_small *t, Intersection_big *p)
{
   float a0,a1,a2,a3,a4,a5,b0,b1,b2,b3,b4,b5,c0,c1,c3;
  float r0,r1,r2,r3,r4,r5;
  float A,B;
  
 
  /* Plucker för linjen*/
  r0 = r->orig[0] * r->end[1] - r->orig[1] * r->end[0];
  r1 = r->orig[0] * r->end[2] - r->orig[2] * r->end[0];
  r2 = r->orig[0] - r->end[0];
  r3 = r->orig[1] * r->end[2] - r->orig[2] * r->end[1];
  r4 = r->orig[2] - r->end[2];
  r5 = r->end[1] - r->orig[1];
  
      /*Plucker koefficienter för triangel sidorna*/
  a0 = t->v0[0] * t->v1[1] - t->v1[0] * t->v0[1];
  a1 = t->v0[0] * t->v1[2] - t->v1[0] * t->v0[2];
  a2 = t->v0[0] - t->v1[0];
  a3 = t->v0[1] * t->v1[2] - t->v1[1] * t->v0[2];
  a4 = t->v0[2] - t->v1[2];
  a5 = t->v1[1] - t->v0[1];

  A = r0*a4 + r1*a5 +r3*a2;

  if(A + r2*a3 + r4*a0 + r5*a1 <0.)
    return 0;

  b0 = t->v1[0] * t->v2[1] - t->v2[0] * t->v1[1];
  b1 = t->v1[0] * t->v2[2] - t->v2[0] * t->v1[2];
  b2 = t->v1[0] - t->v2[0];
  b3 = t->v1[1] * t->v2[2] - t->v2[1] * t->v1[2];
  b4 = t->v1[2] - t->v2[2];
  b5 = t->v2[1] - t->v1[1];
  
 
  B = r0*b4 + r1*b5 + r3*b2;
  
  if(B + r2*b3 + r4*b0 +r5*b1<0.)
    return 0;

  c0 = t->v2[0] * t->v0[1] - t->v0[0] * t->v2[1];
  c1 = t->v2[0] * t->v0[2] - t->v0[0] * t->v2[2];
  c3 = t->v2[1] * t->v0[2] - t->v0[1] * t->v2[2];
  
  if(r2*c3 + r4*c0 + r5*c1 - A - B  < 0.)
    return 0;
 

  return 1;
}




/*
  PU
*/
int plucker_small(Ray_big *r, Triangle_small *t, Intersection_big *p)
{
  float a0,a1,a2,a3,a4,a5,b0,b1,b2,b3,b4,b5,c0,c1,c3;
  float r0,r1,r2,r3,r4,r5;
  float A,B;
int a,b,c;	
   
 
  /* Plucker för linjen*/
  r0 = r->orig[0] * r->end[1] - r->orig[1] * r->end[0];
  r1 = r->orig[0] * r->end[2] - r->orig[2] * r->end[0];
  r2 = r->orig[0] - r->end[0];
  r3 = r->orig[1] * r->end[2] - r->orig[2] * r->end[1];
  r4 = r->orig[2] - r->end[2];
  r5 = r->end[1] - r->orig[1];

    /*Plucker koefficienter för triangel sidorna*/
  a0 = t->v0[0] * t->v1[1] - t->v1[0] * t->v0[1];
  a1 = t->v0[0] * t->v1[2] - t->v1[0] * t->v0[2];
  a2 = t->v0[0] - t->v1[0];
  a3 = t->v0[1] * t->v1[2] - t->v1[1] * t->v0[2];
  a4 = t->v0[2] - t->v1[2];
  a5 = t->v1[1] - t->v0[1];
  
  b0 = t->v1[0] * t->v2[1] - t->v2[0] * t->v1[1];
  b1 = t->v1[0] * t->v2[2] - t->v2[0] * t->v1[2];
  b2 = t->v1[0] - t->v2[0];
  b3 = t->v1[1] * t->v2[2] - t->v2[1] * t->v1[2];
  b4 = t->v1[2] - t->v2[2];
  b5 = t->v2[1] - t->v1[1];
  
  A = r0*a4 + r1*a5 +r3*a2;
  B = r0*b4 + r1*b5 + r3*b2;
  
  a = (A + r2*a3 + r4*a0 + r5*a1 >= 0.f);
  b = (B + r2*b3 + r4*b0 +r5*b1 >= 0.f);

  if(a != b)
    return 0;
  
  c0 = t->v2[0] * t->v0[1] - t->v0[0] * t->v2[1];
  c1 = t->v2[0] * t->v0[2] - t->v0[0] * t->v2[2];
  c3 = t->v2[1] * t->v0[2] - t->v0[1] * t->v2[2];
  c = r2*c3 + r4*c0 + r5*c1 - A - B  >= 0.;
  
  if(c != a)
    return 0;
  
  
  return 1;
}


/*
  CH1p 
*/

int chirkov_other(Ray_big *r, Triangle_plane *t, Intersection_big *p)
{
  float signSrc = t->normal[0]*r->orig[0] + t->normal[1]*r->orig[1] + t->normal[2]*r->orig[2] + t->d;
  float signDst = t->normal[0]*r->end[0] + t->normal[1]*r->end[1] + t->normal[2]*r->end[2] + t->d;
  float ay ;
  float az  ;
  float by  ;
  float bz  ;
  float dely  ;
  float delz ;
  float basey ;
  float basez ;
  float adelxbase;
  float d ;
  float cy;
  float cz;
  int i1 ;
  int i2;
  
  
  if(signSrc*signDst >= 0)
    return 0;

  i1 = t ->  i1;
  i2 = t ->  i2;
  
  d = signSrc - signDst;

  ay =	t->v1[i1] - t->v0[i1];
  az =	t->v1[i2] - t->v0[i2];
  by =	t->v2[i1] - t->v0[i1];
  bz =	t->v2[i2] - t->v0[i2];
  
  dely = r->end[i1] - r->orig[i1] ;
  delz = r->end[i2] - r->orig[i2] ;
  
  basey = r->orig[i1] - t->v0[i1];
  basez = r->orig[i2] - t->v0[i2];
  
  adelxbase = signSrc*(ay *delz - az *dely) + d*(ay*basez - az*basey);
  
  if(adelxbase * (signSrc*(dely*bz - delz*by)	+ d*(basey*bz -	basez*by)) >= 0.0){
    
    
    cy = t->v2[i1] -	t->v1[i1];
    cz = t->v2[i2] -	t->v1[i2];
    basey = r->orig[i1] - t->v1[i1];
    basez = r->orig[i2] - t->v1[i2];
    if(adelxbase * (signSrc*(dely*cz - delz*cy) + d*(basey*cz - basez*cy)) < 0.0){
    
      
      return 1;
    }
  }

  return 0;
}



/* CH2p */

int chirkov2_other(Ray_big *r, Triangle_plane *t, Intersection_big *p)
{
  float ax;
  float ay ;
  float az;
  float bx;
  float by;
  float bz;
  float delx;
  float dely;
  float delz;
  float basex;
  float basey;
  float basez;
  float adelx;
  float adely;
  float adelz;
  float cx;
  float cy;
  float cz;
  float d;
  float signSrc = t->normal[0]*r->orig[0] + t->normal[1]*r->orig[1] + t->normal[2]*r->orig[2] + t->d;
  float signDst = t->normal[0]*r->end[0] + t->normal[1]*r->end[1] + t->normal[2]*r->end[2] + t->d;

  if(signSrc*signDst>=0.0)	
    return 0;
  
  d = signSrc - signDst;
  
  if(t->i0==0)
    {
      ay = t->v1[1] - t->v0[1];
      az = t->v1[2] - t->v0[2];
      by = t->v2[1] - t->v0[1];
      bz = t->v2[2] - t->v0[2];
      dely = r->end[1] - r->orig[1];
      delz = r->end[2] - r->orig[2];
      basey = r->orig[1] - t->v0[1];
      basez = r->orig[2] - t->v0[2];
      
      adelx = signSrc*(ay * delz - az * dely);
      if( (adelx + d*(ay*basez - az*basey)) * ( signSrc*(dely*bz - delz*by) + d*(basey*bz - basez*by)) >=0.0)
	{
	  cy = t->v2[1] - t->v1[1];
	  cz = t->v2[2] - t->v1[2];
	  basey = r->orig[1] - t->v1[1];
	  basez = r->orig[2] - t->v1[2];
	  if( (adelx + d*(ay*basez - az*basey)) * ( signSrc*(dely*cz - delz*cy) + d*(basey*cz - basez*cy)) <0.0){	
	    
	    return 1;
	  }
	}
    }
  else
    if(t->i0==1)
      {
	ax = t->v1[0] - t->v0[0];
	az = t->v1[2] - t->v0[2];
	bx = t->v2[0] - t->v0[0];
	bz = t->v2[2] - t->v0[2];
	delx = r->end[0] - r->orig[0];
	delz = r->end[2] - r->orig[2];
	basex = r->orig[0] - t->v0[0];
	basez = r->orig[2] - t->v0[2];
	adely = signSrc*(az * delx - ax * delz);
	if( (adely + d*(az*basex - ax*basez)) * ( signSrc*(delz*bx - delx*bz) + d*(basez*bx - basex*bz)) >=0.0)
	  {
	    cx = t->v2[0] - t->v1[0];
	    cz = t->v2[2] - t->v1[2];
	    basex = r->orig[0] - t->v1[0];
	    basez = r->orig[2] - t->v1[2];
	    if( (adely + d*(az*basex - ax*basez)) * ( signSrc*(delz*cx - delx*cz) + d*(basez*cx - basex*cz)) <0.0){	
	      return 1;
	    }
	  }
      }
    else
      {
	ax = t->v1[0] - t->v0[0];
	ay = t->v1[1] - t->v0[1];
	bx = t->v2[0] - t->v0[0];
	by = t->v2[1] - t->v0[1];
	delx = r->end[0] - r->orig[0];
	dely = r->end[1] - r->orig[1];
	basex = r->orig[0] - t->v0[0];
	basey = r->orig[1] - t->v0[1];
	adelz = signSrc*(ax * dely - ay * delx);
	
	if( (adelz + d*(ax*basey - ay*basex)) * ( signSrc*(delx*by - dely*bx) + d*(basex*by - basey*bx)) >=0.0)
	  {
	    cx = t->v2[0] - t->v1[0];
	    cy = t->v2[1] - t->v1[1];
	    basex = r->orig[0] - t->v1[0];
	    basey = r->orig[1] - t->v1[1];
	    if( (adelz + d*(ax*basey - ay*basex)) * ( signSrc*(delx*cy - dely*cx) + d*(basex*cy - basey*cx)) <0.0){	
	      return 1;
	    }
	  }
      }
  return 0;
}




/*
  MApl 
*/

int plucker_mahovsky_other(Ray_big *r, Plucker_coords *t, Intersection_big *p)
{

  float r0,r1,r2,r3,r4,r5;
  float A,B;
  
  /* Plucker för linjen*/
  r0 = r->orig[0] * r->end[1] - r->orig[1] * r->end[0];
  r1 = r->orig[0] * r->end[2] - r->orig[2] * r->end[0];
  r2 = r->orig[0] - r->end[0];
  r3 = r->orig[1] * r->end[2] - r->orig[2] * r->end[1];
  r4 = r->orig[2] - r->end[2];
  r5 = r->end[1] - r->orig[1];
  
  
  A = r0*t->a4 + r1*t->a5 +r3*t->a2;
  if(A+ r2*t->a3 +r4*t->a0+r5*t->a1 < 0)
    return 0;
  
  B = r0*t->b4 + r1*t->b5 + r3*t->b2;
  if(B + r2*t->b3 + r4*t->b0 +r5*t->b1 < 0)
    return 0;

  
  if(r2*t->c3 + r4*t->c0 + r5*t->c1 - A - B < 0)
    return 0;
 
  return 1;
}


/*
  PUpl
*/
int plucker_other(Ray_big *r, Plucker_coords *t, Intersection_big *p)
{
  
  float r0,r1,r2,r3,r4,r5;
  float A,B;
  int a,b,c;	
 
  
  /* Plucker för linjen*/
  r0 = r->orig[0] * r->end[1] - r->orig[1] * r->end[0];
  r1 = r->orig[0] * r->end[2] - r->orig[2] * r->end[0];
  r2 = r->orig[0] - r->end[0];
  r3 = r->orig[1] * r->end[2] - r->orig[2] * r->end[1];
  r4 = r->orig[2] - r->end[2];
  r5 = r->end[1] - r->orig[1];
  
  
  A = r0*t->a4 + r1*t->a5 +r3*t->a2;
  B = r0*t->b4 + r1*t->b5 + r3*t->b2;
  a = A + r2*t->a3 +r4*t->a0+r5*t->a1 >=0.;
  b = B + r2*t->b3 + r4*t->b0 +r5*t->b1 >= 0.;
  
  if(a != b)
    return 0;
  
  c = r2*t->c3 + r4*t->c0 + r5*t->c1 - A - B >= 0.;
  
  if(c != a)
    return 0;

  
  return 1;
}

 
/* ORp */

int orourke_other(Ray_big *r, Triangle_plane *t, Intersection_big *p){
  

  float vol0, vol1, vol2; 
  float ax,ay,az,bx,by,bz,cx,cy,cz,dx,dy,dz;
 
  ax = r->orig[0] - r -> end[0];
  ay = r->orig[1] - r -> end[1];
  az = r->orig[2] - r -> end[2];
  bx = t->v0[0] - r -> end[0];
  by = t->v0[1] - r -> end[1];
  bz = t->v0[2] - r -> end[2];
  cx = t->v1[0] - r -> end[0];
  cy = t->v1[1] - r -> end[1];
  cz = t->v1[2] - r -> end[2];
  dx = t->v2[0] - r -> end[0];
  dy = t->v2[1] - r -> end[1];
  dz = t->v2[2] - r -> end[2];

  

  vol0 = (ax * (by*cz - bz*cy) + ay * (bz*cx - bx*cz) + az *(bx*cy - by*cx));
  vol1 = (ax * (cy*dz - cz*dy) + ay * (cz*dx - cx*dz) + az *(cx*dy - cy*dx));
 
  
  if(vol0*vol1<0)
    return 0;


  vol2 = (ax * (dy*bz - dz*by) + ay * (dz*bx - dx*bz) + az *(dx*by - dy*bx));
  
  if(vol0*vol2<0)
    return 0;
  
 return 1;
}


/* ORCp */
int orourke_otherCCW(Ray_big *r, Triangle_plane *t, Intersection_big *p){

  float vol0, vol1, vol2;
  float ax,ay,az,bx,by,bz,cx,cy,cz,dx,dy,dz;
  

  ax = r->orig[0] - r -> end[0];
  ay = r->orig[1] - r -> end[1];
  az = r->orig[2] - r -> end[2];
  bx = t->v0[0] - r -> end[0];
  by = t->v0[1] - r -> end[1];
  bz = t->v0[2] - r -> end[2];
  cx = t->v1[0] - r -> end[0];
  cy = t->v1[1] - r -> end[1];
  cz = t->v1[2] - r -> end[2];


  vol0 = (ax * (by*cz - bz*cy) + ay * (bz*cx - bx*cz) + az *(bx*cy - by*cx));
  
  if(vol0 < 0)
    return 0;
  dx = t->v2[0] - r -> end[0];
  dy = t->v2[1] - r -> end[1];
  dz = t->v2[2] - r -> end[2];
  
  vol1 = (ax * (cy*dz - cz*dy) + ay * (cz*dx - cx*dz) + az *(cx*dy - cy*dx));

  if(vol1<0)
    return 0;
  vol2 = (ax * (dy*bz - dz*by) + ay * (dz*bx - dx*bz) + az *(dx*by - dy*bx));
  
  if(vol2<0)
    return 0;   
  
  return 1;
}

 
 /*
  HFp
*/
int halfplane_other(Ray_big *r, Triangle_plane *t, Intersection_big *p)
{
  
  float n0x,n0y,c0,n1x,n1y,c1;
  int X = t->i1;
  int Y = t->i2;
  float p0, p1;
  float tt,vd,vo;
  int sign1,sign2;
  
  vd = DOT(t->normal, r->dir);
  if(vd == 0)
    return 0;
  vo = -(DOT(t->normal, r->orig) + t->d);
  
  if(vd*vo <= 0)
    {
      
      return 0;
  }
  tt = vo/vd;
  
  p0 = r->orig[X] + tt*r->dir[X];
  p1 = r->orig[Y] + tt*r->dir[Y];
 
  n0x = -(t->v0[Y]- t->v1[Y]);
  n0y = t->v0[X] - t->v1[X];
  c0 = - t->v1[X]*n0x - t->v1[Y]*n0y;
  
  n1x = -(t->v1[Y] - t->v2[Y]);
  n1y = t->v1[X] - t->v2[X];
  c1 = - t->v2[X]*n1x - t->v2[Y]*n1y;
  
  sign1 = p0*n0x + p1*n0y + c0 < 0.;
  sign2 = p0*n1x + p1*n1y + c1 < 0.;
  
  if(sign1 == sign2 )
    {
      n1x = -(t->v2[Y] - t->v0[Y]);
      n1y = t->v2[X] - t->v0[X];
      c1 = - t->v0[X]*n1x - t->v0[Y]*n1y;
      sign2 = p0*n1x + p1*n1y + c1 < 0.;
      if(sign1 == sign2){
	(*p->t) = tt;
	return 1;
      }
    }  
  return 0;
}

/* HF2p */
int halfplane2_other(Ray_big *r, Triangle_plane *t, Intersection_big *p)
{
 

  float n0x,n0y,c0,n1x,n1y,c1;
  int X = t->i1;
  int Y = t->i2;
  float vd,vo;
  float sign1,sign2;

  vd = DOT(t->normal, r->dir);
  vo = - DOT(t->normal, r->orig) - t->d;
  
  if(vd*vo <= 0)
    return 0;
  
  n0x = -(t->v0[Y]- t->v1[Y]);
  n0y = t->v0[X] - t->v1[X];
  c0 = - t->v1[X]*n0x - t->v1[Y]*n0y;
  
  n1x = -(t->v1[Y] - t->v2[Y]);
  n1y = t->v1[X] - t->v2[X];
  c1 = - t->v2[X]*n1x - t->v2[Y]*n1y;

  sign1 = vd*(n0x*r->orig[X] + n0y*r->orig[Y]) + vo*(n0y * r->dir[Y] + n0x * r->dir[X]) + c0*vd; 
  sign2 = vd*(n1x*r->orig[X] + n1y*r->orig[Y]) + vo*(n1y * r->dir[Y] + n1x * r->dir[X]) + c1*vd;
  

  if(sign1*sign2<=0)
    return 0;
  
  n1x = -(t->v2[Y] - t->v0[Y]);
  n1y = t->v2[X] - t->v0[X];
  c1 = - t->v0[X]*n1x - t->v0[Y]*n1y;
  
  sign2 = vd*(n1x*r->orig[X] + n1y*r->orig[Y]) + vo*(n1y * r->dir[Y] + n1x * r->dir[X]) + c1*vd;
  if(sign1*sign2>0)
    return 1;
  
  //tt = vo/vd;
      
  return 0;
}


/* A2Dp */
int area2D_other(Ray_big *r, Triangle_plane *t, Intersection_big *p)
{
   
  float vd,vo,tt,point[2];
 
  float v0x,v0y,v1x,v1y;
  float area1,area2,area3;
  int X = t-> i1;
  int Y = t-> i2;
 
  vd = DOT(t->normal, r->dir);
  if(vd == 0)
    return 0;
  vo = -(DOT(t->normal, r->orig) + t->d);
  
  if(vd*vo <= 0)
    {
      
      return 0;
  }
  tt = vo/vd;
 
  point[0] = r->orig[X] + tt*r->dir[X];
  point[1] = r->orig[Y] + tt*r->dir[Y];

  v0x = t->v1[X] - t->v0[X];
  v0y = t->v1[Y] - t->v0[Y];

  v1x = point[0] - t->v0[X];
  v1y = point[1] - t->v0[Y];

  area1 = v0x*v1y - v1x*v0y;

  v0x = t->v2[X] - t->v1[X];
  v0y = t->v2[Y] - t->v1[Y];

  v1x = point[0] - t->v1[X];
  v1y = point[1] - t->v1[Y];

  area2 = v0x*v1y - v1x*v0y;
  

  if(area1*area2 > 0.){
    v0x = t->v0[X] - t->v2[X];
    v0y = t->v0[Y] - t->v2[Y];
    
    v1x = point[0] - t->v2[X];
    v1y = point[1] - t->v2[Y];
    
    area3 = v0x*v1y - v1x*v0y;
    if(area1*area3 > 0.){
      
      return 1;
    }
  }
  return 0;
  
}



/* ARi */

int arenberg_other_pre(Ray_big *r, Triangle_inv *t, Intersection_big *p)
{
  float num, den, tt,a,b;
  float trans1[3],point[3];
  
  den = DOT(r ->dir, t->bb2);
 
  if(den == 0.){
   return 0;
  }
  
  SUB(trans1, t -> v0, r -> orig);
  num = DOT(trans1, t->bb2);

  tt = num/den;
  
  if(tt <=0.){
    return 0;
  }
  
   
  SUB(trans1, r -> orig, t -> v0);
  point[0] = (tt * r -> dir[0]) + trans1[0];
  point[1] = (tt * r -> dir[1]) + trans1[1];
  point[2] = (tt * r -> dir[2]) + trans1[2];

  a = DOT(point,t->bb0);
  b = DOT(point,t->bb1);
 
  if( a < 0.0 || b < 0.0 || a + b > 1.0){
   return 0;
  }
  return 1;
}

/* AR2i */
int arenberg_other_pre2(Ray_big *r, Triangle_inv *t, Intersection_big *p)
{
  float num, den, a2, b2;
  float trans1[3];

  den = DOT(r ->dir, t->bb2);

  if(den == 0.)
   return 0;
  
  SUB(trans1, t -> v0, r -> orig);
  num = DOT(trans1, t->bb2);

  if(den*num <= 0){
    return 0;
  }
  
  SUB(trans1, r -> orig, t -> v0);

  a2 = DOT(trans1, t->bb0)*den + num*DOT(r->dir,t->bb0);

  if(a2 <0)
    return 0;

  b2 = DOT(trans1, t->bb1)*den + num*DOT(r->dir,t->bb1);
  
  
  if(b2 <0)
    return 0;
  
  if(a2 + b2 > den)
    return 0;
  
  
  return 1;
}



 /*
   HFh
*/
int halfplane_other_pre(Ray_big *r, Triangle_Halfplane *t, Intersection_big *p)
{
 
  int X = t->i1;
  int Y = t->i2;
  float p0, p1;
  float tt,vd,vo;
  float sign1,sign2;
  
  vd = DOT(t->normal, r->dir);
  if(vd == 0)
      return 0;
  vo = -(DOT(t->normal, r->orig) + t->d);
  
  if(vd*vo <= 0)
    return 0;
 
  tt = vo/vd;
 
  p0 = r->orig[X] + tt*r->dir[X];
  p1 = r->orig[Y] + tt*r->dir[Y];
  
  sign1 = p0*t->n0x + p1*t->n0y + t->c0 ;
  sign2 = p0*t->n1x + p1*t->n1y + t->c1 ;
  
  if(sign1*sign2>0 )
    {
      sign2 = p0*t->n2x + p1*t->n2y + t->c2;
      if(sign1*sign2>0){
	return 1;
      }
    }  
  return 0;
}

/* HF2h */
int halfplane_other_pre2(Ray_big *r, Triangle_Halfplane *t, Intersection_big *p)
{
 
  int X,Y;
  float sign1,sign2,sign3;
  float vd,vo;

  vd = DOT(t->normal, r->dir);
 
  vo = - DOT(t->normal, r->orig) - t->d;
  
  if(vd*vo <= 0)
    return 0;
 
  X = t->i1;
  Y = t->i2;

  sign1 = vd*(t->n0x*r->orig[X] + t->n0y*r->orig[Y]) + vo*(t->n0y * r->dir[Y] + t->n0x * r->dir[X]) + t->c0*vd; 
  sign2 = vd*(t->n1x*r->orig[X] + t->n1y*r->orig[Y]) + vo*(t->n1y * r->dir[Y] + t->n1x * r->dir[X]) + t->c1*vd;
 
  if(sign1*sign2<=0)
    return 0;
    
  sign3 = vd*(t->n2x*r->orig[X] + t->n2y*r->orig[Y]) + vo*(t->n2y * r->dir[Y] + t->n2x * r->dir[X]) + t->c2*vd;
  if(sign1*sign3>0)
    return 1;
  
  return 0;
}

/* CH3p */
int chirkov3_other(Ray_big *r, Triangle_plane *t, Intersection_big *p)
{
  float ax;
  float ay ;
  float az;
  float bx;
  float by;
  float bz;
  float delx;
  float dely;
  float delz;
  float basex;
  float basey;
  float basez;
  float cx;
  float cy;
  float cz;
  float d,bary1,bary2,bary3;
  float signSrc = t->normal[0]*r->orig[0] + t->normal[1]*r->orig[1] + t->normal[2]*r->orig[2] + t->d;
  float signDst = t->normal[0]*r->end[0] + t->normal[1]*r->end[1] + t->normal[2]*r->end[2] + t->d;

  if(signSrc*signDst>=0.0)	
    return 0;
  
  d = signSrc - signDst;
  
  if(t->i0==0)
    {
      ay = t->v1[1] - t->v0[1];
      az = t->v1[2] - t->v0[2];
      by = t->v2[1] - t->v0[1];
      bz = t->v2[2] - t->v0[2];
      dely = r->end[1] - r->orig[1];
      delz = r->end[2] - r->orig[2];
      basey = r->orig[1] - t->v0[1];
      basez = r->orig[2] - t->v0[2];

      
      bary1 = ( signSrc*(dely*bz - delz*by) + d*(basey*bz - basez*by));
      bary3 = signSrc*(ay *delz - az *dely) + d*(ay*basez - az*basey);
   
      bary1 = ( signSrc*(dely*bz - delz*by) + d*(basey*bz - basez*by));
      if( (bary3) * bary1 >=0.0)
	{
	  cy = t->v2[1] - t->v1[1];
	  cz = t->v2[2] - t->v1[2];
	  basey = r->orig[1] - t->v1[1];
	  basez = r->orig[2] - t->v1[2];
	  bary2 =  (signSrc*(dely*cz - delz*cy) + d*(basey*cz - basez*cy));
	  if( (bary3) * bary2 <0.0){	
	    return 1;
	  }
	}
    }
  else
    if(t->i0==1)
      {
	ax = t->v1[0] - t->v0[0];
	az = t->v1[2] - t->v0[2];
	bx = t->v2[0] - t->v0[0];
	bz = t->v2[2] - t->v0[2];
	delx = r->end[0] - r->orig[0];
	delz = r->end[2] - r->orig[2];
	basex = r->orig[0] - t->v0[0];
	basez = r->orig[2] - t->v0[2];

	bary3 = signSrc*(az *delx - ax *delz) + d*(az*basex - ax*basez);
	bary1 = ( signSrc*(delz*bx - delx*bz) + d*(basez*bx - basex*bz));
	if( bary3 * bary1 >=0.0)
	  {
	    cx = t->v2[0] - t->v1[0];
	    cz = t->v2[2] - t->v1[2];
	    basex = r->orig[0] - t->v1[0];
	    basez = r->orig[2] - t->v1[2];
	    bary2 = ( signSrc*(delz*cx - delx*cz) + d*(basez*cx - basex*cz));
	    if( bary3 * bary2 <0.0){	
	      
	      return 1;
	    }
	  }
      }
    else
      {
	ax = t->v1[0] - t->v0[0];
	ay = t->v1[1] - t->v0[1];
	bx = t->v2[0] - t->v0[0];
	by = t->v2[1] - t->v0[1];
	delx = r->end[0] - r->orig[0];
	dely = r->end[1] - r->orig[1];
	basex = r->orig[0] - t->v0[0];
	basey = r->orig[1] - t->v0[1];
      
	bary3 = signSrc*(ax *dely - ay *delx) + d*(ax*basey - ay*basex);
	bary1 = ( signSrc*(delx*by - dely*bx) + d*(basex*by - basey*bx));
	if( bary3 * bary1 >=0.0)
	  {
	    cx = t->v2[0] - t->v1[0];
	    cy = t->v2[1] - t->v1[1];
	    basex = r->orig[0] - t->v1[0];
	    basey = r->orig[1] - t->v1[1];
	    bary2 = ( signSrc*(delx*cy - dely*cx) + d*(basex*cy - basey*cx));
	    if( bary3 *  bary2 <0.0){	
	      return 1;
	    }
	  }
      }
  return 0;
}
