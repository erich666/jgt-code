#ifndef TRIANGLE_H
#define TRIANGLE_H

#include <stdio.h>
#include <stdlib.h>

#define CROSS(dest,v1,v2) \
          dest[0]=v1[1]*v2[2]-v1[2]*v2[1]; \
          dest[1]=v1[2]*v2[0]-v1[0]*v2[2]; \
          dest[2]=v1[0]*v2[1]-v1[1]*v2[0];
#define DOT(v1,v2) (v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2])
#define SUB(dest,v1,v2) \
          dest[0]=v1[0]-v2[0]; \
          dest[1]=v1[1]-v2[1]; \
          dest[2]=v1[2]-v2[2]; 

#define NORMALIZE(v) \
{  float s=(float)sqrt(DOT((v),(v)));          \
   if(s>0.00000000001f)                    \
   {  float is=1.0f/s;                   \
      (v)[0]*=is; (v)[1]*=is; (v)[2]*=is; \
   }                                      \
}

#define DET3D(a,b,c,d)(a[0]*b[1]*c[2] - a[0]*b[1]*d[2] - a[0]*b[2]*c[1] + a[0]*b[2]*d[1] + a[0]*c[1]*d[2] - a[0]*c[2]*d[1] - a[1]*b[0]*c[2] + a[1]*b[0]*d[2] + a[1]*b[2]*c[0] - a[1]*b[2]*d[0] - a[1]*c[0]*d[2] + a[1]*d[0]*c[2] + a[2]*b[0]*c[1] - a[2]*b[0]*d[1] - a[2]*b[1]*c[0] + a[2]*b[1]*d[0] + a[2]*c[0]*d[1] - a[2]*c[1]*d[0] - b[0]*c[1]*d[2] + b[0]*c[2]*d[1] + b[1]*c[0]*d[2] - b[1]*c[2]*d[0] - b[2]*c[0]*d[1] + b[2]*c[1]*d[0])



#define SIGNTOINT(a)(a>0. ? 1 : (a==0. ? 0 : -1))



#define EPSILON 0.000001

/* LOTS OF ERIT STUFF*/

#define ZERO EPSILON
#define EPS  0.0


#define SimpleSignEps(x, eps)  \
 ((x <= eps) ? ((x < -eps) ? -1 : 0) : 1)

#define ProjTriEdge(a, b, c, p, q, nr, A, B, C, P, Q) \
{\
   SelectPlane(a, b, c, nr, A, B, C); \
   ProjEdge(p, q, P, Q); }

int project_h_plane;
float project_h_pnt[3], project_h_nr[3];

#define Abs(x) (((x) >= 0.0) ? (x) : -(x))


#define SelectPlane(a, b, c, nr, A, B, C) \
{\
   project_h_nr[0] = Abs(nr[0]); \
   project_h_nr[1] = Abs(nr[1]); \
   project_h_nr[2] = Abs(nr[2]); \
   if ((project_h_nr[0] > project_h_nr[2])  &&  \
       (project_h_nr[0] >= project_h_nr[1])) { \
      if (nr[0] > 0.0) { \
         A[0] = b[2]; \
         B[0] = a[2]; \
         A[1] = b[1]; \
         B[1] = a[1]; \
      } \
      else { \
         A[0] = a[2]; \
         B[0] = b[2]; \
         A[1] = a[1]; \
         B[1] = b[1]; \
      } \
      C[0] = c[2]; \
      C[1] = c[1]; \
      project_h_plane = 2; \
   } \
   else if ((project_h_nr[1] > project_h_nr[2]) &&   \
            (project_h_nr[1] >= project_h_nr[0]))  { \
      if (nr[1] > 0.0)  { \
         A[0] = b[0]; \
         B[0] = a[0]; \
         A[1] = b[2]; \
         B[1] = a[2]; \
      } \
      else { \
         A[0] = a[0]; \
         B[0] = b[0]; \
         A[1] = a[2]; \
         B[1] = b[2]; \
      } \
      C[0] = c[0]; \
      C[1] = c[2]; \
      project_h_plane = 3; \
   } \
   else  { \
      if (nr[2] < 0.0)  { \
         A[0] = b[0]; \
         A[1] = b[1]; \
         B[0] = a[0]; \
         B[1] = a[1]; \
 } \
      else { \
         A[0] = a[0]; \
         A[1] = a[1]; \
         B[0] = b[0]; \
	 B[1] = b[1]; \
      } \
      C[0] = c[0]; \
      C[1] = c[1]; \
      project_h_plane = 1; \
   } \
}



#define ProjEdge(p, q, P, Q) \
{\
   if (project_h_plane == 1)  { \
      P[0] = p[0]; \
      Q[0] = q[0]; \
      P[1] = p[1]; \
      Q[1] = q[1]; \
   } \
   else if (project_h_plane == 2)  { \
      P[0] = p[2]; \
      Q[0] = q[2]; \
      P[1] = p[1]; \
      Q[1] = q[1]; \
   } \
   else { \
      P[0] = p[0]; \
      Q[0] = q[0]; \
      P[1] = p[2]; \
      Q[1] = q[2]; \
   } \
}

#define LinearComb2D(p, q, r, t)  \
 {r[0] = p[0] + (t) * (q[0] - p[0]); \
  r[1] = p[1] + (t) * (q[1] - p[1]); }

#define Det2D(u, v, w) \
 ((u[0] - v[0]) * (v[1] - w[1]) + (v[1] - u[1]) * (v[0] - w[0]))

#define TriPnt2D(A, B, C, P) \
((Det2D(A, B, P) <= 0.0) ? 0 : \
 ((Det2D(B, C, P) <= 0.0) ? 0 : \
  ((Det2D(C, A, P) <= 0.0) ? 0 : 1)))

float martin_h_local;
#define Sign(x) \
(martin_h_local = x, \
 ((martin_h_local > 0.0) ? 1 : ((martin_h_local < 0.0) ? -1 : 0)))

#define TriEdge2D(A, B, C, F, G, sign_t_a, sign_t_b, sign_t_c) \
/*                                                                         */ \
/* check whether the line segment  FG  intersects the  triangle A,B,C.     */ \
/* we first check whether the triangle lies on one side of  FG.            */ \
/*                                                                         */ \
(sign_t_a = Sign(Det2D(F, G, A)), \
 (sign_t_a < 0) \
 ? (sign_t_a = 1, \
    sign_t_b = -Sign(Det2D(F, G, B)), \
    sign_t_c = -Sign(Det2D(F, G, C))) \
 : (sign_t_b = Sign(Det2D(F, G, B)), \
    sign_t_c = Sign(Det2D(F, G, C))), \
 /*                                                                        */ \
 /* note that  sign_t_a >= 0.                                              */ \
 /*                                                                        */ \
 (sign_t_b >= 0) \
 ? ((sign_t_c >= 0) \
    ? 0                                /* triangle on one side of line */ \
    /*                                                                     */ \
    /* A  and  B  lie on a different side of  FG  than C.  thus, the       */ \
    /* supporting line of  FG  intersects  BC  and  CA.                    */ \
    /*                                                                     */ \
    : ((Det2D(B, C, F) <= 0.0) \
       ? ((Det2D(B, C, G) <=  0.0) \
          ? 0                                    /* F,G  right of  BC  */ \
          : 1)                                    /* FG  intersects  BC */ \
       : ((Det2D(C, A, F) > 0.0) \
          ? 1                                     /* F  between  BC, CA */ \
          : ((Det2D(C, A, G) <= 0.0) \
             ? 0                                 /* F,G  right of  CA  */ \
             : 1))))                              /* FG  intersects  CA */ \
 : ((sign_t_c >= 0) \
    /*                                                                     */ \
    /* A  and  C  lie on a different side of  FG  than B.  thus, the       */ \
    /* supporting line of  FG  intersects  AB  and  BC.                    */ \
    /*                                                                     */ \
    ? ((Det2D(A, B, F) <= 0.0) \
       ? ((Det2D(A, B, G) <=  0.0) \
          ? 0                                 /* F,G  right of  AB     */ \
          : 1)                                 /* FG  intersects  AB    */ \
       : ((Det2D(B, C, F) > 0.0) \
          ? 1                                  /* F  between  AB, BC    */ \
          : ((Det2D(B, C, G) <= 0.0) \
             ? 0                              /* F,G  right of  BC     */ \
             : 1)))                            /* FG  intersects  BC    */ \
    : ( \
       /*                                                                  */ \
       /* B  and  C  lie on a different side of  FG  than A.  thus, the    */ \
       /* supporting line of  FG  intersects  AB  and  CA.                 */ \
       /*                                                                  */ \
       (Det2D(A, B, F) <= 0.0) \
        ? ((Det2D(A, B, G) <=  0.0) \
           ? 0                                   /* F,G  right of  AB  */ \
           : 1)                                   /* FG  intersects  AB */ \
        : ((Det2D(C, A, F) > 0.0) \
           ? 1                                    /* F  between  AB, CA */ \
           : ((Det2D(C, A, G) <= 0.0) \
              ? 0                                /* F,G  right of  CA  */ \
              : 1)))))                            /* FG  intersects  CA */ \


int test_hit(float org[3], float dir[3], float vert0[3],float vert1[3], float vert2[3]);

/*
  Struct for planes, 
  a, b, c, d describes the plane equation such as ax + by + cz + d = 0
  i0 is dominatiing axis i1 and i2 the other axises. 
*/
typedef struct
{
  float normal[3];
  float d;
  int i0,i1,i2;
}Plane;

/*
the Plucker coordinates for the triangle
a0 = v0[0] * v1[1] - v1[0] * v0[1];
  a1 = v0[0] * v1[2] - v1[0] * v0[2];
  a2 = v0[0] - v1[0];
  a3 = v0[1] * v1[2] - v1[1] * v0[2];
  a4 = v0[2] - v1[2];
  a5 = v1[1] - v0[1];
  
 b the same for v1 and v2
c for v0 and v2
 
*/


typedef struct
{
  void *p;
}Triangle_other_stuff;

/* 21 floats */
typedef struct
{
  float a0,a1,a2,a3,a4,a5;
  float b0,b1,b2,b3,b4,b5;
  float c0,c1,c2,c3,c4,c5;
  float v0[3];
  Triangle_other_stuff *stuff;
}Plucker_coords;

typedef struct{
  float v0[3];
  float v1[3];
  float v2[3];
  float normal[3];
  float d;
  int i0,i1,i2;
  float a[6];
  float b[6];
  float c[6];
  Triangle_other_stuff *stuff;
}Triangle_big;

/* 9 floats*/
typedef struct{
  float v0[3];
  float v1[3];
  float v2[3];
  Triangle_other_stuff *stuff;
}Triangle_small;

/* 13 floats 3 ints*/
typedef struct{
  float v0[3];
  float v1[3];
  float v2[3];
  float normal[3];
  float d;
  int i0,i1,i2;
  float inv_n;
}Triangle_plane;

/* 14 floats 3 ints*/
typedef struct{
  //float v0[3];
  //float v1[3];
  //float v2[3];
  float normal[3];
  float d;
  int i0,i1,i2;
  float n0x,n0y,c0;
  float n1x,n1y,c1;
  float n2x,n2y,c2;
  float inv_n;
  Triangle_other_stuff *stuff;
}Triangle_Halfplane;

/* 12 floats
*/
typedef struct{
  float v0[3];
  float bb0[3];
  float bb1[3];
  float bb2[3];
  Triangle_other_stuff *stuff;
}Triangle_inv;

typedef struct
{
  float orig[3];
  float dir[3];
  float end[3];
}Ray_big;

typedef struct
{
  float orig[3];
  float dir[3];
}Ray_dir;

typedef struct
{
  float orig[3];
  float end[3];
}Ray_end;

typedef struct
{
  float *t;
  float *u;
  float *v;
  /*  float point[3]; */
}Intersection_big;

typedef struct
{
  float *t;
  float *u;
  float *v;
}Intersection_barycentric;


typedef struct
{
  float point[3];
}Intersection_point;


void mkTriangle_small(float v0[3], float v1[3], float v2[3], Triangle_small *);
void mkPlucker(float v0[3], float v1[3], float v2[3], Plucker_coords *);
void mkTriangle_plane(float v0[3], float v1[3], float v2[3], Triangle_plane*);
void mkTriangle_half(float v0[3], float v1[3], float v2[3], Triangle_Halfplane*);
void mkTriangle_inv(float v0[3], float v1[3], float v2[3], Triangle_inv*);

void mkRay_big(float orig[3], float dir[3], float end[3], Ray_big *);
Intersection_big * mkIntersection_big(float *t, float *u, float *v, float point[3]);
void printTri_big(Triangle_big *);
void printRay_big(Ray_big *);


#endif
