/* Triangle/triangle intersection test routine,
 * by Shen Hao, 2001. 
 * updated from Moller97.c("Tomas Moller 97")
 * About Moller97.c, see article "A Fast Triangle-Triangle Intersection Test",
 * Journal of Graphics Tools, 2(2), 1997
 *
 *
 * updated: 2001-11-22 by Shen Hao
 * Using a new theorem based on separating plane.
 * How to find the separating plane is emphasis.
 *
 * int tri_tri_intersect(float V0[3],float V1[3],float V2[3],
 *                       float U0[3],float U1[3],float U2[3])
 *
 * parameters: vertices of triangle 1: V0,V1,V2
 *             vertices of triangle 2: U0,U1,U2
 * result    : returns 1 if the triangles intersect, otherwise 0
 *
 *
 *
 * Assistant function copied from Moller97.c : 2001-11-20 by Shen Hao
 * int coplanar_tri_tri(float N[3],float V0[3],float V1[3],float V2[3],
 *                                 float U0[3],float U1[3],float U2[3])
 *
 * parameters: normal of the plane   : N
 *             vertices of triangle 1: V0,V1,V2
 *             vertices of triangle 2: U0,U1,U2
 * result    : returns 1 if the triangles intersect, otherwise 0
 */

#include <math.h>

#define FABS(x) (x>=0?x:-x)        /* implement as is fastest on your machine */
/* #define FABS(x) ((float)fabs(x)) */
/* if   is true then we do a check: 
         if |dv|<EPSILON then dv=0.0;
   else no check is done (which is less robust)
*/
#define USE_EPSILON_TEST TRUE
#define EPSILON 0.0000001


/* some macros */
#define CROSS(dest,v1,v2)                      \
              dest[0]=v1[1]*v2[2]-v1[2]*v2[1]; \
              dest[1]=v1[2]*v2[0]-v1[0]*v2[2]; \
              dest[2]=v1[0]*v2[1]-v1[1]*v2[0];

#define DOT(v1,v2) (v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2])

#define SUB(dest,v1,v2) dest[0]=v1[0]-v2[0]; dest[1]=v1[1]-v2[1]; dest[2]=v1[2]-v2[2]; 

#define ADD(dest,v1,v2) dest[0]=v1[0]+v2[0]; dest[1]=v1[1]+v2[1]; dest[2]=v1[2]+v2[2]; 

/* whether signs of d0,d1,d2 are same */
#define SAMESIGN012(d0,d1,d2) ( (d0*d1>0)&&(d0*d2>0) )

/* whether signs of d0,d1 are same, and different from the sign of d2 */
#define SIGN01_DIF_SIGN2(d0,d1,d2)    \
  ( (d0>=0) && (d1>=0) && (d2< 0) ||  \
    (d0<=0) && (d1<=0) && (d2> 0) ||  \
    (d0> 0) && (d1> 0) && (d2<=0) ||  \
    (d0< 0) && (d1< 0) && (d2>=0) )



/* this edge to edge test is based on Franlin Antonio's gem:
   "Faster Line Segment Intersection", in Graphics Gems III,
   pp. 199-202 */ 
#define EDGE_EDGE_TEST(V0,U0,U1)                      \
  Bx=U0[i0]-U1[i0];                                   \
  By=U0[i1]-U1[i1];                                   \
  Cx=V0[i0]-U0[i0];                                   \
  Cy=V0[i1]-U0[i1];                                   \
  f=Ay*Bx-Ax*By;                                      \
  d=By*Cx-Bx*Cy;                                      \
  if((f>0 && d>=0 && d<=f) || (f<0 && d<=0 && d>=f))  \
  {                                                   \
    e=Ax*Cy-Ay*Cx;                                    \
    if(f>0)                                           \
    {                                                 \
      if(e>=0 && e<=f) return 1;                      \
    }                                                 \
    else                                              \
    {                                                 \
      if(e<=0 && e>=f) return 1;                      \
    }                                                 \
  }                                

#define EDGE_AGAINST_TRI_EDGES(V0,V1,U0,U1,U2) \
{                                              \
  float Ax,Ay,Bx,By,Cx,Cy,e,d,f;               \
  Ax=V1[i0]-V0[i0];                            \
  Ay=V1[i1]-V0[i1];                            \
  /* test edge U0,U1 against V0,V1 */          \
  EDGE_EDGE_TEST(V0,U0,U1);                    \
  /* test edge U1,U2 against V0,V1 */          \
  EDGE_EDGE_TEST(V0,U1,U2);                    \
  /* test edge U2,U1 against V0,V1 */          \
  EDGE_EDGE_TEST(V0,U2,U0);                    \
}

#define POINT_IN_TRI(V0,U0,U1,U2)           \
{                                           \
  float a,b,c,d0,d1,d2;                     \
  /* is T1 completly inside T2? */          \
  /* check if V0 is inside tri(U0,U1,U2) */ \
  a=U1[i1]-U0[i1];                          \
  b=-(U1[i0]-U0[i0]);                       \
  c=-a*U0[i0]-b*U0[i1];                     \
  d0=a*V0[i0]+b*V0[i1]+c;                   \
                                            \
  a=U2[i1]-U1[i1];                          \
  b=-(U2[i0]-U1[i0]);                       \
  c=-a*U1[i0]-b*U1[i1];                     \
  d1=a*V0[i0]+b*V0[i1]+c;                   \
                                            \
  a=U0[i1]-U2[i1];                          \
  b=-(U0[i0]-U2[i0]);                       \
  c=-a*U2[i0]-b*U2[i1];                     \
  d2=a*V0[i0]+b*V0[i1]+c;                   \
  if(d0*d1>0.0)                             \
  {                                         \
    if(d0*d2>0.0) return 1;                 \
  }                                         \
}

int coplanar_tri_tri(float N[3],float V0[3],float V1[3],float V2[3],
                     float U0[3],float U1[3],float U2[3])
{
   float A[3];
   short i0,i1;
   /* first project onto an axis-aligned plane, that maximizes the area */
   /* of the triangles, compute indices: i0,i1. */
   A[0]=FABS(N[0]);
   A[1]=FABS(N[1]);
   A[2]=FABS(N[2]);
   if(A[0]>A[1])
   {
      if(A[0]>A[2])  
      {
          i0=1;      /* A[0] is greatest */
          i1=2;
      }
      else
      {
          i0=0;      /* A[2] is greatest */
          i1=1;
      }
   }
   else   /* A[0]<=A[1] */
   {
      if(A[2]>A[1])
      {
          i0=0;      /* A[2] is greatest */
          i1=1;                                           
      }
      else
      {
          i0=0;      /* A[1] is greatest */
          i1=2;
      }
    }               
                
    /* test all edges of triangle 1 against the edges of triangle 2 */
    EDGE_AGAINST_TRI_EDGES(V0,V1,U0,U1,U2);
    EDGE_AGAINST_TRI_EDGES(V1,V2,U0,U1,U2);
    EDGE_AGAINST_TRI_EDGES(V2,V0,U0,U1,U2);
                
    /* finally, test if tri1 is totally contained in tri2 or vice versa */
    POINT_IN_TRI(V0,U0,U1,U2);
    POINT_IN_TRI(U0,V0,V1,V2);

    return 0;
}


int tri_tri_intersect(float V0[3],float V1[3],float V2[3],
                      float U0[3],float U1[3],float U2[3])
{

  float E1[3],E2[3];
  float N1[3],N2[3],d1,d2;
  float tmp[3];
  float du0,du1,du2,dv0,dv1,dv2;
  float N[3];
  float tdu0,tdu1,tdv0,tdv1;
  float du01,du02;
  /* flags of which vertex is single on one side of plane(U0,U1,U2), and plane(V0,V1,V2) */
  /* VV0 is the pointer to the single vertex in (V0,V1,V2) of the plane(U0,U1,U2) */
  /* UU0 is the pointer to the single vertex in (U0,U1,U2) of the plane(V0,V1,V2) */
  float *UU0,*UU1,*UU2,*VV0,*VV1,*VV2;

  /* compute plane equation of triangle(U0,U1,U2) */
  SUB(E1,U1,U0);
  SUB(E2,U2,U0);
  CROSS(N1,E1,E2);
  d1=-DOT(N1,U0);
  /* plane equation 1: N1.X+d1=0 */
  /* put V0,V1,V2 into plane equation 1 to compute signed distances to the plane*/
  dv0=DOT(N1,V0)+d1;
  dv1=DOT(N1,V1)+d1;
  dv2=DOT(N1,V2)+d1;

  /* coplanarity robustness check */
#if USE_EPSILON_TEST==TRUE
  if(FABS(dv0)<EPSILON) dv0=0.0;
  if(FABS(dv1)<EPSILON) dv1=0.0;
  if(FABS(dv2)<EPSILON) dv2=0.0;
#endif

  if(SAMESIGN012(dv0,dv1,dv2)) /* same sign on all of them + not equal 0 ? */
    return 0;               /* no intersection occurs */


  /* compute plane equation of triangle(V0,V1,V2) */
  SUB(E1,V1,V0);
  SUB(E2,V2,V0);
  CROSS(N2,E1,E2);
  d2=-DOT(N2,V0);
  /* plane equation 2: N2.X+d2=0 */

  /* put U0,U1,U2 into plane equation 1 to compute signed distances to the plane*/
  du0=DOT(N2,U0)+d2;
  du1=DOT(N2,U1)+d2;
  du2=DOT(N2,U2)+d2;

  /* coplanarity robustness check */
#if USE_EPSILON_TEST==TRUE
  if(FABS(du0)<EPSILON) du0=0.0;
  if(FABS(du1)<EPSILON) du1=0.0;
  if(FABS(du2)<EPSILON) du2=0.0;
#endif

  if(SAMESIGN012(du0,du1,du2)) /* same sign on all of them + not equal 0 ? */
    return 0;               /* no intersection occurs */

  /* V0,V1,V2,U0,U1,U2 are all in the same plane */
  if ((dv0==0)&&(dv1==0)&&(dv2==0))
  return coplanar_tri_tri(N1,V0,V1,V2,U0,U1,U2);



  /* replace V0,V1,V2 with VV0,VV1,VV2 */
  if      (SIGN01_DIF_SIGN2(dv1,dv2,dv0))
  {
    VV0=V0;VV1=V1;VV2=V2;tdv0=dv0;tdv1=dv1;    /* V0 is single vertex of the plane of triangle (U0,U1,U2) */
  }
  else if (SIGN01_DIF_SIGN2(dv2,dv0,dv1))
  {
    VV0=V1;VV1=V2;VV2=V0;tdv0=dv1;tdv1=dv2;    /* V1 is single vertex of the plane of triangle (U0,U1,U2) */
  }
  else if (SIGN01_DIF_SIGN2(dv0,dv1,dv2))
  {
    VV0=V2;VV1=V0;VV2=V1;tdv0=dv2;tdv1=dv0;    /* V2 is single vertex of the plane of triangle (U0,U1,U2) */
  }


  /* replace U0,U1,U2 with UU0,UU1,UU2 */
  if      (SIGN01_DIF_SIGN2(du1,du2,du0))
  {
    UU0=U0;UU1=U1;UU2=U2;tdu0=du0;tdu1=du1;  /* U0 is single vertex of the plane of triangle (V0,V1,V2) */
  }
  else if (SIGN01_DIF_SIGN2(du2,du0,du1))
  {
    UU0=U1;UU1=U2;UU2=U0;tdu0=du1;tdu1=du2;  /* U1 is single vertex of the plane of triangle (V0,V1,V2) */
  }
  else if (SIGN01_DIF_SIGN2(du0,du1,du2))
  {
    UU0=U2;UU1=U0;UU2=U1;tdu0=du2;tdu1=du0;  /* U2 is single vertex of the plane of triangle (V0,V1,V2) */
  }


  /* compute distance between lines of triangle (V0,V1,V2) and triangle (U0,U1,U2) */
  if ((tdu0>=0) && (tdu1<=0))
  {
    if ((tdv0>=0) && (tdv1<=0))
    {
      SUB(E1,VV0,VV2);
      SUB(E2,UU0,UU2);
      CROSS(N,E1,E2);
      SUB(tmp,UU0,VV0);
      du02=DOT(N,tmp);
#if USE_EPSILON_TEST==TRUE
      if (du02>EPSILON)
#else
      if (du02>0)
#endif
        return 0;
      
      SUB(E1,VV1,VV0);
      SUB(E2,UU1,UU0);
      CROSS(N,E1,E2);
      SUB(tmp,UU0,VV0);
      du01=DOT(N,tmp);
#if USE_EPSILON_TEST==TRUE
      if (du01<-EPSILON)
#else
      if (du01<0)
#endif
        return 0;
    }
    else
    {
      SUB(E1,VV1,VV0);
      SUB(E2,UU0,UU2);
      CROSS(N,E1,E2);
      SUB(tmp,UU0,VV0);
      du02=DOT(N,tmp);
#if USE_EPSILON_TEST==TRUE
      if (du02>EPSILON)
#else
      if (du02>0)
#endif
        return 0;
      
      SUB(E1,VV0,VV2);
      SUB(E2,UU1,UU0);
      CROSS(N,E1,E2);
      SUB(tmp,UU0,VV0);
      du01=DOT(N,tmp);
#if USE_EPSILON_TEST==TRUE
      if (du01<-EPSILON)
#else
      if (du01<0)
#endif
        return 0;
    }
  }
  else
  {
    if ((tdv0>=0) && (tdv1<=0))
    {
      SUB(E1,VV0,VV2);
      SUB(E2,UU1,UU0);
      CROSS(N,E1,E2);
      SUB(tmp,UU0,VV0);
      du01=DOT(N,tmp);
#if USE_EPSILON_TEST==TRUE
      if (du01>EPSILON)
#else
      if (du01>0)
#endif
        return 0;
      
      SUB(E1,VV1,VV0);
      SUB(E2,UU0,UU2);
      CROSS(N,E1,E2);
      SUB(tmp,UU0,VV0);
      du02=DOT(N,tmp);
#if USE_EPSILON_TEST==TRUE
      if (du02<-EPSILON)
#else
      if (du02<0)
#endif
        return 0;
    }
    else
    {
      SUB(E1,VV1,VV0);
      SUB(E2,UU1,UU0);
      CROSS(N,E1,E2);
      SUB(tmp,UU0,VV0);
      du01=DOT(N,tmp);
#if USE_EPSILON_TEST==TRUE
      if (du01>EPSILON)
#else
      if (du01>0)
#endif
        return 0;
      
      SUB(E1,VV0,VV2);
      SUB(E2,UU0,UU2);
      CROSS(N,E1,E2);
      SUB(tmp,UU0,VV0);
      du02=DOT(N,tmp);
#if USE_EPSILON_TEST==TRUE
      if (du02<-EPSILON)
#else
      if (du02<0)
#endif
        return 0;
    }
  }

  return 1;
}
