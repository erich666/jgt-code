/* Ray-Triangle Intersection Test Routines          */
/* Different optimizations of my and Ben Trumbore's */
/* code from journals of graphics tools (JGT)       */
/* http://www.acm.org/jgt/                          */
/* by Tomas Moller, May 2000                        */

#include <math.h>
#include "idata.h"
extern void stat(int a, int m, int d);

#define EPSILON 0.000001
#define CROSS(dest,v1,v2) \
	dest[0]=v1[1]*v2[2]-v1[2]*v2[1]; \
	dest[1]=v1[2]*v2[0]-v1[0]*v2[2]; \
	dest[2]=v1[0]*v2[1]-v1[1]*v2[0];
#define DOT(v1,v2) (v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2])
#define SUB(dest,v1,v2) \
	dest[0]=v1[0]-v2[0]; \
	dest[1]=v1[1]-v2[1]; \
	dest[2]=v1[2]-v2[2]; 

/* the original jgt code */
int intersect_triangle(const RAYTRI *rt)
{
	double u, v;
	double edge1[3], edge2[3], tvec[3], pvec[3], qvec[3];
	double det,inv_det;

	/* find vectors for two edges sharing rt->v0 */
	SUB(edge1, rt->v1, rt->v0);
	SUB(edge2, rt->v2, rt->v0);

	/* begin calculating determinant - also used to calculate U parameter */
	CROSS(pvec, rt->dir, edge2);

	/* if determinant is near zero, ray lies in plane of triangle */
	det = DOT(edge1, pvec);

	if (det > -EPSILON && det < EPSILON)
		return 0;
	inv_det = 1.0 / det;

	/* calculate distance from rt->v0 to ray origin */
	SUB(tvec, rt->org, rt->v0);

	/* calculate U parameter and test bounds */
	u = DOT(tvec, pvec) * inv_det;
	if (u < 0.0 || u > 1.0)
		return 0;

	/* prepare to test V parameter */
	CROSS(qvec, tvec, edge1);

	/* calculate V parameter and test bounds */
	v = DOT(rt->dir, qvec) * inv_det;
	if (v < 0.0 || u + v > 1.0)
		return 0;

	/* calculate t, ray intersects triangle */
	double t = DOT(edge2, qvec) * inv_det;
	if(t<0.0 || t>1.0)	return 0;

	return 1;
}


/* code rewritten to do tests on the sign of the determinant */
/* the division is at the end in the code                    */
int intersect_triangle1(const RAYTRI *rt)
{
	double u, v;
	double edge1[3], edge2[3], tvec[3], pvec[3], qvec[3];
	double det;

	/* find vectors for two edges sharing rt->v0 */
	SUB(edge1, rt->v1, rt->v0);
	SUB(edge2, rt->v2, rt->v0);

	/* begin calculating determinant - also used to calculate U parameter */
	CROSS(pvec, rt->dir, edge2);

	/* if determinant is near zero, ray lies in plane of triangle */
	det = DOT(edge1, pvec);

	if (det > EPSILON)
	{
		/* calculate distance from rt->v0 to ray origin */
		SUB(tvec, rt->org, rt->v0);

		/* calculate U parameter and test bounds */
		u = DOT(tvec, pvec);
		if (u < 0.0 || u > det)
			return 0;

		/* prepare to test V parameter */
		CROSS(qvec, tvec, edge1);

		/* calculate V parameter and test bounds */
		v = DOT(rt->dir, qvec);
		if (v < 0.0 || u + v > det)
			return 0;

	}
	else if(det < -EPSILON)
	{
		/* calculate distance from rt->v0 to ray origin */
		SUB(tvec, rt->org, rt->v0);

		/* calculate U parameter and test bounds */
		u = DOT(tvec, pvec);
		/*      printf("u=%f\n",(float)u); */
		/*      printf("det=%f\n",det); */
		if (u > 0.0 || u < det)
			return 0;

		/* prepare to test V parameter */
		CROSS(qvec, tvec, edge1);

		/* calculate V parameter and test bounds */
		v = DOT(rt->dir, qvec) ;
		if (v > 0.0 || u + v < det)
			return 0;
	}
	else return 0;  /* ray is parallell to the plane of the triangle */


	double inv_det = 1.0 / det;

	/* calculate t, ray intersects triangle */
	double t = DOT(edge2, qvec) * inv_det;
	if(t<0.0 || t>1.0)	return 0;

	return 1;
}


/* code rewritten to do tests on the sign of the determinant */
/* the division is at the end in the code                    */
int intersect_triangle1_(const RAYTRI *rt)
{
	double u, v;
	double edge1[3], edge2[3], tvec[3], pvec[3], qvec[3];
	double det;

	/* find vectors for two edges sharing rt->v0 */
	stat(3,0,0);	SUB(edge1, rt->v1, rt->v0);
	stat(3,0,0);	SUB(edge2, rt->v2, rt->v0);

	/* begin calculating determinant - also used to calculate U parameter */
	stat(3,6,0);	CROSS(pvec, rt->dir, edge2);

	/* if determinant is near zero, ray lies in plane of triangle */
	stat(2,3,0);	det = DOT(edge1, pvec);

	stat(1,0,0);	if (det > EPSILON)
	{
		/* calculate distance from rt->v0 to ray origin */
		stat(3,0,0);	SUB(tvec, rt->org, rt->v0);

		/* calculate U parameter and test bounds */
		stat(2,3,0);	u = DOT(tvec, pvec);
		stat(1,0,0);	if (u < 0.0 || u > det)
			return 0;

		/* prepare to test V parameter */
		stat(3,6,0);	CROSS(qvec, tvec, edge1);

		/* calculate V parameter and test bounds */
		stat(2,3,0);	v = DOT(rt->dir, qvec);
		stat(2,0,0);	if (v < 0.0 || u + v > det)
			return 0;

	}
	else
	{
		stat(1,0,0);	if(det < -EPSILON)
		{
			/* calculate distance from rt->v0 to ray origin */
			stat(3,0,0);	SUB(tvec, rt->org, rt->v0);

			/* calculate U parameter and test bounds */
			stat(2,3,0);	u = DOT(tvec, pvec);
			/*      printf("u=%f\n",(float)u); */
			/*      printf("det=%f\n",det); */
			stat(1,0,0);	if (u > 0.0 || u < det)
				return 0;

			/* prepare to test V parameter */
			stat(3,6,0);	CROSS(qvec, tvec, edge1);

			/* calculate V parameter and test bounds */
			stat(2,3,0);	v = DOT(rt->dir, qvec) ;
			stat(2,0,0);	if (v > 0.0 || u + v < det)
				return 0;
		}
		else return 0;  /* ray is parallell to the plane of the triangle */
	}


	stat(0,0,1);	double inv_det = 1.0 / det;

	/* calculate t, ray intersects triangle */
	stat(2,4,0);	double t = DOT(edge2, qvec) * inv_det;
	stat(1,0,0);	if(t<0.0 || t>1.0)	return 0;

	return 1;
}

/* code rewritten to do tests on the sign of the determinant */
/* the division is before the test of the sign of the det    */
int intersect_triangle2(const RAYTRI *rt)
{
	double u, v;
	double edge1[3], edge2[3], tvec[3], pvec[3], qvec[3];
	double det,inv_det;

	/* find vectors for two edges sharing rt->v0 */
	SUB(edge1, rt->v1, rt->v0);
	SUB(edge2, rt->v2, rt->v0);

	/* begin calculating determinant - also used to calculate U parameter */
	CROSS(pvec, rt->dir, edge2);

	/* if determinant is near zero, ray lies in plane of triangle */
	det = DOT(edge1, pvec);

	/* calculate distance from rt->v0 to ray origin */
	SUB(tvec, rt->org, rt->v0);
	inv_det = 1.0 / det;

	if (det > EPSILON)
	{
		/* calculate U parameter and test bounds */
		u = DOT(tvec, pvec);
		if (u < 0.0 || u > det)
			return 0;

		/* prepare to test V parameter */
		CROSS(qvec, tvec, edge1);

		/* calculate V parameter and test bounds */
		v = DOT(rt->dir, qvec);
		if (v < 0.0 || u + v > det)
			return 0;

	}
	else if(det < -EPSILON)
	{
		/* calculate U parameter and test bounds */
		u = DOT(tvec, pvec);
		if (u > 0.0 || u < det)
			return 0;

		/* prepare to test V parameter */
		CROSS(qvec, tvec, edge1);

		/* calculate V parameter and test bounds */
		v = DOT(rt->dir, qvec) ;
		if (v > 0.0 || u + v < det)
			return 0;
	}
	else return 0;  /* ray is parallell to the plane of the triangle */

	/* calculate t, ray intersects triangle */
	double t = DOT(edge2, qvec) * inv_det;
	if(t<0.0 || t>1.0)	return 0;

	return 1;
}

/* code rewritten to do tests on the sign of the determinant */
/* the division is before the test of the sign of the det    */
int intersect_triangle2_(const RAYTRI *rt)
{
	double u, v;
	double edge1[3], edge2[3], tvec[3], pvec[3], qvec[3];
	double det,inv_det;

	/* find vectors for two edges sharing rt->v0 */
	stat(3,0,0);	SUB(edge1, rt->v1, rt->v0);
	stat(3,0,0);	SUB(edge2, rt->v2, rt->v0);

	/* begin calculating determinant - also used to calculate U parameter */
	stat(3,6,0);	CROSS(pvec, rt->dir, edge2);

	/* if determinant is near zero, ray lies in plane of triangle */
	stat(2,3,0);	det = DOT(edge1, pvec);

	/* calculate distance from rt->v0 to ray origin */
	stat(3,0,0);	SUB(tvec, rt->org, rt->v0);
	stat(0,0,1);	inv_det = 1.0 / det;

	stat(1,0,0);	if (det > EPSILON)
	{
		/* calculate U parameter and test bounds */
		stat(2,3,0);	u = DOT(tvec, pvec);
		stat(1,0,0);	if (u < 0.0 || u > det)
			return 0;

		/* prepare to test V parameter */
		stat(3,6,0);	CROSS(qvec, tvec, edge1);

		/* calculate V parameter and test bounds */
		stat(2,3,0);	v = DOT(rt->dir, qvec);
		stat(2,0,0);	if (v < 0.0 || u + v > det)
			return 0;

	}
	else
	{
		stat(1,0,0);	if(det < -EPSILON)
		{
			/* calculate U parameter and test bounds */
			stat(2,3,0);	u = DOT(tvec, pvec);
			stat(1,0,0);	if (u > 0.0 || u < det)
				return 0;

			/* prepare to test V parameter */
			stat(3,6,0);	CROSS(qvec, tvec, edge1);

			/* calculate V parameter and test bounds */
			stat(2,3,0);	v = DOT(rt->dir, qvec) ;
			stat(2,0,0);	if (v > 0.0 || u + v < det)
				return 0;
		}
		else return 0;  /* ray is parallell to the plane of the triangle */
	}

	/* calculate t, ray intersects triangle */
	stat(2,4,0);	double t = DOT(edge2, qvec) * inv_det;
	stat(1,0,0);	if(t<0.0 || t>1.0)	return 0;

	return 1;
}

/* code rewritten to do tests on the sign of the determinant */
/* the division is before the test of the sign of the det    */
/* and one CROSS has been moved out from the if-else if-else */
int intersect_triangle3(const RAYTRI *rt)
{
	double u, v;
	double edge1[3], edge2[3], tvec[3], pvec[3], qvec[3];
	double det,inv_det;

	/* find vectors for two edges sharing rt->v0 */
	SUB(edge1, rt->v1, rt->v0);
	SUB(edge2, rt->v2, rt->v0);

	/* begin calculating determinant - also used to calculate U parameter */
	CROSS(pvec, rt->dir, edge2);

	/* if determinant is near zero, ray lies in plane of triangle */
	det = DOT(edge1, pvec);

	/* calculate distance from rt->v0 to ray origin */
	SUB(tvec, rt->org, rt->v0);
	inv_det = 1.0 / det;

	CROSS(qvec, tvec, edge1);

	if (det > EPSILON)
	{
		u = DOT(tvec, pvec);
		if (u < 0.0 || u > det)
			return 0;

		/* calculate V parameter and test bounds */
		v = DOT(rt->dir, qvec);
		if (v < 0.0 || u + v > det)
			return 0;

	}
	else if(det < -EPSILON)
	{
		/* calculate U parameter and test bounds */
		u = DOT(tvec, pvec);
		if (u > 0.0 || u < det)
			return 0;

		/* calculate V parameter and test bounds */
		v = DOT(rt->dir, qvec) ;
		if (v > 0.0 || u + v < det)
			return 0;
	}
	else return 0;  /* ray is parallell to the plane of the triangle */

	double t = DOT(edge2, qvec) * inv_det;
	if(t<0.0 || t>1.0)	return 0;

	return 1;
}

