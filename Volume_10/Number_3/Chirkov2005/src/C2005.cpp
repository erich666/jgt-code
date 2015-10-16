/*
Fast 3D Line-segment Triangle Intersection Test
Nick Chirkov
ArtyShock LLC
*/

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


#define	RAYPOINT(result,start,end,dist) { result[0]=start[0]+d*(end[0]-start[0]); result[1]=start[1]+d*(end[1]-start[1]); result[2]=start[2]+d*(end[2]-start[2]); }


//-------------------------------------------------------------------------------------------
//this is a simple implementation with division and without main axis reduction
//no precomputed data is used
//just for clear view of algorithm
//-------------------------------------------------------------------------------------------
int c2005_0(const RAYTRI *rt)
{

	float e0[3],e1[3],e2[3],norm[3],point[3],v[3],av[3],vb[3],vc[3];
	SUB(e0, rt->v1, rt->v0);
	SUB(e1, rt->v2, rt->v0);
	CROSS(norm,e0,e1);

	float pd = DOT(norm, rt->v0);

	float signSrc = DOT(norm, rt->org) -pd;
	float signDst = DOT(norm, rt->end) -pd;

	if(signSrc*signDst > 0.0) return 0;

	float d = signSrc/(signSrc - signDst);

	RAYPOINT(point, rt->org, rt->end,d);
	SUB(v, point, rt->v0);		
	CROSS(av,e0,v);
	CROSS(vb,v,e1);

	if(DOT(av,vb) > 0.0)
	{
		SUB(e2, rt->v1, rt->v2);
		SUB(v, point, rt->v1);
		CROSS(vc,v,e2);
		if(DOT(av,vc) > 0.0) return 1;
	}
	return 0;
}

//-------------------------------------------------------------------------------------------
//no precomputed data, time-based analysis
//-------------------------------------------------------------------------------------------
int c2005_2(const RAYTRI *rt)
{
	float e0x = rt->v1[0] - rt->v0[0];
	float e0y = rt->v1[1] - rt->v0[1];
	float e0z = rt->v1[2] - rt->v0[2];
	float e1x = rt->v2[0] - rt->v0[0];
	float e1y = rt->v2[1] - rt->v0[1];
	float e1z = rt->v2[2] - rt->v0[2];
	float normx = e0y * e1z - e0z * e1y;
	float normy = e0z * e1x - e0x * e1z;
	float normz = e0x * e1y - e0y * e1x;
	float pd = normx*rt->v0[0] + normy*rt->v0[1] + normz*rt->v0[2];

	float signSrc = normx*rt->org[0] + normy*rt->org[1] + normz*rt->org[2] - pd;
	float signDst = normx*rt->end[0] + normy*rt->end[1] + normz*rt->end[2] - pd;
	if(signSrc*signDst > 0.0) return 0;

	float SQ = 0.57735f;//sqrtf(1.0f/3.0f);
	float len = (normx*normx + normy*normy + normz*normz)*SQ;

	float d = signSrc - signDst;
	if(fabs(normx)>len)
	{
		float diry = rt->end[1] - rt->org[1];
		float dirz = rt->end[2] - rt->org[2];
		float basey = rt->org[1] - rt->v0[1];
		float basez = rt->org[2] - rt->v0[2];

		float adelx = signSrc*(e0y * dirz - e0z * diry);
		if( (adelx + d*(e0y*basez - e0z*basey)) * ( signSrc*(diry*e1z - dirz*e1y) + d*(basey*e1z - basez*e1y)) > 0.0)
		{
			float e2y = rt->v1[1] - rt->v2[1];
			float e2z = rt->v1[2] - rt->v2[2];
			basey = rt->org[1] - rt->v1[1];
			basez = rt->org[2] - rt->v1[2];
			if( (adelx + d*(e0y*basez - e0z*basey)) * ( signSrc*(diry*e2z - dirz*e2y) + d*(basey*e2z - basez*e2y)) > 0.0)	return 1;
		}
	}
	else
		if(fabs(normy)>len)
		{
			float dirx = rt->end[0] - rt->org[0];
			float dirz = rt->end[2] - rt->org[2];
			float basex = rt->org[0] - rt->v0[0];
			float basez = rt->org[2] - rt->v0[2];
			float adely = signSrc*(e0z * dirx - e0x * dirz);
			if( (adely + d*(e0z*basex - e0x*basez)) * ( signSrc*(dirz*e1x - dirx*e1z) + d*(basez*e1x - basex*e1z)) > 0.0)
			{
				float e2x = rt->v1[0] - rt->v2[0];
				float e2z = rt->v1[2] - rt->v2[2];
				basex = rt->org[0] - rt->v1[0];
				basez = rt->org[2] - rt->v1[2];
				if( (adely + d*(e0z*basex - e0x*basez)) * ( signSrc*(dirz*e2x - dirx*e2z) + d*(basez*e2x - basex*e2z)) > 0.0)	return 1;
			}
		}
		else
		{
			float dirx = rt->end[0] - rt->org[0];
			float diry = rt->end[1] - rt->org[1];
			float basex = rt->org[0] - rt->v0[0];
			float basey = rt->org[1] - rt->v0[1];
			float adelz = signSrc*(e0x * diry - e0y * dirx);

			if( (adelz + d*(e0x*basey - e0y*basex)) * ( signSrc*(dirx*e1y - diry*e1x) + d*(basex*e1y - basey*e1x)) > 0.0)
			{
				float e2x = rt->v1[0] - rt->v2[0];
				float e2y = rt->v1[1] - rt->v2[1];
				basex = rt->org[0] - rt->v1[0];
				basey = rt->org[1] - rt->v1[1];
				if( (adelz + d*(e0x*basey - e0y*basex)) * ( signSrc*(dirx*e2y - diry*e2x) + d*(basex*e2y - basey*e2x)) > 0.0)	return 1;
			}
		}

		return 0;
}

//-------------------------------------------------------------------------------------------
//no precomputed data, statistical analysis
//-------------------------------------------------------------------------------------------
int c2005_2_(const RAYTRI *rt)
{
	stat(1,0,0); float e0x = rt->v1[0] - rt->v0[0];
	stat(1,0,0); float e0y = rt->v1[1] - rt->v0[1];
	stat(1,0,0); float e0z = rt->v1[2] - rt->v0[2];
	stat(1,0,0); float e1x = rt->v2[0] - rt->v0[0];
	stat(1,0,0); float e1y = rt->v2[1] - rt->v0[1];
	stat(1,0,0); float e1z = rt->v2[2] - rt->v0[2];
	stat(1,2,0); float normx = e0y * e1z - e0z * e1y;
	stat(1,2,0); float normy = e0z * e1x - e0x * e1z;
	stat(1,2,0); float normz = e0x * e1y - e0y * e1x;
	stat(2,3,0); float pd = normx*rt->v0[0] + normy*rt->v0[1] + normz*rt->v0[2];

	stat(3,3,0); float signSrc = normx*rt->org[0] + normy*rt->org[1] + normz*rt->org[2] - pd;
	stat(3,3,0); float signDst = normx*rt->end[0] + normy*rt->end[1] + normz*rt->end[2] - pd;
	stat(0,1,0); if(signSrc*signDst > 0.0) return 0;

	float SQ = 0.57735f;//sqrtf(1.0f/3.0f);
	stat(2,4,0); float len = (normx*normx + normy*normy + normz*normz)*SQ;

	stat(1,0,0); float d = signSrc - signDst;
	stat(1,0,0); if(fabs(normx)>len)
	{
		stat(1,0,0); float diry = rt->end[1] - rt->org[1];
		stat(1,0,0); float dirz = rt->end[2] - rt->org[2];
		stat(1,0,0); float basey = rt->org[1] - rt->v0[1];
		stat(1,0,0); float basez = rt->org[2] - rt->v0[2];

		stat(1,3,0); float adelx = signSrc*(e0y * dirz - e0z * diry);
		stat(5,10,0); if( (adelx + d*(e0y*basez - e0z*basey)) * ( signSrc*(diry*e1z - dirz*e1y) + d*(basey*e1z - basez*e1y)) > 0.0)
		{
			stat(1,0,0); float e2y = rt->v1[1] - rt->v2[1];
			stat(1,0,0); float e2z = rt->v1[2] - rt->v2[2];
			stat(1,0,0); basey = rt->org[1] - rt->v1[1];
			stat(1,0,0); basez = rt->org[2] - rt->v1[2];
			stat(5,10,0); if( (adelx + d*(e0y*basez - e0z*basey)) * ( signSrc*(diry*e2z - dirz*e2y) + d*(basey*e2z - basez*e2y)) > 0.0)	return 1;
		}
	}
	else
	{
		stat(1,0,0); if(fabs(normy)>len)
		{
			stat(1,0,0); float dirx = rt->end[0] - rt->org[0];
			stat(1,0,0); float dirz = rt->end[2] - rt->org[2];
			stat(1,0,0); float basex = rt->org[0] - rt->v0[0];
			stat(1,0,0); float basez = rt->org[2] - rt->v0[2];
			stat(1,3,0); float adely = signSrc*(e0z * dirx - e0x * dirz);
			stat(5,10,0); if( (adely + d*(e0z*basex - e0x*basez)) * ( signSrc*(dirz*e1x - dirx*e1z) + d*(basez*e1x - basex*e1z)) > 0.0)
			{
				stat(1,0,0); float e2x = rt->v1[0] - rt->v2[0];
				stat(1,0,0); float e2z = rt->v1[2] - rt->v2[2];
				stat(1,0,0); basex = rt->org[0] - rt->v1[0];
				stat(1,0,0); basez = rt->org[2] - rt->v1[2];
				stat(5,10,0); if( (adely + d*(e0z*basex - e0x*basez)) * ( signSrc*(dirz*e2x - dirx*e2z) + d*(basez*e2x - basex*e2z)) > 0.0)	return 1;
			}
		}
		else
		{
			stat(1,0,0); float dirx = rt->end[0] - rt->org[0];
			stat(1,0,0); float diry = rt->end[1] - rt->org[1];
			stat(1,0,0); float basex = rt->org[0] - rt->v0[0];
			stat(1,0,0); float basey = rt->org[1] - rt->v0[1];
			stat(1,3,0); float adelz = signSrc*(e0x * diry - e0y * dirx);

			stat(5,10,0); if( (adelz + d*(e0x*basey - e0y*basex)) * ( signSrc*(dirx*e1y - diry*e1x) + d*(basex*e1y - basey*e1x)) > 0.0)
			{
				stat(1,0,0); float e2x = rt->v1[0] - rt->v2[0];
				stat(1,0,0); float e2y = rt->v1[1] - rt->v2[1];
				stat(1,0,0); basex = rt->org[0] - rt->v1[0];
				stat(1,0,0); basey = rt->org[1] - rt->v1[1];
				stat(5,10,0); if( (adelz + d*(e0x*basey - e0y*basex)) * ( signSrc*(dirx*e2y - diry*e2x) + d*(basex*e2y - basey*e2x)) > 0.0)	return 1;
			}
		}
	}
	return 0;
}

//-------------------------------------------------------------------------------------------
//this is a fastest version that uses precomputed data, time-based analysis
//-------------------------------------------------------------------------------------------
int c2005_3(const RAYTRI *rt)
{
	float signSrc = rt->plane.x*rt->org[0] + rt->plane.y*rt->org[1] + rt->plane.z*rt->org[2] - rt->plane.d;
	float signDst = rt->plane.x*rt->end[0] + rt->plane.y*rt->end[1] + rt->plane.z*rt->end[2] - rt->plane.d;
	if(signSrc*signDst > 0.0)	return 0;

	float d = signSrc - signDst;

	if(rt->plane.type==RAYTRI::PLANE::X)
	{
		float e0y = rt->v1[1] - rt->v0[1];
		float e0z = rt->v1[2] - rt->v0[2];
		float e1y = rt->v2[1] - rt->v0[1];
		float e1z = rt->v2[2] - rt->v0[2];
		float basey = rt->org[1] - rt->v0[1];
		float basez = rt->org[2] - rt->v0[2];

		float adelx = signSrc*(e0y * rt->dir[2] - e0z * rt->dir[1]);
		if( (adelx + d*(e0y*basez - e0z*basey)) * ( signSrc*(rt->dir[1]*e1z - rt->dir[2]*e1y) + d*(basey*e1z - basez*e1y)) > 0.0)
		{
			float e2y = rt->v1[1] - rt->v2[1];
			float e2z = rt->v1[2] - rt->v2[2];
			basey = rt->org[1] - rt->v1[1];
			basez = rt->org[2] - rt->v1[2];
			if( (adelx + d*(e0y*basez - e0z*basey)) * ( signSrc*(rt->dir[1]*e2z - rt->dir[2]*e2y) + d*(basey*e2z - basez*e2y)) > 0.0)	return 1;
		}
	}
	else
		if(rt->plane.type==RAYTRI::PLANE::Y)
		{
			float e0x = rt->v1[0] - rt->v0[0];
			float e0z = rt->v1[2] - rt->v0[2];
			float e1x = rt->v2[0] - rt->v0[0];
			float e1z = rt->v2[2] - rt->v0[2];
			float basex = rt->org[0] - rt->v0[0];
			float basez = rt->org[2] - rt->v0[2];
			float adely = signSrc*(e0z * rt->dir[0] - e0x * rt->dir[2]);
			if( (adely + d*(e0z*basex - e0x*basez)) * ( signSrc*(rt->dir[2]*e1x - rt->dir[0]*e1z) + d*(basez*e1x - basex*e1z)) > 0.0)
			{
				float e2x = rt->v1[0] - rt->v2[0];
				float e2z = rt->v1[2] - rt->v2[2];
				basex = rt->org[0] - rt->v1[0];
				basez = rt->org[2] - rt->v1[2];
				if( (adely + d*(e0z*basex - e0x*basez)) * ( signSrc*(rt->dir[2]*e2x - rt->dir[0]*e2z) + d*(basez*e2x - basex*e2z)) > 0.0)	return 1;
			}
		}
		else
		{
			float e0x = rt->v1[0] - rt->v0[0];
			float e0y = rt->v1[1] - rt->v0[1];
			float e1x = rt->v2[0] - rt->v0[0];
			float e1y = rt->v2[1] - rt->v0[1];
			float basex = rt->org[0] - rt->v0[0];
			float basey = rt->org[1] - rt->v0[1];
			float adelz = signSrc*(e0x * rt->dir[1] - e0y * rt->dir[0]);

			if( (adelz + d*(e0x*basey - e0y*basex)) * ( signSrc*(rt->dir[0]*e1y - rt->dir[1]*e1x) + d*(basex*e1y - basey*e1x)) > 0.0)
			{
				float e2x = rt->v1[0] - rt->v2[0];
				float e2y = rt->v1[1] - rt->v2[1];
				basex = rt->org[0] - rt->v1[0];
				basey = rt->org[1] - rt->v1[1];
				if( (adelz + d*(e0x*basey - e0y*basex)) * ( signSrc*(rt->dir[0]*e2y - rt->dir[1]*e2x) + d*(basex*e2y - basey*e2x)) > 0.0)	return 1;
			}
		}
		return 0;
}

//-------------------------------------------------------------------------------------------
//this is a fastest version that uses precomputed data and calculates arithmetic operations executed
//-------------------------------------------------------------------------------------------
int c2005_3_(const RAYTRI *rt)
{
	stat(3,3,0);	float signSrc = rt->plane.x*rt->org[0] + rt->plane.y*rt->org[1] + rt->plane.z*rt->org[2] - rt->plane.d;
	stat(3,3,0);	float signDst = rt->plane.x*rt->end[0] + rt->plane.y*rt->end[1] + rt->plane.z*rt->end[2] - rt->plane.d;
	stat(0,1,0);	if(signSrc*signDst > 0.0)	return 0;

	stat(1,0,0);	float d = signSrc - signDst;

	if(rt->plane.type==RAYTRI::PLANE::X)
	{
		stat(1,0,0);	float e0y = rt->v1[1] - rt->v0[1];
		stat(1,0,0);	float e0z = rt->v1[2] - rt->v0[2];
		stat(1,0,0);	float e1y = rt->v2[1] - rt->v0[1];
		stat(1,0,0);	float e1z = rt->v2[2] - rt->v0[2];
		stat(1,0,0);	float diry = rt->end[1] - rt->org[1];
		stat(1,0,0);	float dirz = rt->end[2] - rt->org[2];
		stat(1,0,0);	float basey = rt->org[1] - rt->v0[1];
		stat(1,0,0);	float basez = rt->org[2] - rt->v0[2];

		stat(1,3,0);	float adelx = signSrc*(e0y * dirz - e0z * diry);
		stat(5,10,0);	if( (adelx + d*(e0y*basez - e0z*basey)) * ( signSrc*(diry*e1z - dirz*e1y) + d*(basey*e1z - basez*e1y)) > 0.0)
		{
			stat(1,0,0);	float e2y = rt->v1[1] - rt->v2[1];
			stat(1,0,0);	float e2z = rt->v1[2] - rt->v2[2];
			stat(1,0,0);	basey = rt->org[1] - rt->v1[1];
			stat(1,0,0);	basez = rt->org[2] - rt->v1[2];
			stat(5,10,0);	if( (adelx + d*(e0y*basez - e0z*basey)) * ( signSrc*(diry*e2z - dirz*e2y) + d*(basey*e2z - basez*e2y)) > 0.0)	return 1;
		}
	}
	else
		if(rt->plane.type==RAYTRI::PLANE::Y)
		{
			stat(1,0,0);	float e0x = rt->v1[0] - rt->v0[0];
			stat(1,0,0);	float e0z = rt->v1[2] - rt->v0[2];
			stat(1,0,0);	float e1x = rt->v2[0] - rt->v0[0];
			stat(1,0,0);	float e1z = rt->v2[2] - rt->v0[2];
			stat(1,0,0);	float dirx = rt->end[0] - rt->org[0];
			stat(1,0,0);	float dirz = rt->end[2] - rt->org[2];
			stat(1,0,0);	float basex = rt->org[0] - rt->v0[0];
			stat(1,0,0);	float basez = rt->org[2] - rt->v0[2];
			stat(1,3,0);	float adely = signSrc*(e0z * dirx - e0x * dirz);
			stat(5,10,0);	if( (adely + d*(e0z*basex - e0x*basez)) * ( signSrc*(dirz*e1x - dirx*e1z) + d*(basez*e1x - basex*e1z)) > 0.0)
			{
				stat(1,0,0);	float e2x = rt->v1[0] - rt->v2[0];
				stat(1,0,0);	float e2z = rt->v1[2] - rt->v2[2];
				stat(1,0,0);	basex = rt->org[0] - rt->v1[0];
				stat(1,0,0);	basez = rt->org[2] - rt->v1[2];
				stat(5,10,0);	if( (adely + d*(e0z*basex - e0x*basez)) * ( signSrc*(dirz*e2x - dirx*e2z) + d*(basez*e2x - basex*e2z)) > 0.0)	return 1;
			}
		}
		else
		{
			stat(1,0,0);	float e0x = rt->v1[0] - rt->v0[0];
			stat(1,0,0);	float e0y = rt->v1[1] - rt->v0[1];
			stat(1,0,0);	float e1x = rt->v2[0] - rt->v0[0];
			stat(1,0,0);	float e1y = rt->v2[1] - rt->v0[1];
			stat(1,0,0);	float dirx = rt->end[0] - rt->org[0];
			stat(1,0,0);	float diry = rt->end[1] - rt->org[1];
			stat(1,0,0);	float basex = rt->org[0] - rt->v0[0];
			stat(1,0,0);	float basey = rt->org[1] - rt->v0[1];
			stat(1,3,0);	float adelz = signSrc*(e0x * diry - e0y * dirx);

			stat(5,10,0);	if( (adelz + d*(e0x*basey - e0y*basex)) * ( signSrc*(dirx*e1y - diry*e1x) + d*(basex*e1y - basey*e1x)) > 0.0)
			{
				stat(1,0,0);	float e2x = rt->v1[0] - rt->v2[0];
				stat(1,0,0);	float e2y = rt->v1[1] - rt->v2[1];
				stat(1,0,0);	basex = rt->org[0] - rt->v1[0];
				stat(1,0,0);	basey = rt->org[1] - rt->v1[1];
				stat(5,10,0);	if( (adelz + d*(e0x*basey - e0y*basex)) * ( signSrc*(dirx*e2y - diry*e2x) + d*(basex*e2y - basey*e2x)) > 0.0)	return 1;
			}
		}

		return 0;
}
