/*****************************************************************/
/* "On Faster Sphere-Box Overlap Testing" by                     */
/* Thomas Larsson, Tomas Akenine-Moller and Eric Lengyel.        */
/*                                                               */
/* Accompanying source code                                      */ 
/*                                                               */
/* Sphere-AABB and Sphere-OBB overlap tests code                 */
/*                                                               */
/*                                                               */
/* Functions:                                                    */
/*                                                               */
/* int overlapSphereAABB_Arvo(Sphere3D & sphere, Box3D & box);   */
/* int overlapSphereAABB_QRI(Sphere3D & sphere, Box3D & box);    */
/* int overlapSphereAABB_QRF(Sphere3D & sphere, Box3D & box);    */
/* int overlapSphereAABB_Cons(Sphere3D & sphere, Box3D & box);   */
/* int overlapSphereAABB_SSE(Sphere3D & sphere, Box3D & box);    */
/*                                                               */
/* int overlapSphereOBB_G_Arvo(Sphere3D & sphere, OBox3D & obox);*/
/* int overlapSphereOBB_QRI(Sphere3D & sphere, OBox3D & obox);   */
/* int overlapSphereOBB_QRF(Sphere3D & sphere, OBox3D & obox);   */
/* int overlapSphereOBB_Cons(Sphere3D & sphere, OBox3D & obox);  */
/* int overlapSphereOBB_SSE(Sphere3D & sphere, OBox3D & obox);   */
/*                                                               */
/* History:                                                      */
/*   2005-12-05: First version of source code created            */
/*   2006-05-08: Added SSE versions of the overlap tests         */
/*                                                               */
/*****************************************************************/

#include <xmmintrin.h>

typedef struct Point3D {
	float x, y, z;
} Point3D;

typedef struct Sphere3D {
	__declspec(align(16)) Point3D c;
	float r;
} Sphere3D;

typedef struct Box3D {
	__declspec(align(16)) Point3D min;
	__declspec(align(16)) Point3D max;
} Box3D;

typedef struct OBox3D {
	__declspec(align(16)) Point3D mid;
	__declspec(align(16)) Point3D ext;
	__declspec(align(16)) Point3D xaxis;
	__declspec(align(16)) Point3D yaxis;
	__declspec(align(16)) Point3D zaxis;
} OBox3D;

inline int overlapSphereAABB_Arvo(const Sphere3D & sphere, const Box3D & box) {
	float dmin = 0;
	float e;
	
	if(sphere.c.x < box.min.x) { 
		e = sphere.c.x - box.min.x;
		dmin += e * e; 
	} else if( sphere.c.x > box.max.x) {
		e = sphere.c.x - box.max.x;
		dmin += e * e; 		
    }

	if(sphere.c.y < box.min.y) { 
		e = sphere.c.y - box.min.y;
		dmin += e * e; 
	} else if( sphere.c.y > box.max.y) {
		e = sphere.c.y - box.max.y;
		dmin += e * e; 		
    }

	if(sphere.c.z < box.min.z) { 
		e = sphere.c.z - box.min.z;
		dmin += e * e; 
	} else if( sphere.c.z > box.max.z) {
		e = sphere.c.z - box.max.z;
		dmin += e * e; 		
    }

	if( dmin <= sphere.r * sphere.r ) return 1;
	return 0;
}

inline int overlapSphereAABB_QRI(const Sphere3D & sphere, const Box3D & box) {
	float dmin = 0;
	float e;
	
	if((e = sphere.c.x - box.min.x) < 0) { 
		if (e < -sphere.r) return 0;
		dmin += e * e; 
	} else if((e = sphere.c.x - box.max.x) > 0) {
		if (e > sphere.r) return 0;
		dmin += e * e; 		
    }

	if((e = sphere.c.y - box.min.y) < 0) { 
		if (e < -sphere.r) return 0;
		dmin += e * e; 
	} else if((e = sphere.c.y - box.max.y) > 0) {
		if (e > sphere.r) return 0;
		dmin += e * e; 		
    }

	if((e = sphere.c.z - box.min.z) < 0) { 
		if (e < -sphere.r) return 0;
		dmin += e * e; 
	} else if((e = sphere.c.z - box.max.z) > 0) {
		if (e > sphere.r) return 0;
		dmin += e * e; 		
    }

	if( dmin <= sphere.r * sphere.r ) return 1;
	return 0;
}

inline int overlapSphereAABB_QRF(const Sphere3D & sphere, const Box3D & box) {
	float dmin = 0;
	float e;

	if (sphere.c.x < box.min.x - sphere.r || sphere.c.x > box.max.x + sphere.r) return 0;
	if (sphere.c.y < box.min.y - sphere.r || sphere.c.y > box.max.y + sphere.r) return 0;
	if (sphere.c.z < box.min.z - sphere.r || sphere.c.z > box.max.z + sphere.r) return 0;

	if(sphere.c.x < box.min.x) { 
		e = sphere.c.x - box.min.x;
		dmin += e * e; 
	} else if( sphere.c.x > box.max.x) {
		e = sphere.c.x - box.max.x;
		dmin += e * e; 		
    }

	if(sphere.c.y < box.min.y) { 
		e = sphere.c.y - box.min.y;
		dmin += e * e; 
	} else if( sphere.c.y > box.max.y) {
		e = sphere.c.y - box.max.y;
		dmin += e * e; 		
    }

	if(sphere.c.z < box.min.z) { 
		e = sphere.c.z - box.min.z;
		dmin += e * e; 
	} else if( sphere.c.z > box.max.z) {
		e = sphere.c.z - box.max.z;
		dmin += e * e; 		
    }

	if( dmin <= sphere.r * sphere.r ) return 1;
	return 0;
}

inline int overlapSphereAABB_Cons(const Sphere3D & sphere, const Box3D & box) {
	if (sphere.c.x < box.min.x - sphere.r || sphere.c.x > box.max.x + sphere.r) return 0;
	if (sphere.c.y < box.min.y - sphere.r || sphere.c.y > box.max.y + sphere.r) return 0;
	if (sphere.c.z < box.min.z - sphere.r || sphere.c.z > box.max.z + sphere.r) return 0;
	return 1;
}

static bool overlapSphereAABB_SSE(const Sphere3D& sphere, const Box3D& box)
{
	/*float e = MaxZero(box.min.x - sphere.c.x) + MaxZero(sphere.c.x - box.max.x);
	float dmin = e * e;
	
	e = MaxZero(box.min.y - sphere.c.y) + MaxZero(sphere.c.y - box.max.y);
	dmin += e * e;
	
	e = MaxZero(box.min.z - sphere.c.z) + MaxZero(sphere.c.z - box.max.z);
	dmin += e * e;
	
	float r = sphere.r;
	return (dmin <= r * r);*/
	
	
	__m128 zero = _mm_setzero_ps();
	__m128 center = *(__m128 *) &sphere.c;
	__m128 boxmin = *(__m128 *) &box.min;
	__m128 boxmax = *(__m128 *) &box.max;
	
	__m128 e = _mm_add_ps(_mm_max_ps(_mm_sub_ps(boxmin, center), zero), _mm_max_ps(_mm_sub_ps(center, boxmax), zero));
	e = _mm_mul_ps(e, e);
	
	const Point3D *p = (Point3D *) &e;
	float r = sphere.r;
	return (p->x + p->y + p->z <= r * r);
}

inline int overlapSphereOBB_G_Arvo(const Sphere3D & sphere, const OBox3D & obox) {		
	float d, e, dmin = 0;
	Point3D v;
	
	v.x = sphere.c.x - obox.mid.x; 
	v.y = sphere.c.y - obox.mid.y;
	v.z = sphere.c.z - obox.mid.z;

	d = v.x * obox.xaxis.x + v.y * obox.xaxis.y + v.z * obox.xaxis.z;
	if(d < -obox.ext.x) { 
		e = d + obox.ext.x;
		dmin += e * e; 
	} else if(d > obox.ext.x) {
		e = d - obox.ext.x;
		dmin += e * e; 
    }
	
	d = v.x * obox.yaxis.x + v.y * obox.yaxis.y + v.z * obox.yaxis.z; 
	if(d < -obox.ext.y) { 
		e = d + obox.ext.y;
		dmin += e * e;
	} else if(d > obox.ext.y) {
		e = d - obox.ext.y;
		dmin += e * e; 			
    }

	d = v.x * obox.zaxis.x + v.y * obox.zaxis.y + v.z * obox.zaxis.z; 
	if(d < -obox.ext.z) { 
		e = d + obox.ext.z;
		dmin += e * e;
	} else if(d > obox.ext.z) {
		e = d - obox.ext.z;
		dmin += e * e; 			
    }

	if( dmin <= sphere.r * sphere.r ) return 1;
	return 0;
}

inline int overlapSphereOBB_QRI(const Sphere3D & sphere, const OBox3D & obox) {
	float d, e, dmin = 0;
	Point3D v;

	v.x = sphere.c.x - obox.mid.x;
	v.y = sphere.c.y - obox.mid.y;
	v.z = sphere.c.z - obox.mid.z;

	d = v.x * obox.xaxis.x + v.y * obox.xaxis.y + v.z * obox.xaxis.z; 
	if((e = d + obox.ext.x) < 0) { 
		if (-e > sphere.r) return 0;
		dmin += e * e; 
	} else if((e = d - obox.ext.x) > 0) {	
		if (e > sphere.r) return 0;
		dmin += e * e; 		
    }

	d = v.x * obox.yaxis.x + v.y * obox.yaxis.y + v.z * obox.yaxis.z;
	if((e = d + obox.ext.y) < 0) { 
		if (-e > sphere.r) return 0;
		dmin += e * e; 
	} else if((e = d - obox.ext.y) > 0) {	
		if (e > sphere.r) return 0;
		dmin += e * e; 		
    }

	d = v.x * obox.zaxis.x + v.y * obox.zaxis.y + v.z * obox.zaxis.z;
	if((e = d + obox.ext.z) < 0) { 
		if (-e > sphere.r) return 0;
		dmin += e * e; 
	} else if((e = d - obox.ext.z) > 0) {	
		if (e > sphere.r) return 0;
		dmin += e * e; 		
    }

	if( dmin <= sphere.r * sphere.r ) return 1;
	return 0;
}

inline int overlapSphereOBB_QRF(const Sphere3D & sphere, const OBox3D & obox) {
	float dx, dy, dz, e, dmin = 0;
	Point3D v;
	
	v.x = sphere.c.x - obox.mid.x; 
	v.y = sphere.c.y - obox.mid.y;
	v.z = sphere.c.z - obox.mid.z;

	dx = v.x * obox.xaxis.x + v.y * obox.xaxis.y + v.z * obox.xaxis.z;
	if (dx < - obox.ext.x - sphere.r || dx > obox.ext.x + sphere.r) return 0;
	dy = v.x * obox.yaxis.x + v.y * obox.yaxis.y + v.z * obox.yaxis.z;
	if (dy < - obox.ext.y - sphere.r || dy > obox.ext.y + sphere.r) return 0;
	dz = v.x * obox.zaxis.x + v.y * obox.zaxis.y + v.z * obox.zaxis.z;
	if (dz < - obox.ext.z - sphere.r || dz > obox.ext.z + sphere.r) return 0;

	if(dx < -obox.ext.x) { 
		e = dx + obox.ext.x;
		dmin += e * e; 
	} else if(dx > obox.ext.x) {
		e = dx - obox.ext.x;
		dmin += e * e; 
    }

	if(dy < -obox.ext.y) { 
		e = dy + obox.ext.y;
		dmin += e * e;
	} else if(dy > obox.ext.y) {
		e = dy - obox.ext.y;
		dmin += e * e; 			
    }

	if(dz < -obox.ext.z) { 
		e = dz + obox.ext.z;
		dmin += e * e;
	} else if(dz > obox.ext.z) {
		e = dz - obox.ext.z;
		dmin += e * e; 			
    }

	if( dmin <= sphere.r * sphere.r ) return 1;
	return 0;
}

inline int overlapSphereOBB_Cons(const Sphere3D & sphere, const OBox3D & obox) {
	float dx, dy, dz;
	Point3D v;
	
	v.x = sphere.c.x - obox.mid.x; 
	v.y = sphere.c.y - obox.mid.y;
	v.z = sphere.c.z - obox.mid.z;

	dx = v.x * obox.xaxis.x + v.y * obox.xaxis.y + v.z * obox.xaxis.z;
	if (dx < - obox.ext.x - sphere.r || dx > obox.ext.x + sphere.r) return 0;
	dy = v.x * obox.yaxis.x + v.y * obox.yaxis.y + v.z * obox.yaxis.z;
	if (dy < - obox.ext.y - sphere.r || dy > obox.ext.y + sphere.r) return 0;
	dz = v.x * obox.zaxis.x + v.y * obox.zaxis.y + v.z * obox.zaxis.z;
	if (dz < - obox.ext.z - sphere.r || dz > obox.ext.z + sphere.r) return 0;
	return 1;
}

static bool overlapSphereOBB_SSE(const Sphere3D& sphere, const OBox3D& box)
{
	__m128 zero = _mm_setzero_ps();
	__m128 center = *(__m128 *) &sphere.c;
	__m128 ext = *(__m128 *) &box.ext;
	__m128 mid = *(__m128 *) &box.mid;
	__m128 xaxis = *(__m128 *) &box.xaxis;
	__m128 yaxis = *(__m128 *) &box.yaxis;
	__m128 zaxis = *(__m128 *) &box.zaxis;

	__m128 v = _mm_sub_ps(center, mid);
	__m128 xmul = _mm_mul_ps(v, xaxis);
	__m128 ymul = _mm_mul_ps(v, yaxis);
	__m128 zmul = _mm_mul_ps(v, zaxis);

	// Sequential adds  
	/*
	__m128 d;
	d.m128_f32[0] = xmul.m128_f32[0] + xmul.m128_f32[1] + xmul.m128_f32[2];
	d.m128_f32[1] = ymul.m128_f32[0] + ymul.m128_f32[1] + ymul.m128_f32[2];
	d.m128_f32[2] = zmul.m128_f32[0] + zmul.m128_f32[1] + zmul.m128_f32[2];
	*/

	// Re-arrange elements so we can use parallell adds
	__m128 d, d1, d2;
    __m128 tmp0, tmp2, tmp1, tmp3;
	tmp0   = _mm_shuffle_ps(xmul, ymul, 0x44);
	tmp2   = _mm_shuffle_ps(xmul, ymul, 0xEE);
	tmp1   = _mm_shuffle_ps(zmul, zero, 0x44);
	tmp3   = _mm_shuffle_ps(zmul, zero, 0xEE);
	d = _mm_shuffle_ps(tmp0, tmp1, 0x88);
	d1 = _mm_shuffle_ps(tmp0, tmp1, 0xDD);
	d2 = _mm_shuffle_ps(tmp2, tmp3, 0x88);
	d = _mm_add_ps(d, d1);
	d = _mm_add_ps(d, d2);

	__m128 e = _mm_add_ps(_mm_min_ps(_mm_add_ps(d, ext), zero), _mm_max_ps(_mm_sub_ps(d, ext), zero));
	e = _mm_mul_ps(e, e);
	
	const Point3D *p = (Point3D *) &e;
	float r = sphere.r;
	return (p->x + p->y + p->z <= r * r);
}
