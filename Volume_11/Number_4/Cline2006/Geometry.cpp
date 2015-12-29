
//--------------------------------------------------------------------------//
//	Geometry.cpp 
//--------------------------------------------------------------------------//

#include <time.h>
#include <math.h>
#include <float.h>
#include "Geometry.h"
#include "LBVH.h"
using namespace std;

//--------------------------------------------------------------------------//
//	TIME function 
//--------------------------------------------------------------------------//

double TIME(void) 
{	
	return (double)(clock()) / (double)CLOCKS_PER_SEC;
}

//--------------------------------------------------------------------------//
// Triangle implementations
//--------------------------------------------------------------------------//

// RAY TRIANGLE INTERSECTION ADAPTED FROM
/* Ray-Triangle Intersection Test Routines          */
/* Different optimizations of my and Ben Trumbore's */
/* code from journals of graphics tools (JGT)       */
/* http://www.acm.org/jgt/                          */
/* by Tomas Moller, May 2000                        */

#define CROSS(dest,v1,v2) \
          dest[0]=v1[1]*v2[2]-v1[2]*v2[1]; \
          dest[1]=v1[2]*v2[0]-v1[0]*v2[2]; \
          dest[2]=v1[0]*v2[1]-v1[1]*v2[0];
#define DOT(v1,v2) (v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2])
#define SUB(dest,v1,v2) \
          dest[0]=v1[0]-v2[0]; \
          dest[1]=v1[1]-v2[1]; \
          dest[2]=v1[2]-v2[2]; 

bool Triangle::intersectRay(Ray &ray, TriangleMesh &mesh, 
							Intersection *intersection, float maxT)
{
	float *orig  = ray.origin.A;
	float *dir   = ray.direction.A;
	float *vert0 = mesh.vertices[v0].A;
	float *vert1 = mesh.vertices[v1].A;
	float *vert2 = mesh.vertices[v2].A;
	float t, u, v, alpha, beta, gamma;

	// Local variables in jgt function
	float edge1[3], edge2[3], tvec[3], pvec[3], qvec[3];
	float det, inv_det;

	/* find vectors for two edges sharing vert0 */
	SUB(edge1, vert1, vert0);
	SUB(edge2, vert2, vert0);

	/* begin calculating determinant - also used to calculate U parameter */
	CROSS(pvec, dir, edge2);

	/* if determinant is near zero, ray lies in plane of triangle */
	det = DOT(edge1, pvec);
	if (det > -0.000000001f && det < 0.000000001f) return false;
	inv_det = 1.0f / det;

	/* calculate distance from vert0 to ray origin */
	SUB(tvec, orig, vert0);

	/* calculate U parameter and test bounds */
	u = DOT(tvec, pvec) * inv_det;
	if (u < 0.0f || u > 1.0f) return false;

	/* prepare to test V parameter */
	CROSS(qvec, tvec, edge1);

	/* calculate V parameter and test bounds */
	v = DOT(dir, qvec) * inv_det;
	if (v < 0.0f || u + v > 1.0f) return false;

	/* calculate t, ray intersects triangle */
	t = DOT(edge2, qvec) * inv_det;

	if (!intersection) { 
		if (t < EPSILON || t > maxT) return false;
		return true;
	}
	if (t < EPSILON || t > intersection->tval) return false;

	// put values back into my data structures
	beta  = u;
	gamma = v;
	alpha = 1.0f - (beta+gamma);

	intersection->tval = t;
	intersection->material = &mesh.material;
	intersection->location = ray.origin + t*ray.direction;
	getNormal(intersection->normal, alpha, beta, gamma, mesh);
	if (intersection->normal.dot(ray.direction) > 0) intersection->normal *= -1.0f;

	return true;
}

//--------------------------------------------------------------------------//
// Scene implementations
//--------------------------------------------------------------------------//

Scene::~Scene() 
{
	int i;
	delete rootBoundingVolume;	
	for (i=0; i<(int)lights.size(); i++) delete lights[i];
	for (i=0; i<(int)meshes.size(); i++) delete meshes[i];
}

//--------------------------------------------------------------------------//
// Camera implementations
//--------------------------------------------------------------------------//

Point3 Camera::traceRay(Ray &ray)
{
	LBVH *rootNode = scene->rootBoundingVolume;
	Intersection intersection;	
	intersection.tval = FLT_MAX;
	Ray shadowRay;
	int i;
	Point3 color;
	DirectionalLight *light;

	if (rootNode && rootNode->intersectRay(ray, &intersection, FLT_MAX)) {
		
		// ADD AMBIENT COLOR
		color = scene->ambientLight * intersection.material->diffuse;

		// ADD CONTRIBUTION FROM EACH LIGHT
		for (i=0; i<(int)scene->lights.size(); i++) {
			light = scene->lights[i];
			if (intersection.normal.dot(light->direction) < 0.0f) continue;
			shadowRay.set(intersection.location, light->direction);
			if (!rootNode->intersectRay(shadowRay, NULL, FLT_MAX)) {
				color += intersection.evaluateLighting(-ray.direction, *light);
			}
		}

		return color;
	}
	
	return scene->backgroundColor;
}

//--------------------------------------------------------------------------//

void Camera::captureImage(Scene *s, char *imageName)
{
	double startTime = TIME();

	int i,j;
	Point3 U,V,N;
	Point3 Xinc, Yinc;
	Point3 *image = new Point3[imageWidth*imageHeight];
	int w = imageWidth;
	int h = imageHeight;
	float tanVal = 2.0f * tanf(fieldOfView * 0.5f);
	Ray ray;
	unsigned char c;
	scene = s;

	// SET UP INCREMENTS
	N = (lookFrom-lookAt); N.normalize();
	U = (viewUp.cross(N)); U.normalize();
	V = (N.cross(U)); V.normalize();
	Xinc = U * (tanVal / h);
	Yinc = V * (-tanVal / h);

	// ITERATE THROUGH IMAGE PIXELS
	for (j=0; j<h; j+=1) {
		printf("\rRENDERING: %d / %d  ", j+1, h);
		for (i=0; i<w; i+=1) {
			ray.set(lookFrom, (-N) + ((i-w*0.5f)*Xinc) + ((j-h*0.5f)*Yinc));
			image[j*w + i] = traceRay(ray);
		}
	}

	double endTime = TIME();
	printf("\rRENDERING: done. (%0.3f seconds)\n", endTime-startTime);

	// WRITE PPM FILE
	FILE *F = fopen(imageName, "w+b");
	fprintf(F, "P6\n%d %d\n255\n", w, h); // PPM header
	for (i=0; i<w*h; i++) {
		image[i].clamp(0.0f, 1.0f);
		for (j=0; j<3; j++) {
			c = (unsigned char)(image[i][j]*255.0f);
			fwrite(&c, 1, 1, F);
		}
	}
	fclose(F);

	delete[] image;
}
//--------------------------------------------------------------------------//
