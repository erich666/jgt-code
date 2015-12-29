
//--------------------------------------------------------------------------//
// LBVH.cpp: 
//--------------------------------------------------------------------------//

#include <float.h>
#include <stdio.h>
#include <math.h>
#include <functional>
#include <algorithm>
#include "Geometry.h"
#include "LBVH.h"

using namespace std;

//--------------------------------------------------------------------------//

LBVH::LBVH()
{
	dx = dy = dz = 0.0f;
	bvh = NULL;
	bvhSize = 0; 
	objectsPerLeaf = 1;
	triangles = NULL;
	boundingVolumes = NULL;
}
//--------------------------------------------------------------------------//

LBVH::~LBVH()
{
	if (bvh) delete[] bvh;
	if (triangles) delete triangles;
	// boundingVolumes deleted by Scene
}
//--------------------------------------------------------------------------//

bool LBVH::intersectRaySimple(Ray &ray, Intersection *intersection, float maxT)
{
	int i;
	bool rval = false;

	// IF WE ARE BOUNDING TRIANGLES
	if (triangles) {
		for (i=0; i<(int)triangles->size(); i++) {
			if (triangles->intersectTriangle(i, ray, intersection, maxT)) {
				if (intersection==NULL) return true;
				rval = true;	
			}
		}

	// IF WE ARE BOUNDING LBVHs
	} else {
		for (i=0; i<(int)boundingVolumes->size(); i++) {
			if ((*boundingVolumes)[i]->intersectRay(ray, intersection, maxT)) {
				if (intersection==NULL) return true;
				rval = true;	
			}
		}
	}

	return rval;
}
//--------------------------------------------------------------------------//

bool LBVH::intersectRay(Ray &ray, Intersection *intersection, float maxT)
{
	if (bvhSize == 0) return intersectRaySimple(ray, intersection, maxT);

	Point3 *origin = &ray.origin;
	Point3 *direction = &(ray.direction);
	int nodeNum;
	int childNum;
	LBVHnode *node;
	int start, end, i;
	bool rval = false;
	int numObjects = getNumObjects();

	// THE STACK
	int stackDepth = 0;
	int stack[256];
	
	// SETUP CONSTANTS
	float xx = 1.0f / direction->x();
	float yy = 1.0f / direction->y();
	float zz = 1.0f / direction->z();
	float kx1 = (bounds.min.x() - origin->x()) / dx;
	float ky1 = (bounds.min.y() - origin->y()) / dy;
	float kz1 = (bounds.min.z() - origin->z()) / dz;
	float kx2 = dx * xx;
	float ky2 = dy * yy;
	float kz2 = dz * zz;

	float tmin, tmax;
	float tminX, tminY, tminZ, tmaxX, tmaxY, tmaxZ;
	float *bminX, *bminY, *bminZ, *bmaxX, *bmaxY, *bmaxZ;

	// SET CASES DEPENDING ON DIRECTION OF RAY
	//
	if (xx >= 0.0f) { bminX=&tminX; bmaxX=&tmaxX; } 
	else            { bminX=&tmaxX; bmaxX=&tminX; }
	//
	if (yy >= 0.0f) { bminY=&tminY; bmaxY=&tmaxY; } 
	else            { bminY=&tmaxY; bmaxY=&tminY; }
	//
	if (zz >= 0.0f) { bminZ=&tminZ; bmaxZ=&tmaxZ; } 
	else            { bminZ=&tmaxZ; bmaxZ=&tminZ; }

	// PUT ROOT ON THE STACK
	stack[0] = 0;
	nodeNum = 0;

	// TRAVERSE THE HIERARCHY
	while (stackDepth >= 0) {
		nodeNum = stack[stackDepth--]; // pop node off stack
		node = &bvh[nodeNum];

		// CHECK TO SEE IF THE RAY INTERSECTS THE NODE
		*bminX = (node->xmin + kx1) * kx2;  
		*bmaxX = (node->xmax + kx1) * kx2;
		//
		*bminY = (node->ymin + ky1) * ky2;  
		*bmaxY = (node->ymax + ky1) * ky2;
		//
		*bminZ = (node->zmin + kz1) * kz2;  
		*bmaxZ = (node->zmax + kz1) * kz2;
		//
		tmin = MAX(tminX, MAX(tminY, tminZ));  
		tmax = MIN(tmaxX, MIN(tmaxY, tmaxZ));
		if (tmin>tmax || tmax<0.0f || tmin>maxT) continue;

		// PUSH CHILD NODES ONTO STACK IF WE ARE NOT AT A LEAF
		childNum = nodeNum * 4; // Assuming a branch factor of 4
		if (childNum + 4 < bvhSize) {
			stack[stackDepth+1] = childNum+1;
			stack[stackDepth+2] = childNum+2;
			stack[stackDepth+3] = childNum+3;
			stack[stackDepth+4] = childNum+4;
			stackDepth += 4;

		// IF A LEAF NODE WAS HIT, INTERSECT ANYTHING INSIDE IT
		} else {

			// IF WE ARE BOUNDING TRIANGLES
			if (triangles) {
				start = nodeNum - firstLeafOnBottomRow;
				if (start < 0) start += numLeaves - 1;
				start *= objectsPerLeaf;
				end = start + objectsPerLeaf;
				if (end > numObjects) end = numObjects;

				for (i=start; i<end; i++) {
					if (triangles->intersectTriangle(i, ray, intersection, maxT)) {
						if (intersection==NULL) return true;
						rval = true;
						maxT = intersection->tval;
					}
				}

			// IF WE ARE BOUNDING LBVHs
			} else {
				start = nodeNum - firstLeafOnBottomRow;
				if (start < 0) start += numLeaves - 1;
				start *= objectsPerLeaf;
				end = start + objectsPerLeaf;
				if (end > numObjects) end = numObjects;

				for (i=start; i<end; i++) {
					if ((*boundingVolumes)[i]->intersectRay(ray, intersection, maxT)) {
						if (intersection==NULL) return true;
						rval = true;	
						maxT = intersection->tval;
					}
				}
			}
			//

		}
	}

	return rval;
}

//--------------------------------------------------------------------------//

void LBVH::getPartialBoundingBox(BoundingBox &BB, int start, int end)
{
	int i;
	BoundingBox bbTemp;
	BB.init();

	if (triangles) {
		for (i=start; i<=end; i++) {
			if (i < triangles->size()) {
				triangles->getTriangleBounds(bbTemp, i);
				BB.expand(bbTemp);
			}
		}
	} else if (boundingVolumes) {
		for (i=start; i<=end; i++) {
			if (i < (int) boundingVolumes->size()) {
				BB.expand((*boundingVolumes)[i]->bounds);
			}
		}
	}
}
//--------------------------------------------------------------------------//

struct TriangleComparator : public std::binary_function< Triangle&, Triangle&, bool >
{
	TriangleMesh *mesh;
	int axis;
	TriangleComparator(TriangleMesh *m, int a) {mesh=m; axis=a;}
	bool operator()(const Triangle &A, const Triangle &B) {
		return (mesh->vertices[A.v0][axis] < mesh->vertices[B.v0][axis]);
	}
};
//-------------------

struct LBVHComparator : public std::binary_function< LBVHptr&, LBVHptr&, bool >
{
	int axis;
	LBVHComparator(int a) {axis=a;}
	bool operator()(const LBVHptr A, const LBVHptr B) {
		return (A->getBounds().min[axis] < B->getBounds().min[axis]);
	}
};

//--------------------------------------------------------------------------//

void LBVH::sortObjects(int axis, int start, int end)
{
	if (triangles != NULL) {
		TriangleComparator comparator(triangles, axis);
		if (start >= (int)triangles->size()) return;
		if (end   >= (int)triangles->size()) end = (int)triangles->size()-1;
		sort(&triangles->triangles[start], &triangles->triangles[end], comparator);
	} else {
		LBVHComparator comparator(axis);
		if (start >= (int)boundingVolumes->size()) return;
		if (end   >= (int)boundingVolumes->size()) end = (int)boundingVolumes->size()-1;
		sort(&(*boundingVolumes)[start], &(*boundingVolumes)[end], comparator);
	}
}
//--------------------------------------------------------------------------//

void LBVH::initBVHnode(int nodeNum, int start, int end, BoundingBox &BB)
{
	LBVHnode *node = &bvh[nodeNum];
	int *child, n1,n2,n3,n4;
	BoundingBox bbTemp;
	int axis;

	node->clear();
	if (end < start) {
		return;
	}

	// SET BOUNDING BOX OF NODE
	if (nodeNum != 0) getPartialBoundingBox(BB, start, end);
	node->xmin = (NUM_TYPE) floorf((BB.min.x() - bounds.min.x()) / dx);
	node->xmax = (NUM_TYPE) ceilf ((BB.max.x() - bounds.min.x()) / dx);
	//
	node->ymin = (NUM_TYPE) floorf((BB.min.y() - bounds.min.y()) / dy);
	node->ymax = (NUM_TYPE) ceilf ((BB.max.y() - bounds.min.y()) / dy);
	//
	node->zmin = (NUM_TYPE) floorf((BB.min.z() - bounds.min.z()) / dz);
	node->zmax = (NUM_TYPE) ceilf ((BB.max.z() - bounds.min.z()) / dz);

	if ((nodeNum*4 + 4)<bvhSize) {

		child = (int*) (&bvh[nodeNum*4+1]);
		n1 = *child;
		child = (int*) (&bvh[nodeNum*4+2]);
		n2 = *child;
		child = (int*) (&bvh[nodeNum*4+3]);
		n3 = *child;
		child = (int*) (&bvh[nodeNum*4+4]);
		n4 = *child;

		// SORT START TO END
		axis = BB.getMajorAxis();
		if (end-start >= 4) sortObjects(axis, start, end);

		// SORT FIRST TWO n1,n2
		if (n1 + n2 > 2) {
			getPartialBoundingBox(bbTemp, start, (start + n1 + n2 - 1));
			axis = bbTemp.getMajorAxis();
			sortObjects(axis, start, start+n1-1);
		}
		initBVHnode(1+nodeNum*4, start,    start+n1-1, bbTemp);
		initBVHnode(2+nodeNum*4, start+n1, start+n1+n2-1, bbTemp);

		// SORT LAST TWO n3,n4
		if (n3 + n4 > 2) {
			getPartialBoundingBox(bbTemp, start+n1+n2, start+n1+n2+n3+n4-1);
			axis = bbTemp.getMajorAxis();
			sortObjects(axis, start+n1+n2, start+n1+n2+n3+n4-1);
		}
		initBVHnode(3+nodeNum*4, start+n1+n2,    start+n1+n2+n3-1,    bbTemp);
		initBVHnode(4+nodeNum*4, start+n1+n2+n3, start+n1+n2+n3+n4-1, bbTemp);
	}
}
//--------------------------------------------------------------------------//

void LBVH::determineObjectsPerNode(void)
{
	int *node;
	int i, n;

	// assign objectsPerLeaf objects to leaf nodes
	for (i=firstLeaf; i<=bvhSize; i++) {
		node = (int*) (&bvh[i]);
		*node = objectsPerLeaf;
	}
	
	// Calculate number of objects in internal nodes based on children
	for (i=firstLeaf-1; i>=0; i--) {
		n = 0;
		node = (int*) (&bvh[4*i+1]);
		n += *node;
		node = (int*) (&bvh[4*i+2]);
		n += *node;
		node = (int*) (&bvh[4*i+3]);
		n += *node;
		node = (int*) (&bvh[4*i+4]);
		n += *node;
		//
		node = (int*) (&bvh[i]);
		*node = n;
	}
}

//--------------------------------------------------------------------------//

void LBVH::initHierarchy(int obsPerLeaf)
{
	objectsPerLeaf = obsPerLeaf;

	calculateBounds();
	Point3 extent = bounds.max - bounds.min;
	dx = extent.x() / BOX_DIVISIONS;
	dy = extent.y() / BOX_DIVISIONS;
	dz = extent.z() / BOX_DIVISIONS;

	if (bvh) delete[] bvh;
	bvh = NULL;

	// 4/3 as many nodes as leaves
	int numObjects = getNumObjects();
	while (numObjects % objectsPerLeaf != 0) numObjects++;
	bvhSize = (numObjects / objectsPerLeaf) * 4 / 3; 
	// make sure all internal nodes have 4 children
	while (bvhSize % 4 != 1) bvhSize++; 

	if (bvhSize < 5) {
		bvhSize = 0;
		return;
	}
	
	firstLeaf = 1 + (bvhSize-2)/4;
	numLeaves = 1 + bvhSize - firstLeaf;
	firstLeafOnBottomRow = 0;
	while (1) {
		if (firstLeafOnBottomRow*4 + 1 >= bvhSize) break;
		firstLeafOnBottomRow = firstLeafOnBottomRow*4 + 1;
	}

	bvh = new LBVHnode[bvhSize];
	determineObjectsPerNode();
	initBVHnode(0, 0, numObjects-1, bounds);
}
//--------------------------------------------------------------------------//

void LBVH::calculateBounds(void) 
{
	if (triangles != NULL) {
		triangles->getBounds(bounds);
		bounds.expand(EPSILON);
	} else {
		bounds.init();
		for (int i=0; i<(int)boundingVolumes->size(); i++) {
			bounds.expand((*boundingVolumes)[i]->getBounds());
		}
	}
}
//--------------------------------------------------------------------------//
