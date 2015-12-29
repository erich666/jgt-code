
//--------------------------------------------------------------------------//
// LBVH.h: A lightweigh bounding volume hierarchy
//--------------------------------------------------------------------------//

#ifndef _LBVH_
#define _LBVH_

#include "Geometry.h"
using namespace std;

//--------------------------------------------------------------------------//
// LBVHnode - a ligthwieght bounding volume node 
//--------------------------------------------------------------------------//

// TWO BYTE VALUES
// note: for some reason signed shorts are faster than unsigned ones.
#define NUM_TYPE short 
#define BOX_DIVISIONS 32766.0f;

/*
// ONE BYTE VALUES
#define NUM_TYPE unsigned char
#define BOX_DIVISIONS 254.0f
*/

class LBVHnode
{
public:
	NUM_TYPE xmin,xmax, ymin,ymax, zmin,zmax;

	void clear(void) {
		xmin=ymin=zmin=1; xmax=ymax=zmax=0;
	}
	void print(void) {
		printf("(%d,%d,%d) - (%d,%d,%d)", xmin,ymin,zmin, xmax,ymax,zmax);
	}
};

//--------------------------------------------------------------------------//
// LBVH - a lightweight bounding volume
//--------------------------------------------------------------------------//

class LBVH 
{
protected:
	BoundingBox bounds; // the world space bounds of the hierarchy
	float dx,dy,dz;     // scale factors for scaling the ray
	LBVHnode *bvh;      // the bounding volume hierarchy
	int bvhSize;        // the number of nodes in the hierarchy
	int objectsPerLeaf; // the number of objects enclosed by leaf nodes
	int numLeaves;      // the number of leaf nodes in the hierarchy
	int firstLeaf;      // the index of the first leaf node
	int firstLeafOnBottomRow; // the index of the first leaf node on the bottom row

	TriangleMesh *triangles;        // a triangle mesh that we are bounding OR
	vector<LBVH*> *boundingVolumes; // the bounding volumes that we are bounding

public:
	LBVH();
	~LBVH();

	const BoundingBox &getBounds(void) const {return bounds;}
	void setTriangleMesh(TriangleMesh *mesh) {triangles = mesh;}
	void setBoundingVolumes(vector<LBVH*> *bv) {boundingVolumes = bv;}

	// Intersecting rays
	bool intersectRaySimple(Ray &ray, Intersection *intersection, float maxT);
	bool intersectRay(Ray &ray, Intersection *intersection, float maxT);
	
	// Initialization
	void getPartialBoundingBox(BoundingBox &BB, int start, int end);
	void sortObjects(int axis, int start, int end);
	void initBVHnode(int nodeNum, int start, int end, BoundingBox &bb);
	void determineObjectsPerNode(void);
	void initHierarchy(int obsPerLeaf);

	// Other
	void calculateBounds(void); 
	BoundingBox &getBounds(void) {
		return bounds;
	}
	int getNumObjects(void) {
		if (triangles) return (int) triangles->size();
		else return (int) boundingVolumes->size();
	}
};

#endif
