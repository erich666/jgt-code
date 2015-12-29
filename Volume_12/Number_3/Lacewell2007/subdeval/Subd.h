/* Subd.h - Brent Burley, Feb 2005
   Catmull-Clark subdivision implementation.
   Modified by Dylan Lacewell:
   Added limit surface evaluation at any (faceid, u, v) location.
*/

#ifndef subd_h
#define subd_h

#include <string>

class SubdInternal;

class Subd {
 public:
    Subd(int nverts, const float* verts, int nfaces, const int* nvertsPerFace,
	 const int* faceverts);
    virtual ~Subd();
    void subdivide(int levels=1);
    bool eval(int faceid, double u, double v, double* p, double* dPdU, double* dPdV);
    int nverts();
    int nfaces();
    int nfaceverts();
    const float* verts();
    const int* nvertsPerFace();
    const int* faceverts();
    const float* normals();
    const float* limitverts();
 private:
    SubdInternal* impl;
};
#endif

