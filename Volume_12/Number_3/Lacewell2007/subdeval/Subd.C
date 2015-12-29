/* Subd.C - Brent Burley, Feb 2005
   Catmull-Clark subdivision implementation.
   Modified by Dylan Lacewell:
   Added limit surface evaluation at any (faceid, u, v) location.
   */

#include <algorithm>
#include <map>
#include <numeric>
#include <vector>
#include <math.h>
#include "Subd.h"
#include "SubdEigenData.h"
#include "IntPairMap.h"


namespace {
    struct Vec3 {
	float x,y,z;
	Vec3() : x(0), y(0), z(0) {}
	Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

	Vec3 operator+ (const Vec3& c) const {return Vec3(x+c.x,y+c.y,z+c.z);}
	Vec3 operator- (const Vec3& c) const {return Vec3(x-c.x,y-c.y,z-c.z);}
	Vec3 operator* (float c) const	{ return Vec3(x*c, y*c, z*c); }
	Vec3 operator/ (float c) const	{ return *this * (1/c); }
	Vec3& operator+= (const Vec3& c) { return *this = *this + c; }
	Vec3& operator-= (const Vec3& c) { return *this = *this - c; }
	Vec3& operator*= (float c) { return *this = *this * c; }
	Vec3& operator/= (float c) { return *this = *this * (1/c); }
	friend Vec3 operator* (float c, const Vec3& v) { return v * c; }
	Vec3 cross(const Vec3& c) const { return Vec3(y*c.z - z*c.y,
		z*c.x - x*c.z,
		x*c.y - y*c.x); }
	double length() const { return sqrt(x*x+y*y+z*z); }
	void normalize() { *this = this->normalized(); }
	Vec3 normalized() const {
	    double l = length();
	    return l > 0 ? *this / length() : Vec3(0,0,0);
	}
    };
            
}


class SubdInternal {
    public:
	SubdInternal(int nverts, const float* verts,
		int nfaces, const int* nvertsPerFace, const int* faceverts);
	bool eval(int faceid, double u, double v, Vec3& p, Vec3& dPdU, Vec3& dPdV);
	void subdivide(int levels);
	int nverts()		{ return _verts.size(); }
	int nfaces()		{ return _nvertsPerFace.size(); }
	int nfaceverts()	{ return _faceverts.size(); }
	Vec3* verts()		{ return &_verts[0]; }
	int* nvertsPerFace()	{ return &_nvertsPerFace[0]; }
	int* faceverts()	{ return &_faceverts[0]; }
	Vec3* normals();
	Vec3* limitverts();

    private:
	template<class vec>
	    struct Vertex {
		Vertex() : n(0), boundary(0) {}
		vec esum, fsum;  // sum of surrounding edge/face points
		unsigned char n; // valence (num edges)
		bool boundary;
		void addFacePoint(const vec& f) { fsum += f; }

		// jdl note: valences are updated in addEdge()
		void addEdgePoint(const vec& e)	{ esum += e; /*n++;*/ }
		void computeSubd(vec& v);
		void computeLimit(vec& v);
	    };
	typedef Vertex<Vec3> Vertex3;

	struct Edge {
	    int facea, faceb;	// adjacent faces
	    int edgeindexa;     // edge index in first face, in range [0, _nvertsPerFace[facea]]
	    int edgeindexb;     // edge index in second face, in range [0, _nvertsPerFace[faceb]]
	    int v0, v1;		// vertex id's
	    bool boundary;	// true if edge is a boundary
	};

	// Limit surface evaluation data structures start here
	struct EigenStruct {
	    int      eigenval;      	// eigen values [K]
	    int      inverseEigenVecs;  // inverse of eigen vectors [K][K]
	    int      coeffs[3];   	// coeffs of the splines [3][K][16]
	};

	struct BoundaryEigenStruct {
	    int eigenval;
	    int U0;
	    int W1_0;
	    int W1;
	    std::vector<int> U1;
	    std::vector<int> coeffs[3];
	};

	struct FaceControlPointCacheEntry {
	    int boundaryEdge;
	    std::vector<Vec3> controls;
	};

	enum CacheEntryType {kCellRegular, kCellEV, kCellBoundaryEV, kCellNone};

	struct CellControlPointCacheEntry {
	    CacheEntryType type;
	    int valence;
	    int localFaceIndex;  // these two fields are for boundary EVs only
	    bool flip;           //
	    std::vector<Vec3> controls;
	};

	typedef std::map< int, FaceControlPointCacheEntry > FaceControlPointCache;
	typedef std::map< std::pair<int, int>, CellControlPointCacheEntry > CellControlPointCache;

	static void computeWeights(int valence, double* w)
	{
	    // compute weight values used for subdividing w/ a given valence
	    if (valence < 1) return;
	    double n = valence;
	    w[0] = (n-2)/n;		// subd vert weight
	    w[1] = 1/(n*n);		// subd neighbor vert & face point weight
	    w[2] = (n-1)/(n+5);	// limit vert weight
	    w[3] = 2/(n*(n+5));	// limit neighbor vert weight
	    w[4] = 4/(n*(n+5));	// limit face point weight
	}

	static double* getWeights(int valence)
	{
	    static const int MaxValence = 20;
	    static double weights[MaxValence+1][5];
	    static bool initialized = 0;
	    if (!initialized) {
		// precompute weights up to MaxValence
		initialized = 1;
		for (int valence = 1; valence <= MaxValence; valence++)
		    computeWeights(valence, weights[valence]);
	    }
	    if (valence <= 1) return weights[0];
	    // use cached weight if possible
	    if (valence <= MaxValence) return weights[valence];
	    // beyond MaxValence: compute each time
	    // use slot 1 which is otherwise unused (slot 0 is used for errors)
	    computeWeights(valence, weights[1]); 
	    return weights[1];
	}

	void clearResult();
	bool buildMesh(bool checkQuads=false);
	void clearMesh();
	void computeLimit();
	void buildEdges();
	int addEdge(int faceid, int edgeindex, int v0, int v1);
	void markBoundaryVerts();
	void computeFacePoints(bool subdivide);
	void computeEdgePoints(bool subdivide);
	void computeSubdVerts();
	void computeLimitVerts();
	void subdivideFaces();
	void computeNormals();

	int facemod(int faceid, int x)
	{
	    return (x + 4) % 4;
	}

	void rotateFaceUvs(int cell, double u, double v, double& uprime, double& vprime);
	void rotateTangents(int cell, Vec3& dPdU, Vec3& dPdV);
	int getAdjacentFace(int faceid, int edgeindex, int& faceid1, int& edgeindex1);
	void getEdgeVertIds(int faceid, int edgeindex, int& v0, int& v1);
	void bsplineCoeffs(double u, double bu[4], double dBdU[4]);
	void evalBspline(const std::vector<Vec3>& controls, double u, double v, Vec3& p, Vec3& dPdU, Vec3& dPdV);
	void evalEigenBasis(const std::vector<Vec3>& controls, int valence, double u, double v, 
		Vec3& p, Vec3& dPdU, Vec3& dPdV);
	void evalBoundaryEigenBasis(const std::vector<Vec3>& controls, int valence, int face, double u, double v, 
		Vec3& p, Vec3& dPdU, Vec3& dPdV);
	void projectControlPoints(int valence, const std::vector<Vec3>& controls, std::vector<Vec3>& Cp);
	void projectBoundaryControlPoints(int valence, int face, const std::vector<Vec3>& controls, std::vector<Vec3>& Cp);
	void computeFacePoint(int faceid, Vec3& facepoint);
	Vec3 computeFaceAndEdgePoints(int faceid, int vindex, 
		std::vector<int>& edgepointIds, std::vector<int>& facepointIds, int& phase, bool startAtBoundary);
	Vec3 computeFaceAndEdgePoints(int faceid, int vindex, 
		std::vector<int>& edgepointIds, std::vector<int>& facepointIds)
	{
	    int phase;
	    return computeFaceAndEdgePoints(faceid, vindex, edgepointIds, facepointIds, 
		    phase, /*startAtBoundary*/ false);
	}
	void buildBoundaryEigenIndices(int maxvalence = SUBD_MAX_BOUNDARY_VALENCE);

#define MAX_FACE_CONTROL_CACHE 500
#define MAX_CELL_CONTROL_CACHE 100	  

	// Note: if the cache gets too large, we do the simple thing: clear it and start fresh.  
	void cacheFaceControls(int faceid, int boundaryEdge, const std::vector<Vec3>& controls)
	{
	    if (_faceControlCache.size() >= MAX_FACE_CONTROL_CACHE)
		_faceControlCache.clear();
	    FaceControlPointCacheEntry& entry = _faceControlCache[faceid];
	    entry.boundaryEdge = boundaryEdge;
	    entry.controls = controls;
	}

	void cacheCellControls(int faceid, int cell, CacheEntryType type, int valence, 
		int localFaceIndex, bool flip, const std::vector<Vec3>& controls)
	{
	    if (_cellControlCache.size() >= MAX_CELL_CONTROL_CACHE)
		_cellControlCache.clear();
	    CellControlPointCacheEntry& entry = _cellControlCache[std::pair<int,int>(faceid, cell)];
	    entry.type = type;
	    entry.valence = valence;
	    entry.localFaceIndex = localFaceIndex;
	    entry.flip = flip;
	    entry.controls = controls;
	}

	void cacheCellControls(int faceid, int cell, CacheEntryType type, int valence, 
		const std::vector<Vec3>& controls)
	{
	    cacheCellControls(faceid, cell, type, valence, /*localface*/ 0, /*flip*/ false, controls);
	}
	
	
	std::vector<int> _faceVertIdOffsets;
	static EigenStruct _eigenIndices[48];	
	static EigenStruct _cornerEigenIndices[1];
	static BoundaryEigenStruct _boundaryEigenIndices[SUBD_MAX_BOUNDARY_VALENCE];

	// caches used during evaluation
	std::map<int, Vec3> _facepoints;  // faceid --> facepoint
	std::map<int, Vec3> _edgepoints;  // edgeid --> edgepoint
	FaceControlPointCache _faceControlCache;  // faceid --> vector of control points
	CellControlPointCache _cellControlCache;  // (faceid,cell) --> vector of control points

	// surface definition
	std::vector<Vec3> _verts;	     // list of verts
	std::vector<int> _nvertsPerFace; // list of nverts per face
	std::vector<int> _faceverts;     // packed face vert ids

	// subdivision mesh (intermediate) data
	int _faceVertOffset;	     // index of first new face vert
	int _edgeVertOffset;	     // index of first new edge point
	std::vector<Vertex3> _vertinfo;  // temp vertex data
	std::vector<Edge> _edges;	     // temp edge data
	std::vector<int> _faceedges;     // edge ids per face vert
	IntPairMap _edgemap;	     // map from v1,v2 to edge id

	// result data
	std::vector<Vec3> _normals;	     // normals per vertex
	std::vector<Vec3> _limitverts;   // verts projected to limit surface
};


Subd::Subd(int nverts, const float* verts,
	int nfaces, const int* nvertsPerFace, const int* faceverts)
: impl(new SubdInternal(nverts, verts, nfaces, nvertsPerFace, faceverts)) 
{}
Subd::~Subd() { delete impl; }
void Subd::subdivide(int levels)       { impl->subdivide(levels); }
bool Subd::eval(int faceid, double u, double v, double * p, double * dPdU, double * dPdV)
{ 
    Vec3 pvec(p[0], p[1], p[2]);
    Vec3 dPdUvec(dPdU[0], dPdU[1], dPdU[2]);
    Vec3 dPdVvec(dPdV[0], dPdV[1], dPdV[2]);
    
    bool ret = impl->eval(faceid, u, v, pvec, dPdUvec, dPdVvec); 

    p[0]=pvec.x; p[1]=pvec.y; p[2]=pvec.z;
    dPdU[0]=dPdUvec.x; dPdU[1]=dPdUvec.y; dPdU[2]=dPdUvec.z; 
    dPdV[0]=dPdVvec.x; dPdV[1]=dPdVvec.y; dPdV[2]=dPdVvec.z;

    return ret;
}
int Subd::nverts()                     { return impl->nverts(); }
int Subd::nfaces()                     { return impl->nfaces(); }
int Subd::nfaceverts()                 { return impl->nfaceverts(); }
const float* Subd::verts()             { return (float*)impl->verts(); }
const int* Subd::nvertsPerFace()       { return impl->nvertsPerFace(); }
const int* Subd::faceverts()           { return impl->faceverts(); }
const float* Subd::normals()           { return (float*)impl->normals(); }
const float* Subd::limitverts()        { return (float*)impl->limitverts(); }


SubdInternal::SubdInternal(int nverts, const float* verts,
	int nfaces, const int* nvertsPerFace, const int* faceverts)
{
    _verts.assign((Vec3*)verts, ((Vec3*)verts)+nverts);
    _nvertsPerFace.assign(nvertsPerFace, nvertsPerFace + nfaces);
    int nfaceverts = std::accumulate(_nvertsPerFace.begin(),
	    _nvertsPerFace.end(), 0);
    _faceverts.assign(faceverts, faceverts + nfaceverts);

    _faceVertOffset = _edgeVertOffset = 0;

    _faceVertIdOffsets.resize(_nvertsPerFace.size());
    _faceVertIdOffsets[0] = 0;
    for (int i = 0; i < _nvertsPerFace.size()-1; i++) {
	_faceVertIdOffsets[i+1] = _faceVertIdOffsets[i] + _nvertsPerFace[i]; 
    }
    
}


void SubdInternal::clearResult()
{
    _normals.clear();
    _limitverts.clear();
}


void SubdInternal::subdivide(int levels)
{
    while (levels-- > 0) {
	buildMesh();

	// make room for new face points and edge points in vertex list
	_faceVertOffset = nverts();
	_edgeVertOffset = _faceVertOffset + nfaces();
	_verts.resize(_edgeVertOffset + _edges.size());

	// compute subd points
	computeSubdVerts();

	// divide each face into set of new quad faces
	subdivideFaces();

	// cleanup
	clearMesh();
    }

    // clear result data (which are no longer valid)
    clearResult();
}




Vec3* SubdInternal::normals()
{
    if (_normals.empty()) computeNormals();
    return &_normals[0];
}


Vec3* SubdInternal::limitverts()
{
    if (_limitverts.empty()) computeLimit();
    return &_limitverts[0];
}


void SubdInternal::computeLimit()
{
    // build mesh structure
    buildMesh();

    // compute limit points
    computeLimitVerts();

    // cleanup
    clearMesh();
}


bool SubdInternal::buildMesh(bool checkQuads)
{
    // lazy build
    if (_edges.empty()) {
	
	if (checkQuads) 
	    if (4*nfaces() != nfaceverts())
		return false;

	// alloc mesh data
	_vertinfo.resize(nverts());
	_edges.reserve(nverts()*2);
	_faceedges.resize(nfaceverts());
	_edgemap.reserve(nverts()*4);

	// build mesh structures
	buildEdges();
	markBoundaryVerts();
    } 

    return true;
}


void SubdInternal::clearMesh()
{
    _vertinfo.clear();
    _faceedges.clear();
    _edges.clear();
    _edgemap.clear();

    _facepoints.clear();
    _edgepoints.clear();
    _faceControlCache.clear();
    _cellControlCache.clear();
}


void SubdInternal::buildEdges()
{
    const int* vert = &_faceverts[0];
    int* e = &_faceedges[0];

    // build list of edges from face verts
    for (int face = 0; face < nfaces(); face++) {
	int nverts = _nvertsPerFace[face];
	for (int i = 0; i < nverts; i++) {
	    int i2 = (i+1)%nverts;
	    *e++ = addEdge(face, i, vert[i], vert[i2]);
	}
	vert += nverts;
    }
}

int SubdInternal::addEdge(int faceid, int edgeindex, int v0, int v1)
{
    // put verts in canonical order for lookup
    if (v0 > v1) { std::swap(v0, v1); }

    // look for existing edge
    int& id = _edgemap.find(v0, v1);
    if (id < 0) {
	// edge not found, create a new one and set first face point
	id = _edges.size();
	_edges.resize(_edges.size()+1);
	Edge& e = _edges[id];
	e.facea = faceid;
	e.faceb = -1;
	e.edgeindexa = edgeindex;
	e.edgeindexb = -1;
	e.v0 = v0;
	e.v1 = v1;
	e.boundary = 1;   // assume boundary for now
	// update vertex valences
	_vertinfo[v0].n++;
	_vertinfo[v1].n++;
    }
    else {
	// found edge, add second face and update flags
	Edge& e = _edges[id];
	e.faceb = faceid;
	e.edgeindexb = edgeindex;
	e.boundary = 0;
    }
    return id;
}



void SubdInternal::markBoundaryVerts()
{
    // propagate edge boundary flags to vertices
    std::vector<Edge>::iterator iter;
    for (iter = _edges.begin(); iter != _edges.end(); iter++) {
	Edge& e = *iter;

	// if edge is a boundary, mark verts as boundary
	if (e.boundary) {
	    _vertinfo[e.v0].boundary = 1;
	    _vertinfo[e.v1].boundary = 1;
	}
    }
}



void SubdInternal::computeFacePoints(bool subdivide)
{
    const int* vertid = &_faceverts[0];
    for (int i = 0; i < nfaces(); i++) {
	// compute new face point (avg of face verts)
	int nverts = _nvertsPerFace[i];
	Vec3 p;
	for (int vi = 0; vi < nverts; vi++) {
	    p += _verts[vertid[vi]];
	}
	p /= nverts;

	// add p to vert face point sums
	for (int vi = 0; vi < nverts; vi++) {
	    _vertinfo[*vertid++].addFacePoint(p);
	}

	if (subdivide) {
	    // store new face point
	    _verts[_faceVertOffset + i] = p;
	}
    }
}


void SubdInternal::computeEdgePoints(bool subdivide)
{
    for (int i = 0; i < _edges.size(); i++) {
	Edge& e = _edges[i];

	if (subdivide) {
	    // compute new edge point based on verts and face points
	    Vec3 p;
	    if (e.boundary) {
		p = 0.5 * (_verts[e.v0] + _verts[e.v1]);
	    }
	    else {
		p = 0.25 * (_verts[e.v0] + _verts[e.v1] +
			_verts[_faceVertOffset + e.facea] +
			_verts[_faceVertOffset + e.faceb]);
	    }
	    _verts[_edgeVertOffset + i] = p;
	}

	/* distribute verts to neighbor verts along edges
Note: restrict to boundary edges for verts on boundary.
I.e. if edge is on boundary or if target vert is not on boundary,
then it's ok to add the edge vert to the target vert. */
	Vertex3& v0 = _vertinfo[e.v0];
	Vertex3& v1 = _vertinfo[e.v1];
	if (e.boundary || !v0.boundary) v0.addEdgePoint(_verts[e.v1]);
	if (e.boundary || !v1.boundary) v1.addEdgePoint(_verts[e.v0]);
    }
}


void SubdInternal::computeSubdVerts()
{
    
    computeFacePoints(/*subdivide*/ true);
    computeEdgePoints(/*subdivide*/ true);
    
    // verts are subdivided in place
    for (int i = 0, n = _faceVertOffset; i < n; i++)
	_vertinfo[i].computeSubd(_verts[i]);
}


void SubdInternal::computeLimitVerts()
{
    computeFacePoints(/*subdivide*/ false);
    computeEdgePoints(/*subdivide*/ false);

    // copy verts to limit verts
    _limitverts = _verts;

    // adjust limit verts based on edge/face points
    for (int i = 0, n = nverts(); i < n; i++)
	_vertinfo[i].computeLimit(_limitverts[i]);
}


    template<class vec>
void SubdInternal::Vertex<vec>::computeSubd(vec& v)
{
    // move vertex to it's position after 1 subdivision
    if (boundary) {
	v = (6*v + esum) * (1.0/8);
    }
    else { // interior
	double* w = SubdInternal::getWeights(n);
	v = w[0]*v + w[1]*(esum + fsum);
    }
}


    template<class vec>
void SubdInternal::Vertex<vec>::computeLimit(vec& v)
{
    // move vertex to it's limit position
    if (boundary) {
	v = (4*v + esum) * (1.0/6);
    }
    else { // interior
	double* w = SubdInternal::getWeights(n);
	v = w[2]*v + w[3]*esum + w[4]*fsum;
    }
}


void SubdInternal::subdivideFaces()
{
    // extract a quad for every vert of every face
    int nnewfaces = nfaceverts();
    std::vector<int> newfaceverts(nnewfaces*4);
    int* vertid = &_faceverts[0];
    int* e = &_faceedges[0];
    int* newvertid = &newfaceverts[0];

    for (int f = 0; f < nfaces(); f++)
    {
	int nverts = _nvertsPerFace[f];
	int prevedge = nverts-1;
	// make one quad for each vert: [vert, edge pt, face pt, prev edge pt]
	for (int i = 0; i < nverts; i++) {
	    // find adjacent edges
	    int e1id = e[i];
	    int e2id = e[prevedge];
	    Edge& e1 = _edges[e1id];
	    Edge& e2 = _edges[e2id];

	    // copy vert ids
	    *newvertid++ = *vertid++;
	    *newvertid++ = _edgeVertOffset + e1id;
	    *newvertid++ = _faceVertOffset + f;
	    *newvertid++ = _edgeVertOffset + e2id;
	    
	    prevedge = i;	    
	}
	e += nverts;
    }
    std::swap(_faceverts, newfaceverts);

    // set every face to be a quad (4 verts)
    _nvertsPerFace.resize(nnewfaces);
    std::fill(_nvertsPerFace.begin(), _nvertsPerFace.end(), 4);
}


void SubdInternal::computeNormals()
{
    _normals.clear();
    _normals.resize(nverts());

    // for each face
    Vec3* lverts = limitverts();
    const int* vert = &_faceverts[0];
    for (int f = 0; f < nfaces(); f++)
    {
	// find face normal as avg of face vert normals
	Vec3 faceNormal;
	int nverts = _nvertsPerFace[f];
	int pv = vert[nverts-1];
	for (int i = 0; i < nverts; i++) {
	    // find normal from adjacent verts
	    int v = vert[i];
	    int nv = vert[(i+1)%nverts];
	    Vec3& v0 = lverts[pv];
	    Vec3& v1 = lverts[v];
	    Vec3& v2 = lverts[nv];
	    faceNormal += (v2-v1).cross(v0-v1).normalized();
	    pv = v;
	    v++;
	}
	faceNormal.normalize();

	// distribute face normal to surrounding verts
	for (int i = 0; i < nverts; i++) {
	    // find normal from adjacent verts
	    int v = vert[i];
	    _normals[v] += faceNormal;
	}

	vert += nverts;
    }
    for (int i = 0; i < _normals.size(); i++)
	_normals[i].normalize();
}


// Limit surface valuation code starts here


// Jump across an edge, returning the adjacent face and edge index relative to that face
int SubdInternal::getAdjacentFace(int faceid, int edgeindex, int& faceid1, int& edgeindex1)
{
    edgeindex = facemod(faceid, edgeindex);
    int fvertid = _faceVertIdOffsets[faceid];
    int edgeid = _faceedges[fvertid + edgeindex];
    if (edgeid >= 0) {
	const Edge& edge = _edges[edgeid];
	if (edge.facea == faceid) {
	    faceid1 = edge.faceb;
	    edgeindex1 = edge.edgeindexb;

	} else {
	    faceid1 = edge.facea;
	    edgeindex1 = edge.edgeindexa;
	}
    }
    return edgeid;
}


// Given a local edge index (modulo 4), return the two edge vertex indices in
// the correct winding order.  
void SubdInternal::getEdgeVertIds(int faceid, int edgeindex, int& v0, int& v1)
{
    edgeindex = facemod(faceid, edgeindex);

    int fvertid = _faceVertIdOffsets[faceid];
    int nverts = _nvertsPerFace[faceid];
    v0 = _faceverts[fvertid + edgeindex];

    if (++edgeindex == nverts) edgeindex = 0;
    v1 = _faceverts[fvertid + edgeindex];
}


void SubdInternal::bsplineCoeffs(double u, double bu[4], double dBdU[4])
{
    static double c = 1.0 / 6.0;
    static double c4 = 4.0 / 6.0;
    double u2 = u*u;
    double u3 = u2*u;
    bu[0] = c + (0.5)*(u2 - u) - c*u3;
    bu[1] = c4 - u2 + (0.5)*u3;
    bu[2] = c + (0.5)*(u + u2 - u3);
    bu[3] = c*u3; 

    dBdU[0] = -0.5 + u - 0.5*u2;
    dBdU[1] = -2*u + 1.5*u2;
    dBdU[2] = 0.5 + u - 1.5*u2;
    dBdU[3] = 0.5*u2;
}


void SubdInternal::evalBspline(const std::vector<Vec3>& controls, double u, double v, Vec3& p, Vec3& dPdU, Vec3& dPdV)
{
    double bu[4], bv[4], dBdU[4], dBdV[4];

    bsplineCoeffs(u, bu, dBdU);
    bsplineCoeffs(v, bv, dBdV);

    Vec3 row0 = controls[0] * bu[0] + controls[1] * bu[1] + controls[2] * bu[2] + controls[3] * bu[3];
    Vec3 row1 = controls[4] * bu[0] + controls[5] * bu[1] + controls[6] * bu[2] + controls[7] * bu[3];
    Vec3 row2 = controls[8] * bu[0] + controls[9] * bu[1] + controls[10] * bu[2] + controls[11] * bu[3];
    Vec3 row3 = controls[12] * bu[0] + controls[13] * bu[1] + controls[14] * bu[2] + controls[15] * bu[3];

    p = row0 * bv[0] + row1 * bv[1] + row2 * bv[2] + row3 * bv[3];

    dPdU = (controls[0] * dBdU[0] + controls[1] * dBdU[1] + controls[2] * dBdU[2] + controls[3] * dBdU[3]) * bv[0] +
	(controls[4] * dBdU[0] + controls[5] * dBdU[1] + controls[6] * dBdU[2] + controls[7] * dBdU[3]) * bv[1] +
	(controls[8] * dBdU[0] + controls[9] * dBdU[1] + controls[10] * dBdU[2] + controls[11] * dBdU[3]) * bv[2] +
	(controls[12] * dBdU[0] + controls[13] * dBdU[1] + controls[14] * dBdU[2] + controls[15] * dBdU[3]) * bv[3];

    dPdV = row0 * dBdV[0] + row1 * dBdV[1] + row2 * dBdV[2] + row3 * dBdV[3];
}


// This is straight from Stam 98.  
// Cp are the projected control points.
// Important Note: (u,v) are also ordered according to Stam, with the origin at the EV
void SubdInternal::evalEigenBasis(const std::vector<Vec3>& Cp, int valence, double u, double v, 
	Vec3& p, Vec3& dPdU, Vec3& dPdV)
{
    double bu[4], bv[4], dBdU[4], dBdV[4];

    if (u <= 1e-6) u = 1e-6;
    if (v <= 1e-6) v = 1e-6;

    int n = (int)floor(std::min( -log2(u),-log2(v) ))+1;
    //float pow2 = pow(2.0,n);
    int pow2 = 1 << n;
    u *= pow2; v *= pow2;

    int k;
    if (v < 1.0) {
	k=0; u=u-1; v=v;
    }
    else if (u < 1.0) {
	k=2; u=u; v=v-1;
    }
    else {
	k=1; u=u-1; v=v-1;
    }

    bsplineCoeffs(u, bu, dBdU);
    bsplineCoeffs(v, bv, dBdV);

    //assert(valence >= 3);
    int index = valence-3;
    const double * eigenvals = &eigendata[_eigenIndices[index].eigenval];
    const double * coeffs = &eigendata[_eigenIndices[index].coeffs[k]];

    int K = 2*valence + 8;
    p = dPdU = dPdV = Vec3(0.0, 0.0, 0.0);
    for (const Vec3* cp = &Cp[0], * cpend = cp + K; cp != cpend; cp++) {
	Vec3 r = pow(*eigenvals++, n-1) * *cp;

	double row0 = coeffs[0]*bu[0] + coeffs[1]*bu[1] + coeffs[2]*bu[2] + coeffs[3]*bu[3];
	double row1 = coeffs[4]*bu[0] + coeffs[5]*bu[1] + coeffs[6]*bu[2] + coeffs[7]*bu[3];
	double row2 = coeffs[8]*bu[0] + coeffs[9]*bu[1] + coeffs[10]*bu[2] + coeffs[11]*bu[3];
	double row3 = coeffs[12]*bu[0] + coeffs[13]*bu[1] + coeffs[14]*bu[2] + coeffs[15]*bu[3];

	p += (row0*bv[0] + row1*bv[1] + row2*bv[2] + row3*bv[3])*r;

	dPdU += ((coeffs[0]*dBdU[0] + coeffs[1]*dBdU[1] + coeffs[2]*dBdU[2] + coeffs[3]*dBdU[3])*bv[0] +
		(coeffs[4]*dBdU[0] + coeffs[5]*dBdU[1] + coeffs[6]*dBdU[2] + coeffs[7]*dBdU[3])*bv[1] +
		(coeffs[8]*dBdU[0] + coeffs[9]*dBdU[1] + coeffs[10]*dBdU[2] + coeffs[11]*dBdU[3])*bv[2] +
		(coeffs[12]*dBdU[0] + coeffs[13]*dBdU[1] + coeffs[14]*dBdU[2] + coeffs[15]*dBdU[3])*bv[3])*r;

	dPdV += (row0*dBdV[0] + row1*dBdV[1] + row2*dBdV[2] + row3*dBdV[3])*r;

	coeffs += 16;

    }
    dPdU = pow2 * dPdU;
    dPdV = pow2 * dPdV;
}



void SubdInternal::projectControlPoints(int valence, const std::vector<Vec3>& controls, std::vector<Vec3>& Cp)
{
    Cp.resize(controls.size());

    int index = valence-3;      int K = 2*valence+8;
    double * iV = &eigendata[_eigenIndices[index].inverseEigenVecs];
    for (Vec3* cp = &Cp[0], * cpend = cp + K; cp != cpend; cp++) {
	*cp = *iV++ * controls[0];
	for (const Vec3* c = &controls[1], * cend = c + K - 1; c != cend; c++) {
	    *cp += *iV++ * *c;
	}
    } 
}

// An extension of Stam 98 to the boundary case.
// Cp are the projected control points.
// Important Note: (u,v) are also ordered according to Stam, with the origin at the corner EV
void SubdInternal::evalBoundaryEigenBasis(const std::vector<Vec3>& Cp, int valence, int face, double u, double v, 
	Vec3& p, Vec3& dPdU, Vec3& dPdV)
{
    int K = Cp.size();
    double bu[4], bv[4], dBdU[4], dBdV[4];

    if (u <= 1e-6) u = 1e-6;
    if (v <= 1e-6) v = 1e-6;

    int n = (int)floor(std::min( -log2(u),-log2(v) ))+1;
    //float pow2 = pow(2.0,n);
    int pow2 = 1 << n;
    u *= pow2; v *= pow2;

    int k;
    if (v < 1.0) {
	k=0; u=u-1; v=v;
    }
    else if (u < 1.0) {
	k=2; u=u; v=v-1;
    }
    else {
	k=1; u=u-1; v=v-1;
    }
    
    bsplineCoeffs(u, bu, dBdU);
    bsplineCoeffs(v, bv, dBdV);
    
    const double * eigenvals = &boundaryEigenData[_boundaryEigenIndices[valence-2].eigenval];
    const double * coeffs = &boundaryEigenData[_boundaryEigenIndices[valence-2].coeffs[k][face]];
    
    p = dPdU = dPdV = Vec3(0.0, 0.0, 0.0);
    double eigenX[K], dXdU[K], dXdV[K];
    double *x = eigenX, *dxdu = dXdU, *dxdv = dXdV;
        
    for (const Vec3* cp = &Cp[0], * cpend = cp + K; cp != cpend; cp++) {
	Vec3 r = pow(*eigenvals++, n-1) * *cp;

	double row0 = coeffs[0]*bu[0] + coeffs[1]*bu[1] + coeffs[2]*bu[2] + coeffs[3]*bu[3];
	double row1 = coeffs[4]*bu[0] + coeffs[5]*bu[1] + coeffs[6]*bu[2] + coeffs[7]*bu[3];
	double row2 = coeffs[8]*bu[0] + coeffs[9]*bu[1] + coeffs[10]*bu[2] + coeffs[11]*bu[3];
	double row3 = coeffs[12]*bu[0] + coeffs[13]*bu[1] + coeffs[14]*bu[2] + coeffs[15]*bu[3];

	(*x) = row0*bv[0] + row1*bv[1] + row2*bv[2] + row3*bv[3];
	p += (*x)*r;

	(*dxdu) = (coeffs[0]*dBdU[0] + coeffs[1]*dBdU[1] + coeffs[2]*dBdU[2] + coeffs[3]*dBdU[3])*bv[0] +
		(coeffs[4]*dBdU[0] + coeffs[5]*dBdU[1] + coeffs[6]*dBdU[2] + coeffs[7]*dBdU[3])*bv[1] +
		(coeffs[8]*dBdU[0] + coeffs[9]*dBdU[1] + coeffs[10]*dBdU[2] + coeffs[11]*dBdU[3])*bv[2] +
		(coeffs[12]*dBdU[0] + coeffs[13]*dBdU[1] + coeffs[14]*dBdU[2] + coeffs[15]*dBdU[3])*bv[3];
	
	dPdU += (*dxdu)*r;

	(*dxdv) = row0*dBdV[0] + row1*dBdV[1] + row2*dBdV[2] + row3*dBdV[3];
	dPdV += (*dxdv)*r;

	coeffs += 16;
	x++; dxdu++; dxdv++;

    }

    // add cross term, since for some valences the eigenvectors do not form a complete basis
    if (n > 1) {
	int m = (valence+1) % 4;
	if (m > 0) {
	    int index;
	    float lambda;
	    if (m == 2) {
		index = (valence-1)/2;
		lambda = 0.5;
	    } else {
		index = valence;
		lambda = 0.25;
	    }	
	    Vec3 r = (((double)(n-1)) * (pow(lambda, n-2))) * Cp[index+1];
	    p += eigenX[index] * r;
	    dPdU += dXdU[index] * r;
	    dPdV += dXdV[index] * r;
	}
    }
        
    dPdU = pow2 * dPdU;
    dPdV = pow2 * dPdV;
}


void SubdInternal::projectBoundaryControlPoints(int valence, int face, const std::vector<Vec3>& controls, std::vector<Vec3>& Cp)
{
    Cp.resize(controls.size());

    int K = controls.size();

    
    // blockwise matrix mult
    int index = valence-2;
    
    const double * U0 = &boundaryEigenData[_boundaryEigenIndices[index].U0];
    const double * U1 = &boundaryEigenData[_boundaryEigenIndices[index].U1[face]];
    const double * W1;
    if (face == 0) 
	W1 = &boundaryEigenData[_boundaryEigenIndices[index].W1_0];
    else
	W1 = &boundaryEigenData[_boundaryEigenIndices[index].W1];
    
    int n = 2*valence;
    // U0 block
    for (Vec3* cp = &Cp[0], *cpend = cp + n; cp != cpend; cp++) {
	*cp = *U0++ * controls[0];
	for (const Vec3* c = &controls[1], *cend = c + n - 1; c != cend; c++) {
	    *cp += *U0++ * *c;
	}
    }

    // U1 block
    for (Vec3* cp = &Cp[n], *cpend = cp + K - n; cp != cpend; cp++) {
	*cp = *U1++ * controls[0];
	for (const Vec3* c = &controls[1], *cend = c + n - 1; c != cend; c++) {
	   *cp += *U1++ * *c;
	}
    }    

    // W1 block
    for (Vec3* cp = &Cp[n], *cpend = cp + K - n; cp != cpend; cp++) {
	*cp += *W1++ * controls[n];
	for (const Vec3* c = &controls[n+1], *cend = c+K-n-1; c != cend; c++) {
	    *cp += *W1++ * *c;
	}
    }
}


void SubdInternal::computeFacePoint(int faceid, Vec3& facepoint)
{
    // check the cache first
    if (_facepoints.find(faceid) != _facepoints.end()) {
	facepoint = _facepoints[faceid];
	return;
    }

    int facevertid = _faceVertIdOffsets[faceid];
    int nv = _nvertsPerFace[faceid];
    for (int j = 0; j < nv; j++)
	facepoint += _verts[_faceverts[facevertid + j]];
    facepoint /= nv;
    _facepoints[faceid] = facepoint;
}


// Compute all face/edge points about a vertex, cache them, and return the new position of the vertex
// Also return indices into the caches for the points we create
// The traversal is ccw starting at edge [vindex, vindex+1] and continuing until we wrap around.  
// We might cross a boundary during the traversal.  If so, we return an optional argument, the "phase" 
// of the face, which is the number of faces traversed cw before hitting the boundary edge. 
//
// Design note: this function is only needed for evaluating near an EV.  As an alternative we could 
// subdivide the mesh everywhere using ::subdivide(), but that would be wasteful; subdivision is not 
// needed to evaluate regular patches.
Vec3 SubdInternal::computeFaceAndEdgePoints(int faceid, int vindex, 
	std::vector<int>& faceIds, std::vector<int>& edgeIds, int& phase, bool startAtBoundary)
{
    // intermediate vectors to allow for reording
    std::vector<int> facepointIds, edgepointIds;

    vindex = facemod(faceid, vindex);

    int fvertid = _faceVertIdOffsets[faceid];
    int vertid = _faceverts[fvertid+vindex];
    int numverts = _nvertsPerFace[faceid];

    Vec3 facesum, edgesum;

    phase = 0;
    int fid=-1, eindex=-1;
    int edgeid = getAdjacentFace(faceid, vindex, fid, eindex);
    int vindex0 = vindex;
    int faceid0 = faceid;
    int v0, v1;
    getEdgeVertIds(faceid, vindex, v0, v1);
    if (_vertinfo[vertid].boundary) {
	// traverse cw to find the boundary edge
	edgeid = getAdjacentFace(faceid, vindex, fid, eindex);
	while (fid >= 0) {
	    faceid0 = fid;
	    vindex0 = eindex+1;
	    getEdgeVertIds(fid, eindex+1, v0, v1);
	    edgeid = getAdjacentFace(fid, eindex+1, fid, eindex);
	    phase++;
	}
	_edgepoints[edgeid] = 0.5 * (_verts[v0] + _verts[v1]);
    } else {
	Vec3 facepoint0;
	computeFacePoint(faceid, facepoint0);
	edgeid = getAdjacentFace(faceid, vindex, fid, eindex);
	//assert(fid >= 0);
	Vec3 facepoint;
	computeFacePoint(fid, facepoint);
	if (_edgepoints.find(edgeid) == _edgepoints.end())
	    _edgepoints[edgeid] = 0.25 * (facepoint + facepoint0 + _verts[v0] + _verts[v1]);
    }
    edgepointIds.push_back(edgeid);

    // traverse ccw until we loop around, or hit a boundary
    getEdgeVertIds(faceid0, vindex0-1, v0, v1);
    edgesum += _verts[v0];
    Vec3 prevfacepoint, facepoint0;
    computeFacePoint(faceid0, facepoint0);
    facepointIds.push_back(faceid0);
    facesum = facepoint0;
    prevfacepoint = facepoint0;
    edgeid = getAdjacentFace(faceid0, vindex0-1, fid, eindex);
    while (fid >= 0 && fid != faceid0) {
	Vec3 facepoint, edgepoint;
	computeFacePoint(fid, facepoint);
	facepointIds.push_back(fid);
	facesum += facepoint;
	if (_edgepoints.find(edgeid) == _edgepoints.end())
	    _edgepoints[edgeid] = 0.25 * (prevfacepoint + facepoint + _verts[v0] + _verts[v1]);
	edgepointIds.push_back(edgeid);
	prevfacepoint = facepoint;
	getEdgeVertIds(fid, eindex-1, v0, v1);
	edgesum += _verts[v0];
	edgeid = getAdjacentFace(fid, eindex-1, fid, eindex);
    }

    if (fid < 0) {
	if (_edgepoints.find(edgeid) == _edgepoints.end())
	    _edgepoints[edgeid] = 0.5 * (_verts[v0] + _verts[v1]);
	edgepointIds.push_back(edgeid);
    }

    // copy face and edge point in order requested by caller
    int index = (startAtBoundary ? 0 : phase );
    int nfacepoints = facepointIds.size();
    faceIds.resize(nfacepoints);
    for (int i = 0; i < nfacepoints; i++) {
	faceIds[i] = facepointIds[index++];
	if (index >= nfacepoints) index = 0;
    }

    index = (startAtBoundary ? 0 : phase );
    int nedgepoints = edgepointIds.size();
    edgeIds.resize(nedgepoints);
    for (int i = 0; i < nedgepoints; i++) { 
	edgeIds[i] = edgepointIds[index++];
	if (index >= nedgepoints) index = 0;
    }
    
    if (fid < 0) {
	return 0.25 * (2*_verts[v1] + _edgepoints[edgeid] + _edgepoints[edgepointIds[0]]);
    } else {
	double * w = SubdInternal::getWeights(_vertinfo[vertid].n);
	return w[0]*_verts[v1] + w[1]*(edgesum + facesum);	
    }

}


// Rotate and scale face uvs to cell uvs
void SubdInternal::rotateFaceUvs(int cell, double u, double v, double& uprime, double& vprime)
{
    switch(cell) {
	case 0:
	    uprime = u; vprime = v;
	    break;
	case 1:
	    uprime = v; vprime = 1-u;
	    break;
	case 2:
	    uprime = 1-u; vprime = 1-v;
	    break;
	case 3:
	    uprime = 1-v; vprime = u;
	    break;
    }
    uprime = 2.0 * uprime;
    vprime = 2.0 * vprime;
}

// Rotate and scale tangents from cell uv space to face uv space
void SubdInternal::rotateTangents(int cell, Vec3& dPdU, Vec3& dPdV)
{
    Vec3 tmp;
    switch (cell) {
	case 0:
	    break;
	case 1:
	    tmp = dPdU;
	    dPdU = -1.0 * dPdV;
	    dPdV = tmp;
	    break;
	case 2:
	    dPdU = -1.0 * dPdU;
	    dPdV = -1.0 * dPdV;
	    break;
	case 3:
	    tmp = dPdU;
	    dPdU = dPdV;
	    dPdV = -1.0 * tmp;
	    break;
    }
    dPdU = 2.0 * dPdU;
    dPdV = 2.0 * dPdV;
}


/*
 * Evaluate the limit surface at a given face, and local (u,v) location.
 *
 * Notes and assumptions:
 * - Assume every face is a quad
 * - Assume all face vert ids have a consistent winding order
 * - For each face, the local (u,v) origin is at vertex 0.  The u axis points toward vertex 1, and
 *   the v axis points toward vertex 3
 * - We use the Stam evaluation near interior EVs of valence 3, 5, and higher.
 * - Boundary EVs with valence >= 3 are not handled
 * - Corners (boundary EVs with valence 2) are evaluated correctly using an extension of Stam's method.
 * 
 */
bool SubdInternal::eval(int faceid, double u, double v, Vec3& p, Vec3& dPdU, Vec3& dPdV)
{
    // Lazy build
    if (!buildMesh(/*checkQuads*/ true)) {
	printf("SubdInternal::eval(): the mesh contains non-quads, aborting\n");
	return false;
    }

    // Check the regular BSpline control point cache
    FaceControlPointCache::const_iterator facecache_it;
    if ( (facecache_it = _faceControlCache.find(faceid)) != _faceControlCache.end()) {
	Vec3 limit;
	if (facecache_it->second.boundaryEdge < 0) {
	    evalBspline(facecache_it->second.controls, u, v, p, dPdU, dPdV);
	} else {
	    double uprime, vprime;
	    rotateFaceUvs(facecache_it->second.boundaryEdge, u, v, uprime, vprime);	
	    uprime *= 0.5; 
	    vprime *= 0.5;
	    evalBspline(facecache_it->second.controls, uprime, vprime, p, dPdU, dPdV);
	    rotateTangents(facecache_it->second.boundaryEdge, dPdU, dPdV);
	    dPdU = 0.5 * dPdU;
	    dPdV = 0.5 * dPdV;
	}
	return true;
    }

    int cell = 0;
    if (u < 0.5) {
	if (v < 0.5) cell = 0;
	else cell = 3;
    } else  {
	if (v < 0.5) cell = 1;
	else cell = 2;
    }

    // Check the cell control point cache
    CellControlPointCache::const_iterator cache_it;
    if ( (cache_it = _cellControlCache.find(std::pair<int,int>(faceid, cell))) != _cellControlCache.end()) {
	Vec3 limit;
	double uprime, vprime;
	rotateFaceUvs(cell, u, v, uprime, vprime);
	if (cache_it->second.type == kCellRegular) {
	    evalBspline(cache_it->second.controls, uprime, vprime, p, dPdU, dPdV);
	} else if (cache_it->second.type == kCellEV) {
	    evalEigenBasis(cache_it->second.controls, cache_it->second.valence, uprime, vprime, p, dPdU, dPdV);	
	} else if (cache_it->second.type == kCellBoundaryEV) {
	    if (cache_it->second.flip)
		evalBoundaryEigenBasis(cache_it->second.controls, cache_it->second.valence, cache_it->second.localFaceIndex,
			vprime, uprime, p, dPdV, dPdU);
	    else
		evalBoundaryEigenBasis(cache_it->second.controls, cache_it->second.valence, cache_it->second.localFaceIndex,
			uprime, vprime, p, dPdU, dPdV);
		
	} else {
	    printf("empty cache entry for faceid %d, cell %d (this should not happen)\n", faceid, cell);
	    return false;
	}
	rotateTangents(cell, dPdU, dPdV);
	return true;
    }	

    //printf("CACHE MISS\n");

    // Collect initial info about the face (# EVs and boundary edges).  
    int boundaryEdge = -1;
    int numBoundaryEdges = 0;

    if (faceid >= nfaces()) {
	printf("Subd::eval() called with faceid %d >= upper bound of %d\n", faceid, nfaces());
	return false;
    }
    
    int ev = -1;	
    int numverts = _nvertsPerFace[faceid];
    if (numverts != 4) {
	printf("Subd::eval(%d, %f, %f) called on a non-quad with %d verts\n", faceid, u, v, numverts);
	return false;
    }
    int fvertid = _faceVertIdOffsets[faceid];
    int valences[numverts];
    bool boundary[numverts];
    for (int i = 0; i < numverts; i++) {
	int vertid = _faceverts[fvertid+i];

	valences[i] = _vertinfo[vertid].n;
	boundary[i] = _vertinfo[vertid].boundary;
	if (!_vertinfo[vertid].boundary && _vertinfo[vertid].n != 4 ||
		_vertinfo[vertid].boundary && _vertinfo[vertid].n != 3) {

	    //printf("found EV %d\n", vertid);
	    ev = i;
	}

	int edgeid = _faceedges[fvertid+i];
	if (_edges[edgeid].boundary) {
	    //printf("found boundary edge %d\n", i);
	    boundaryEdge = i;
	    numBoundaryEdges++;
	}
    }

    if (ev < 0 ) {

	std::vector<Vec3> controls(16);

	if (numBoundaryEdges == 0) {

	    // regular interior case
	    //printf("regular interior case\n");
	    static int vertorder[] = {6, 2, 3, 7, 10, 11, 15, 14, 9, 13, 12, 8, 5, 4, 0, 1};
	    int *orderp = vertorder;
	    for (int i = 0; i < 4; i++) {
		int nextfaceid = -1, nextEdgeIndex = -1;
		getAdjacentFace(faceid, i, nextfaceid, nextEdgeIndex);
		nextEdgeIndex--;
		if (nextEdgeIndex < 0) nextEdgeIndex += 4;
		getAdjacentFace(nextfaceid, nextEdgeIndex, nextfaceid, nextEdgeIndex);
		int fvid = _faceVertIdOffsets[nextfaceid];
		for (int j = 0; j < 4; j++) 
		    controls[*orderp++] = _verts[_faceverts[fvid + (nextEdgeIndex+j)%4]];

	    }
	    evalBspline(controls, u, v, p, dPdU, dPdV);

	    cacheFaceControls(faceid, /*boundaryEdge*/ -1, controls);

	    return true;

	} else if (numBoundaryEdges == 1) {
	    // regular boundary case
	    //printf("regular boundary case\n");

	    static int vertorder[] = {5, 6, 10, 9, 7, 11, 15, 14, 13, 12, 8, 4};
	    const int * orderp = vertorder;
	    for (int i = 0; i < 4; i++) 
		controls[*orderp++] = _verts[_faceverts[fvertid+(boundaryEdge+i)%4]];

	    for (int i = 1; i < 4; i++) {
		int v0, v1;
		int nextfaceid = -1, nextEdgeIndex = -1;
		getAdjacentFace(faceid, boundaryEdge+i, nextfaceid, nextEdgeIndex);
		getEdgeVertIds(nextfaceid, nextEdgeIndex+2, v0, v1);
		controls[*orderp++] = _verts[v0];
		controls[*orderp++] = _verts[v1];
		getAdjacentFace(nextfaceid, nextEdgeIndex+3, nextfaceid, nextEdgeIndex);
		if (nextEdgeIndex >= 0) {
		    getEdgeVertIds(nextfaceid, nextEdgeIndex+1, v0, v1);
		    controls[*orderp++] = _verts[v1];
		}
	    }

	    // extrapolate boundary
	    for (int i = 0; i < 4; i++) 
		controls[i] = 2 * (controls[i+4]) - controls[i+8];

	    double uprime, vprime;
	    rotateFaceUvs(boundaryEdge, u, v, uprime, vprime);	
	    uprime *= 0.5; 
	    vprime *= 0.5;
	    evalBspline(controls, uprime, vprime, p, dPdU, dPdV);
	    rotateTangents(boundaryEdge, dPdU, dPdV);
	    dPdU = 0.5 * dPdU;
	    dPdV = 0.5 * dPdV;

	    cacheFaceControls(faceid, boundaryEdge, controls);

	    return true;
	}

    }

    // If we got to here, then the face needs to be subdivded into cells.
    // Build the cell control points, evaluate the limit, and update the cache

    double uprime, vprime;
    rotateFaceUvs(cell, u, v, uprime, vprime);

    if (boundary[cell] == false) {
	if (valences[cell] == 4) {

	    // The cell is in the interior and not on an EV.
	    // Build face and edge points then do a regular BSpline eval
	    std::vector<Vec3> controls(16);
	    //printf("evaluating regular interior point on a face with an EV\n");
	    std::vector<int> facepointIds, edgepointIds;
	    
	    static int order[] = {5, 6, 10, 9, 8, 4, 0, 1, 2, 7, 11, 3, 15, 13, 14, 12};
	    const int * orderp = order;
	    controls[*orderp++] = computeFaceAndEdgePoints(faceid, cell, facepointIds, edgepointIds);
	    for (int i = 0; i < 4; i++) {
		controls[*orderp++] = _edgepoints[edgepointIds[i]];
		controls[*orderp++] = _facepoints[facepointIds[i]];
	    }

	    controls[*orderp++] = computeFaceAndEdgePoints(faceid, cell+1, facepointIds, edgepointIds);
	    controls[*orderp++] = _edgepoints[edgepointIds[0]];
	    controls[*orderp++] = _edgepoints[edgepointIds[2]];

	    controls[*orderp++] = computeFaceAndEdgePoints(faceid, cell+2, facepointIds, edgepointIds);

	    controls[*orderp++] = computeFaceAndEdgePoints(faceid, cell+3, facepointIds, edgepointIds);
	    controls[*orderp++] = _edgepoints[edgepointIds[1]];
	    controls[*orderp++] = _edgepoints[edgepointIds.back()];

	    evalBspline(controls, uprime, vprime, p, dPdU, dPdV);
	    rotateTangents(cell, dPdU, dPdV);
	    cacheCellControls(faceid, cell, kCellRegular, /*valence*/ 4, controls);

	    return true;
	}
	else {
	    //printf("stam eval\n");
	    // The cell is in the interior and on an EV.
	    // Do the interior Stam eval
	    int n = valences[cell];
	    if (n > SUBD_MAX_INTERIOR_VALENCE) {
		printf("valence %d exceeds max interior valence of %d\n", 
			n, SUBD_MAX_INTERIOR_VALENCE);
		return false;
	    }
	    
	    std::vector<Vec3> controls(2*n+8);

	    // first vertex (EV)	
	    int order[n];   		// this maps my traversal order to Stam order
	    order[0] = 4; order[1] = 2;
	    for (int i = 0; i < n-2; i++) 
		order[i+2] = 2*(n-i);

	    std::vector<int> facepointIds, edgepointIds;
	    controls[0] = computeFaceAndEdgePoints(faceid, cell, facepointIds, edgepointIds);
	    for (int i = 0; i < n; i++) 
		controls[order[i]] = _facepoints[facepointIds[i]];
	    for (int i = 0; i < n; i++)
		controls[(order[i]+1)%(2*n)] = _edgepoints[edgepointIds[i]];

	    Vec3 * controlp = &controls[2*n];

	    // second vertex
	    controlp[3] = computeFaceAndEdgePoints(faceid, cell+1, facepointIds, edgepointIds);
	    controlp[2] = _edgepoints[edgepointIds[0]];
	    controlp[4] = _edgepoints[edgepointIds[2]];

	    // third vertex
	    controlp[1] = computeFaceAndEdgePoints(faceid, cell+2, facepointIds, edgepointIds);

	    // fourth vertex
	    controlp[6] = computeFaceAndEdgePoints(faceid, cell+3, facepointIds, edgepointIds);
	    controlp[5] = _edgepoints[edgepointIds[1]];
	    controlp[7] = _edgepoints[edgepointIds.back()];

	    std::vector<Vec3> Cp(controls.size());
	    projectControlPoints(n, controls, Cp);

	    evalEigenBasis(Cp, n, uprime, vprime, p, dPdU, dPdV);
	    rotateTangents(cell, dPdU, dPdV);
	    cacheCellControls(faceid, cell, kCellEV, n, Cp);

	    return true;
	}
    } else {
	std::vector<int> facepointIds, edgepointIds;
	
	// The cell is on the boundary

	if (valences[cell] == 3) {
	    // the cell is not on an EV so can still do Bspline eval
	    //printf("evaluating regular boundary point on a face with an EV\n");

	    std::vector<Vec3> controls(16);
	    
	    // the traversal order depends on which edge of the cell is on the boundary
	    if (_edges[_faceedges[fvertid+cell]].boundary) {
		//printf("case 1\n");
		static int order[] = {5, 6, 10, 9, 8, 4, 7, 11, 15, 13, 14, 12};
		const int * orderp = order;
		controls[*orderp++] = computeFaceAndEdgePoints(faceid, cell, facepointIds, edgepointIds);
		for (int i = 0; i < 2; i++) {
		    controls[*orderp++] = _edgepoints[edgepointIds[i]];
		    controls[*orderp++] = _facepoints[facepointIds[i]];
		}
		controls[*orderp++] = _edgepoints[edgepointIds.back()];

		controls[*orderp++] = computeFaceAndEdgePoints(faceid, cell+1, facepointIds, edgepointIds);
		controls[*orderp++] = _edgepoints[edgepointIds[0]];

		controls[*orderp++] = computeFaceAndEdgePoints(faceid, cell+2, facepointIds, edgepointIds);

		controls[*orderp++] = computeFaceAndEdgePoints(faceid, cell+3, facepointIds, edgepointIds);
		controls[*orderp++] = _edgepoints[edgepointIds[1]];
		controls[*orderp++] = _edgepoints[edgepointIds.back()];

		// extrapolate boundary
		for (int i = 0; i < 4; i++) 
		    controls[i] = 2 * (controls[i+4]) - controls[i+8];

	    } else {
		//printf("case 2\n");
		static int order[] = {5, 6, 10, 9, 2, 1, 7, 11, 3, 15, 13, 14};
		const int * orderp = order;
		controls[*orderp++] = computeFaceAndEdgePoints(faceid, cell, facepointIds, edgepointIds);
		for (int i = 0; i < 2; i++) {
		    controls[*orderp++] = _edgepoints[edgepointIds[i]];
		    controls[*orderp++] = _facepoints[facepointIds[i]];
		}
		controls[*orderp++] = _edgepoints[edgepointIds.back()];

		controls[*orderp++] = computeFaceAndEdgePoints(faceid, cell+1, facepointIds, edgepointIds);
		controls[*orderp++] = _edgepoints[edgepointIds[0]];
		controls[*orderp++] = _edgepoints[edgepointIds[2]];

		controls[*orderp++] = computeFaceAndEdgePoints(faceid, cell+2, facepointIds, edgepointIds);

		controls[*orderp++] = computeFaceAndEdgePoints(faceid, cell+3, facepointIds, edgepointIds);
		controls[*orderp++] = _edgepoints[edgepointIds[1]];

		// extrapolate boundary
		for (int i = 0; i < 4; i++) { 
		    int j = 4*i;
		    controls[j] = 2 * (controls[j+1]) - controls[j+2];
		}
	    }

	    evalBspline(controls, uprime, vprime, p, dPdU, dPdV);
	    rotateTangents(cell, dPdU, dPdV);
	    cacheCellControls(faceid, cell, kCellRegular, /*valence*/ 3, controls);

	    return true;

	} else if (valences[cell] == 2) {
	    
	    // Convex corner case
	    // Note that this is not a pinned corner; we still use Bspline rules to form the limit
	    // printf("convex corner case\n");
	    
	    if (_boundaryEigenIndices[0].coeffs[0].size() == 0)
		buildBoundaryEigenIndices();

	    std::vector<Vec3> controls(9);
	    
	    static int order[] = {0, 3, 2, 1, 6, 5, 4, 8, 7};
	    const int * orderp = order;
	    controls[*orderp++] = computeFaceAndEdgePoints(faceid, cell, facepointIds, edgepointIds);
	    controls[*orderp++] = _edgepoints[edgepointIds[0]];
	    controls[*orderp++] = _facepoints[facepointIds[0]];
	    controls[*orderp++] = _edgepoints[edgepointIds[1]];

	    controls[*orderp++] = computeFaceAndEdgePoints(faceid, cell+1, facepointIds, edgepointIds);
	    controls[*orderp++] = _edgepoints[edgepointIds[0]];

	    controls[*orderp++] = computeFaceAndEdgePoints(faceid, cell+2, facepointIds, edgepointIds);

	    controls[*orderp++] = computeFaceAndEdgePoints(faceid, cell+3, facepointIds, edgepointIds);
	    controls[*orderp++] = _edgepoints[edgepointIds[1]];
	    
	    std::vector<Vec3> Cp(controls.size());
	    projectBoundaryControlPoints(/*valence*/ 2, /*face*/ 0, controls, Cp);
	    
	    evalBoundaryEigenBasis(Cp, /*valence*/ 2, /*face*/ 0, uprime, vprime, p, dPdU, dPdV);
	    rotateTangents(cell, dPdU, dPdV); 
	    cacheCellControls(faceid, cell, kCellBoundaryEV, /*valence*/ 2, /*localface*/ 0, /*flip*/ false, Cp);
	    
	    return true;

	} else {
	    // Boundary EV case, with valence > 3

	    if (_boundaryEigenIndices[0].coeffs[0].size() == 0)
		buildBoundaryEigenIndices();
	    
	    int n = valences[cell];
	    if (n > SUBD_MAX_BOUNDARY_VALENCE) {
		printf("valence %d exceeds max boundary valence of %d\n", 
			n, SUBD_MAX_BOUNDARY_VALENCE);
		return false;
	    }
	    
	    std::vector<Vec3> controls(2*n+7);
	    std::vector<int> facepointIds, edgepointIds;
	    int phase = -1;
	    controls[0] = computeFaceAndEdgePoints(faceid, cell, facepointIds, edgepointIds, 
		    phase, /*startAtBoundary*/ true);
	    
	    int order[2*n];
	    bool flip = (phase < (n+1)/2 - 1);
	    int face ;  // ccw rotational index about the ev
	    if (flip) {
		for (int i = 0; i < 2*n-1; i++)
		    order[i] = i+1;
		if (phase == 0)
		    controls.resize(2*n+6);
		face = phase;
	    } else {
		for (int i = 0; i < 2*n-1; i++) 
		    order[i] = 2*n-i-1;
		if (phase == n-2)
		    controls.resize(2*n+6);
		face = n - 2 - phase;
	    }
	   
	    int * orderp = order; 
	    for (int i = 0; i < n-1; i++) {
		controls[*orderp++] = _edgepoints[edgepointIds[i]];
		controls[*orderp++] = _facepoints[facepointIds[i]];
	    }
	    controls[*orderp++] = _edgepoints[edgepointIds[n-1]];
	    
	    Vec3 * controlp = &controls[2*n];
	    if (flip) {
		controlp[5] = computeFaceAndEdgePoints(faceid, cell+1, facepointIds, edgepointIds);
		controlp[4] = _edgepoints[edgepointIds[0]];
		if (phase > 0) 
		    controlp[6] = _edgepoints[edgepointIds[2]];
		
		controlp[0] = computeFaceAndEdgePoints(faceid, cell+2, facepointIds, edgepointIds);

		controlp[2] = computeFaceAndEdgePoints(faceid, cell+3, facepointIds, edgepointIds);
		controlp[1] = _edgepoints[edgepointIds[1]];
		controlp[3] = _edgepoints[edgepointIds.back()];	
	    } else {
		controlp[2] = computeFaceAndEdgePoints(faceid, cell+1, facepointIds, edgepointIds);
		controlp[1] = _edgepoints[edgepointIds[0]];
		controlp[3] = _edgepoints[edgepointIds[2]];

		controlp[0] = computeFaceAndEdgePoints(faceid, cell+2, facepointIds, edgepointIds);
		
		controlp[5] = computeFaceAndEdgePoints(faceid, cell+3, facepointIds, edgepointIds);
		controlp[4] = _edgepoints[edgepointIds[1]];
		if (phase < n-2) 
		    controlp[6] = _edgepoints[edgepointIds.back()];
	    }

	    std::vector<Vec3> Cp(controls.size());
	    projectBoundaryControlPoints(n, face, controls, Cp);

	    if (flip) 
		evalBoundaryEigenBasis(Cp, n, face, vprime, uprime, p, dPdV, dPdU);
	    else 
		evalBoundaryEigenBasis(Cp, n, face, uprime, vprime, p, dPdU, dPdV);
	    
	    rotateTangents(cell, dPdU, dPdV);
	    cacheCellControls(faceid, cell, kCellBoundaryEV, n, face, flip, Cp);
	    
	    return true;
	    
	}
    }

    return true;
}


void SubdInternal::buildBoundaryEigenIndices(int maxvalence)
{
    // index of W1 for N=2 (5x5): 0
    // + W1 for N>2, f=0 (6x6): 25
    // + W1 for N>3, f>0 (7x7): 61
    int index = 110;
     
    {
	// n = 2;
	BoundaryEigenStruct * s = &_boundaryEigenIndices[0];
	s->eigenval = index;
	index += 9; //K
	s->U0 = index;
	index += 16; //2*N*2*N
	s->W1 = 0;
	s->W1_0 = 0;
	s->U1.resize(1);
	s->U1[0] = index;
	index += 20; //2*n*5
	for (int k = 0; k < 3; k++) {
	    s->coeffs[k].resize(1);
	    s->coeffs[k][0] = index;
	    index += 144;  //K*16
	}
    }
    
    for (int n = 3; n <= maxvalence; n++) {
	int p = 2*n;
	BoundaryEigenStruct * s = &_boundaryEigenIndices[n-2];
	s->eigenval = index;
	index += (n > 3 ? p+7 : p+6);
	s->U0 = index;
	index += 4*n*n;
	if (n > 3) {
	    s->W1 = 61;
	    s->W1_0 = 25;
	} else {
	    s->W1_0 = s->W1 = 25;
	}
	
	int m = n/2;
	s->U1.resize(m);
	s->U1[0] = index;
	index += p*6;
	for (int k = 0; k < 3; k++) {
	    s->coeffs[k].resize(m);
	    s->coeffs[k][0] = index;
	    index += (p+6)*16;
	}
	for (int j = 1; j < m; j++) {
	    s->U1[j] = index;
	    index += p*7;
	    for (int k = 0; k < 3; k++) {
		s->coeffs[k][j] = index;
		index += (p+7)*16;
	    }
	}
    }
    
}



// Indices into interior eigenbasis data

SubdInternal::EigenStruct SubdInternal::_eigenIndices[48] = {
    {  0,  14,  { 210, 434, 658 }},
    {  882,  898,  { 1154, 1410, 1666 }},
    {  1922,  1940,  { 2264, 2552, 2840 }},
    {  3128,  3148,  { 3548, 3868, 4188 }},
    {  4508,  4530,  { 5014, 5366, 5718 }},
    {  6070,  6094,  { 6670, 7054, 7438 }},
    {  7822,  7848,  { 8524, 8940, 9356 }},
    {  9772,  9800,  { 10584, 11032, 11480 }},
    {  11928,  11958,  { 12858, 13338, 13818 }},
    {  14298,  14330,  { 15354, 15866, 16378 }},
    {  16890,  16924,  { 18080, 18624, 19168 }},
    {  19712,  19748,  { 21044, 21620, 22196 }},
    {  22772,  22810,  { 24254, 24862, 25470 }},
    {  26078,  26118,  { 27718, 28358, 28998 }},
    {  29638,  29680,  { 31444, 32116, 32788 }},
    {  33460,  33504,  { 35440, 36144, 36848 }},
    {  37552,  37598,  { 39714, 40450, 41186 }},
    {  41922,  41970,  { 44274, 45042, 45810 }},
    {  46578,  46628,  { 49128, 49928, 50728 }},
    {  51528,  51580,  { 54284, 55116, 55948 }},
    {  56780,  56834,  { 59750, 60614, 61478 }},
    {  62342,  62398,  { 65534, 66430, 67326 }},
    {  68222,  68280,  { 71644, 72572, 73500 }},
    {  74428,  74488,  { 78088, 79048, 80008 }},
    {  80968,  81030,  { 84874, 85866, 86858 }},
    {  87850,  87914,  { 92010, 93034, 94058 }},
    {  95082,  95148,  { 99504, 100560, 101616 }},
    {  102672,  102740,  { 107364, 108452, 109540 }},
    {  110628,  110698,  { 115598, 116718, 117838 }},
    {  118958,  119030,  { 124214, 125366, 126518 }},
    {  127670,  127744,  { 133220, 134404, 135588 }},
    {  136772,  136848,  { 142624, 143840, 145056 }},
    {  146272,  146350,  { 152434, 153682, 154930 }},
    {  156178,  156258,  { 162658, 163938, 165218 }},
    {  166498,  166580,  { 173304, 174616, 175928 }},
    {  177240,  177324,  { 184380, 185724, 187068 }},
    {  188412,  188498,  { 195894, 197270, 198646 }},
    {  200022,  200110,  { 207854, 209262, 210670 }},
    {  212078,  212168,  { 220268, 221708, 223148 }},
    {  224588,  224680,  { 233144, 234616, 236088 }},
    {  237560,  237654,  { 246490, 247994, 249498 }},
    {  251002,  251098,  { 260314, 261850, 263386 }},
    {  264922,  265020,  { 274624, 276192, 277760 }},
    {  279328,  279428,  { 289428, 291028, 292628 }},
    {  294228,  294330,  { 304734, 306366, 307998 }},
    {  309630,  309734,  { 320550, 322214, 323878 }},
    {  325542,  325648,  { 336884, 338580, 340276 }},
    {  341972,  342080,  { 353744, 355472, 357200 }},
};

SubdInternal::EigenStruct SubdInternal::_cornerEigenIndices[1] = {
    { 0, 9, {90, 234, 378}}
};
    
SubdInternal::BoundaryEigenStruct SubdInternal::_boundaryEigenIndices[SUBD_MAX_BOUNDARY_VALENCE];

