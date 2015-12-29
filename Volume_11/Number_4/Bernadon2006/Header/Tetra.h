//-----------------------------------------------------------------------------
// File: Tetra.h
// Desc: class to open tetrahedra meshes
// Copyright (C) 2005, Joao Comba, Fabio Bernardon, UFRGS-Brasil
//-----------------------------------------------------------------------------
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
//-----------------------------------------------------------------------------

#ifndef __TETRA_H__
#define __TETRA_H__

#ifdef WIN32
#  include <windows.h>
#endif

#include <vector>
#include <list>
#include <map>
#include <math.h>
#include <assert.h>

#include "algebra3.h"

typedef enum {OFF, NRRD} MeshFormat;

double const EPS = 0.00001f;

using namespace std;

struct Vertex
{
   FLOAT x, y, z; // x, y, and z coordinates.
   FLOAT r, g, b;
};

//----------------------------------------------------------------------------
//
// class TetraFace
//
//----------------------------------------------------------------------------
class TetraFace
{
public:

	int         _nt;    // number of tetrahedra this face belongs to
	int         _t[2];  // pointers to the (up to two) tetrahedra
	int			_fInd[2];	// tells the index of this face (0,1,2 or 3) this face is
						// referenced in the tetrahedra
	int          _v[4];  // indices to the vertices

	double        _a, _b, _c, _d;   // plane equation for this face

	vec3         _centroid;
	double        _dist;

	inline void addT(int t, int f)    { _t[_nt] = t; _fInd[_nt] = f; _nt++;  }
	inline int  getT(int t)    { return _t[t];        }
	inline int  nT()           { return _nt;          }
	inline vec3 getNormal()    { return vec3(_a, _b, _c); }
	void  getAdjTetra(int t, int &tAdj, int &fAdj);

  TetraFace(int v0, int v1, int v2, int v3)
    { _v[0] = v0; _v[1] = v1; _v[2] = v2; _v[3] = v3; _nt = 0; 
      _t[0] = _t[1] = -1; }
  TetraFace() {}

  void getVertexIndex(int *v) { v[0] = _v[0]; v[1] = _v[1]; v[2] = _v[2]; v[3] = _v[3]; }

  bool computeABCD(vector<vec3> &vertices_VEC, vector<int> &tetra_VEC, int& invertNormals);
  bool computeCentroid(vector<vec3> &vertices_VEC)
  {
    _centroid = (vertices_VEC[_v[0]] + 
		 vertices_VEC[_v[1]] + 
		 vertices_VEC[_v[2]])/3.;
    return true;
  }

  friend inline bool operator<(const TetraFace& a, const TetraFace& b) 
  {
    return a._dist > b._dist;
  }
};


//----------------------------------------------------------------------------
//
// class IndexedTetraSet
//
//----------------------------------------------------------------------------
class IndexedTetraSet
{
public:

  int                _nV;
  int                _nT;
  vector<vec3>       _vertices_VEC;
  vector<double>      _scalar_VEC;
  vector<int>        _tetra_VEC;
  vector<int>        _tetra_face_VEC;
  vector<TetraFace>  _face_VEC;
  vector<double>      _den_VEC;
  vector<vec3>       _col_VEC;
  vector<vec3>		 _grad_VEC;
  vector<TetraFace>  _boundary_fac_VEC;

  vector<double>      _den_map_VEC;
  vector<double>      _col_map_VEC;

  double              _minScalar;
  double              _maxScalar;

  double              _minX, _maxX;
  double              _minY, _maxY;
  double              _minZ, _maxZ;

  int				_invertNormals;

  int				_nBoundaryFaces;
  int				_nMeshCells;
  int				_nFillerCells;
  int				_nTotalCells;

  double		_maxDistance;

  int  _getF(unsigned a, unsigned b, unsigned c, unsigned d,
	     map< unsigned, vector<int>, less<int> > &vertices_useset_MAP);

  inline double getMaxDistance() {return _maxDistance; }
  inline double  getMinX() { return _minX; }
  inline double  getMaxX() { return _maxX; }
  inline double  getMinY() { return _minY; }
  inline double  getMaxY() { return _maxY; }
  inline int nTetra() { return _nT; }
  inline int nMeshCells() { return _nMeshCells; }
  inline int nBoundaryFaces() { return _nBoundaryFaces; }

  IndexedTetraSet() {}
  ~IndexedTetraSet() 
    {
      _vertices_VEC.erase(_vertices_VEC.begin(), _vertices_VEC.end());
      _scalar_VEC.erase(_scalar_VEC.begin(), _scalar_VEC.end());
      _tetra_VEC.erase(_tetra_VEC.begin(), _tetra_VEC.end());
	  _grad_VEC.erase(_grad_VEC.begin(), _grad_VEC.end());
      _face_VEC.erase(_face_VEC.begin(), _face_VEC.end());
      _tetra_face_VEC.erase(_tetra_face_VEC.begin(), _tetra_face_VEC.end());
      _den_VEC.erase(_den_VEC.begin(), _den_VEC.end());
      _col_VEC.erase(_col_VEC.begin(), _col_VEC.end());
    }

  bool readDATASET(char *filename);
  bool readGLOBAL(char *filename);
  bool readXYZ(char *filename);
  bool readVERTS(char *filename);
  bool readSCALAR(char *filename);
  bool readMAPS(char *colorname, char *densityname);
  bool makeMAPS();

  bool readOff(char *filename);
  bool readNRRD(char *filename);
  bool readFromFile(char *filename, MeshFormat mformat);

  bool computeFaces();
  bool computeFacesPlaneEquations();

  bool getBoundaryFaces(vector<TetraFace>& bfaces_VEC);
  bool getBoundaryFacesWithVertices(vector<TetraFace>& bfaces_VEC, 
				    vector<vec3>& vert_VEC);
  bool saveOffBoundaryFaces(char *filename);

  // Tetrahedron data queries
  vec3 getVertex(int vInd) { return _vertices_VEC[vInd]; }
  vec3 getTetraVertex(int t, int vInd);
  vec3 getTetraFaceNormal(int t, int fInd);
  void getTetraAdjToTetraFace (int t, int fInd, int &tAdj, int &fAdjInd);
  double getTetraScalarValue(int t);
  void computeIntersection(int t, vec3& eye, vec3& ray, float &lambda);
  int computeFirstIntersection(vec3& eye, vec3& ray, float &lambda);
  int computeFirstIntersection(int t, vec3& eye, vec3& ray, 
							float &lambdaNear, float &lambdaFar);

  Vertex* IndexedTetraSet::createVertexBuffer(int meshTextureSize) ;

  float squareDistance(vec3 begin, vec3 end);

};

void computeUV2D(int t, int& u, int &v, int texDimension);
void computeUV2DNormalized(int t, float& uf, float&vf, int texDimension);
void computeTetraFromUVNormalized(float uf, float vf, int &t, int texDimension);

#endif
