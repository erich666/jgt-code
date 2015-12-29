//-----------------------------------------------------------------------------
// File: Tetra.cpp
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

#include "../Header/defines.h"
#include "../Header/Tetra.h"
#include <algorithm>
#include <stdio.h>

#include <D3DX9.h>

using namespace std;

//-----------------------------------------------------------------------------
// IndexedTetraSet::getTetraScalarValue
//-----------------------------------------------------------------------------
vec3 IndexedTetraSet::getTetraVertex(int t, int vInd) {
	// returns the vertex vInd of tetrahedron t 
	return _vertices_VEC[_tetra_VEC[4*t+vInd]];;
}

//-----------------------------------------------------------------------------
// IndexedTetraSet::getTetraFaceNormal
//-----------------------------------------------------------------------------
vec3 IndexedTetraSet::getTetraFaceNormal(int t, int fInd) {
	// returns the normal of the face fInd of tetrahedron t 
	if(_face_VEC[_tetra_face_VEC[4*t+fInd]]._t[0] == t)
		return (_face_VEC[_tetra_face_VEC[4*t+fInd]].getNormal());
	//else - invert the normal
	return -(_face_VEC[_tetra_face_VEC[4*t+fInd]].getNormal());
}

//-----------------------------------------------------------------------------
// IndexedTetraSet::getTetraAdjToTetraFace 
//-----------------------------------------------------------------------------
void IndexedTetraSet::getTetraAdjToTetraFace (int t, int fInd, int &tAdj, int &fAdjInd) {
	// returns the index of the tetrahedron adjacent to a given
	// face fInd of a tetrahedron t. If it is a boundary face
	// return -1
	_face_VEC[_tetra_face_VEC[4*t+fInd]].getAdjTetra(t, tAdj, fAdjInd);
}

//-----------------------------------------------------------------------------
// IndexedTetraSet::getTetraScalarValue
//-----------------------------------------------------------------------------
double IndexedTetraSet::getTetraScalarValue(int t) {
	// returns one scalar representing the scalar value inside the
	// tetrahedron. Since our model has one scalar value for each vertex
	// defining the tetrahedron, we either pick the value from one of the
	// vertices or compute an average of the scalar values
	
	// We use the scalar value of the FIRST vertex because the
	// ray casting computation in the .fx files uses this vertex
	// as base vertex
	return _scalar_VEC[_tetra_VEC[4*t]];;
}

//-----------------------------------------------------------------------------
// IndexedTetraSet::computeFirstIntersection
//-----------------------------------------------------------------------------
int IndexedTetraSet::computeFirstIntersection(vec3& eye, vec3& ray, 
											float &lambda) {
	vec3 v[4];
	vec3 n[4];

	float nearTet=-1;
	lambda = 9999999;

	for (int t=0; t < _nT; t++) {
		float lnear, lfar;
		if (computeFirstIntersection(t, eye, ray, lnear, lfar)) {
			if (lnear < lambda) {
				lambda = lnear;
				nearTet = (float)t;
			}
		}
	}
	return (int)nearTet;
}					  

//-----------------------------------------------------------------------------
// IndexedTetraSet::computeFirstIntersection
//-----------------------------------------------------------------------------
int
IndexedTetraSet::computeFirstIntersection(int t, vec3& eye, vec3& ray, 
								    float &lambdaNear, float &lambdaFar) {
	vec3 v[4];
	vec3 n[4];
	double num, den;

	for (int i=0; i<4; i++) {
		v[i] = getTetraVertex(t, i)-eye;
		n[i] = getTetraFaceNormal(t, i);
	}

	if (t == 12684 || t == 4295)
		int found = 1;

	lambdaNear = -1; 
	lambdaFar = 99999999.0f;

	for (int i=0; i<4; i++) {
		num = v[3-i] * n[i];
		den = ray * n[i];
		if (den < 0) 
			lambdaNear = (num/den > lambdaNear) ? (float)(num/den) : lambdaNear;
		if (den > 0)
			lambdaFar = (num/den < lambdaFar) ? (float)(num/den) : lambdaFar;
	}
	if (lambdaNear < lambdaFar && lambdaNear > 0) return 1; else return 0;
}

//-----------------------------------------------------------------------------
// IndexedTetraSet::computeIntersection
//-----------------------------------------------------------------------------
void IndexedTetraSet::computeIntersection(int t, vec3& eye, vec3& ray, 
										  float &lambda) {
	vec3 v[4];
	vec3 n[4];
	double num, den, lambdas[4];

	for (int i=0; i<4; i++) {
		v[i] = getTetraVertex(t, i)-eye;
		n[i] = getTetraFaceNormal(t, i);
	}

	for (int i=0; i<4; i++) {
		num = v[3-i] * n[i];
		den = ray * n[i];
		lambdas[i]=num/den;
	}

	lambda = (lambdas[0] < lambdas[1]) ? (float)lambdas[0]: (float)lambdas[1];
	lambda = (lambdas[2] < lambda) ? (float)lambdas[2]: lambda;
	lambda = (lambdas[3] < lambda) ? (float)lambdas[3]: lambda;
}

//-----------------------------------------------------------------------------
// computeUV2D
// computes u and v coordinates to be used to index mesh data store into textures
//-----------------------------------------------------------------------------
void
computeUV2D(int t, int& u, int &v, int texDimension) {
	// - changed u and v assignments so that consecutive vertices are stored in
	// consecutive positions
	// - need to multiply by 3 the u index
	// - add 1 to u value (do not use first position)
	v = t / (texDimension / 3);
	u = (t % (texDimension / 3))*3+1;
}

//-----------------------------------------------------------------------------
// computeUV2DNormalized
// computes u and v coordinates to be used to index mesh data store into textures
//-----------------------------------------------------------------------------
void
computeUV2DNormalized(int t, float& uf, float&vf, int texDimension) {
	// - changed u and v assignments so that consecutive vertices are stored in
	// consecutive positions
	// - need to multiply by 3 the u index
	// - add 1 to u value (do not use first position)
	int u, v;
	computeUV2D(t, u, v, texDimension);
	float texIncr = 1.0f / (float)(2.0*texDimension);
	uf = texIncr + (float)u / (float)texDimension;
	vf = texIncr + (float)v / (float)texDimension;
}

//-----------------------------------------------------------------------------
// computeTetraFromUVNormalized
//-----------------------------------------------------------------------------
void
computeTetraFromUVNormalized(float uf, float vf, int &t, int texDimension) {
	// - changed u and v assignments so that consecutive vertices are stored in
	// consecutive positions
	// - need to multiply by 3 the u index
	// - add 1 to u value (do not use first position)
	int u, v;
	float texIncr = 1.0f / (float)(2.0*texDimension);
	u = (int)((uf-texIncr) * texDimension);
	v = (int)((vf-texIncr) * texDimension);
	t = (u-1)/3 + v * (texDimension / 3);
}

//----------------------------------------------------------------------------
// float IndexedTetraSet::squareDistance()
//----------------------------------------------------------------------------
float IndexedTetraSet::squareDistance(vec3 begin, vec3 end){
	return (float)(pow((begin[0]-end[0]),2)+pow((begin[1]-end[1]),2)+pow((begin[2]-end[2]),2));
}

//----------------------------------------------------------------------------
// bool IndexedTetraSet::readNRRD()
//----------------------------------------------------------------------------
bool IndexedTetraSet::readNRRD(char *filename)
{
	char dummy[100];

	// Now we need to read the vertex file
	sprintf(dummy,"%s-vertex.nrrd", filename);
	FILE *fd = fopen(dummy,"r");

	if (fd==NULL) return false;

	fscanf (fd,"%s", dummy);
	fscanf (fd,"%s", dummy);
	fscanf (fd,"%s", dummy);
	fscanf (fd,"%s", dummy);
	fscanf (fd,"%s", dummy);
	fscanf (fd,"%s", dummy);
	fscanf (fd,"%s", dummy);
	fscanf (fd,"%d", &_nV);
	fscanf (fd,"%s", dummy);
	fscanf (fd,"%s", dummy);

	for (int i=0; i<_nV; i++)
	{
		double x,y,z;
		if (fscanf (fd,"%lf %lf %lf", &x, &y, &z) != 3)
		return false;

		vec3 v(x, y, z);

		_vertices_VEC.push_back(v);

		if (x < _minX) _minX = x;
		if (x > _maxX) _maxX = x;
		if (y < _minY) _minY = y;
		if (y > _maxY) _maxY = y;
		if (z < _minZ) _minZ = z;
		if (z > _maxZ) _maxZ = z;
	}

	fclose(fd);

	// we need to read the tetra file
	sprintf(dummy,"%s-elem.nrrd", filename);
	fd = fopen(dummy,"r");

	if (fd==NULL) return false;

	fscanf (fd,"%s", dummy);
	fscanf (fd,"%s", dummy);
	fscanf (fd,"%s", dummy);
	fscanf (fd,"%s", dummy);
	fscanf (fd,"%s", dummy);
	fscanf (fd,"%s", dummy);
	fscanf (fd,"%s", dummy);
	fscanf (fd,"%d", &_nT);
	fscanf (fd,"%s", dummy);
	fscanf (fd,"%s", dummy);

	// read tetra
	int nreads=0;
	double dist = 0.0f, newDist;
	for (int i=0; i<_nT; i++)
	{
		unsigned a,b,c,d;
		if (fscanf (fd,"%u %u %u %u", &a, &b, &c, &d) != 4)
				return false;

		_tetra_VEC.push_back(a);
		_tetra_VEC.push_back(b);
		_tetra_VEC.push_back(c);
		_tetra_VEC.push_back(d);
		newDist = squareDistance(_vertices_VEC[a], _vertices_VEC[b]);
		dist = (dist > newDist) ? dist : newDist;
  		newDist = squareDistance(_vertices_VEC[a], _vertices_VEC[c]);
		dist = (dist > newDist) ? dist : newDist;
  		newDist = squareDistance(_vertices_VEC[a], _vertices_VEC[d]);
		dist = (dist > newDist) ? dist : newDist;
  		newDist = squareDistance(_vertices_VEC[b], _vertices_VEC[c]);
		dist = (dist > newDist) ? dist : newDist;
  		newDist = squareDistance(_vertices_VEC[b], _vertices_VEC[d]);
		dist = (dist > newDist) ? dist : newDist;
  		newDist = squareDistance(_vertices_VEC[c], _vertices_VEC[d]);
		dist = (dist > newDist) ? dist : newDist;
		nreads += 4;
	}

	_maxDistance = sqrt(dist);

	fclose(fd);
	int nsize = _tetra_VEC.size();

	// Now we need to read the scalar file
	sprintf(dummy,"%s-scalar.nrrd", filename);
	fd = fopen(dummy,"r");

	if (fd==NULL) return false;

	fscanf (fd,"%s", dummy);
	fscanf (fd,"%s", dummy);
	fscanf (fd,"%s", dummy);
	fscanf (fd,"%s", dummy);
	fscanf (fd,"%s", dummy);
	fscanf (fd,"%s", dummy);
	fscanf (fd,"%s", dummy);
	fscanf (fd,"%d", &_nV);
	fscanf (fd,"%s", dummy);
	fscanf (fd,"%s", dummy);

	for (int i=0; i<_nV; i++)
	{
		double s;
		if (fscanf (fd,"%lf", &s) != 1) 
			return false;

		_scalar_VEC.push_back(s);

		if (s < _minScalar) _minScalar = s;
		if (s > _maxScalar) _maxScalar = s;
	}

	fclose(fd);

	return true;
}

//----------------------------------------------------------------------------
// bool IndexedTetraSet::readOff()
//----------------------------------------------------------------------------
bool IndexedTetraSet::readOff(char *filename)
{
	char filename2[100];
	sprintf (filename2,"%s.off", filename);

  FILE *f = fopen(filename2,"rb");

  if (f==NULL) return false;
  
  if (fscanf (f,"%u %u", &_nV, &_nT) != 2)    return false;

  // read points
  int i;
  for (i=0; i<_nV; i++)
    {
      double x,y,z,s;
      if (fscanf (f,"%lf %lf %lf %lf", &x, &y, &z, &s) != 4)
		return false;

      vec3 v(x, y, z);

  	  if (i == 1634) 
		  int found = 1;

      _vertices_VEC.push_back(v);
      _scalar_VEC.push_back(s);

      if (x < _minX) _minX = x;
      if (x > _maxX) _maxX = x;
      if (y < _minY) _minY = y;
      if (y > _maxY) _maxY = y;
      if (z < _minZ) _minZ = z;
      if (z > _maxZ) _maxZ = z;
	  if (s < _minScalar) _minScalar = s;
	  if (s > _maxScalar) _maxScalar = s;
    }

  // read tetra
  int nreads=0;
  double dist = 0.0f, newDist;
  for (i=0; i<_nT; i++)
    {
      unsigned a,b,c,d;
      if (fscanf (f,"%u %u %u %u", &a, &b, &c, &d) != 4)
				return false;

      _tetra_VEC.push_back(a);
      _tetra_VEC.push_back(b);
      _tetra_VEC.push_back(c);
      _tetra_VEC.push_back(d);
	  newDist = squareDistance(_vertices_VEC[a], _vertices_VEC[b]);
	  dist = (dist > newDist) ? dist : newDist;
  	  newDist = squareDistance(_vertices_VEC[a], _vertices_VEC[c]);
	  dist = (dist > newDist) ? dist : newDist;
  	  newDist = squareDistance(_vertices_VEC[a], _vertices_VEC[d]);
	  dist = (dist > newDist) ? dist : newDist;
  	  newDist = squareDistance(_vertices_VEC[b], _vertices_VEC[c]);
	  dist = (dist > newDist) ? dist : newDist;
  	  newDist = squareDistance(_vertices_VEC[b], _vertices_VEC[d]);
	  dist = (dist > newDist) ? dist : newDist;
  	  newDist = squareDistance(_vertices_VEC[c], _vertices_VEC[d]);
	  dist = (dist > newDist) ? dist : newDist;
	 nreads += 4;
    }

	_maxDistance = sqrt(dist);

  fclose(f);

  return true;
  /*
  int nsize = _tetra_VEC.size();

  // translate the dataset
  { 
    float lx,ly,lz,x,y,z;
    int i;

    lx = -(float)(_minX+((_maxX-_minX)/2.0));
    ly = -(float)(_minY+((_maxY-_minY)/2.0));
    lz = -(float)(_minZ+((_maxZ-_minZ)/2.0));

    _minX = _minY = _minZ = 1e20;
    _maxX = _maxY = _maxZ = -1e20;

    vec3 l(lx, ly, lz);
    
    for (i=0; i<_nV;i++) {
      _vertices_VEC[i] += l;
	  _scalar_VEC[i] /= _maxScalar;
      
      x = (float)_vertices_VEC[i][0];
      y = (float)_vertices_VEC[i][1];
      z = (float)_vertices_VEC[i][2];
      
      if (x < _minX) _minX = x;
      if (x > _maxX) _maxX = x;
      if (y < _minY) _minY = y;
      if (y > _maxY) _maxY = y;
      if (z < _minZ) _minZ = z;
      if (z > _maxZ) _maxZ = z;
    }
  }
  */
}

//----------------------------------------------------------------------------
// bool IndexedTetraSet::readFromFile()
//----------------------------------------------------------------------------
bool IndexedTetraSet::readFromFile(char *filename, MeshFormat mformat)
{
	_minX = _minY = _minZ = _minScalar = 1e20;
	_maxX = _maxY = _maxZ = _maxScalar = -1e20;
	_nT = _nV = 0;

	bool result;

	switch (mformat) {
		case OFF: result = readOff(filename);break;
		case NRRD: result = readNRRD(filename); break;
		default: printf("mesh format invalid"); exit(-1);
	}

	if (result == false) return result;

	// translate the dataset
	float lx,ly,lz,x,y,z;
	int i;

	lx = (float)-(_minX+((_maxX-_minX)/2.0f));
	ly = (float)-(_minY+((_maxY-_minY)/2.0f));
	lz = (float)-(_minZ+((_maxZ-_minZ)/2.0f));

	_minX = _minY = _minZ = 1e20;
	_maxX = _maxY = _maxZ = -1e20;

  //TODO: TEST: fabio - fix max scalar to allow the visualization of other time stamps.
  _maxScalar = 3.1860000999999998;

	vec3 l(lx, ly, lz);

	double bias = 0.5/(double)_nV;
	double scale = (1.0 - (1.0/(double)_nV))/ (_maxScalar - _minScalar);

	for (i=0; i<_nV;i++) {
		_vertices_VEC[i] += l;
		_scalar_VEC[i] = scale*(_scalar_VEC[i]-_minScalar) + bias;

		x = (float)_vertices_VEC[i][0];
		y = (float)_vertices_VEC[i][1];
		z = (float)_vertices_VEC[i][2];
	    
		if (x < _minX) _minX = x;
		if (x > _maxX) _maxX = x;
		if (y < _minY) _minY = y;
		if (y > _maxY) _maxY = y;
		if (z < _minZ) _minZ = z;
		if (z > _maxZ) _maxZ = z;
	}

	for (i=0; i<_nT; i++)
	{
		unsigned a,b,c,d; // Four vertices
		a = _tetra_VEC[4*i+0];      b = _tetra_VEC[4*i+1];
		c = _tetra_VEC[4*i+2];      d = _tetra_VEC[4*i+3];
		vec3 v1, v2, v3, v4;
		v1 = _vertices_VEC[a];
  		v2 = _vertices_VEC[b];
  		v3 = _vertices_VEC[c];
  		v4 = _vertices_VEC[d];
		mat4 mat(vec4(v1, 1), vec4(v2, 1), vec4(v3, 1), vec4(v4, 1));
		vec4 scalar(_scalar_VEC[a], _scalar_VEC[b], 
			_scalar_VEC[c], _scalar_VEC[d]);
		vec4 plane = mat.inverse() * scalar;
		vec3 grad(plane, 3);

		_grad_VEC.push_back(grad);
		vec3 grad2 = _grad_VEC[i];
	}

	cout << "_minX = "  << _minX << " _maxX = " << _maxX
		<< " _minY = " << _minY << " _maxY = " << _maxY 
		<< " _minZ = " << _minZ << " _maxZ = " << _maxZ << endl; 

	return true;
}


//----------------------------------------------------------------------------
//
// int IndexedTetraSet::_getF()
//
//----------------------------------------------------------------------------

int IndexedTetraSet::_getF(unsigned a, unsigned b, unsigned c, unsigned d,
			   map<unsigned, vector<int>, less<int> > &vertices_useset_MAP)
{
  map< unsigned, vector<int>, less<int> >::iterator vi;
  vi = vertices_useset_MAP.find(a);
  if(vi != vertices_useset_MAP.end())
  {
      unsigned i;
      vector<int> & t_VEC = vertices_useset_MAP[a];
      for(i = 0; i < t_VEC.size(); i++)
	{
	  int t = t_VEC[i];

	  int tp[3];
	  tp[0] = _face_VEC[t]._v[0];
	  tp[1] = _face_VEC[t]._v[1];
	  tp[2] = _face_VEC[t]._v[2];

	  if (tp[0]==a &&
	      (tp[1]==b && tp[2]==c ||
	       tp[1]==c && tp[2]==b)
	      ||
	      tp[0]==b &&
	      (tp[1]==a && tp[2]==c || 
	       tp[1]==c && tp[2]==a)
	      ||
	      tp[0]==c &&
	      (tp[1]==a && tp[2]==b || 
	       tp[1]==b && tp[2]==a))
	    return t;
	}
  }//end if vi

  int tn = _face_VEC.size();
  _face_VEC.push_back(TetraFace(a, b, c, d));
  vertices_useset_MAP[ a ].push_back(tn);
  vertices_useset_MAP[ b ].push_back(tn);
  vertices_useset_MAP[ c ].push_back(tn);
  return tn;
}

//----------------------------------------------------------------------------
//
// bool IndexedTetraSet::computeFaces()
//
//----------------------------------------------------------------------------

bool IndexedTetraSet::computeFaces()
{
  map< unsigned, vector<int>, less<int> > vertices_useset_MAP;

  vector<int> tmp((size_t) _tetra_VEC.size(), -1);

  _tetra_face_VEC.swap(tmp);

  int i;
  for (i=0; i<_nT; i++)
    {
		if(i == 827000)
			int found =1;

      unsigned a,b,c,d; // Four vertices
      a = _tetra_VEC[4*i+0];      b = _tetra_VEC[4*i+1];
      c = _tetra_VEC[4*i+2];      d = _tetra_VEC[4*i+3];

      int f = _getF(b, c, d, a, vertices_useset_MAP);      
      _face_VEC[f].addT(i, 0);
      _tetra_face_VEC[4*i + 0] = f;
      f = _getF(a, c, d, b, vertices_useset_MAP);          
      _face_VEC[f].addT(i, 1);      
      _tetra_face_VEC[4*i + 1] = f;
      f = _getF(a, b, d, c, vertices_useset_MAP);          
      _face_VEC[f].addT(i, 2);
      _tetra_face_VEC[4*i + 2] = f;
      f = _getF(a, b, c, d, vertices_useset_MAP);          
      _face_VEC[f].addT(i, 3);
      _tetra_face_VEC[4*i + 3] = f;
      if (i%100 == 0) std::cout << i << " " << flush;
    }

  return true;
}

//----------------------------------------------------------------------------
//
// bool IndexedTetraSet::createVertexBuffer()
//
//----------------------------------------------------------------------------
Vertex* IndexedTetraSet::createVertexBuffer(int meshTextureSize) 
// The texture size is important because it is used to encode the tetrahedron
// information into r and b coordinates of the color of each face. The
// color is generated as follows:
// - r = t / textureSize
// - g = t % textureSize
// - b = face index
{
	int nFaces = _boundary_fac_VEC.size();
	Vertex *vb = new Vertex [3*nFaces];

	vector<TetraFace>::iterator fi;
	int i;
	int vInd[4];
	vec3 v[3];

	for(fi = _boundary_fac_VEC.begin(), i=0; i < nFaces /*fi != _boundary_fac_VEC.end()*/; fi++, i++) {
		fi->getVertexIndex(vInd);
		v[0] = getVertex(vInd[0]); 
		v[1] = getVertex(vInd[1]);
		v[2] = getVertex(vInd[2]);

		int t = fi->_t[0];

		if (t == 12684)
			int found = 1;
		int f = fi->_fInd[0];
		float r;
		float g;

		computeUV2DNormalized(t, r, g, meshTextureSize);

    for (int j=0; j<3; j++) {
			vb[3*i+j].x = (float) v[j][0];
			vb[3*i+j].y = (float) v[j][1];
			vb[3*i+j].z = (float) v[j][2];
			vb[3*i+j].r = r;
			vb[3*i+j].g = g;
			vb[3*i+j].b = 0.0f;
		}
	}
	return vb;
}

//----------------------------------------------------------------------------
//
// bool IndexedTetraSet::getBoundaryFaces()
//
//----------------------------------------------------------------------------

bool IndexedTetraSet::getBoundaryFaces(vector<TetraFace>& bfaces_VEC)
{
  bfaces_VEC.erase(bfaces_VEC.begin(), bfaces_VEC.end());

  // collect boundary faces
  vector<TetraFace>::iterator fi;
  for(fi = _face_VEC.begin(); fi != _face_VEC.end(); fi++)
    if(fi->nT()==1)	
      bfaces_VEC.push_back(*fi);

  return true;
}

//----------------------------------------------------------------------------
//
// bool IndexedTetraSet::getBoundaryFacesWithVertices()
//
//----------------------------------------------------------------------------

bool IndexedTetraSet::getBoundaryFacesWithVertices(vector<TetraFace>& bfaces_VEC, 
						   vector<vec3>& vert_VEC)
{
  vector<int> vertex_index((size_t) _vertices_VEC.size(), -1);
  vector<int> outVerInd_VEC;

  int nOutVer = 0;
  _nBoundaryFaces = 0;
  vector<TetraFace>::iterator fi;
  for(fi = _face_VEC.begin(); fi != _face_VEC.end(); fi++)
    if(fi->nT()==1)	
      {
	_nBoundaryFaces++;
	for(int i= 0; i < 3; i++)
	  if(vertex_index[fi->_v[i]] == -1)
	    {
	      vertex_index[fi->_v[i]] = nOutVer; 
	      outVerInd_VEC.push_back(fi->_v[i]);
	      nOutVer++; 
	    }
      }

  int i;
  for(i = 0; i < nOutVer; i++)
    vert_VEC.push_back(_vertices_VEC[outVerInd_VEC[i]]);

  for(fi = _face_VEC.begin(); fi != _face_VEC.end(); fi++)
    if(fi->nT()==1)	
      {
	TetraFace tmp = *fi;
	
	for(int i= 0; i < 3; i++)
	  tmp._v[i] = vertex_index[fi->_v[i]];

	bfaces_VEC.push_back(tmp);
      }

  return true;
}

//----------------------------------------------------------------------------
//
// bool IndexedTetraSet::saveOffBoundaryFaces()
//
//----------------------------------------------------------------------------

bool IndexedTetraSet::saveOffBoundaryFaces(char *filename)
{
  vector<int> vertex_index((size_t) _vertices_VEC.size(), -1);
  vector<int> outVerInd_VEC;

  int nOutVer = 0;
  int boundaryFaces = 0;
  vector<TetraFace>::iterator fi;
  for(fi = _face_VEC.begin(); fi != _face_VEC.end(); fi++)
    if(fi->nT()==1)	
      {
	boundaryFaces++;

	for(int i= 0; i < 3; i++)
	  if(vertex_index[fi->_v[i]] == -1)
	    { 
	      vertex_index[fi->_v[i]] = nOutVer; 
	      outVerInd_VEC.push_back(fi->_v[i]);
	      nOutVer++; 
	    }
      }

  ofstream offFile(filename, ios::out);
  if (!offFile)     return false;

  offFile << "OFF" << endl;
  offFile << nOutVer << " " << boundaryFaces << " " << 0 << "\n";

  int i;
  for(i = 0; i < nOutVer; i++)
    {
      vec3 v = _vertices_VEC[outVerInd_VEC[i]];
      offFile << v[0] << " " << v[1] << " " << v[2] << endl;
    }

  for(fi = _face_VEC.begin(); fi != _face_VEC.end(); fi++)
    if(fi->nT()==1)	
      {
	offFile << "3 ";
	for(int i= 0; i < 3; i++)
	  offFile << vertex_index[fi->_v[i]] << " ";
	offFile << endl;
      }

  offFile.close();

  cout << "\nNumber of vertices in boundary file : " << nOutVer;
  cout << "\nNumber of faces in boundary file    : " << boundaryFaces;
  
  return true;
}


//----------------------------------------------------------------------------
// void  TetraFace::getAdjTetra();
//----------------------------------------------------------------------------  
void
TetraFace::getAdjTetra(int t, int &tAdj, int &fAdj) {
	if (_nt == 1) {
		tAdj = -1;
		fAdj = -1;
		return;
	}
	if (_t[0] == t) {
		tAdj = _t[1];
		fAdj = _fInd[1];
	}
	else if (_t[1] == t) {
		tAdj = _t[0];
		fAdj = _fInd[0];
	}
	else {
		printf("\nError while accessing adjacent tetrahedron");
		exit(-1);
	}
}

//----------------------------------------------------------------------------
// bool TetraFace::computeABCD()
//----------------------------------------------------------------------------

bool TetraFace::computeABCD(vector<vec3> &vertices_VEC, vector<int> &tetra_VEC, int& invertNormals)
{
  vec3 &fv0 = vertices_VEC[_v[0]];
  vec3 &fv1 = vertices_VEC[_v[1]];
  vec3 &fv2 = vertices_VEC[_v[2]];
  vec3 &fv3 = vertices_VEC[_v[3]];

  vec3 n = (fv2-fv0)^(fv1-fv0);

	if (n[0] == 0.0 && n[1] == 0.0 && n[2] == 0.0) {
		_a = _b = _c = _d = 0.0;
		return true;
	}

  n.normalize();

  _a = n[0];
  _b = n[1];
  _c = n[2];
  _d = - n * fv0;

  double ndot = n * fv3 + _d;

  if (ndot > 0.0) {
    int aux = _v[2];
    _v[2] = _v[1];
    _v[1] = aux;
    _a = -_a;
    _b = -_b;
    _c = -_c;
    _d = -_d;
    invertNormals++;
  }

  return true;
}


//----------------------------------------------------------------------------
// bool IndexedTetraSet::computeFacesPlaneEquations()
//----------------------------------------------------------------------------
bool IndexedTetraSet::computeFacesPlaneEquations()
{
	_invertNormals=0;
	int nFaces=0;
	vector<TetraFace>::iterator fi;
	_nBoundaryFaces = 0;
	for(fi = _face_VEC.begin(); fi != _face_VEC.end(); fi++)
	{
		if(fi->_nt < 2) {
			fi->computeCentroid(_vertices_VEC);
			_nBoundaryFaces++;
		}else{
			fi->computeCentroid(_vertices_VEC);
		}

		fi->computeABCD(_vertices_VEC, _tetra_VEC, _invertNormals);
		nFaces++;
	}

	cout << "\n**** Inverted :"  << _invertNormals << " of " << nFaces << "\n\n" << flush;

	// cache the boundary faces to avoid having to determine
	// them multiple times
	getBoundaryFaces(_boundary_fac_VEC);
	return true;
}