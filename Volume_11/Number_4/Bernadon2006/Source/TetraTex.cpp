//-----------------------------------------------------------------------------
// File: TetraTex.cpp
// Desc: Load a tetrahedral mesh into textures
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

#define WIN32_LEAN_AND_MEAN   // Exclude the stuff we don't need in libraies.
#define WIN32_EXTRA_LEAN      // Exclude even further.

#include "../Header/defines.h"

#include <stdio.h>
#include <d3dx9.h>             // The main header you will need for Direct3d.
#include "../Header/resource.h"
#include "../Header/tetratex.h"
#include "../Header/tetra.h"


//-----------------------------------------------------------------------------
// setTetrahedralDataInTexture2D
// stores 3 4D-vectors in one texture
//-----------------------------------------------------------------------------
void
setTetrahedralDataInTexture2D(D3DLOCKED_RECT &lockedRect, BYTE* pSliceStart, 
              							  int texDimension, int t, D3DXVECTOR4 &value1,
                              D3DXVECTOR4 &value2, D3DXVECTOR4 &value3) {

	// Need to compute the u and v coordinates
	int u, v;
	computeUV2D(t, u, v, texDimension);
	lockedRect.pBits = pSliceStart + v*lockedRect.Pitch;

	// store the computed value
	((D3DXVECTOR4*)lockedRect.pBits)[u] = value1;
	((D3DXVECTOR4*)lockedRect.pBits)[u+1] = value2;
	((D3DXVECTOR4*)lockedRect.pBits)[u+2] = value3;
}


//-----------------------------------------------------------------------------
// setTetrahedralDataInTexture2D
//-----------------------------------------------------------------------------
void setTetrahedralDataInTexture2D (D3DLOCKED_RECT &LockedRect,
                                    BYTE* pSliceStart, int texDimension,
                                    int t, D3DXVECTOR4 &value) {

	// Need to compute the u and v coordinates
	int u = t / texDimension;
	int v = t % texDimension;
	LockedRect.pBits = pSliceStart + v*LockedRect.Pitch;

	// store the computed value
	((D3DXVECTOR4*)LockedRect.pBits)[u] = value;
}


//-----------------------------------------------------------------------------
// LoadMeshIntoNewTextures
//-----------------------------------------------------------------------------
HRESULT LoadMeshIntoNewTextures(IndexedTetraSet *mesh, 
				 LPDIRECT3DDEVICE9 D3D_Device,
				 LPDIRECT3DTEXTURE9 *verticesTEX,
				 LPDIRECT3DTEXTURE9 *normalsTEX,
				 LPDIRECT3DTEXTURE9 *neighborTEX,
				 int texDimension) {

	HRESULT hr;

	int width=texDimension, height=texDimension, depth=4; 

	// Create vertices texture
	hr = D3DXCreateTexture(D3D_Device, width, height, 1, 0,
							D3DFMT_A32B32G32R32F, D3DPOOL_MANAGED, verticesTEX);
	if (hr != D3D_OK) {
		MessageBox(NULL, "Unable to create vertices texture", NULL, MB_OK);
		return false;
	}

	// Create facenormals texture
	hr = D3DXCreateTexture(D3D_Device, width, height, 1, 0,
							D3DFMT_A32B32G32R32F, D3DPOOL_MANAGED, normalsTEX);
	if (hr != D3D_OK)  {
		MessageBox(NULL, "Unable to create normal texture", NULL, MB_OK);
		return false;
	}

	// Create neighbor data texture
	hr = D3DXCreateTexture(D3D_Device, width, height, 1, 0,
							D3DFMT_A32B32G32R32F, D3DPOOL_MANAGED, neighborTEX);
	if (hr != D3D_OK)  {
		MessageBox(NULL, "Unable to create volume texture", NULL, MB_OK);
		return false;
	}

	D3DLOCKED_RECT verticesLockedRect, neighborLockedRect;

    if (FAILED(hr = (*verticesTEX)->LockRect(0, &verticesLockedRect, 0, 0))) return hr;
	D3DLOCKED_RECT normalsLockedRect;
    if (FAILED(hr = (*normalsTEX)->LockRect(0, &normalsLockedRect, 0, 0))) return hr;
    if (FAILED(hr = (*neighborTEX)->LockRect(0, &neighborLockedRect, 0, 0))) return hr;

	BYTE* startVertices    = (BYTE*)verticesLockedRect.pBits;
	BYTE* startNormals = (BYTE*)normalsLockedRect.pBits;
	BYTE* startNeighbor    = (BYTE*)neighborLockedRect.pBits;

	int nTetra = mesh->nTetra();

	float testFloat = 0;
	int nModified = 0;

	for (int t=0; t< nTetra; t++){
		D3DXVECTOR4 vertices[3];
		D3DXVECTOR4 normals[3];

		vec3 fourthVertex = mesh->getTetraVertex(t, 3);
		vec3 fourthNormal = mesh->getTetraFaceNormal(t, 3);

		vec3 v[3];
		v[0] = mesh->getTetraVertex(t, 0);
		v[1] = mesh->getTetraVertex(t, 1);
		v[2] = mesh->getTetraVertex(t, 2);

		vec3 n[3];
		n[0] = mesh->getTetraFaceNormal(t, 0);
		n[1] = mesh->getTetraFaceNormal(t, 1);
		n[2] = mesh->getTetraFaceNormal(t, 2);

		int adj0, adj1, adj2, adj3, fAdj;
		mesh->getTetraAdjToTetraFace(t, 0, adj0, fAdj);
		mesh->getTetraAdjToTetraFace(t, 1, adj1, fAdj);
		mesh->getTetraAdjToTetraFace(t, 2, adj2, fAdj);
		mesh->getTetraAdjToTetraFace(t, 3, adj3, fAdj);

		vec3 n1 = (v[2]-v[0])^(v[1]-v[0]);
		n1.normalize();
		vec3 tCentre = (v[0]+v[1]+v[2])/3;
		if((n1 * (fourthVertex - tCentre)) < 0.0){
			// same direction - invert vertices and neighbors
			vec3 aux = v[2];
			v[2] = v[1];
			v[1] =  aux;
			aux  = n[2];
			n[2] = n[1];
			n[1] =  aux;
			int nAux = adj2;
			adj2 = adj1;
			adj1 = nAux;
			nModified++;
		}

		// Loop over the vertices and faces
		for(int i=0; i<3; i++){
			//vec3 pt = mesh->getTetraVertex(t, i);
			vec3 pt = v[i];
			//vec3 normal = mesh->getTetraFaceNormal(t, i);
			vec3 normal = n[i];

			vertices[i] = D3DXVECTOR4((float)pt[0], (float)pt[1], (float)pt[2], 0.0f);
			normals[i] = D3DXVECTOR4((float)normal[0], (float)normal[1], (float)normal[2], 0.0f);
		}

    vertices[0].w = (float)fourthVertex[0];
		vertices[1].w = (float)fourthVertex[1];
		vertices[2].w = (float)fourthVertex[2];

		normals[0].w = (float)fourthNormal[0];
		normals[1].w = (float)fourthNormal[1];
		normals[2].w = (float)fourthNormal[2];

		// Get the scalar value associated with the tetrahedron
		// Need to scale it with largest value so it can be accessed
		// correctly in the transfer function
		double s = mesh->getTetraScalarValue(t);///mesh->_maxScalar;
		if (s > 1.0) {
			s = mesh->getTetraScalarValue(t);
			s = 1.0;
		}
		if (s < 0.0) {
			s = mesh->getTetraScalarValue(t);
			s = 0.0;
		}

		// gather scalar and neighbors
		vec3 grad = mesh->_grad_VEC[t];
		// calculate the gradient scalar
		s += ((-grad) * vec3(vertices[0].x, vertices[0].y, vertices[0].z));
		/*s += ((-grad) * vec3(mesh->getTetraFaceNormal(t, 2)[0],
								mesh->getTetraFaceNormal(t, 2)[1],
								mesh->getTetraFaceNormal(t, 2)[2]));
		*/
		D3DXVECTOR4 scalar((float)grad[0], (float)grad[1], (float)grad[2], (FLOAT) s);
		D3DXVECTOR4 neighbor1, neighbor2;

		//********************************************************************************************************************

// load the neighbor in two channels
		float uf, vf;
		if(adj0 != -1){
			computeUV2DNormalized(adj0, uf, vf, texDimension);
			neighbor1.x = uf;
			neighbor1.y = vf;
		}else{
			neighbor1.x = 0.0f;
			neighbor1.y = 0.0f;
		}

		if(adj1 != -1){
			computeUV2DNormalized(adj1, uf, vf, texDimension);
			neighbor1.z = uf;
			neighbor1.w = vf;
		}else{
			neighbor1.z = 0.0f;
			neighbor1.w = 0.0f;
		}

		if(adj2 != -1){
			computeUV2DNormalized(adj2, uf, vf, texDimension);
			neighbor2.x = uf;
			neighbor2.y = vf;
		}else{
			neighbor2.x = 0.0f;
			neighbor2.y = 0.0f;
		}

		if(adj3 != -1){
			computeUV2DNormalized(adj3, uf, vf, texDimension);
			neighbor2.z = uf;
			neighbor2.w = vf;
		}else{
			neighbor2.z = 0.0f;
			neighbor2.w = 0.0f;
		}
		//********************************************************************************************************************

		// create the methods to load the information on textures, just like the above ones.
		setTetrahedralDataInTexture2D(verticesLockedRect, startVertices, texDimension, t, vertices[0], vertices[1], vertices[2]);
		setTetrahedralDataInTexture2D(normalsLockedRect, startNormals, texDimension, t, normals[0], normals[1], normals[2]);
		setTetrahedralDataInTexture2D(neighborLockedRect, startNeighbor, texDimension, t, neighbor1, neighbor2, scalar);
	}

	(*verticesTEX)->UnlockRect(0);
	(*normalsTEX)->UnlockRect(0);
	(*neighborTEX)->UnlockRect(0);

	nModified += 0;

	return S_OK;
}
