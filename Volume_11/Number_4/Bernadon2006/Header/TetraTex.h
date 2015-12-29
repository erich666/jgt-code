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

#ifndef TETRATEX_H
#define TETRATEX_H

#define WIN32_LEAN_AND_MEAN   // Exclude the stuff we don't need in libraies.
#define WIN32_EXTRA_LEAN      // Exclude even further.

#include "defines.h"

#include <stdio.h>
#include <d3dx9.h>             // The main header you will need for Direct3d.
#include "resource.h"
#include "tetra.h"

HRESULT LoadMeshIntoNewTextures(IndexedTetraSet *mesh, 
				 LPDIRECT3DDEVICE9 D3D_Device,
				 LPDIRECT3DTEXTURE9 *verticesTEX,
				 LPDIRECT3DTEXTURE9 *normalsTEX,
				 LPDIRECT3DTEXTURE9 *neighborTEX,
				 int texDimension);

void DisplayShaderERR(LPD3DXBUFFER ErrorMessages, char *filename, HRESULT hr);


#endif