//-----------------------------------------------------------------------------
// File: text.h
// Desc: class to enable drawing of text
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

#ifndef TEXT_H
#define TEXT_H

#include <tchar.h>
#include <D3D9.h>


//-----------------------------------------------------------------------------
// Name: class CText
// Desc: Texture-based font class for doing text in a 3D scene.
//-----------------------------------------------------------------------------
class CText {

  char                    m_aszFontFilename[MAX_PATH];
  IDirect3DDevice9       *m_pD3DDevice;
  IDirect3DTexture9      *m_pTexture;
  IDirect3DVertexBuffer9 *m_pVB;
  int                     m_nTexWidth, m_nTexHeight;
  float                   m_aafTexCoords[136][4];

  //stateblocks for setting and restoring render states
  IDirect3DStateBlock9   *m_pStateBlockSaved;
  IDirect3DStateBlock9   *m_pStateBlockDrawText;

public:

  //text drawing function
  bool DrawText(float fSX, float fSY, DWORD wwColor, const char* aszText);

  //initializing and destroying device-dependent objects
  bool InitDeviceObjects(IDirect3DDevice9 *pd3dDevice);
  bool RestoreDeviceObjects();
  bool InvalidateDeviceObjects();
  bool DeleteDeviceObjects();

  //constructor / destructor
  CText(const char* pszFontFilename);
  ~CText();
};


#endif //TEXT_H


