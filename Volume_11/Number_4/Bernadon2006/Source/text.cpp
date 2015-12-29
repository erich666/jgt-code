//-----------------------------------------------------------------------------
// File: text.cpp
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

#include <D3DX9.h>
#include "../Header/defines.h"
#include "../Header/text.h"


//-----------------------------------------------------------------------------
// Custom vertex types for rendering text
//-----------------------------------------------------------------------------
#define MAX_NUM_VERTICES 50*6

struct FONT2DVERTEX {
  D3DXVECTOR4 p;
  DWORD color;
  float tu, tv;
};

#define D3DFVF_FONT2DVERTEX (D3DFVF_XYZRHW|D3DFVF_DIFFUSE|D3DFVF_TEX1)

inline FONT2DVERTEX InitFont2DVertex(const D3DXVECTOR4& p, D3DCOLOR color,
                                     float tu, float tv) {

  //aux vars
  FONT2DVERTEX v;

  v.p = p;
  v.color = color;
  v.tu = tu;
  v.tv = tv;

  return v;
}


//-----------------------------------------------------------------------------
// Name: CText()
// Desc: Font class constructor
//-----------------------------------------------------------------------------
CText::CText(const char* pszFontFilename) {

  strcpy(m_aszFontFilename, pszFontFilename);

  m_pD3DDevice           = NULL;
  m_pTexture             = NULL;
  m_pVB                  = NULL;

  m_pStateBlockSaved     = NULL;
  m_pStateBlockDrawText  = NULL;
}


//-----------------------------------------------------------------------------
// Name: ~CText()
// Desc: Font class destructor
//-----------------------------------------------------------------------------
CText::~CText() {

  InvalidateDeviceObjects();
  DeleteDeviceObjects();
}


//-----------------------------------------------------------------------------
// Name: InitDeviceObjects()
// Desc: Initializes device-dependent objects, including the vertex buffer used
//       for rendering text and the texture map nWhich stores the font image.
//-----------------------------------------------------------------------------
bool CText::InitDeviceObjects(LPDIRECT3DDEVICE9 pd3dDevice) {

  //aux vars
  int nFontHeight = 20;

  if (D3DXCreateTextureFromFileEx(pd3dDevice, m_aszFontFilename, D3DX_DEFAULT, D3DX_DEFAULT, 1, 0,
    D3DFMT_A8R8G8B8, D3DPOOL_MANAGED, D3DX_DEFAULT, D3DX_DEFAULT, 0xFF000000, NULL, NULL,
    &m_pTexture) != D3D_OK)
    return false;

  m_nTexWidth = 256;
  m_nTexHeight = 256;

  int x, y, dx, dy, nChar;

  dx = 17;
  dy = 20;

  y = 0;
  for(int nRow = 0; nRow < 8; nRow++) {

    x = 0;
    for (int nCol = 0; nCol < 13; nCol++) {

      nChar = 32 + 13 * nRow + nCol;

      m_aafTexCoords[nChar-32][0] = ((float)(x + 0)) / (float)m_nTexWidth;
      m_aafTexCoords[nChar-32][1] = ((float)(y + 0)) / (float)m_nTexHeight;
      m_aafTexCoords[nChar-32][2] = ((float)(x + dx)) / (float)m_nTexWidth;
      m_aafTexCoords[nChar-32][3] = ((float)(y + dy)) / (float)m_nTexHeight;

      x += dx;
    }

    y += dy;
  }

  // Keep a local copy of the device
  m_pD3DDevice = pd3dDevice;

  return true;
}


//-----------------------------------------------------------------------------
// Name: RestoreDeviceObjects()
// Desc:
//-----------------------------------------------------------------------------
bool CText::RestoreDeviceObjects() {

  //aux vars
  HRESULT hr;
  bool bSupportsAlphaBlend;
  int vertexSize, nWhich;
  LPDIRECT3D9 pd3d9 = NULL;
  D3DCAPS9 Caps;
  D3DDISPLAYMODE Mode;
  LPDIRECT3DSURFACE9 pSurf = NULL;
  D3DSURFACE_DESC Desc;

  // Create vertex buffer for the letters
  vertexSize = sizeof(FONT2DVERTEX);
  if(FAILED(hr = m_pD3DDevice->CreateVertexBuffer(MAX_NUM_VERTICES * vertexSize,
    D3DUSAGE_WRITEONLY | D3DUSAGE_DYNAMIC, 0, D3DPOOL_DEFAULT, &m_pVB, NULL)))

    return false;

  bSupportsAlphaBlend = true;
  if(m_pD3DDevice->GetDirect3D(&pd3d9) == D3D_OK) {

    m_pD3DDevice->GetDeviceCaps(&Caps);
    m_pD3DDevice->GetDisplayMode(0, &Mode);
    if(m_pD3DDevice->GetRenderTarget(0, &pSurf) == D3D_OK) {

      pSurf->GetDesc(&Desc);
      if(pd3d9->CheckDeviceFormat(Caps.AdapterOrdinal, Caps.DeviceType, Mode.Format,
        D3DUSAGE_RENDERTARGET | D3DUSAGE_QUERY_POSTPIXELSHADER_BLENDING, D3DRTYPE_SURFACE, 
        Desc.Format) != D3D_OK)

        bSupportsAlphaBlend = false;

      SafeRelease(pSurf);
    }
    SafeRelease(pd3d9);
  }

  // Create the state blocks for rendering text
  for(nWhich=0; nWhich<2; nWhich++) {

    m_pD3DDevice->BeginStateBlock();
    m_pD3DDevice->SetTexture(0, m_pTexture);

    if(bSupportsAlphaBlend) {

      m_pD3DDevice->SetRenderState(D3DRS_ALPHABLENDENABLE, TRUE);
      m_pD3DDevice->SetRenderState(D3DRS_SRCBLEND,   D3DBLEND_SRCALPHA);
      m_pD3DDevice->SetRenderState(D3DRS_DESTBLEND,  D3DBLEND_INVSRCALPHA);
    }
    else
      m_pD3DDevice->SetRenderState(D3DRS_ALPHABLENDENABLE, FALSE);

    m_pD3DDevice->SetRenderState(D3DRS_ZENABLE, FALSE);
    m_pD3DDevice->SetRenderState(D3DRS_ALPHATESTENABLE,  TRUE);
    m_pD3DDevice->SetRenderState(D3DRS_ALPHAREF,         0x08);
    m_pD3DDevice->SetRenderState(D3DRS_ALPHAFUNC,  D3DCMP_GREATEREQUAL);
    m_pD3DDevice->SetRenderState(D3DRS_FILLMODE,   D3DFILL_SOLID);
    m_pD3DDevice->SetRenderState(D3DRS_CULLMODE,   D3DCULL_CCW);
    m_pD3DDevice->SetRenderState(D3DRS_STENCILENABLE,    FALSE);
    m_pD3DDevice->SetRenderState(D3DRS_CLIPPING,         TRUE);
    m_pD3DDevice->SetRenderState(D3DRS_CLIPPLANEENABLE,  FALSE);
    m_pD3DDevice->SetRenderState(D3DRS_VERTEXBLEND,      D3DVBF_DISABLE);
    m_pD3DDevice->SetRenderState(D3DRS_INDEXEDVERTEXBLENDENABLE, FALSE);
    m_pD3DDevice->SetRenderState(D3DRS_FOGENABLE,        FALSE);
    m_pD3DDevice->SetRenderState(D3DRS_COLORWRITEENABLE,
      D3DCOLORWRITEENABLE_RED  | D3DCOLORWRITEENABLE_GREEN |
      D3DCOLORWRITEENABLE_BLUE | D3DCOLORWRITEENABLE_ALPHA);
    m_pD3DDevice->SetTextureStageState(0, D3DTSS_COLOROP,   D3DTOP_MODULATE);
    m_pD3DDevice->SetTextureStageState(0, D3DTSS_COLORARG1, D3DTA_TEXTURE);
    m_pD3DDevice->SetTextureStageState(0, D3DTSS_COLORARG2, D3DTA_DIFFUSE);
    m_pD3DDevice->SetTextureStageState(0, D3DTSS_ALPHAOP,   D3DTOP_MODULATE);
    m_pD3DDevice->SetTextureStageState(0, D3DTSS_ALPHAARG1, D3DTA_TEXTURE);
    m_pD3DDevice->SetTextureStageState(0, D3DTSS_ALPHAARG2, D3DTA_DIFFUSE);
    m_pD3DDevice->SetTextureStageState(0, D3DTSS_TEXCOORDINDEX, 0);
    m_pD3DDevice->SetTextureStageState(0, D3DTSS_TEXTURETRANSFORMFLAGS, D3DTTFF_DISABLE);
    m_pD3DDevice->SetTextureStageState(1, D3DTSS_COLOROP,   D3DTOP_DISABLE);
    m_pD3DDevice->SetTextureStageState(1, D3DTSS_ALPHAOP,   D3DTOP_DISABLE);
    m_pD3DDevice->SetSamplerState(0, D3DSAMP_MINFILTER, D3DTEXF_LINEAR);
    m_pD3DDevice->SetSamplerState(0, D3DSAMP_MAGFILTER, D3DTEXF_LINEAR);
    m_pD3DDevice->SetSamplerState(0, D3DSAMP_MIPFILTER, D3DTEXF_NONE);

    if(nWhich==0)
      m_pD3DDevice->EndStateBlock(&m_pStateBlockSaved);
    else
      m_pD3DDevice->EndStateBlock(&m_pStateBlockDrawText);
  }

  return true;
}


//-----------------------------------------------------------------------------
// Name: InvalidateDeviceObjects()
// Desc: Destroys all device-dependent objects
//-----------------------------------------------------------------------------
bool CText::InvalidateDeviceObjects() {

  SafeRelease(m_pVB);
  SafeRelease(m_pStateBlockSaved);
  SafeRelease(m_pStateBlockDrawText);

  return true;
}


//-----------------------------------------------------------------------------
// Name: DeleteDeviceObjects()
// Desc: Destroys all device-dependent objects
//-----------------------------------------------------------------------------
bool CText::DeleteDeviceObjects() {

  SafeRelease(m_pTexture);
  m_pD3DDevice = NULL;

  return true;
}


//-----------------------------------------------------------------------------
// Name: DrawText()
// Desc: Draws 2D text. Note that fSX and fSY are in pixels
//-----------------------------------------------------------------------------
bool CText::DrawText(float fSX, float fSY, DWORD wwColor, const char* aszText) {

  //aux vars
  const char *pszText;
  float fStartX, tx1, ty1, tx2, ty2, w, h;
  FONT2DVERTEX* pVertices = NULL;
  DWORD dwNumTriangles = 0;
  char szC;

  pszText = aszText;

  if (m_pD3DDevice == NULL)
    return false;

  //setup renderstate
  m_pStateBlockSaved->Capture();
  m_pStateBlockDrawText->Apply();
  m_pD3DDevice->SetFVF(D3DFVF_FONT2DVERTEX);
  m_pD3DDevice->SetPixelShader(NULL);
  m_pD3DDevice->SetStreamSource(0, m_pVB, 0, sizeof(FONT2DVERTEX));

  m_pD3DDevice->SetSamplerState(0, D3DSAMP_MINFILTER, D3DTEXF_LINEAR);
  m_pD3DDevice->SetSamplerState(0, D3DSAMP_MAGFILTER, D3DTEXF_LINEAR);

  //adjust for character spacing
  fStartX = fSX;

  //fill vertex buffer
  m_pVB->Lock(0, 0, (void**)&pVertices, D3DLOCK_DISCARD);

  while(*pszText) {

    szC = *pszText++;

    if(szC == '\n') {

      fSX = fStartX;
      fSY += (m_aafTexCoords[0][3]-m_aafTexCoords[0][1])*m_nTexHeight;
    }

    if((szC-32) < 0 || (szC-32) >= 128-32)
      continue;

    tx1 = m_aafTexCoords[szC-32][0];
    ty1 = m_aafTexCoords[szC-32][1];
    tx2 = m_aafTexCoords[szC-32][2];
    ty2 = m_aafTexCoords[szC-32][3];

    w = (tx2-tx1) *  m_nTexWidth;
    h = (ty2-ty1) * m_nTexHeight;

    if(szC != ' ') {

      *pVertices++ = InitFont2DVertex(D3DXVECTOR4(fSX + 0, fSY + h, 0.9f, 1.0f), wwColor, tx1, ty2);
      *pVertices++ = InitFont2DVertex(D3DXVECTOR4(fSX + 0, fSY + 0, 0.9f, 1.0f), wwColor, tx1, ty1);
      *pVertices++ = InitFont2DVertex(D3DXVECTOR4(fSX + w, fSY + h, 0.9f, 1.0f), wwColor, tx2, ty2);
      *pVertices++ = InitFont2DVertex(D3DXVECTOR4(fSX + w, fSY + 0, 0.9f, 1.0f), wwColor, tx2, ty1);
      *pVertices++ = InitFont2DVertex(D3DXVECTOR4(fSX + w, fSY + h, 0.9f, 1.0f), wwColor, tx2, ty2);
      *pVertices++ = InitFont2DVertex(D3DXVECTOR4(fSX + 0, fSY + 0, 0.9f, 1.0f), wwColor, tx1, ty1);
      dwNumTriangles += 2;

      if(dwNumTriangles*3 > (MAX_NUM_VERTICES-6)) {

        //unlock, render, and relock the vertex buffer
        m_pVB->Unlock();
        m_pD3DDevice->DrawPrimitive(D3DPT_TRIANGLELIST, 0, dwNumTriangles);
        pVertices = NULL;
        m_pVB->Lock(0, 0, (void**)&pVertices, D3DLOCK_DISCARD);
        dwNumTriangles = 0L;
      }
    }

    fSX += w;
  }

  //unlock and render the vertex buffer
  m_pVB->Unlock();
  if(dwNumTriangles > 0)
    m_pD3DDevice->DrawPrimitive(D3DPT_TRIANGLELIST, 0, dwNumTriangles);

  //restore the modified renderstates
  m_pStateBlockSaved->Apply();

  return true;
}
