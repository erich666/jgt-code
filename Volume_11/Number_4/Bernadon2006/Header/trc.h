//-----------------------------------------------------------------------------
// File: trc.h
// Desc: Tiled Ray Casting program
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

#include "text.h"

#ifndef TRC_H
#define TRC_H

//-----------------------------------------------------------------------------
//Name: HRCApp
//Desc: class that executes the Ray Casting algorithm
//-----------------------------------------------------------------------------
class HRCApp {

protected:
  //class to draw text
  CText                   *m_pFont;

  //class to handle tetrahedra meshes
  IndexedTetraSet         *m_its;

  //class to handle 3D transfer functions
  LUT3D                   *m_poLut3D;

  //class to handle 2D transfer functions
  LUT2D                   *m_poLut2D;

  //class to handle 1D transfer function - used to create 2D and 3D TF
  LUT1D                   m_oTF;

  //direct 3d main interface and device
  IDirect3D9              *m_pD3D;
  IDirect3DDevice9        *m_pD3DDevice;
  D3DPRESENT_PARAMETERS    m_stD3DPresentParameters;

  //mesh textures
  IDirect3DTexture9       *m_pNewVerticesTEX;
  IDirect3DTexture9       *m_pNewNormalsTEX;
  IDirect3DTexture9       *m_pNewNeighborTEX;
  int				               m_meshTextureSize;
  float                    m_fTextureIncr1, m_fTextureIncr2; //texel's size

  //transfer functions 
  IDirect3DVolumeTexture9 *m_pLutTEX;
  IDirect3DTexture9       *m_pLut2DTEX;
  int                      m_nLutType;

  //traversal textures and surfaces
  IDirect3DTexture9       *m_pCurrentCellTEX[2];
  IDirect3DTexture9       *m_pColorTEX[2];

  IDirect3DSurface9       *m_pCurrentCellTARGET[2];
  IDirect3DSurface9       *m_pColorTARGET[2];

  //frame buffer
  IDirect3DSurface9       *m_pOriginalTarget;
  D3DSURFACE_DESC          m_stBBDesc;

  //screen-sized rectangle buffers
  IDirect3DVertexBuffer9  *m_rectangle;
  IDirect3DVertexBuffer9  *m_pRectBuffer;

  //boundary faces d3d mesh
  IDirect3DVertexBuffer9  *m_pBoundaryBuffer;
  Vertex			            *m_pBoundaryFaces;
  int				               m_nBoundaryFaces;

  //main effect interface and technique handles
  ID3DXEffect             *m_pEffect;
  D3DXHANDLE               m_hTechniqueFH, m_hTechniqueGDP, m_hTechniqueRC,
                           m_hTechniqueOT, m_hTechniqueRCT, m_hTechniqueRDL,
                           m_hTechniqueRFC, m_hTechniqueR3DT;

  //occlusion queries to control program termination
  IDirect3DQuery9         **m_ppOcclusionQuery;

  //depth peeling structures
  vector<IDirect3DTexture9*> m_apDepthVector;
  vector<IDirect3DSurface9*> m_apDepthSurfaceVector;
  int depthCounter;

  //control the number of passes
  int m_nNumPass, m_nPassToOcclusion;

  //control result view
  int m_nView;

  //eye point and interpolation parameter
  D3DXVECTOR4 m_afEye, m_afLrpParam;

  //control creation of queries and resize of tile partitioning
  bool m_bFirstTime;

  //control the number of rectangles drawn in a scene
  int  m_nHRect, m_nVRect, m_nQuantFinishedRect;
  int *m_pPendingResult;
  int *m_pPassCounter;
  int *m_pnFinishedRect;

  //control program behaviour
  bool m_nManualPass, m_nOcclusionFirstPass;

  //keep track of mouse position
  int m_nCurrentMousePositionX, m_nCurrentMousePositionY;

  //camera parameters
  float m_fEyeX;
  float m_fEyeY;
  float m_fEyeZ;

  float m_fUpX;
  float m_fUpY;
  float m_fUpZ;

  //transformation matrices
  D3DXMATRIX m_stMatView, m_stMatProj;

  //mesh bounding box
  float m_fBBDiagonal;
  float m_fBBMaxX, m_fBBMaxY, m_fBBMaxZ, m_fBBMinX, m_fBBMinY, m_fBBMinZ;

  // Indicates which one is the current source and target traversals
  int m_nSourceTraversal, m_nTargetTraversal;

protected:

  //initialize d3d objects only once
  HRESULT InitDeviceObjects();

  //restore d3d objects after device's reset
  HRESULT RestoreDeviceObjects();

  //release d3d objects to enable the device's reset operation
  HRESULT InvalidateDeviceObjects();

  //final release to all device objects
  HRESULT DeleteDeviceObjects();

  //main render loop
  HRESULT Render();

  //rende boundary faces to our first level
  HRESULT RenderIntoTraversalStructures();

  //generate all depth levels
  HRESULT GenerateDepthPeeling();

  //ray casting procedure
  HRESULT RayCasting(int nPass);

  //occlusion test
  HRESULT OcclusionTest(int oPass);

  //render the final result image
  HRESULT RenderColorTarget(int passId);

  //render tetrahedra faces using its index as color
  HRESULT RenderFaceColored(int passId);

  //render the generated depth levels1
  HRESULT RenderDepthLevels(int passId);

  //create vertex buffers structures
  HRESULT CreateVertexBuffers();

  //create render target structures
  bool CreateTargetTextures();

  //release render target structures
  bool  ReleaseTargetTextures();

  //change screen tile partitioning
  HRESULT ChangeScreenRect();
  HRESULT ReleaseScreenRect();

  //retrieve techniques handles from effect file
  HRESULT LoadTechniques();

  //set render states
  HRESULT SetRenderStates();

  //set effect matrices
  HRESULT SetCommonMatrix();

  //set effect textures
  HRESULT SetCommonTextures();

  //set effect vectors
  HRESULT SetCommonVectors();

  //set projection matices
  void SetupViewProjectionMatrices();

  //load tetrahedra mesh
  void LoadMesh();

  //erase remaining tile's counter
  void ClearFinishedRect();
  void ClearPendingResults();

  //generate program statistics
  void SaveStatistics();

  //handles mouse messages to move the camera
  void MoveCameraU(float fDelta);
  void MoveCameraV(float fDelta);
  void MoveCameraN(float fDelta);
  void RotCameraN(float fDelta);

  //reset device
  bool Reset();

public:

  //class constructor
  HRCApp();

  //class destructor
  ~HRCApp();

  //initialize program instance
  bool Create(HINSTANCE hInst);

  //main program loop
  int Run();

  //initialize camera position
  void SetCameraPosZ(float fPos) { m_fEyeZ = fPos; }

  //class message procedure
  LRESULT MsgProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
};


#endif //TRC_H