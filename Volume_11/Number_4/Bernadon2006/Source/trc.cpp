//-----------------------------------------------------------------------------
// File: trc.cpp
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

#pragma comment (lib, "d3dx9.lib")

#define STRICT

#include <Windows.h>
#include <commctrl.h>
#include <D3DX9.h>

#include <sys/timeb.h>
#include <time.h>

#include "../Header/defines.h"

#include "../Header/text.h"
#include "../Header/resource.h"
#include "../Header/TetraTex.h"
#include "../Header/lut.h"

#ifndef WM_MOUSEWHEEL
  #define WM_MOUSEWHEEL 0x020A
#endif

#include "../Header/trc.h"

//main window
HWND g_hMainWindow;

//program object
HRCApp *g_poD3DApp = NULL;

//command line arguments
char g_acFilenames[NARGS][100];

//vertex formats
#define D3DFVF_D3DVERTEX (D3DFVF_XYZ|D3DFVF_NORMAL)
#define D3DFVF_CUSTOMVERTEX (D3DFVF_XYZ|D3DFVF_TEX1|D3DFVF_TEX2|D3DFVF_TEX3)


//-----------------------------------------------------------------------------
// Name: checkD3D
// Desc: checks if the result of a d3d operation is correct
//-----------------------------------------------------------------------------
void checkD3D(HRESULT hr) {
  if(hr == E_OUTOFMEMORY){
    MessageBox(NULL, "D3D OUTOFMEMORY error.", NULL, MB_OK);
    exit(-1);
  }
  if(hr == D3DERR_NOTAVAILABLE){
    MessageBox(NULL, "D3D NOTAVAILABLE error.", NULL, MB_OK);
    exit(-1);
  }
  if (hr != D3D_OK)  {
    MessageBox(NULL, "Invalid D3D Call", NULL, MB_OK);
    exit(-1);
  }
}

//------------------------------------------------------------------------------
// DisplayShaderERR
// Puts up a message box that shows the err message give at shader assemble
// time.
//------------------------------------------------------------------------------
void DisplayShaderERR(LPD3DXBUFFER ErrorMessages, char *filename, HRESULT hr) { 

  char buffer[512], *c;
  if(ErrorMessages != NULL)
    c = (char *)ErrorMessages->GetBufferPointer();
  else 
    c = "App: No Error Message in error message buffer";

  sprintf(buffer,"|%s|%s %X", filename, c, hr);
  MessageBox(NULL, buffer, filename, MB_OK);
}


//-----------------------------------------------------------------------------
// MsgProc
// simply pass the message to our main class
//-----------------------------------------------------------------------------
LRESULT WINAPI MsgProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {

  if ((msg == WM_CLOSE) ||
      ((msg == WM_KEYDOWN) && (wParam == VK_ESCAPE)))
    SafeDelete(g_poD3DApp);

  if (g_poD3DApp)
    g_poD3DApp->MsgProc(hWnd, msg, wParam, lParam);

  return DefWindowProc(hWnd, msg, wParam, lParam);
}


//-----------------------------------------------------------------------------
// Name: WinMain()
// Desc: Entry point to the program. Initializes everything, and goes into a
//       message-processing loop. Idle time is used to render the scene.
//-----------------------------------------------------------------------------
INT WINAPI WinMain(HINSTANCE hInst, HINSTANCE, LPSTR s, INT in)
{
  //aux vars
  float fZ;
  int i, j, k;
  i = 0;
  j = 0;
  k = 0;

  //copy g_acFilenames
  g_acFilenames[STATISTICS][0] = NULL;

  for(i=0; i < (int)strlen(s)+1; i++){
    if(s[i] != ' '){
      g_acFilenames[k][j] = s[i];
      j++;
    }else{
      g_acFilenames[k][j] = NULL;
      k++;
      j = 0;
    }
  }

  WNDCLASSEX stWndClass = { sizeof(WNDCLASSEX), CS_CLASSDC, MsgProc, 0L, 0L,
    GetModuleHandle(NULL), NULL, NULL, NULL, NULL,
    "Tetrahedron Ray Casting", NULL };
  RegisterClassEx(&stWndClass);

  g_hMainWindow = CreateWindow("Tetrahedron Ray Casting", "Tetrahedron Ray Casting",
    WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX,
    50, 50, SCREEN_DIMENSION, SCREEN_DIMENSION, GetDesktopWindow(), NULL,
    stWndClass.hInstance, NULL);

  RECT stRect;
  GetWindowRect(g_hMainWindow, &stRect);

  SetWindowPos(g_hMainWindow, HWND_NOTOPMOST,
               stRect.left, stRect.top,
               (stRect.right - stRect.left),
               (stRect.bottom - stRect.top),
               SWP_SHOWWINDOW);

  g_poD3DApp = new HRCApp();

  //init eye position
  fZ = (float)atof(g_acFilenames[EYEPOS]);
  if(fZ != 0.0f){
    g_poD3DApp->SetCameraPosZ(fZ);
  }

  InitCommonControls();

  if(FAILED(g_poD3DApp->Create(hInst)))
    return 0;

  return g_poD3DApp->Run();
}


//-----------------------------------------------------------------------------
// readTetrahedralMesh
// Load the tetrahedral mesh into the internal data structure
//-----------------------------------------------------------------------------
IndexedTetraSet* readTetrahedralMesh(char *filename) {

  IndexedTetraSet *t = new IndexedTetraSet();

  if (t->readFromFile(filename, OFF)==false) {

    printf("\nUnable to open mesh %s", filename);
    exit(-1);
  }
  t->computeFaces();
  t->computeFacesPlaneEquations();

  return t;
}


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
bool HRCApp::Create(HINSTANCE hInst) {

  SetClassLong(g_hMainWindow, GCL_HCURSOR, HandleToLong(LoadCursor(NULL, IDC_SIZEALL)));

  // Initialize the app's device-dependent objects
  HRESULT hr;
  hr = InitDeviceObjects();
  if(hr != S_OK)
    DeleteDeviceObjects();
  else {

    hr = RestoreDeviceObjects();
    if(hr != S_OK)
      InvalidateDeviceObjects();
    else
      return true;
  }

  return false;
}


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
int HRCApp::Run() {

  MSG msg;
  ZeroMemory(&msg, sizeof(msg));
  Reset();
  while(msg.message!=WM_QUIT) {

    if(PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE)) {

      TranslateMessage(&msg);
      DispatchMessage(&msg);
    }
    else
      g_poD3DApp->Render();
  }

  return (INT)msg.wParam;
}


//-----------------------------------------------------------------------------
// Name: ~HRCApp()
// Desc: Class destructor
//-----------------------------------------------------------------------------
HRCApp::~HRCApp(){

  //aux vars
  int nTotalReferences;

  DeleteDeviceObjects();

  SafeDelete(m_pFont);
  SafeDelete(m_poLut3D);
  SafeDelete(m_its);
  SafeDelete(m_pBoundaryFaces);

  SafeRelease(m_pD3DDevice);
  nTotalReferences = m_pD3D->Release();

  if (nTotalReferences > 0)
    MessageBox(NULL, "There are not released D3D references", "Bad cleanup",
    MB_OK | MB_ICONERROR);

  DestroyWindow(g_hMainWindow);
  PostQuitMessage(0);
  g_hMainWindow = NULL;
}


//-----------------------------------------------------------------------------
// Name: SetupViewProjectionMatrices()
// Desc: Sets up the world, view, and projection transform matrices.
//-----------------------------------------------------------------------------
VOID HRCApp::SetupViewProjectionMatrices() {

  //aux vars
  float fBiggest, fSmallest;

  //our camera will always look at (0,0,0)
  m_afEye = D3DXVECTOR4(m_fEyeX, m_fEyeY, m_fEyeZ, 1.0f);
  D3DXVECTOR3 stEyePt(m_fEyeX, m_fEyeY, m_fEyeZ);
  D3DXVECTOR3 stLookatPt(0.0f, 0.0f, 0.0f);
  D3DXVECTOR3 stUpVec(m_fUpX, m_fUpY, m_fUpZ);

  D3DXMatrixLookAtLH(&m_stMatView, &stEyePt, &stLookatPt, &stUpVec);
  m_pD3DDevice->SetTransform(D3DTS_VIEW, &m_stMatView);

  //set up projection matrix
  fBiggest = m_stBBDesc.Width > m_stBBDesc.Height ?
    (float)m_stBBDesc.Width : (float)m_stBBDesc.Height;
  fSmallest = m_stBBDesc.Width < m_stBBDesc.Height ?
    (float)m_stBBDesc.Width : (float)m_stBBDesc.Height;

  D3DXMatrixPerspectiveLH(&m_stMatProj, (float)m_stBBDesc.Width/fSmallest,
    (float)m_stBBDesc.Height/fSmallest, 1.0f, 2000.0f);

  m_pD3DDevice->SetTransform(D3DTS_PROJECTION, &m_stMatProj);
}


//-----------------------------------------------------------------------------
// Name: HRCApp()
// Desc: Application constructor. Sets attributes for the app.
//-----------------------------------------------------------------------------
HRCApp::HRCApp() {

  RECT stMainRect;

  //create D3D
  m_pD3D = Direct3DCreate9(D3D_SDK_VERSION);
  if(m_pD3D == NULL) {

    DisplayError("Could not create Direct3D.\nProgram will exit.");
    exit(-1);
  }

  //create device
  ZeroMemory(&m_stD3DPresentParameters, sizeof(m_stD3DPresentParameters));

  GetClientRect(g_hMainWindow, &stMainRect);

  m_stD3DPresentParameters.Windowed = TRUE;
  m_stD3DPresentParameters.BackBufferCount = 1;
  m_stD3DPresentParameters.MultiSampleType = D3DMULTISAMPLE_NONE;
  m_stD3DPresentParameters.MultiSampleQuality = 0;
  m_stD3DPresentParameters.SwapEffect = D3DSWAPEFFECT_DISCARD;
  m_stD3DPresentParameters.BackBufferFormat = D3DFMT_X8R8G8B8;
  m_stD3DPresentParameters.EnableAutoDepthStencil = TRUE;
  m_stD3DPresentParameters.AutoDepthStencilFormat = D3DFMT_D16;
  m_stD3DPresentParameters.Flags = D3DPRESENTFLAG_DISCARD_DEPTHSTENCIL;
  m_stD3DPresentParameters.hDeviceWindow = g_hMainWindow;

  m_stD3DPresentParameters.BackBufferWidth  = stMainRect.right - stMainRect.left;
  m_stD3DPresentParameters.BackBufferHeight = stMainRect.bottom - stMainRect.top;
  m_stD3DPresentParameters.FullScreen_RefreshRateInHz = 0;
  m_stD3DPresentParameters.PresentationInterval = D3DPRESENT_INTERVAL_IMMEDIATE;

  if(m_pD3D->CreateDevice(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, g_hMainWindow,
                          D3DCREATE_PUREDEVICE | D3DCREATE_HARDWARE_VERTEXPROCESSING,
                          &m_stD3DPresentParameters, &m_pD3DDevice) != S_OK) {

    DisplayError("Could not create D3D device.\nProgram will exit.");
    SafeRelease(m_pD3D);
    exit(-1);
  }

  // font
  m_pFont = new CText("Resource/fontes.png");

  m_its = readTetrahedralMesh(g_acFilenames[MESHFILE]);

  m_oTF.BuildTransferFunction(g_acFilenames[TFFILE], g_acFilenames[TFOPFILE],
    m_its->_maxScalar, m_its->_minScalar, m_its->_maxDistance);

  //choose a transfer function
  if(_strnicmp(g_acFilenames[LUTPOS], "3D", 2) == 0)
    // lut 3D
    m_nLutType = 0;
  else if(_strnicmp(g_acFilenames[LUTPOS], "2Df", 3) == 0)
    // lut 2D fixa
    m_nLutType = 1;
  else
    // lut 2D "slice"
    m_nLutType = 2;

  switch(m_nLutType) {

    case 0: // 3D transfer function
      m_poLut3D = new LUT3D(sizeTF, sizeTF, sizeTF);
      m_poLut3D->ComputeLUT3D(m_oTF._tf, m_its->_maxDistance);
      break;
    case 1:	// 2D fixed transfer function
      m_poLut2D = new LUT2D();
      m_poLut2D->ComputeLUT2DFixed();
      break;
    case 2: // 2D slice of 3D
      m_poLut2D = new LUT2D();
      m_poLut3D = new LUT3D(sizeTF, sizeTF, sizeTF);
      m_poLut3D->ComputeLUT3D(m_oTF._tf, m_its->_maxDistance);
      m_poLut2D->ComputeLUT2D(m_poLut3D, 127);
      break;
  }

  //init mesh bounding box
  float m_fBBDiagonal = 1.0f;
  float m_fBBMaxX = 1.0f, m_fBBMaxY = 1.0f, m_fBBMaxZ = 1.0f, 
  m_fBBMinX = -1.0f, m_fBBMinY = -1.0f, m_fBBMinZ = -1.0f;

  //init camera parameters
  m_fEyeX = 0.0f;
  m_fEyeY = 0.0f;
  m_fEyeZ = -12.0f;

  m_fUpX = 0.0f;
  m_fUpY = 1.0f;
  m_fUpZ = 0.0f;

  //init source and target textures
  m_nSourceTraversal = 0;
  m_nTargetTraversal = 1;

  m_nView = 0;
  m_pBoundaryBuffer = NULL;
  m_pEffect         = NULL;
  m_pBoundaryFaces  = NULL;
  m_pOriginalTarget = NULL;
}


//-----------------------------------------------------------------------------
// Name: HRCApp::LoadMesh()
// Desc: Application constructor. Sets attributes for the app.
//-----------------------------------------------------------------------------
void HRCApp::LoadMesh() {

  HRESULT hr = LoadMeshIntoNewTextures(m_its, m_pD3DDevice, &m_pNewVerticesTEX, 
    &m_pNewNormalsTEX, &m_pNewNeighborTEX, m_meshTextureSize);

  float diffX = (float)(m_its->_maxX - m_its->_minX);
  float diffY = (float)(m_its->_maxY - m_its->_minY);
  float diffZ = (float)(m_its->_maxZ - m_its->_minZ);

  m_fBBDiagonal = sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ);

  m_fBBMaxX = (float)(m_its->_maxX);
  m_fBBMaxY = (float)(m_its->_maxY);
  m_fBBMaxZ = (float)(m_its->_maxZ); 
  m_fBBMinX = (float)(m_its->_minX);
  m_fBBMinY = (float)(m_its->_minY);
  m_fBBMinZ = (float)(m_its->_minZ);

  if(m_nLutType == 0){	//m_poLut3D
    // Create LUT texture
    hr = D3DXCreateVolumeTexture(m_pD3DDevice, sizeTF, sizeTF, sizeTF, 1, 0, 
      D3DFMT_A8B8G8R8, D3DPOOL_MANAGED, &m_pLutTEX);
    if (hr != D3D_OK) {
      MessageBox(NULL, "Unable to create LUT texture", NULL, MB_OK);
      return;
    }

    D3DLOCKED_BOX lutLockedBox;
    if (FAILED(hr = m_pLutTEX->LockBox(0, &lutLockedBox, 0, 0))) return;
    BYTE* startLUT = (BYTE*)lutLockedBox.pBits;

    int dj, dk;
    for (int k=0; k < sizeTF; k++) {
      dk = lutLockedBox.SlicePitch * k;
      for (int j=0; j < sizeTF; j++) {
        dj = lutLockedBox.RowPitch * j;
        for (int i=0; i< sizeTF; i++) {
          lutLockedBox.pBits = startLUT + dk + dj;
          int r = (int)(m_poLut3D->get(i,j,k)[0] * 255);
          int g = (int)(m_poLut3D->get(i,j,k)[1] * 255);
          int b = (int)(m_poLut3D->get(i,j,k)[2] * 255);
          int a = (int)(m_poLut3D->get(i,j,k)[3] * 255);
          D3DCOLOR c = D3DCOLOR_RGBA(r,g,b,a);
          ((D3DCOLOR*)lutLockedBox.pBits)[i] = c;
        }
      }
    }

    m_pLutTEX->UnlockBox(0);

  }else{		//m_poLut2D - any

    // Create 2D LUT texture
    hr = D3DXCreateTexture(m_pD3DDevice, sizeTF, sizeTF, 1, 0, 
      D3DFMT_A8B8G8R8, D3DPOOL_MANAGED, &m_pLut2DTEX);
    if (hr != D3D_OK) {
      MessageBox(NULL, "Unable to create LUT texture", NULL, MB_OK);
      return;
    }

    D3DLOCKED_RECT lut2DLockedRect;
    if (FAILED(hr = m_pLut2DTEX->LockRect(0, &lut2DLockedRect, 0, 0))) return;
    BYTE* startlut2D = (BYTE*)lut2DLockedRect.pBits;

    for (int j=0; j < sizeTF; j++) {
      for (int i=0; i< sizeTF; i++) {
        int r = (int)(m_poLut2D->_lutTF2D[i][j][0] * 255);
        int g = (int)(m_poLut2D->_lutTF2D[i][j][1] * 255);
        int b = (int)(m_poLut2D->_lutTF2D[i][j][2] * 255);
        int a = (int)(m_poLut2D->_lutTF2D[i][j][3] * 255);
        D3DCOLOR c = D3DCOLOR_RGBA(r,g,b,a);

        lut2DLockedRect.pBits = startlut2D + j*lut2DLockedRect.Pitch;
        ((D3DCOLOR*)lut2DLockedRect.pBits)[i] = c;
      }
    }

    m_pLut2DTEX->UnlockRect(0);
  }
}


//-----------------------------------------------------------------------------
// Name: RenderIntoTraversalStructures()
// Desc: Render the boundary faces into a texture
//-----------------------------------------------------------------------------
HRESULT HRCApp::RenderIntoTraversalStructures()
{
  // Set render targets
  // IMPORTANT: a render target clear actually CLEAR ALL TARGETS, not
  // just the current one
  m_pD3DDevice->SetRenderTarget(0,m_pCurrentCellTARGET[m_nTargetTraversal]);
  // clear the color target here, once per frame
  m_pD3DDevice->SetRenderTarget(1, m_pColorTARGET[m_nTargetTraversal]);
  m_pD3DDevice->SetRenderTarget(2, m_pColorTARGET[m_nSourceTraversal]);
  m_pD3DDevice->Clear(0L, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER,
    D3DCOLOR_RGBA(0,0,0,0), 1.0f, 0L);

  m_pD3DDevice->SetRenderTarget(1, NULL);
  m_pD3DDevice->SetRenderTarget(2, NULL);

  // render pass
  if(SUCCEEDED(m_pD3DDevice->BeginScene()))
  {
    UINT numPasses;
    UINT iPass;

    //clear the screen tiles
    ClearFinishedRect();

    //*************************************************************************
    // First hit technique
    //*************************************************************************

    m_pEffect->SetTechnique(m_hTechniqueFH);

    m_pEffect->Begin(&numPasses, 0);
    for(iPass = 0; iPass < numPasses; iPass ++)
    {
      m_pEffect->BeginPass(iPass);
      m_pD3DDevice->SetFVF(D3DFVF_D3DVERTEX);
      m_pD3DDevice->SetStreamSource(0, m_pBoundaryBuffer, 0, sizeof(Vertex));
      m_pD3DDevice->DrawPrimitive(D3DPT_TRIANGLELIST, 0, m_nBoundaryFaces);
      m_pEffect->EndPass();
    }
    m_pEffect->End();

    // End the scene
    m_pD3DDevice->EndScene();
  }

  m_pD3DDevice->SetRenderTarget(0, m_pOriginalTarget);
  m_pD3DDevice->SetRenderTarget(1, NULL);

  GenerateDepthPeeling();

  // Switch the texture targets
  m_nSourceTraversal = !m_nSourceTraversal;
  m_nTargetTraversal = !m_nTargetTraversal;

  return S_OK;
}


//-----------------------------------------------------------------------------
// Name: GenerateDepthPeeling()
// Desc: generate all the depth levels from our model
//-----------------------------------------------------------------------------
HRESULT HRCApp::GenerateDepthPeeling()
{
  LPDIRECT3DTEXTURE9 currentDepthTex;

  currentDepthTex = m_pCurrentCellTEX[m_nTargetTraversal];
  DWORD quantFrag = 100;

  depthCounter = 0;

  while(depthCounter < MAX_DEPTH)
  {
    checkD3D(m_pD3DDevice->SetRenderTarget(0, m_apDepthSurfaceVector[depthCounter]));
    checkD3D(m_pD3DDevice->SetRenderTarget(1, NULL));

    checkD3D(m_pD3DDevice->Clear(0L,NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER,
      D3DCOLOR_RGBA(0, 0, 255, 255), 1.0f, 0L));

    checkD3D(m_pD3DDevice->BeginScene());

    UINT numPasses;
    UINT iPass;

    //*************************************************************************
    // Depth Peeling technique
    //*************************************************************************
    checkD3D(m_pEffect->SetTechnique(m_hTechniqueGDP));

    checkD3D(m_pEffect->SetTexture("tCurrentCell", currentDepthTex));

    m_pEffect->Begin(&numPasses, 0);
    for(iPass = 0; iPass < numPasses; iPass++)
    {
      m_pEffect->BeginPass(iPass);
      m_pD3DDevice->SetFVF(D3DFVF_D3DVERTEX);
      m_pD3DDevice->SetStreamSource(0, m_pBoundaryBuffer, 0, sizeof(Vertex));
      m_pD3DDevice->DrawPrimitive(D3DPT_TRIANGLELIST, 0, m_nBoundaryFaces);
      m_pEffect->EndPass();
    }
    m_pEffect->End();

    m_pD3DDevice->EndScene();

    if(depthCounter < MAX_DEPTH){
      currentDepthTex = m_apDepthVector[depthCounter];
    }
    depthCounter++;
  }

  return S_OK;
}


//-----------------------------------------------------------------------------
// Name: RayCasting()
// Desc: Advance one ray at a time inside the tetrahedral mesh
//-----------------------------------------------------------------------------
HRESULT HRCApp::RayCasting(int nPass)
{
  // Set the texture as the render targets
  // set a render target and clear it and the viewport
  m_pD3DDevice->SetRenderTarget(0, m_pCurrentCellTARGET[m_nTargetTraversal]);
  m_pD3DDevice->SetRenderTarget(1, NULL);
  m_pD3DDevice->Clear(0L, NULL, D3DCLEAR_TARGET,
    D3DCOLOR_RGBA(0,0,0,0), 1.0f, 0L);
  m_pD3DDevice->SetRenderTarget(1, m_pColorTARGET[m_nTargetTraversal]);

  // Tell Direct 3D we are now going to start drawing.
  UINT numPasses, iPass;
  m_pD3DDevice->BeginScene();
  HRESULT hr;

  //*************************************************************************
  // Ray Casting technique
  //*************************************************************************
  //	m_pEffect->SetTechnique(m_hTechniqueRC);
  // this is called just once, and is not removed. If other technique remove
  // it, that one has to set it back to its place after finishing

  // Make the mesh and traversal textures  available
  hr = m_pEffect->SetTexture("tCurrentCell", m_pCurrentCellTEX[m_nSourceTraversal]);
  hr = m_pEffect->SetTexture("tColor", m_pColorTEX[m_nSourceTraversal]);

  m_pD3DDevice->SetFVF (D3DFVF_CUSTOMVERTEX);

  DWORD quantFrag;
  m_pEffect->Begin(&numPasses, 0);
  for(iPass = 0; iPass < numPasses; iPass ++) {

    m_pEffect->BeginPass(iPass);
    // This will bind the vertex data in the buffer to the Direct3D device.
    m_pD3DDevice->SetStreamSource(0, m_pRectBuffer, 0, sizeof(CUSTOMVERTEX2));
    m_pD3DDevice->SetFVF(D3DFVF_CUSTOMVERTEX);
    for(int yRect=0; yRect<(int)m_nVRect; yRect++)
      for(int xRect=0; xRect<(int)m_nHRect; xRect++)
        if(!m_pnFinishedRect[yRect*m_nHRect + xRect]) {

          m_pPassCounter[yRect*m_nHRect+xRect]++;

          // starts a occlusion query if needed
          m_pD3DDevice->DrawPrimitive(D3DPT_TRIANGLESTRIP, 
            ((1+m_nHRect)*2*yRect+2*xRect), 2);

          if (m_pPendingResult[yRect*m_nHRect+xRect]){
            if(m_ppOcclusionQuery[yRect*m_nHRect+xRect]->GetData(
              (void*)&quantFrag, sizeof(DWORD), D3DGETDATA_FLUSH) == S_OK){
                m_pPendingResult[yRect*m_nHRect+xRect] = 0;
                if(quantFrag == 0){
                  m_pnFinishedRect[yRect*m_nHRect + xRect] = true;
                  m_nQuantFinishedRect--;
                  continue;
                }
              }
          } // pendigResult
        } // m_pnFinishedRect
        m_pEffect->EndPass();
  }

  m_pEffect->End();

  // We are done drawing for this scene.
  m_pD3DDevice->EndScene();

  // Switch the texture targets
  m_nSourceTraversal = !m_nSourceTraversal;
  m_nTargetTraversal = !m_nTargetTraversal;

  return S_OK;
}


//-----------------------------------------------------------------------------
// Name: OcclusionTest()
// Desc: Test if the fragments still are written in the frame buffer
//-----------------------------------------------------------------------------
HRESULT HRCApp::OcclusionTest(int oPass)
{
  // Set the texture as the render targets
  m_pD3DDevice->SetRenderTarget(1, NULL);
  m_pD3DDevice->SetRenderTarget(0, m_pOriginalTarget);

  if(m_nOcclusionFirstPass){
    m_pD3DDevice->Clear(0L, NULL, D3DCLEAR_ZBUFFER,
      D3DCOLOR_RGBA(0,0,0,255), 1.0f, 0L);
    m_nOcclusionFirstPass = false;
  }

  // Tell Direct 3D we are now going to start drawing.
  UINT numPasses, iPass;
  m_pD3DDevice->BeginScene();
  HRESULT hr;

  //*************************************************************************
  // Occlusion technique
  //*************************************************************************
  m_pEffect->SetTechnique(m_hTechniqueOT);

  // Make the mesh and traversal textures  available
  hr = m_pEffect->SetTexture("tCurrentCell", m_pCurrentCellTEX[m_nSourceTraversal]);

  m_pD3DDevice->SetFVF (D3DFVF_CUSTOMVERTEX);

  m_pEffect->Begin(&numPasses, 0);
  for(iPass = 0; iPass < numPasses; iPass ++)
  {
    m_pEffect->BeginPass(iPass);

    //This will bind the vertex data in the buffer to the Direct3D device.
    m_pD3DDevice->SetStreamSource(0, m_pRectBuffer, 0, sizeof(CUSTOMVERTEX2));
    m_pD3DDevice->SetFVF(D3DFVF_CUSTOMVERTEX);

    for(int yRect=0; yRect<(int)m_nVRect; yRect++)
      for(int xRect=0; xRect<(int)m_nHRect; xRect++)
      {
        if (!m_pPendingResult[yRect*m_nHRect+xRect] && !m_pnFinishedRect[yRect*m_nHRect + xRect])
        {
          if(iPass > 0){
            m_ppOcclusionQuery[yRect*m_nHRect+xRect]->Issue(D3DISSUE_BEGIN);
          }
          m_pD3DDevice->DrawPrimitive(D3DPT_TRIANGLESTRIP,
            ((1+m_nHRect)*2*yRect+2*xRect), 2);
          if(iPass > 0){
            m_ppOcclusionQuery[yRect*m_nHRect+xRect]->Issue(D3DISSUE_END);
            m_pPendingResult[yRect*m_nHRect+xRect] = 1;
          }
        }
      }
      m_pEffect->EndPass();
  }
  m_pEffect->End();

  // We are done drawing for this scene.
  m_pD3DDevice->EndScene();

  return S_OK;
}


//-----------------------------------------------------------------------------
// Name: RenderColorTarget()
// Desc: Render the final texture, with the accumulated colors.
//-----------------------------------------------------------------------------
HRESULT HRCApp::RenderColorTarget(int passId) {

  //restore the original render target and clear the viewport
  m_pD3DDevice->SetRenderTarget(1, NULL);
  m_pD3DDevice->SetRenderTarget(0, m_pOriginalTarget);
  m_pD3DDevice->Clear(0L, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER,
    D3DCOLOR_RGBA(0,0,0,0), 1.0f, 0L);

  UINT numPasses;
  UINT iPass;

  // Tell Direct 3D we are now going to start drawing.
  m_pD3DDevice->BeginScene();

  HRESULT hr;

  m_pEffect->SetTechnique(m_hTechniqueRCT);

  hr = m_pEffect->SetTexture("tColor", m_pColorTEX[m_nSourceTraversal]);

  m_pD3DDevice->SetFVF (D3DFVF_CUSTOMVERTEX);

  m_pEffect->Begin(&numPasses, 0);
  for(iPass = 0; iPass < numPasses; iPass ++) {

    m_pEffect->BeginPass(iPass);
    m_pD3DDevice->SetStreamSource(0, m_rectangle, 0, sizeof(CUSTOMVERTEX2));
    m_pD3DDevice->SetFVF(D3DFVF_CUSTOMVERTEX);
    m_pD3DDevice->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 2);
    m_pEffect->EndPass();
  }
  m_pEffect->End();

  // We are done drawing for this scene.
  m_pD3DDevice->EndScene();

  return S_OK;
}

//-----------------------------------------------------------------------------
// Name: RenderFaceColored()
// Desc: Render the current traversal structure using face indices to define
//		fixed color
//-----------------------------------------------------------------------------
HRESULT HRCApp::RenderFaceColored(int passId)
{
  //disable the alpha blend
  m_pD3DDevice->SetRenderState(D3DRS_ALPHABLENDENABLE,  FALSE);

  //restore the original render target and clear the viewport
  m_pD3DDevice->SetRenderTarget(1, NULL);
  m_pD3DDevice->SetRenderTarget(0, m_pOriginalTarget);
  m_pD3DDevice->Clear(0L, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER,
    D3DCOLOR_RGBA(200,200,255,255), 1.0f, 0L);

  UINT numPasses;
  UINT iPass;

  // Tell Direct 3D we are now going to start drawing.
  m_pD3DDevice->BeginScene();

  HRESULT hr;

  // set technique
  m_pEffect->SetTechnique(m_hTechniqueRFC);

  hr = m_pEffect->SetTexture("tNeighbor", m_pCurrentCellTEX[m_nSourceTraversal]);//ok

  m_pD3DDevice->SetFVF (D3DFVF_CUSTOMVERTEX);

  m_pEffect->Begin(&numPasses, 0);
  for(iPass = 0; iPass < numPasses; iPass ++)
  {
    m_pEffect->BeginPass(iPass);
    m_pD3DDevice->SetStreamSource(0, m_rectangle, 0, sizeof(CUSTOMVERTEX2));
    m_pD3DDevice->SetFVF(D3DFVF_CUSTOMVERTEX);
    m_pD3DDevice->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 2);
    m_pEffect->EndPass();
  }
  m_pEffect->End();

  char pass[100];

  float incrx = 0.25f / (float)m_nHRect;
  float incry = 0.25f / (float)m_nVRect;
  for(int yRect=0; yRect < m_nVRect; yRect++)
    for(int xRect=0; xRect < m_nHRect; xRect++) {

      sprintf(pass, "%3d", m_pPassCounter[yRect*m_nHRect + xRect]);
      float xscr, yscr;
      xscr = incrx + (float)xRect / (float)m_nHRect;
      yscr = incry + (float)yRect / (float)m_nVRect;
      m_pFont->DrawText(
        xscr * (float)SCREEN_DIMENSION, yscr * (float)SCREEN_DIMENSION,
        D3DCOLOR_RGBA(255, 0, 0,255), pass);
    }

    // We are done drawing for this scene.
    m_pD3DDevice->EndScene();

    return S_OK;
}


//-----------------------------------------------------------------------------
// Name: RenderDepthLevels()
// Desc: Render the choosed depth level
//-----------------------------------------------------------------------------
HRESULT HRCApp::RenderDepthLevels(int passId) {

  //disable the alpha blend
  m_pD3DDevice->SetRenderState(D3DRS_ALPHABLENDENABLE,  FALSE);

  //restore the original render target and clear the viewport
  m_pD3DDevice->SetRenderTarget(1, NULL);
  m_pD3DDevice->SetRenderTarget(0, m_pOriginalTarget);
  m_pD3DDevice->Clear(0L, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER,
    D3DCOLOR_RGBA(255,0,0,255), 1.0f, 0L);

  UINT numPasses;
  UINT iPass;

  // Tell Direct 3D we are now going to start drawing.
  m_pD3DDevice->BeginScene();

  HRESULT hr;

  float uf, vf;
  computeUV2DNormalized(501, uf, vf, 256);

  // set technique
  m_pEffect->SetTechnique(m_hTechniqueRDL);

  // Make the mesh and traversal textures  available
  if(passId == 0)
    hr = m_pEffect->SetTexture("tCurrentCell", m_pCurrentCellTEX[m_nSourceTraversal]);
  else {

    passId = (passId <= MAX_DEPTH) ? passId : MAX_DEPTH;
    hr = m_pEffect->SetTexture("tCurrentCell", m_apDepthVector[passId-1]);
  }

  m_pD3DDevice->SetFVF (D3DFVF_CUSTOMVERTEX);

  m_pEffect->Begin(&numPasses, 0);
  for(iPass = 0; iPass < numPasses; iPass ++) {

    m_pEffect->BeginPass(iPass);
    m_pD3DDevice->SetStreamSource(0, m_rectangle, 0, sizeof(CUSTOMVERTEX2));
    m_pD3DDevice->SetFVF(D3DFVF_CUSTOMVERTEX);
    m_pD3DDevice->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 2);
    m_pEffect->EndPass();
  }
  m_pEffect->End();

  char pass[100];
  sprintf(pass, "depth level %d", passId);
  m_pFont->DrawText((float)5, (float)(SCREEN_DIMENSION - 25),
    D3DCOLOR_RGBA(255, 255, 255 ,255), pass);

  // We are done drawing for this scene.
  m_pD3DDevice->EndScene();

  return S_OK;
}


//-----------------------------------------------------------------------------
// Name: SetRenderStates()
// Desc: set the DX states
//-----------------------------------------------------------------------------
HRESULT HRCApp::SetRenderStates() {

  checkD3D(m_pD3DDevice->SetRenderState(D3DRS_ZENABLE, D3DZB_TRUE));
  checkD3D(m_pD3DDevice->SetRenderState(D3DRS_ALPHABLENDENABLE,  FALSE));
  checkD3D(m_pD3DDevice->SetRenderState(D3DRS_CULLMODE, D3DCULL_CW));
  checkD3D(m_pD3DDevice->SetRenderState(D3DRS_SRCBLEND, D3DBLEND_SRCALPHA));
  return S_OK;
}


//-----------------------------------------------------------------------------
// Name: SetCommonMatrix()
// Desc: set effect's Common matrix
//-----------------------------------------------------------------------------
HRESULT HRCApp::SetCommonMatrix() {

  D3DXMATRIXA16 matViewT, matModelT, matProjT, mat;

  checkD3D(m_pEffect->SetMatrix("matView", &m_stMatView));
  checkD3D(m_pEffect->SetMatrix("matProjection", &m_stMatProj));

  D3DXMATRIXA16 matModelInv, matProjInverse;

  checkD3D(m_pEffect->SetMatrix("matModelInverse", &matModelInv));

  D3DXMatrixInverse(&matProjInverse, NULL, &m_stMatProj);
  checkD3D(m_pEffect->SetMatrix("matProjectionInverse", &matProjInverse));

  D3DXMATRIX matViewInverse, matViewInverseTransposed;
  D3DXMatrixInverse(&matViewInverse, NULL, &m_stMatView);
  D3DXMatrixTranspose(&matViewInverseTransposed, &matViewInverse);
  checkD3D(m_pEffect->SetMatrix("matViewInverseTransposed", &matViewInverseTransposed));
  checkD3D(m_pEffect->SetMatrix("matViewInverse", &matViewInverse));

  D3DXMATRIX matViewInverseTransposedInverse;
  D3DXMatrixInverse(&matViewInverseTransposedInverse, NULL, &matViewInverseTransposed);
  checkD3D(m_pEffect->SetMatrix("matViewInverseTransposedInverse", &matViewInverseTransposedInverse));

  return S_OK;
}


//-----------------------------------------------------------------------------
// Name: SetCommonTextures()
// Desc: set effect's Common textures
//-----------------------------------------------------------------------------
HRESULT HRCApp::SetCommonTextures() {

  checkD3D(m_pEffect->SetTexture("tNewVertex", m_pNewVerticesTEX));
  checkD3D(m_pEffect->SetTexture("tNewNormals", m_pNewNormalsTEX));
  checkD3D(m_pEffect->SetTexture("tNewNeighbor", m_pNewNeighborTEX));
  if (m_nLutType)
    checkD3D(m_pEffect->SetTexture("tLut2D", m_pLut2DTEX));
  else
    checkD3D(m_pEffect->SetTexture("tLut", m_pLutTEX));
  return S_OK;
}


//-----------------------------------------------------------------------------
// Name: SetCommonVectors()
// Desc: set effect's Common vectors
//-----------------------------------------------------------------------------
HRESULT HRCApp::SetCommonVectors() {

  // texture displacement passed in passcolor
  float fVComp = m_fTextureIncr1 / 2.0f - SIGMA;
  float fUComp = fVComp + m_fTextureIncr1 - SIGMA;
  D3DXVECTOR4 passColor(m_fTextureIncr1, m_fTextureIncr2, fUComp, fVComp);

  if(m_pEffect->SetVector("passColor", (D3DXVECTOR4*)&passColor) != D3D_OK) {
    MessageBox(NULL, "Float array error!", "Effect error", MB_OK | MB_ICONERROR);
    return -1;
  }

  // set the local m_afEye point for RC
  D3DXVECTOR4 eyeNew(0,0,-2,1);

  if(m_pEffect->SetVector("eyeLocal", (D3DXVECTOR4*)&eyeNew) != D3D_OK)
    MessageBox(NULL, "eyeNew array error!", "Effect error", MB_OK | MB_ICONERROR);

  // set the LRP parameter
  m_afLrpParam = D3DXVECTOR4(1.0f/SCREEN_DIMENSION, 1.0f/SCREEN_DIMENSION ,
    (1.0f - (1.0f / (float)sizeTF))/(FLOAT)m_its->_maxDistance,
    1.0f / (2.0f*(float)sizeTF));
  m_afLrpParam = D3DXVECTOR4(0.0f, 0.0f, 1.0f, 1.0f);
  if(m_pEffect->SetVector("lrpParam", (D3DXVECTOR4*)&m_afLrpParam) != D3D_OK)
    MessageBox(NULL, "Float array error!", "Effect error", MB_OK | MB_ICONERROR);

  // set the texture dimension
  if(m_pEffect->SetFloat("textureDimension", (float)m_meshTextureSize) != D3D_OK)
    MessageBox(NULL, "Float error!", "Effect error", MB_OK | MB_ICONERROR);

  // set the integer of texture dimension / 3 and its normalized version
  int nTextureDimensionDiv3 = (m_meshTextureSize / 3);
  float fTexDiv3 = (float)nTextureDimensionDiv3 / (float)m_meshTextureSize;
  if(m_pEffect->SetFloat("normalizedTexDimDiv3", (float)fTexDiv3) != D3D_OK)
    MessageBox(NULL, "Float error!", "Effect error", MB_OK | MB_ICONERROR);

  if(m_pEffect->SetFloat("textureDimensionDiv3", (float)nTextureDimensionDiv3) != D3D_OK)
    MessageBox(NULL, "Float error!", "Effect error", MB_OK | MB_ICONERROR);

  return S_OK;
}


//-----------------------------------------------------------------------------
// Name: LoadTechniques()
// Desc: Load techniques
//-----------------------------------------------------------------------------
HRESULT HRCApp::LoadTechniques() {

  m_hTechniqueFH = m_pEffect->GetTechniqueByName("FirstHit");
  m_hTechniqueGDP = m_pEffect->GetTechniqueByName("DepthPeeling");
  m_hTechniqueOT = m_pEffect->GetTechniqueByName("OcclusionTest");
  m_hTechniqueRCT = m_pEffect->GetTechniqueByName("ShowColorBuffer");
  m_hTechniqueRDL = m_pEffect->GetTechniqueByName("RenderDepthPeeling");
  m_hTechniqueRFC = m_pEffect->GetTechniqueByName("FaceColoredRender");

  if(m_nLutType == 0)
    m_hTechniqueRC = m_pEffect->GetTechniqueByName("RayCasting");
  else
    m_hTechniqueRC = m_pEffect->GetTechniqueByName("RayCastingTF2D");

  return S_OK;
}


//-----------------------------------------------------------------------------
// Name: SaveStatistics()
// Desc: Save program statistics
//-----------------------------------------------------------------------------
void HRCApp::SaveStatistics() {

  //aux vars
  IDirect3DTexture9       *pOriginalCellTEX[2];
  IDirect3DSurface9       *pOriginalCellTARGET[2];

  FILE *fp = fopen(g_acFilenames[STATISTICS],"w");
  fprintf(fp, "ViewPos\tTiles\tminTime\tmaxTime\tminPass\tmaxPass\n");

  __timeb64 timeBegin, timeEnd;
  long frameTime;
  int nTilesPass, nViewPos, maxTime, minTime, maxPass, minPass, i, highPass;
  float fAux = 0.0f;
  float fDistFromCenter = m_fBBDiagonal / 2 + m_fBBDiagonal;
  float norma = m_fBBDiagonal / 2;

  // gather statistic information
  for(nTilesPass = NTILES; nTilesPass > 0; nTilesPass--){	//NTILES
    m_nVRect = nTilesPass;
    m_nHRect = nTilesPass;
    ChangeScreenRect();
    maxTime = 0;
    minTime = 9999999;
    maxPass = 0;
    minPass = 9999999;

    for(nViewPos = 0; nViewPos < 14; nViewPos++){
      highPass = 0;
      _ftime64(&timeBegin);

      switch(nViewPos){
        // faces
        case 0:		// -Z
          m_fEyeX = 0.0f;	m_fEyeY = 0.0f;	m_fEyeZ = m_fBBMinZ - m_fBBDiagonal;
          m_fUpX = 0.0f;	m_fUpY = 1.0f;	m_fUpZ = 0.0f;
          break;
        case 1:		// Z
          m_fEyeX = 0.0f;	m_fEyeY = 0.0f;	m_fEyeZ = m_fBBMaxZ + m_fBBDiagonal;
          m_fUpX = 0.0f;	m_fUpY = 1.0f;	m_fUpZ = 0.0f;
          break;
        case 2:		// -X
          m_fEyeX = m_fBBMinX - m_fBBDiagonal;	m_fEyeY = 0.0f;	m_fEyeZ = 0.0f;
          m_fUpX = 0.0f;	m_fUpY = 1.0f;	m_fUpZ = 0.0f;
          break;
        case 3:		// X
          m_fEyeX = m_fBBMaxX + m_fBBDiagonal;	m_fEyeY = 0.0f;	m_fEyeZ = 0.0f;
          m_fUpX = 0.0f;	m_fUpY = 1.0f;	m_fUpZ = 0.0f;
          break;
        case 4:		// -Y
          m_fEyeX = 0.0f;	m_fEyeY = m_fBBMinY - m_fBBDiagonal;	m_fEyeZ = 0.0f;
          m_fUpX = 0.0f;	m_fUpY = 0.0f;	m_fUpZ = 1.0f;
          break;
        case 5:		// Y
          m_fEyeX = 0.0f;	m_fEyeY = m_fBBMaxX + m_fBBDiagonal;	m_fEyeZ = 0.0f;
          m_fUpX = 0.0f;	m_fUpY = 0.0f;	m_fUpZ = 1.0f;
          break;
          // diagonal
        case 6:		// -X -Y -Z
          m_fEyeX = (m_fBBMinX / norma) * m_fBBDiagonal;
          m_fEyeY = (m_fBBMinY / norma) * m_fBBDiagonal;
          m_fEyeZ = (m_fBBMinZ / norma) * m_fBBDiagonal;
          m_fUpX = 0.0f;	m_fUpY = 1.0f;	m_fUpZ = 0.0f;
          break;
        case 7:		// -X -Y +Z
          m_fEyeX = (m_fBBMinX / norma) * m_fBBDiagonal;
          m_fEyeY = (m_fBBMinY / norma) * m_fBBDiagonal;
          m_fEyeZ = (m_fBBMaxZ / norma) * m_fBBDiagonal;
          m_fUpX = 0.0f;	m_fUpY = 1.0f;	m_fUpZ = 0.0f;
          break;
        case 8:		// -X +Y -Z
          m_fEyeX = (m_fBBMinX / norma) * m_fBBDiagonal;
          m_fEyeY = (m_fBBMaxY / norma) * m_fBBDiagonal;
          m_fEyeZ = (m_fBBMinZ / norma) * m_fBBDiagonal;
          m_fUpX = 0.0f;	m_fUpY = 1.0f;	m_fUpZ = 0.0f;
          break;
        case 9:		// -X +Y +Z
          m_fEyeX = (m_fBBMinX / norma) * m_fBBDiagonal;
          m_fEyeY = (m_fBBMaxY / norma) * m_fBBDiagonal;
          m_fEyeZ = (m_fBBMaxZ / norma) * m_fBBDiagonal;
          m_fUpX = 0.0f;	m_fUpY = 1.0f;	m_fUpZ = 0.0f;
          break;
        case 10:		// +X -Y -Z
          m_fEyeX = (m_fBBMaxX / norma) * m_fBBDiagonal;
          m_fEyeY = (m_fBBMinY / norma) * m_fBBDiagonal;
          m_fEyeZ = (m_fBBMinZ / norma) * m_fBBDiagonal;
          m_fUpX = 0.0f;	m_fUpY = 1.0f;	m_fUpZ = 0.0f;
          break;
        case 11:		// +X -Y +Z
          m_fEyeX = (m_fBBMaxX / norma) * m_fBBDiagonal;
          m_fEyeY = (m_fBBMinY / norma) * m_fBBDiagonal;
          m_fEyeZ = (m_fBBMaxZ / norma) * m_fBBDiagonal;
          m_fUpX = 0.0f;	m_fUpY = 1.0f;	m_fUpZ = 0.0f;
          break;
        case 12:		// +X +Y -Z
          m_fEyeX = (m_fBBMaxX / norma) * m_fBBDiagonal;
          m_fEyeY = (m_fBBMaxY / norma) * m_fBBDiagonal;
          m_fEyeZ = (m_fBBMinZ / norma) * m_fBBDiagonal;
          m_fUpX = 0.0f;	m_fUpY = 1.0f;	m_fUpZ = 0.0f;
          break;
        case 13:		// +X +Y +Z
          m_fEyeX = (m_fBBMaxX / norma) * m_fBBDiagonal;
          m_fEyeY = (m_fBBMaxY / norma) * m_fBBDiagonal;
          m_fEyeZ = (m_fBBMaxZ / norma) * m_fBBDiagonal;
          m_fUpX = 0.0f;	m_fUpY = 1.0f;	m_fUpZ = 0.0f;
          break;
      }

      SetupViewProjectionMatrices();

      SetCommonMatrix();
      SetCommonVectors();		// just after set the matrix

      RenderIntoTraversalStructures();
      ClearFinishedRect();

      pOriginalCellTARGET[0] = m_pCurrentCellTARGET[0];
      pOriginalCellTARGET[1] = m_pCurrentCellTARGET[1];
      pOriginalCellTEX[0] = m_pCurrentCellTEX[0];
      pOriginalCellTEX[1] = m_pCurrentCellTEX[1];
      ClearPendingResults();

      m_pEffect->SetTechnique(m_hTechniqueRC);
      for(int numDepth = 0; numDepth < MAX_DEPTH; numDepth++){
        i = 9;
        while(!m_nManualPass && m_nQuantFinishedRect>0  && i < 1000){
          RayCasting(i);
          if((++i % m_nPassToOcclusion) == 0){
            OcclusionTest(i);
            m_pEffect->SetTechnique(m_hTechniqueRC);
          }
        }

        // change the depth level
        m_pCurrentCellTARGET[m_nSourceTraversal] = m_apDepthSurfaceVector[numDepth];
        m_pCurrentCellTEX[m_nSourceTraversal] = m_apDepthVector[numDepth];
        ClearFinishedRect();
      }

      RenderColorTarget(i);

      // restore default targets and textures
      m_pCurrentCellTARGET[0] = pOriginalCellTARGET[0];
      m_pCurrentCellTARGET[1] = pOriginalCellTARGET[1];
      m_pCurrentCellTEX[0] = pOriginalCellTEX[0];
      m_pCurrentCellTEX[1] = pOriginalCellTEX[1];

      m_nOcclusionFirstPass = true;

      _ftime64(&timeEnd);

      frameTime = (long)((timeEnd.time*1000+timeEnd.millitm) - (timeBegin.time*1000+timeBegin.millitm));

      for(int k=0; k < nTilesPass; k++){
        for(int j=0; j < nTilesPass; j++){
          highPass = (highPass > (int)m_pPassCounter[k*nTilesPass + j]) ? highPass : m_pPassCounter[k*nTilesPass + j];
        }
      }

      minTime = (frameTime < minTime) ? frameTime : minTime;
      maxTime = (frameTime > maxTime) ? frameTime : maxTime;
      minPass = (highPass < minPass) ? highPass : minPass;
      maxPass = (highPass > maxPass) ? highPass : maxPass;

      // Show the frame on the primary surface.
      HRESULT hr = m_pD3DDevice->Present(NULL, NULL, NULL, NULL);
      fprintf(fp, "%i\t%i\t%i\t%i\t%i\t%i\n", nViewPos, nTilesPass, minTime, maxTime, minPass, maxPass);
    }
  }

  fclose(fp);
}


//-----------------------------------------------------------------------------
// Name: Render()
// Desc: Render the scene.
//-----------------------------------------------------------------------------
HRESULT HRCApp::Render() {

  //aux vars
  IDirect3DTexture9       *pOriginalCellTEX[2];
  IDirect3DSurface9       *pOriginalCellTARGET[2];

  if((int)(strlen(g_acFilenames[STATISTICS])) != 0){
    SaveStatistics();
    return -1;
  }

  int i=0;
  __timeb64 timeBegin, timeEnd;
  _ftime64(&timeBegin);

  SetCommonMatrix();
  SetCommonVectors();		// just after set the matrix

  RenderIntoTraversalStructures();

  ClearFinishedRect();

  pOriginalCellTARGET[0] = m_pCurrentCellTARGET[0];
  pOriginalCellTARGET[1] = m_pCurrentCellTARGET[1];
  pOriginalCellTEX[0] = m_pCurrentCellTEX[0];
  pOriginalCellTEX[1] = m_pCurrentCellTEX[1];

  // save the reference to the depth levels
  ClearPendingResults();
  if(m_nManualPass){			// manual
    int numDepth = 0;
    m_pEffect->SetTechnique(m_hTechniqueRC);
    for (i=0; i < m_nNumPass; i++) {
      RayCasting(i);

      if((i % m_nPassToOcclusion) == 0){
        OcclusionTest(i);
        m_pEffect->SetTechnique(m_hTechniqueRC);
      }
      if(m_nQuantFinishedRect == 0 && numDepth < MAX_DEPTH){

        m_pCurrentCellTARGET[m_nSourceTraversal] = m_apDepthSurfaceVector[numDepth];
        m_pCurrentCellTEX[m_nSourceTraversal] = m_apDepthVector[numDepth];
        m_nOcclusionFirstPass = true;
        ClearFinishedRect();
        numDepth++;
      }
    }
  }
  else{				// automatic
    m_pEffect->SetTechnique(m_hTechniqueRC);
    for(int numDepth = 0; numDepth <= MAX_DEPTH; numDepth++){

      i = 9;
      while(!m_nManualPass && m_nQuantFinishedRect>0 && i < 1000){		// modified to prevent endless loop
        RayCasting(i);
        if((++i % m_nPassToOcclusion) == 0){
          OcclusionTest(i);
          m_pEffect->SetTechnique(m_hTechniqueRC);
        }
      }

      if (numDepth < MAX_DEPTH) {
        // change the depth level
        m_pCurrentCellTARGET[m_nSourceTraversal] = m_apDepthSurfaceVector[numDepth];
        m_pCurrentCellTEX[m_nSourceTraversal] = m_apDepthVector[numDepth];
        m_nOcclusionFirstPass = true;
        ClearFinishedRect();
      }
    }
  }

  if (i & 1)
    RayCasting(i);

  //switch view style
  switch (m_nView) {
    case 0:
      RenderColorTarget(i);
      break;

    case 1:
      RenderFaceColored(i);
      break;

    case 2:
      RenderDepthLevels(i);
      break;

  }

  // restore default targets and textures
  m_pCurrentCellTARGET[0] = pOriginalCellTARGET[0];
  m_pCurrentCellTARGET[1] = pOriginalCellTARGET[1];
  m_pCurrentCellTEX[0] = pOriginalCellTEX[0];
  m_pCurrentCellTEX[1] = pOriginalCellTEX[1];

  // restore the reference to the depth levels
  m_nOcclusionFirstPass = true;

  _ftime64(&timeEnd);

  if(m_nView == 1){

    m_pD3DDevice->BeginScene();

    char sTime[20];
    sprintf(sTime, "TIME(%i)",
      ((timeEnd.time*1000+timeEnd.millitm) -
      (timeBegin.time*1000+timeBegin.millitm)));

    m_pFont->DrawText(20.0f, (float)(SCREEN_DIMENSION - 25),
      D3DCOLOR_RGBA(255, 0, 0,255), sTime);

    sprintf(sTime, "#pOcc(%i)", m_nPassToOcclusion);
    m_pFont->DrawText( (float)(SCREEN_DIMENSION - (strlen(sTime) * 17 + 17)),
      (float)(SCREEN_DIMENSION - 25), D3DCOLOR_RGBA(255, 0, 0,255), sTime);

    m_pD3DDevice->EndScene();
  }

  m_pD3DDevice->Present(NULL, NULL, NULL, NULL);

  return S_OK;
}


//-----------------------------------------------------------------------------
// Name: InitDeviceObjects()
// Desc: This creates all device-dependent managed objects.
//-----------------------------------------------------------------------------
HRESULT HRCApp::InitDeviceObjects() {

  //aux vars
  HRESULT hr;

  //initialize the screen rects variables
  m_ppOcclusionQuery = NULL;
  m_pnFinishedRect = NULL;
  m_pRectBuffer = NULL;
  m_pPendingResult = NULL;
  m_pPassCounter = NULL;
  m_nHRect = NTILES;
  m_nVRect = NTILES;
  m_bFirstTime = true;
  m_nManualPass = false;
  m_nOcclusionFirstPass = true;

  //initialize the number of passes
  m_nNumPass = 0;

  //one occlusion test for each 10 RC passes
  m_nPassToOcclusion = 10;

  SetupViewProjectionMatrices();

  m_afEye = D3DXVECTOR4(m_fEyeX, m_fEyeY, m_fEyeZ, 1.0f);

  // Initialize the font's internal textures
  m_pFont->InitDeviceObjects(m_pD3DDevice);

  LPD3DXBUFFER ppCompilationErrors;
  if(FAILED(hr = D3DXCreateEffectFromFile(m_pD3DDevice, "Source/RayCasting.fx", NULL, NULL, 
    0, NULL, &m_pEffect, &ppCompilationErrors))) {

    DisplayShaderERR(ppCompilationErrors, "RayCasting.fx", hr);
    MessageBox(NULL, "FX Compile error!", NULL, MB_OK);
    ppCompilationErrors->Release();
    return hr;
  }

  m_nBoundaryFaces = m_its->nTetra();

  int nT = m_its->nTetra();

  m_meshTextureSize = (int)ceil(sqrt((float)(3*nT)));
  m_meshTextureSize += 1;	// one position must remain empty

  if(m_meshTextureSize % 3 == 0)
    m_meshTextureSize += 1;		// can't be divisible by 3

  if(m_meshTextureSize > 2044) {

    char szBuff[512];
    sprintf (szBuff, "\nUnable to load mesh in 3D texture - too many tetrahedron %d", nT);
    MessageBox(NULL, szBuff, "Error", MB_OK | MB_ICONERROR);
    exit(-1);
  }

  //load the mesh into textures
  LoadMesh();

  m_fTextureIncr1 = 1.0f / (m_meshTextureSize);
  m_fTextureIncr2 = m_fTextureIncr1 * 2.0f;

  // create mesh vertex bufer
  m_pBoundaryFaces = m_its->createVertexBuffer(m_meshTextureSize);
  m_nBoundaryFaces = m_its->nBoundaryFaces();

  // create device-dependent vertex buffers
  CreateVertexBuffers();

  LoadTechniques();
  SetRenderStates();
  SetCommonTextures();

  ChangeScreenRect();	//initialize the screen tiles

  return S_OK;
}


//-----------------------------------------------------------------------------
// Name: DeleteDeviceObjects()
// Desc: delete all device-independent objects
//-----------------------------------------------------------------------------
HRESULT HRCApp::DeleteDeviceObjects(){

  //aux vars
  int nCount;

  //release render targets
  ReleaseTargetTextures();

  //release tile buffer and occlusion queries
  ReleaseScreenRect();

  //vertex buffers
  SafeRelease(m_pBoundaryBuffer);
  SafeRelease(m_rectangle);
  SafeRelease(m_pRectBuffer);

  //textures
  SafeRelease(m_pNewVerticesTEX);
  SafeRelease(m_pNewNormalsTEX);
  SafeRelease(m_pNewNeighborTEX);

  if(m_nLutType != 0){
    SafeRelease(m_pLut2DTEX);
  }

  //volumes
  if(m_nLutType == 0){
    SafeRelease(m_pLutTEX);
  }

  //occlusion queries
  for(nCount=0; nCount < (int)NTILES * NTILES; nCount++){
    SafeRelease(m_ppOcclusionQuery[nCount]);
  }

  //effect
  SafeRelease(m_pEffect);

  return S_OK;
}


//-----------------------------------------------------------------------------
// Name: InvalidateDeviceObjects()
// Desc: Called when the device-dependent objects are about to be lost.
//-----------------------------------------------------------------------------
HRESULT HRCApp::InvalidateDeviceObjects() {

  ReleaseTargetTextures();
  ReleaseScreenRect();

  m_pFont->InvalidateDeviceObjects();

  if(m_pEffect != NULL)
    m_pEffect->OnLostDevice();

  return S_OK;
}



//-----------------------------------------------------------------------------
// Name: RestoreDeviceObjects()
// Desc: Restore device-memory objects and state after a device is created or
//       resized.
//-----------------------------------------------------------------------------
HRESULT HRCApp::RestoreDeviceObjects()
{
  m_pFont->RestoreDeviceObjects();

  // Update projection matrix based on window dimensions
  FLOAT fAspect = (FLOAT)((FLOAT)m_stBBDesc.Width / (FLOAT)m_stBBDesc.Height);

  if(m_pEffect != NULL)
    m_pEffect->OnResetDevice();

  // restore screen tiles
  ChangeScreenRect();

  // create render targets
  CreateTargetTextures();

  return S_OK;
}


//-----------------------------------------------------------------------------
// Name: ClearFinishedRect()
// Desc: reset all tile's
//-----------------------------------------------------------------------------
void HRCApp::ClearFinishedRect() {

  m_pD3DDevice->Clear(0L, NULL, D3DCLEAR_ZBUFFER,
    D3DCOLOR_RGBA(255,0,0,255), 1.0f, 0L);

  m_nQuantFinishedRect = m_nVRect*m_nHRect;
  for(int i=0; i<(int)m_nQuantFinishedRect; i++){
    m_pnFinishedRect[i] = false;
    m_pPendingResult[i] = 0;
  }
}


//-----------------------------------------------------------------------------
// Name: ClearPendingResults()
// Desc: erase pass counter and reset pending results
//-----------------------------------------------------------------------------
void HRCApp::ClearPendingResults() {

  m_nQuantFinishedRect = m_nVRect*m_nHRect;
  for(int i=0; i<(int)m_nQuantFinishedRect; i++){
    m_pPendingResult[i] = 0;
    m_pPassCounter[i] = 0;
  }
}


//-----------------------------------------------------------------------------
// Name: ChangeScreenRect()
// Desc: Change the quantity of screen's rect
//-----------------------------------------------------------------------------
HRESULT HRCApp::ChangeScreenRect() {

  m_nVRect = (m_nVRect<=0) ? 1 : m_nVRect;
  m_nHRect = (m_nHRect<=0) ? 1 : m_nHRect;

  float epsilonWidth  = 0.5f/(float)m_stBBDesc.Width;
  float epsilonHeight = 0.5f/(float)m_stBBDesc.Height;
  float top = 1.0f, left = -1.0f;

  float dx = (-left)*2 / (float)m_nHRect;
  float dy = (-top)*2 / (float)m_nVRect;

  float dxt = 1.0f / (float)m_nHRect;
  float dyt = 1.0f / (float)m_nVRect;

  int quantRect = (m_nHRect+1) * (2*m_nVRect);

  int quant = (NTILES+1) * (2*NTILES);
  CUSTOMVERTEX2 *v = new CUSTOMVERTEX2[quant];
  CUSTOMVERTEX2 *v2 = (CUSTOMVERTEX2*)malloc(sizeof(CUSTOMVERTEX2)*quant);

  VOID* pVertices;

  HRESULT hr;
  if(m_bFirstTime){
    m_pnFinishedRect = new int[NTILES * NTILES];
    m_pPendingResult = new int[NTILES * NTILES];
    m_pPassCounter = new int[NTILES * NTILES];
    m_ppOcclusionQuery = new IDirect3DQuery9*[NTILES * NTILES];

    //create the occlusion query
    for(int i=0; i<(int)(NTILES * NTILES); i++){
      checkD3D(m_pD3DDevice->CreateQuery(D3DQUERYTYPE_OCCLUSION, &m_ppOcclusionQuery[i]));
    }

    if(FAILED(hr=m_pD3DDevice->CreateVertexBuffer(quantRect*sizeof(CUSTOMVERTEX2), 0, D3DFVF_CUSTOMVERTEX,
      D3DPOOL_MANAGED, &m_pRectBuffer, NULL))){
        return E_FAIL;
      }

    m_bFirstTime = false;
  }

  for(int y=0; y<(int)(m_nVRect); y++){
    for(int x=0; x<(int)(m_nHRect+1); x++){
      //xyz
      v[2*y*(1+m_nHRect) + x*2].x = left + dx * x;
      v[2*y*(1+m_nHRect) + x*2].y = top + dy * y;
      v[2*y*(1+m_nHRect) + x*2].z = 0.0f;
      //xyz next line
      v[2*y*(1+m_nHRect) + x*2+1].x = left + dx * x;
      v[2*y*(1+m_nHRect) + x*2+1].y = top + dy * (y+1);
      v[2*y*(1+m_nHRect) + x*2+1].z = 0.0f;
      //texture coords
      v[2*y*(1+m_nHRect) + x*2].tu1 = (x*dxt)+epsilonWidth;
      v[2*y*(1+m_nHRect) + x*2].tv1 = (y*dyt)+epsilonHeight;
      v[2*y*(1+m_nHRect) + x*2].tu2 = (x*dxt)+epsilonWidth;
      v[2*y*(1+m_nHRect) + x*2].tv2 = (y*dyt)+epsilonHeight;
      v[2*y*(1+m_nHRect) + x*2].tu3 = (x*dxt)+epsilonWidth;
      v[2*y*(1+m_nHRect) + x*2].tv3 = (y*dyt)+epsilonHeight;
      //texture coords next line
      v[2*y*(1+m_nHRect) + x*2+1].tu1 = (x*dxt)+epsilonWidth;
      v[2*y*(1+m_nHRect) + x*2+1].tv1 = ((y+1)*dyt)+epsilonHeight;
      v[2*y*(1+m_nHRect) + x*2+1].tu2 = (x*dxt)+epsilonWidth;
      v[2*y*(1+m_nHRect) + x*2+1].tv2 = ((y+1)*dyt)+epsilonHeight;
      v[2*y*(1+m_nHRect) + x*2+1].tu3 = (x*dxt)+epsilonWidth;
      v[2*y*(1+m_nHRect) + x*2+1].tv3 = ((y+1)*dyt)+epsilonHeight;

      v2[2*y*(1+m_nHRect) + x*2] = v[2*y*(1+m_nHRect) + x*2];
      v2[2*y*(1+m_nHRect) + x*2+1] = v[2*y*(1+m_nHRect) + x*2+1];
    }
  }

  if(FAILED(hr=m_pRectBuffer->Lock(0, sizeof(CUSTOMVERTEX2)*quantRect, (void**)&pVertices, 0))){
    MessageBox(NULL, "It was not possible to lock the Rectangle Vertex Buffer.", NULL, MB_OK);
    return E_FAIL;
  }

  memcpy(pVertices, v2, sizeof(CUSTOMVERTEX2)*quantRect);
  m_pRectBuffer->Unlock();

  delete(v);
  free(v2);

  return S_OK;
}


//-----------------------------------------------------------------------------
// Name: ReleaseScreenRect()
// Desc: Release queries and a vertex buffer
//-----------------------------------------------------------------------------
HRESULT HRCApp::ReleaseScreenRect() {

  //create the occlusion query
  for(int i=0; i<(int)(NTILES * NTILES); i++){

    SafeRelease(m_ppOcclusionQuery[i]);
  }

  SafeRelease(m_pRectBuffer);

  m_bFirstTime = true;

  return S_OK;

}


//-----------------------------------------------------------------------------
// Name: CreateVertexBuffers()
// Desc: Create the device-dependent vertex buffers
//-----------------------------------------------------------------------------
HRESULT HRCApp::CreateVertexBuffers() {

  //aux vars
  Vertex* Vertices;

  // Create the vertex buffer that will hold the triangle.
  if(FAILED(m_pD3DDevice->CreateVertexBuffer(sizeof(Vertex)*m_nBoundaryFaces*3, 0, D3DFVF_D3DVERTEX, 
    D3DPOOL_MANAGED, &m_pBoundaryBuffer, NULL))){
      MessageBox(NULL, "Error creating the vertex buffer", NULL, MB_OK);
  }

  //lock and copy boundary vertex data
  if(FAILED(m_pBoundaryBuffer->Lock(0, sizeof(Vertex)*m_nBoundaryFaces*3, (void**)&Vertices, 0))){
    MessageBox(NULL, "Failed to lock the boundary buffer.", NULL, MB_OK);
  }
  memcpy(Vertices, m_pBoundaryFaces, sizeof(Vertex)*m_nBoundaryFaces*3);
  m_pBoundaryBuffer->Unlock();

  // Initialize three vertices for rendering a triangle
  CUSTOMVERTEX2 v[4];

  float epsilonWidth  = 0.5f/(float)m_stBBDesc.Width;
  float epsilonHeight = 0.5f/(float)m_stBBDesc.Height;

  v[0].tu1 = 0.0f+epsilonWidth;
  v[0].tv1 =0.0f+epsilonHeight;
  v[1].tu1 = 0.0f+epsilonWidth;
  v[1].tv1 =1.0f-epsilonHeight;
  v[2].tu1 = 1.0f-epsilonWidth;
  v[2].tv1 =0.0f+epsilonHeight;
  v[3].tu1 = 1.0f-epsilonWidth;
  v[3].tv1 =1.0f-epsilonHeight;

  v[0].tu2 = 0.0f+epsilonWidth;
  v[0].tv2 =0.0f+epsilonHeight;
  v[1].tu2 = 0.0f+epsilonWidth;
  v[1].tv2 =1.0f-epsilonHeight;
  v[2].tu2 = 1.0f-epsilonWidth;
  v[2].tv2 =0.0f+epsilonHeight;
  v[3].tu2 = 1.0f-epsilonWidth;
  v[3].tv2 =1.0f-epsilonHeight;

  v[0].tu3 = 0.0f+epsilonWidth;
  v[0].tv3 =0.0f+epsilonHeight;
  v[1].tu3 = 0.0f+epsilonWidth;
  v[1].tv3 =1.0f-epsilonHeight;
  v[2].tu3 = 1.0f-epsilonWidth;
  v[2].tv3 =0.0f+epsilonHeight;
  v[3].tu3 = 1.0f-epsilonWidth;
  v[3].tv3 =1.0f-epsilonHeight;

  //initialize the FVF square with screen dimension
  //(0,0) is the center of screen
  float s = 1.0f, t = 1.0f;
  //up-left
  v[0].x = -s;
  v[0].y = t;
  v[0].z = 0.0f;
  //down-left
  v[1].x = -s;
  v[1].y = -t;
  v[1].z = 0.0f;
  //up-right
  v[2].x = s;
  v[2].y = t;
  v[2].z = 0.0f;
  //down-right
  v[3].x = s;
  v[3].y = -t;
  v[3].z = 0.0f;

  if(FAILED(m_pD3DDevice->CreateVertexBuffer(4*sizeof(CUSTOMVERTEX2),
    0, D3DFVF_CUSTOMVERTEX,
    D3DPOOL_MANAGED, &m_rectangle, NULL)))
  {
    MessageBox(NULL, "Error creating the vertex buffer", NULL, MB_OK);
    return E_FAIL;
  }

  // fill the vertex buffer
  VOID* pVertices;
  if(FAILED(m_rectangle->Lock(0, sizeof(v), (void**)&pVertices, 0)))
    return E_FAIL;

  memcpy(pVertices, v, sizeof(v));
  m_rectangle->Unlock();

  return S_OK;
}


//-----------------------------------------------------------------------------
// Name: CreateTargetTextures()
// Desc: Create the device-dependent vertex buffers
//-----------------------------------------------------------------------------
bool HRCApp::CreateTargetTextures(){

  //aux vars
  HRESULT hr;
  int i = 0;

  // first check the width and height of primary render surface
  m_pD3DDevice->GetRenderTarget(0, &m_pOriginalTarget);
  m_pOriginalTarget->GetDesc(&m_stBBDesc);

  //initialize the Depth peeling variables
  m_apDepthVector.clear();
  m_apDepthSurfaceVector.clear();
  if(m_apDepthVector.size() > 0)
    MessageBox(NULL, "m_apDepthVector.size() > 0!!!", NULL, MB_OK);
  if(m_apDepthSurfaceVector.size() > 0)
    MessageBox(NULL, "m_apDepthSurfaceVector.size() > 0!!!", NULL, MB_OK);

  LPDIRECT3DTEXTURE9 newDepthTex[MAX_DEPTH];
  LPDIRECT3DSURFACE9 depthTarget[MAX_DEPTH];
  for(depthCounter = 0; depthCounter < MAX_DEPTH; depthCounter++){
    hr = D3DXCreateTexture(m_pD3DDevice, m_stBBDesc.Width, m_stBBDesc.Height, 1,
      D3DUSAGE_RENDERTARGET, D3DFMT_A32B32G32R32F, D3DPOOL_DEFAULT,
      &newDepthTex[depthCounter]);
    if (hr != D3D_OK) return false;

    hr = newDepthTex[depthCounter]->GetSurfaceLevel(0, &depthTarget[depthCounter]);

    m_apDepthVector.push_back(newDepthTex[depthCounter]);
    m_apDepthSurfaceVector.push_back(depthTarget[depthCounter]);
  }
  depthCounter = 0;

  // Create the textures as render targets
  for (i = 0; i < 2; i++) {
    hr = D3DXCreateTexture(m_pD3DDevice, m_stBBDesc.Width, m_stBBDesc.Height, 1, D3DUSAGE_RENDERTARGET,
      D3DFMT_A32B32G32R32F, D3DPOOL_DEFAULT, &m_pCurrentCellTEX[i]);
    if (hr != D3D_OK) return false;

    hr = D3DXCreateTexture(m_pD3DDevice, m_stBBDesc.Width, m_stBBDesc.Height, 1, D3DUSAGE_RENDERTARGET,
      D3DFMT_A32B32G32R32F, D3DPOOL_DEFAULT, &m_pColorTEX[i]);
    if (hr != D3D_OK) return false;

    m_pCurrentCellTEX[i]->GetSurfaceLevel(0, &m_pCurrentCellTARGET[i]);
    m_pColorTEX[i]->GetSurfaceLevel(0, &m_pColorTARGET[i]);
  }

  return true;
}


//-----------------------------------------------------------------------------
// Name: ReleaseTargetTextures()
// Desc: Release the device-dependent textures
//-----------------------------------------------------------------------------
bool HRCApp::ReleaseTargetTextures(){

  //aux vars
  int nCount;

  SafeRelease(m_pOriginalTarget);

  //initialize the Depth peeling variables
  for(nCount = 0; nCount < MAX_DEPTH; nCount++){

    SafeRelease(m_apDepthSurfaceVector[nCount]);
    SafeRelease(m_apDepthVector[nCount]);
  }

  m_apDepthVector.clear();
  m_apDepthSurfaceVector.clear();

  // Create the textures as render targets
  for (nCount = 0; nCount < 2; nCount++) {

    SafeRelease(m_pCurrentCellTARGET[nCount]);
    SafeRelease(m_pColorTARGET[nCount]);
    SafeRelease(m_pCurrentCellTEX[nCount]);
    SafeRelease(m_pColorTEX[nCount]);
  }

  return true;
}



//-----------------------------------------------------------------------------
// Name: MoveCameraU()
// Desc: Move Camera on its U axis
//-----------------------------------------------------------------------------
void HRCApp::MoveCameraU(float fDelta) {

  D3DXVECTOR3 vEyePt(m_fEyeX, m_fEyeY, m_fEyeZ);
  D3DXVECTOR3 vLookatPt(0.0f, 0.0f, 0.0f);
  D3DXVECTOR3 vUpVec(m_fUpX, m_fUpY, m_fUpZ);

  D3DXMatrixLookAtLH(&m_stMatView, &vEyePt, &vLookatPt, &vUpVec);

  D3DXVECTOR3 afU, afV, afN, afNewPos;

  afU = D3DXVECTOR3(m_stMatView._11, m_stMatView._21, m_stMatView._31);
  afV = D3DXVECTOR3(m_stMatView._12, m_stMatView._22, m_stMatView._32);

  float fDist = sqrt(m_fEyeX * m_fEyeX + m_fEyeY * m_fEyeY + m_fEyeZ * m_fEyeZ);

  afNewPos = D3DXVECTOR3(	m_fEyeX + afU[0] * fDelta,
    m_fEyeY + afU[1] * fDelta,
    m_fEyeZ + afU[2] * fDelta);

  D3DXVec3Normalize(&afNewPos, &afNewPos);
  afN = -afNewPos;
  afNewPos *= fDist;

  D3DXVec3Cross(&afU, &afN, &afV);
  D3DXVec3Normalize(&afU, &afU);

  D3DXVec3Cross(&afV, &afU, &afN);
  D3DXVec3Normalize(&afV, &afV);

  m_fEyeX = afNewPos[0];
  m_fEyeY = afNewPos[1];
  m_fEyeZ = afNewPos[2];

  m_fUpX = afV[0];
  m_fUpY = afV[1];
  m_fUpZ = afV[2];
}


//-----------------------------------------------------------------------------
// Name: MoveCameraV()
// Desc: Move Camera on its V axis
//-----------------------------------------------------------------------------
void HRCApp::MoveCameraV(float fDelta) {

  D3DXVECTOR3 vEyePt(m_fEyeX, m_fEyeY, m_fEyeZ);
  D3DXVECTOR3 vLookatPt(0.0f, 0.0f, 0.0f);
  D3DXVECTOR3 vUpVec(m_fUpX, m_fUpY, m_fUpZ);

  D3DXMatrixLookAtLH(&m_stMatView, &vEyePt, &vLookatPt, &vUpVec);

  D3DXVECTOR3 afU, afV, afN, afNewPos;

  afU = D3DXVECTOR3(m_stMatView._11, m_stMatView._21, m_stMatView._31);
  afV = D3DXVECTOR3(m_stMatView._12, m_stMatView._22, m_stMatView._32);

  float fDist = sqrt(m_fEyeX * m_fEyeX + m_fEyeY * m_fEyeY + m_fEyeZ * m_fEyeZ);

  afNewPos = D3DXVECTOR3(	m_fEyeX + afV[0] * fDelta,
    m_fEyeY + afV[1] * fDelta,
    m_fEyeZ + afV[2] * fDelta);

  D3DXVec3Normalize(&afNewPos, &afNewPos);
  afN = -afNewPos;
  afNewPos *= fDist;

  D3DXVec3Cross(&afU, &afN, &afV);
  D3DXVec3Normalize(&afU, &afU);

  D3DXVec3Cross(&afV, &afU, &afN);
  D3DXVec3Normalize(&afV, &afV);

  m_fEyeX = afNewPos[0];
  m_fEyeY = afNewPos[1];
  m_fEyeZ = afNewPos[2];

  m_fUpX = afV[0];
  m_fUpY = afV[1];
  m_fUpZ = afV[2];
}


//-----------------------------------------------------------------------------
// Name: MoveCameraN()
// Desc: Move Camera on its N axis
//-----------------------------------------------------------------------------
void HRCApp::MoveCameraN(float fDelta) {

  D3DXVECTOR3 vEyePt(m_fEyeX, m_fEyeY, m_fEyeZ);
  D3DXVECTOR3 vLookatPt(0.0f, 0.0f, 0.0f);
  D3DXVECTOR3 vUpVec(m_fUpX, m_fUpY, m_fUpZ);

  D3DXMatrixLookAtLH(&m_stMatView, &vEyePt, &vLookatPt, &vUpVec);

  D3DXVECTOR3 afU, afV, afN, afNewPos;

  afU = D3DXVECTOR3(m_stMatView._11, m_stMatView._21, m_stMatView._31);
  afV = D3DXVECTOR3(m_stMatView._12, m_stMatView._22, m_stMatView._32);
  afN = D3DXVECTOR3(m_stMatView._13, m_stMatView._23, m_stMatView._33);

  afNewPos = D3DXVECTOR3(	m_fEyeX + afN[0] * fDelta,
    m_fEyeY + afN[1] * fDelta,
    m_fEyeZ + afN[2] * fDelta);

  m_fEyeX = afNewPos[0];
  m_fEyeY = afNewPos[1];
  m_fEyeZ = afNewPos[2];
}


//-----------------------------------------------------------------------------
// Name: RotCameraN()
// Desc: Rotation Camera on its N axis
//-----------------------------------------------------------------------------
void HRCApp::RotCameraN(float fDelta) {

  D3DXVECTOR3 vEyePt(m_fEyeX, m_fEyeY, m_fEyeZ);
  D3DXVECTOR3 vLookatPt(0.0f, 0.0f, 0.0f);
  D3DXVECTOR3 vUpVec(m_fUpX, m_fUpY, m_fUpZ);

  D3DXMatrixLookAtLH(&m_stMatView, &vEyePt, &vLookatPt, &vUpVec);

  D3DXVECTOR3 afU, afV, afN, afNewPos;
  D3DXMATRIX matRot;

  afU = D3DXVECTOR3(m_stMatView._11, m_stMatView._21, m_stMatView._31);
  afV = D3DXVECTOR3(m_stMatView._12, m_stMatView._22, m_stMatView._32);
  afN = D3DXVECTOR3(m_stMatView._13, m_stMatView._23, m_stMatView._33);

  D3DXMatrixRotationAxis(&matRot, &afN, (float)(fDelta * PI / 180.0f));

  D3DXVECTOR4 vAux;
  D3DXVec3Transform(&vAux, &afV, &matRot);
  afV = D3DXVECTOR3(vAux);

  D3DXVec3Transform(&vAux, &afU, &matRot);
  afU = D3DXVECTOR3(vAux);

  m_fUpX = afV[0];
  m_fUpY = afV[1];
  m_fUpZ = afV[2];
}


//-----------------------------------------------------------------------------
/// \param none
/// \return true if reset was successful, false otherwise
//-----------------------------------------------------------------------------
bool HRCApp::Reset() {

  RECT stWndRect;

  GetWindowRect(g_hMainWindow, &stWndRect);

  m_stD3DPresentParameters.BackBufferHeight = stWndRect.bottom - stWndRect.top;
  m_stD3DPresentParameters.BackBufferWidth  = stWndRect.right - stWndRect.left;

  InvalidateDeviceObjects();

  if (m_pD3DDevice->Reset(&m_stD3DPresentParameters) != D3D_OK) {

    MessageBox(g_hMainWindow, "Could not reset D3D Device.", "Reset failed", MB_OK | MB_ICONERROR);
    return false;
  }

  RestoreDeviceObjects();

  SetupViewProjectionMatrices();

  return true;
}


//-----------------------------------------------------------------------------
// Name: MsgProc()
// Desc: Message proc function to handle user input
//-----------------------------------------------------------------------------
LRESULT HRCApp::MsgProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {

  //aux vars
  float auxX, auxY;

  switch(uMsg) {

    case WM_EXITSIZEMOVE:
      Reset();
      break;

    case WM_RBUTTONDOWN:
    case WM_LBUTTONDOWN:
      m_nCurrentMousePositionX = LOWORD(lParam);
      m_nCurrentMousePositionY = HIWORD(lParam);
      break;

    case WM_MOUSEWHEEL:
      MoveCameraN((float)((short)HIWORD(wParam)) / 120.0f);
      break;

    case WM_MOUSEMOVE:
      if (wParam & MK_LBUTTON) {

        MoveCameraU((float)(LOWORD(lParam) - m_nCurrentMousePositionX) / 2.0f);
        MoveCameraV((float)(HIWORD(lParam) - m_nCurrentMousePositionY) / 2.0f);
      }

      if (wParam & MK_RBUTTON) {

        auxX = (float)(LOWORD(lParam) - m_nCurrentMousePositionX) / 2.0f;
        auxY = (float)(HIWORD(lParam) - m_nCurrentMousePositionY) / 2.0f;
        RotCameraN(-auxX);
      }

      m_nCurrentMousePositionX = LOWORD(lParam);
      m_nCurrentMousePositionY = HIWORD(lParam);
      break;

    case WM_KEYDOWN:
      switch (wParam) {

        case VK_ADD:
          m_fEyeZ += 0.1f;
          SetupViewProjectionMatrices();
          break;
        case VK_SUBTRACT:
          m_fEyeZ -= 0.1f;
          SetupViewProjectionMatrices();
          break;
        case 39:	// left arrow
          m_nNumPass+=1;
          break;
        case 37:	// right arrow
          m_nNumPass = (m_nNumPass > 0) ? m_nNumPass-1 : 0;
          break;
        case VK_UP:	// up arrow
          m_nPassToOcclusion++;
          break;	// down arrow
        case VK_DOWN:
          m_nPassToOcclusion = (m_nPassToOcclusion > 1) ? m_nPassToOcclusion-1 : m_nPassToOcclusion ;
          break;
          // config options
        case VK_PRIOR: // PageUp
          m_nVRect = (m_nVRect < NTILES) ? m_nVRect+1 : NTILES;
          m_nHRect = (m_nHRect < NTILES) ? m_nHRect+1 : NTILES;
          ChangeScreenRect();
          break;
        case VK_NEXT:	// PageDown
          m_nVRect = (m_nVRect <= 1) ? 1 : m_nVRect-1;
          m_nHRect = (m_nHRect <= 1) ? 1 : m_nHRect-1;
          ChangeScreenRect();
          break;

        // display options
        case 'r':
        case 'R':
          m_nView = (m_nView != 1) ? 1 : 0;
          break;

        case 'p':
        case 'P':
          m_nView = (m_nView != 2) ? 2 : 0;
          break;

        case 'e':
        case 'E':
          m_nManualPass = !m_nManualPass;
          break;
      }
      return 0;
  }

  return 0;
}
