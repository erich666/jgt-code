/*
 * DISPLAY.C
 *
 *	Windows display code, created by Chris Babcock (babcock@rtp.idt.com)
 *	Click the left mouse button to exit.
 */

#include <windows.h>
#include <stdio.h>
#include <math.h>
#include <commctrl.h>

#define APPCLASSNAME "DISPLAY"

HWND      hWnd;
HDC       hDC;
HINSTANCE hInst;

char szAppName[] = "DISPLAY";
char szTitle[]   = "DISPLAY";
char szAppClass[32];

int Width  = 640;
int Height = 480;
int InitOk = 0;

#define GAMMA 1.8

// Gamma table
BYTE gamma[65536];

LRESULT CALLBACK WndProc(
      HWND hWnd,         // window handle
      UINT message,      // type of message
      WPARAM uParam,     // additional information
      LPARAM lParam      // additional information
)
{                               /*  Message Handler */
  static float xo,yo;
  PAINTSTRUCT ps;

  switch (message) {
    case WM_CREATE:
         hDC = GetDC( hWnd);
         break;

    case WM_CLOSE:
         DestroyWindow(hWnd);
         break;

    case WM_SIZE:
         Width = LOWORD(lParam);
         Height = HIWORD(lParam);
         break ;

    case WM_PAINT:
         BeginPaint (hWnd, &ps);
         EndPaint(hWnd, &ps);
         break;

    case WM_DESTROY:  // message: window being destroyed
         ReleaseDC(hWnd,hDC);
         PostQuitMessage(0);
         break;

    case WM_LBUTTONDOWN:       // left mouse button down = quit
         PostQuitMessage(0);
         break;

    default:          // Passes it on if unproccessed
         return (DefWindowProc(hWnd, message, uParam, lParam));
  }
  return (0);
}


BOOL InitApplication(HINSTANCE hInstance,COLORREF background)
{       /* called for first instance of app */
  WNDCLASS wc;

  wc.style = (CS_HREDRAW | CS_VREDRAW | CS_OWNDC | CS_DBLCLKS | CS_BYTEALIGNCLIENT);
  wc.lpfnWndProc   = (WNDPROC)WndProc;             // Window Procedure
  wc.cbClsExtra    = 0;                            // No per-class extra data.
  wc.cbWndExtra    = sizeof (DWORD);               // 4-bytes extra data.
  wc.hInstance     = hInstance;                    // Owner of this class
  wc.hIcon         = 0;
  wc.hCursor       = LoadCursor(NULL, IDC_ARROW);  // Cursor
  wc.hbrBackground = CreateSolidBrush(background); // Default color
  wc.lpszMenuName  = NULL;                         // Menu name from .RC
  wc.lpszClassName = szAppClass;                   // Name to register as
  if (RegisterClass(&wc)==FALSE) return FALSE;
  return TRUE;
}


BOOL InitInstance(HINSTANCE hInstance, WORD nCmdShow)
{
  RECT WinRect;
  HWND wHdl;

  SetRect(&WinRect,0,0,Width,Height);
  AdjustWindowRect(&WinRect,WS_OVERLAPPEDWINDOW,0); // 1 = has menu

  hInst = hInstance; // save instance handle
  wHdl = CreateWindow(szAppClass,                 // See RegisterClass()
                      szTitle,                    // Window title bar
                      WS_OVERLAPPEDWINDOW,        // Window style.
                      CW_USEDEFAULT,              // init x pos
                      CW_USEDEFAULT,              // init y pos
                      WinRect.right-WinRect.left, // init x size
                      WinRect.bottom-WinRect.top, // init y size
                      NULL,                       // Overlapped windows = no parent
                      NULL,                       // Use the window class menu.
                      hInstance,                  // This instance owns this window
                      NULL                        // We don't use any data
                     );
  if (!wHdl) return (FALSE);
  ShowWindow(wHdl, nCmdShow);             // Show the window

  UpdateWindow(wHdl);                 // Sends WM_PAINT message
  hWnd = wHdl;
  return (TRUE);                      // We succeeded...
}


void init_screen(int background)
{
  MSG msg;
  double inten;
  int n;

  hInst = GetModuleHandle(NULL);
  sprintf(szAppClass,"%s%d", APPCLASSNAME, hInst);
  if (!InitApplication(hInst,(COLORREF)background)) return;
  if (!InitInstance(hInst,SW_SHOWNORMAL)) return;

  while (PeekMessage(&msg,NULL,0,0,PM_REMOVE)) {
    TranslateMessage(&msg);
    DispatchMessage(&msg);
  }
  InitOk = 1;

  // generate gamma table
  for (n=0; n<65536; n++) {
    inten = pow((double)n/65536.0,1.0/GAMMA);
    if (inten > 1.0) inten = 1.0;
    gamma[n] = (BYTE)(inten * 255.0);
  }
}

void close_screen(void)
{
  MSG msg;

  if (!InitOk) return;
  while (1) {
    if (!GetMessage(&msg,NULL,0,0)) return;
    if (msg.message == WM_QUIT) return;
    TranslateMessage(&msg);
    DispatchMessage(&msg);
  }
}

void draw_pixel(long int x, long int y, unsigned short r, unsigned short g, unsigned short b)
{
  if (!InitOk) return;
  SetPixel(hDC,x,y,RGB(gamma[r],gamma[g],gamma[b]));
}

