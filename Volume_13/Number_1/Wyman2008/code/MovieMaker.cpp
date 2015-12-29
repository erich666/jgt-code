
/*************************************************
** MovieMaker.cpp                               **
** -----------                                  **
**                                              **
** Code for capturing movies from the GL window **
**    This is from an nVidia demo, and it only  **
**    works under Windows systems.              **
**                                              **
** Chris Wyman (9/07/2006)                      **
*************************************************/

#include "MovieMaker.h"
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <windowsx.h>
#include <GL/glut.h>

HANDLE  MakeDib( HBITMAP hbitmap, UINT bits );
HBITMAP LoadBMPFromFB( int w, int h );

/*
    ===============================================
        Constructors, Destructor
    ===============================================
*/


MovieMaker::MovieMaker()
{
    sprintf( fname, "movie.avi" );
    width  = -1;
    height = -1;

    bOK = true;
    nFrames = 0;
	ready = 1;

  	pfile = NULL;
	ps = NULL;
	psCompressed = NULL;
	psText = NULL;
	aopts[0] = &opts;

    // Check VFW version.
	WORD wVer = HIWORD( VideoForWindowsVersion() );
	if ( wVer < 0x010A )
        {
        fprintf( stderr, "VFW version is too old.\n" );
        exit( -1 );
	    }
	else
	    {
		AVIFileInit();
	    }
}

void MovieMaker::PrepareForCapture( void )
{
	/* make sure everything from the last capture is clear */
	if (ps)
		AVIStreamClose(ps);
	if (psCompressed)
		AVIStreamClose(psCompressed);
	if (psText)
		AVIStreamClose(psText);
	if (pfile)
        {
		AVIFileClose(pfile);
        }

	WORD wVer = HIWORD(VideoForWindowsVersion());
	if (wVer >= 0x010A)
	    {
		AVIFileExit();
	    }

	/* reset error bounds & screen res */
	bOK = true;
	ready = 1;
	nFrames = 0;
	estFramesPerSecond = 30;
	width = -1;
	height = -1;

	/* reset internals */
	pfile = NULL;
	ps = NULL;
	psCompressed = NULL;
	psText = NULL;
	aopts[0] = &opts;

	/* initialize a new AVI file */
	if (wVer >= 0x010A)
	{
		AVIFileInit();
	}
	else
	{
		fprintf( stderr, "VFW version is too old.\n" );
        exit( -1 );
	}
}


MovieMaker::~MovieMaker()
{
	if (ps)
		AVIStreamClose(ps);

	if (psCompressed)
		AVIStreamClose(psCompressed);

	if (psText)
		AVIStreamClose(psText);

	if (pfile)
        {
		AVIFileClose(pfile);
        }

	WORD wVer = HIWORD(VideoForWindowsVersion());
	if (wVer >= 0x010A)
	    {
		AVIFileExit();
	    }
}

void MovieMaker::StartCapture( const char *name, int framesPerSecond )
{
	if (!ready) PrepareForCapture();

    strcpy( fname, name );

    // Get the width and height.
    width = glutGet( GLUT_WINDOW_WIDTH  );
    height = glutGet( GLUT_WINDOW_HEIGHT  );

    fprintf( stderr, "Starting %d x %d capture to file: %s\n", width, height, fname );
	estFramesPerSecond = framesPerSecond;
	ready = 0;
}

void MovieMaker::EndCapture()
{
	if (ps)				AVIStreamClose(ps);
 	if (psCompressed)	AVIStreamClose(psCompressed);
	if (psText)			AVIStreamClose(psText);
 	if (pfile)			AVIFileClose(pfile);
  
	ps = psCompressed = psText = NULL;
	pfile = NULL;

	WORD wVer = HIWORD(VideoForWindowsVersion());
	if (wVer >= 0x010A) AVIFileExit();
}

int MovieMaker::Snap()
{
	HRESULT hr;

	if (!bOK) return 0;

    // Get an image and stuff it into a bitmap.
    HBITMAP bmp;
    bmp = LoadBMPFromFB( width, height );

	LPBITMAPINFOHEADER alpbi = (LPBITMAPINFOHEADER)GlobalLock(MakeDib(bmp, 32));
    DeleteObject( bmp );

	if (alpbi == NULL)
        {
        bOK = false;
		return 0;
        }
	if (width>=0 && width != alpbi->biWidth)
	{
		GlobalFreePtr(alpbi);
        bOK = false;
		return 0;
	}
	if (height>=0 && height != alpbi->biHeight)
	{
		GlobalFreePtr(alpbi);
        bOK = false;
		return 0;
	}
	width = alpbi->biWidth;
	height = alpbi->biHeight;
	if (nFrames == 0)
	{
		hr = AVIFileOpen(&pfile,					// returned file pointer
			       fname,							// file name
				   OF_WRITE | OF_CREATE,		    // mode to open file with
				   NULL);							// use handler determined
													// from file extension....
		if (hr != AVIERR_OK)
		{
			GlobalFreePtr(alpbi);
			bOK = false;
			return 0;
		}
		_fmemset(&strhdr, 0, sizeof(strhdr));
		strhdr.fccType                = streamtypeVIDEO;	// stream type
		strhdr.fccHandler             = 0;
		strhdr.dwScale                = 1;					// frames per second = dwRate / dwScale
		strhdr.dwRate                 = estFramesPerSecond; 
		strhdr.dwSuggestedBufferSize  = alpbi->biSizeImage;
		SetRect(&strhdr.rcFrame, 0, 0,						// rectangle for stream
			(int) alpbi->biWidth,
			(int) alpbi->biHeight);

		// And create the stream;
		hr = AVIFileCreateStream(pfile,		    // file pointer
						         &ps,		    // returned stream pointer
								 &strhdr);	    // stream header
		if (hr != AVIERR_OK)
		{
			GlobalFreePtr(alpbi);
			bOK = false;
			return 0;
		}

		_fmemset(&opts, 0, sizeof(opts));

		if (!AVISaveOptions(NULL, 0, 1, &ps, (LPAVICOMPRESSOPTIONS FAR *) &aopts))
		{
            fprintf( stderr, "AVISaveOptions failed.\n" );
			GlobalFreePtr(alpbi);
			bOK = false;
			return 0;
		}

		hr = AVIMakeCompressedStream(&psCompressed, ps, &opts, NULL);
		if (hr != AVIERR_OK)
		{
            fprintf( stderr, "AVIMakeCompressedStream failed.\n" );
			GlobalFreePtr(alpbi);
			bOK = false;
			return 0;
		}

		hr = AVIStreamSetFormat(psCompressed, 0,
					   alpbi,			 // stream format
				       alpbi->biSize +   // format size
				       alpbi->biClrUsed * sizeof(RGBQUAD));
		if (hr != AVIERR_OK)
		{
            fprintf( stderr, "AVIStreamSetFormat failed.\n" );
			GlobalFreePtr(alpbi);
			bOK = false;
			return 0;
		}
	}

	// Now actual writing
	hr = AVIStreamWrite(psCompressed,	// stream pointer
		nFrames * 1,					// "time" (in # of frames) of this frame
		1,								// number of frames to write
		(LPBYTE) alpbi +				// pointer to data
			alpbi->biSize +
			alpbi->biClrUsed * sizeof(RGBQUAD),
			alpbi->biSizeImage,			// size of this frame
		AVIIF_KEYFRAME,					// flags....
		NULL,
		NULL);
	if (hr != AVIERR_OK)
	{
        fprintf( stderr, "AVIStreamWrite failed.\n" );
		GlobalFreePtr(alpbi);
		bOK = false;
		return 0;
	}

	GlobalFreePtr(alpbi);
	nFrames++;

	return nFrames;
}

static HANDLE  MakeDib( HBITMAP hbitmap, UINT bits )
{
	HANDLE              hdib ;
	HDC                 hdc ;
	BITMAP              bitmap ;
	UINT                wLineLen ;
	DWORD               dwSize ;
	DWORD               wColSize ;
	LPBITMAPINFOHEADER  lpbi ;
	LPBYTE              lpBits ;
	
	GetObject(hbitmap,sizeof(BITMAP),&bitmap) ;

	//
	// DWORD align the width of the DIB
	// Figure out the size of the colour table
	// Calculate the size of the DIB
	//
	wLineLen = (bitmap.bmWidth*bits+31)/32 * 4;
	wColSize = sizeof(RGBQUAD)*((bits <= 8) ? 1<<bits : 0);
	dwSize = sizeof(BITMAPINFOHEADER) + wColSize +
		(DWORD)(UINT)wLineLen*(DWORD)(UINT)bitmap.bmHeight;

	//
	// Allocate room for a DIB and set the LPBI fields
	//
	hdib = GlobalAlloc(GHND,dwSize);
	if (!hdib)
		return hdib ;

	lpbi = (LPBITMAPINFOHEADER)GlobalLock(hdib) ;

	lpbi->biSize = sizeof(BITMAPINFOHEADER) ;
	lpbi->biWidth = bitmap.bmWidth ;
	lpbi->biHeight = bitmap.bmHeight ;
	lpbi->biPlanes = 1 ;
	lpbi->biBitCount = (WORD) bits ;
	lpbi->biCompression = BI_RGB ;
	lpbi->biSizeImage = dwSize - sizeof(BITMAPINFOHEADER) - wColSize ;
	lpbi->biXPelsPerMeter = 0 ;
	lpbi->biYPelsPerMeter = 0 ;
	lpbi->biClrUsed = (bits <= 8) ? 1<<bits : 0;
	lpbi->biClrImportant = 0 ;

	//
	// Get the bits from the bitmap and stuff them after the LPBI
	//
	lpBits = (LPBYTE)(lpbi+1)+wColSize ;

	hdc = CreateCompatibleDC(NULL) ;

	GetDIBits(hdc,hbitmap,0,bitmap.bmHeight,lpBits,(LPBITMAPINFO)lpbi, DIB_RGB_COLORS);

	// Fix this if GetDIBits messed it up....
	lpbi->biClrUsed = (bits <= 8) ? 1<<bits : 0;

	DeleteDC(hdc) ;
	GlobalUnlock(hdib);

	return hdib ;
}


static HBITMAP LoadBMPFromFB( int w, int h )
{
    // Create a normal DC and a memory DC for the entire screen. The 
    // normal DC provides a "snapshot" of the screen contents. The 
    // memory DC keeps a copy of this "snapshot" in the associated 
    // bitmap. 
 
    HDC hdcScreen = wglGetCurrentDC();
    HDC hdcCompatible = CreateCompatibleDC(hdcScreen); 
 
    // Create a compatible bitmap for hdcScreen. 

    HBITMAP hbmScreen = CreateCompatibleBitmap(hdcScreen, w,h ); 

    if (hbmScreen == 0)
        {
			fprintf( stderr, "hbmScreen == NULL\nExiting.\n" );
			exit( -1 );
        }
 
    // Select the bitmaps into the compatible DC. 
 
    if (!SelectObject(hdcCompatible, hbmScreen)) 
        {
			fprintf( stderr, "Couldn't SelectObject()\nExiting.\n" );
			exit( -1 );
        }
 
    //Copy color data for the entire display into a 
    //bitmap that is selected into a compatible DC. 
 
    if (!BitBlt(hdcCompatible, 
                 0,0, 
                 w, h, 
                 hdcScreen, 
                 0, 0,
                 SRCCOPY)) 
        {
			fprintf( stderr, "Screen to Compat Blt Failed\nExiting.\n" );
			exit( -1 );
        }

    DeleteDC( hdcCompatible );
    return( hbmScreen );
}


