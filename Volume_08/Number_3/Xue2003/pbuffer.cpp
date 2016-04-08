/*
Copyright (C) 2000-2001 Adrian Welbourn

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.

*/
/*
	File:		pbuffer.c

	Function:	Pixel buffer extensions class.
*/

#include "pbuffer.h"


#ifndef WIN32

/* Assume LINUX */

	/*
	 * I'm not going to try supporting GLX pixel buffers until my Linux
	 * X-Windows system is up and running again.
	 */

#else

/* WINDOWS */

#define MAX_PFORMATS	256
#define MAX_ATTRIBS		32

pbf pbf_create(int w, int h, int mode, bool share)
{
	pbf p = (pbf)malloc(sizeof(struct pbf_str));
	if (p)
		pbf_init(p, w, h, mode, share);
	return p;
}

void pbf_init(pbf p, int w, int h, int mode, bool share)
{
	int iattribs[2*MAX_ATTRIBS];
	float fattribs[2*MAX_ATTRIBS];
	int pformat[MAX_PFORMATS];
	int nfattribs = 0;
	int niattribs = 0;
	unsigned int nformats;
	int format;
	HDC dc = wglGetCurrentDC();
	HGLRC rc = wglGetCurrentContext();

	if (p) {
		memset(p, 0, sizeof(struct pbf_str));
		memset(iattribs, 0, sizeof(iattribs));
		memset(fattribs, 0, sizeof(fattribs));

		p->width = w;
		p->height = h;
		p->mode = mode;
		p->share = share;
		p->glutWin = glutGetWindow();
		p->glutWinDc = dc;
		p->glutWinRc = rc;
		p->isValid = false;

		/* pBuffer pixel format must be "p-buffer capable" */
		iattribs[2*niattribs    ] = WGL_DRAW_TO_PBUFFER_ARB;
		iattribs[2*niattribs + 1] = true;
		niattribs++;

		if (mode & GLUT_INDEX) {
			iattribs[2*niattribs    ] = WGL_PIXEL_TYPE_ARB;
			iattribs[2*niattribs + 1] = WGL_TYPE_COLORINDEX_ARB;
			niattribs++;
		}
		else {
			iattribs[2*niattribs    ] = WGL_PIXEL_TYPE_ARB;
			iattribs[2*niattribs + 1] = WGL_TYPE_RGBA_ARB;
			niattribs++;
		}

		if (mode & GLUT_DOUBLE) {
			iattribs[2*niattribs    ] = WGL_DOUBLE_BUFFER_ARB;
			iattribs[2*niattribs + 1] = true;
			niattribs++;
		}

		if (mode & GLUT_DEPTH) {
			iattribs[2*niattribs    ] = WGL_DEPTH_BITS_ARB;
			iattribs[2*niattribs + 1] = true;
			niattribs++;
		}
		if (mode & GLUT_STENCIL) {
			iattribs[2*niattribs    ] = WGL_STENCIL_BITS_ARB;
			iattribs[2*niattribs + 1] = true;
			niattribs++;
		}
		if (mode & GLUT_ACCUM) {
			iattribs[2*niattribs    ] = WGL_ACCUM_BITS_ARB;
			iattribs[2*niattribs + 1] = true;
			niattribs++;
		}

		iattribs[2*niattribs    ] = WGL_SUPPORT_OPENGL_ARB;
		iattribs[2*niattribs + 1] = true;
		niattribs++;

		if (!wglChoosePixelFormatARB(
				dc, iattribs, fattribs, MAX_PFORMATS, pformat, &nformats))
			return;
		format = pformat[0];

		iattribs[0] = 0;
		p->pBuffer = wglCreatePbufferARB(
											dc, format, w, h, iattribs);
		if (!p->pBuffer)
			return;

		p->dc = wglGetPbufferDCARB(p->pBuffer);
		if (!p->dc) {
			wglDestroyPbufferARB(p->pBuffer);
			return;
		}

		p->rc = wglCreateContext(p->dc);
		if (!p->rc) {
			wglReleasePbufferDCARB(p->pBuffer, p->dc);
			wglDestroyPbufferARB(p->pBuffer);
			return;
		}

		if (share) {
			if(!wglShareLists(rc, p->rc))
				p->share = false;
		}

		/* Determine actual width and height */
		wglQueryPbufferARB(p->pBuffer,
								WGL_PBUFFER_WIDTH_ARB, &p->width);
		wglQueryPbufferARB(p->pBuffer,
								WGL_PBUFFER_HEIGHT_ARB, &p->height);

		p->isValid = true;
	}
}

void pbf_makeCurrent(pbf p)
{
	if (p->isValid) {
		wglMakeCurrent(p->dc, p->rc);
	}
}

void pbf_makeRead(pbf p)
{
//	glutSetWindow(p->glutWin);
//	if (p->isValid) {
//		wglMakeContextCurrentARB(p->glutWinDc, p->dc, p->glutWinRc);
//	}
}

void pbf_makeGlutWindowCurrent(pbf p)
{
	glutSetWindow(p->glutWin);
}

void pbf_restore(pbf p)
{
	int lost = 0;
	if (p->isValid) {
		/*
		 * Check to see if pbuffer memory was lost due to a display
		 * mode change.
		 */
		wglQueryPbufferARB(p->pBuffer, WGL_PBUFFER_LOST_ARB, &lost);
		if (lost) {
			wglDeleteContext(p->rc);
			wglReleasePbufferDCARB(p->pBuffer, p->dc);
			wglDestroyPbufferARB(p->pBuffer);
			pbf_init(p, p->width, p->height, p->mode, p->share);
		}
	}
}

void pbf_destroy(pbf p)
{
	if (p) {
		if (p->isValid) {
			wglDeleteContext(p->rc);
			wglReleasePbufferDCARB(p->pBuffer, p->dc);
			wglDestroyPbufferARB(p->pBuffer);
		}
		free(p);
	}
}

#endif
