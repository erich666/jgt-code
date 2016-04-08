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
	File:		pbuffer.h

	Function:	Pixel buffer extensions class.
*/

#ifndef PBUFFER_H
#define PBUFFER_H


#include <gl/glut.h>
#include "general.h"
#include "glh_extensions.h"

typedef struct pbf_str {

#ifndef WIN32

/* Assume LINUX */

	/*
	 * I'm not going to try supporting GLX pixel buffers until my Linux
	 * X-Windows system is up and running again.
	 */

	 int dummy;

#else

/* WINDOWS - must resist Hungarian notation, must resist Hungarian notation, ... */

	int width;
	int height;
	int mode;
	bool share;

	HPBUFFERARB pBuffer;
	HDC dc;
	HGLRC rc;
	bool isValid;

	int glutWin;
	HDC glutWinDc;
	HGLRC glutWinRc;

#endif

} *pbf;

pbf pbf_create(int w, int h, int mode, bool share);
void pbf_init(pbf p, int w, int h, int mode, bool share);
void pbf_makeCurrent(pbf p);
void pbf_makeRead(pbf p);
void pbf_makeGlutWindowCurrent(pbf p);
void pbf_restore(pbf p);
void pbf_destroy(pbf p);

#endif /* PBUFFER_H */
