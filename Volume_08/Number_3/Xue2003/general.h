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
	File:		general.h

	Function:	miscellaneous stuff to (hopefully) make life easier.
*/

#ifndef GENERAL_H
#define GENERAL_H

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include <errno.h>

#ifndef WIN32

/* Assume LINUX */

#define TEMPDIR "/tmp/"

#define stricmp strcasecmp

#else

/* WINDOWS */

#include <windows.h>
#include <mmsystem.h>

/*
 * Disable stupid "converting from double to float,
 * possible loss of data" warning.
 */
#pragma warning (disable: 4244)

#define TEMPDIR "c:\\temp\\"

#endif

#ifndef M_PI
#define M_PI		3.14159265358979323846f
#endif

#ifndef bool
#define bool int
#define false	0
#define true	1
#endif

#define SCMP(x,y)	(strcmp(x,y) == 0)
#define SICMP(x,y)	(stricmp(x,y) == 0)
#define SCPY(x,y,z)	{strncpy(x,y,z-1);x[z-1]='\0';}
#define SCAT(x,y,z)	{strncat(x,y,z-strlen(x)-1);x[z-1]='\0';}
#define SQR(x)		((x) * (x))
#define MAX(a,b)	((a) > (b) ? (a) : (b))
#define MIN(a,b)	((a) < (b) ? (a) : (b))
#define DEG2RAD(a)	(((a) * M_PI) / 180.0f)
#define RAD2DEG(a)	(((a) * 180.0f) / M_PI)

#endif //GENERAL_H
