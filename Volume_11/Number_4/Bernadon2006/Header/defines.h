//-----------------------------------------------------------------------------
// File: defines.h
// Desc: holds the controller defines of the program
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

#ifndef _DEFINES_H_
#define _DEFINES_H_

#define SafeRelease(p) if (p) { p->Release(); p = NULL; }
#define SafeDelete(p) if (p) { delete(p); p = NULL; }
#define SafeDeleteArray(p) if (p) { delete[] p; p = NULL;}

#define PI 3.14159265358979

#define DisplayError(str) MessageBox(NULL, str, "Error", MB_OK | MB_ICONERROR);

// program default - currently the code only works with squared windows
#define SCREEN_DIMENSION 512

// control the number of depth levels to compute on each frame
#define MAX_DEPTH 3

// control the screen tiles subdivision number. The program can handle
// a diferent value per width/height of the window, but we use a single
// value for simplicity.
#define NTILES 6

// Control when the program will use High Level code or optimized assembler
// version. To use ps_2_0 shader profile, use the optimized assembler code,
// since current compilers do not seen to be able to generate a code full
// optimized.

//#define HLSL_SHADERS

#define VS_COMPILER_TARGET vs_2_a
#define PS_COMPILER_TARGET ps_2_a

//#define VS_COMPILER_TARGET vs_3_0
//#define PS_COMPILER_TARGET ps_3_0

//command line args
#define NARGS       6
#define MESHFILE    0
#define TFFILE      1
#define TFOPFILE    2
#define LUTPOS      3
#define EYEPOS      4
#define STATISTICS  5

#define SIGMA (1.0f/8192.0f)

struct CUSTOMVERTEX2 {

  float x, y, z;
	float tu1,tv1;
	float tu2,tv2;
	float tu3,tv3;
};


#endif // _DEFINES_H_
