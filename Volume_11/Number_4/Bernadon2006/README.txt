//-----------------------------------------------------------------------------
// File: README.txt
// Desc: program instructions
// Copyright (C) 2005, Joao Comba, Fabio Bernardon, UFRGS-Brasil
//
// for further information, please contact one of te authors:
//   {comba, fabiofb}@inf.ufrgs.br
//
//-----------------------------------------------------------------------------
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
//-----------------------------------------------------------------------------


This package implements the algorithm proposed in:

"GPU-based Tiled Ray Casting using Depth Peeling", F. Bernardon,
C. Pagot, J. Comba, and C. Silva, Journal of Graphics Tools, to
appear.

This code is for educational purposes only. No guarantees of any kind
are given. 


Instructions
----------------

Here you can find some instructions to use the Ray Casting program.

This program was developed under Microsoft Windows XP with Microsoft
Visual Studio .NET 2003.

We used the DirectX 9.0c SDK update February 2005, NVidia GeForce 6800
GT with graphics driver version 71.89.

The program was designed to work with Shader Model 2.0, and should
work on all graphics card with this properties, plus the need of
Multiple Render Targets and Occlusion Queries. The HLSL compiler
present on DirectX SDK does not generates the optimal code for PS 2.0,
so we had to optimize the assembler code to work under PS 2.0. The
program, by default, is configured to load the assembler optimized
code, but you can change it on the "defines.h" file, and the program
will compile the high level code for PS 2.a. This code was also tested
under the NVidia graphics card.

Command line parameters:

HRC.exe mesh colormap opacity TFType posZ [statFile] 
	- mesh: the mesh to open. It must be in the .off format, and you
	remove the extension of the filename (ex.: spxc).
	- colormap: the transfer function file (ex.: spx.col). This
	file must have its extension.
	- opacity: the opacity map for the transfer function (ex.:
	spx.op). This file also needs the extension.
	- TFType: 2D or 3D (a 2D slice of the 3D TF or the entire volume)
	- posZ: camera Z position
optional:
	- statFile: a file where the program will write some
	statistics (see function saveStatistics())

You can change some program attributes just using the file
"defines.h", including:
 - window dimension.
 - use of HLSL shader code or optimized assembler code.
 - number of depth levels to generate.
 - number of screen tiles subdivision.

Below is a command line to run the spx file:
HRC.exe Data/spxc Data/spx.col Data/spx.op 3D -12

To save statistics:
HRC.exe Data/spxc Data/spx.col Data/spx.op 3D -12 stat.txt



Acknowledgement
----------------

This work was partially supported by the U.S. National Science
Foundation and the U.S. Department of Energy.

The SPX dataset (included in the package) is courtesy of 
Bruno Notrosso (Electricite de France).

