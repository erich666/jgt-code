

This archive contains source code for a ray tracer that implements
lightweight bounding volumes.

Copyright 2006 David Cline.  
All Rights Reserved.

Permission to use, copy, modify and distribute this software and its 
documentation for educational, research and non-profit purposes, without fee, 
and without a written agreement is hereby granted, provided that the above 
copyright notice and following disclaimer appears in all copies.

IN NO EVENT SHALL DAVID CLINE OR BYU BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, 
SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT 
OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THEY HAVE BEEN ADVISED 
OF THE POSSIBILITY OF SUCH DAMAGES.

DAVID CLINE AND BYU SPECIFICALLY DISCLAIM ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. 
THE SOFTWARE PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND DAVID CLINE AND BYU HAVE NO 
OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

--------------------------------------------------------

This archive contains 5 source code files that implements a
ray tracer with lightweight bounding volumes:

	Geometry.h, Geometry.cpp, LBVH.h, LBVH.cpp and lbvhTrace.cpp. 

The LBVH implementation is contained in LBVH.h and LBVH.cpp.
The other source files are supporting code for the lbvhTrace.cpp
main program.

The archive also contains three example scenes:

	bunnyScene.txt, bunnies.txt and buddha.txt
	
as well as thumbnails of the scenes in jpeg format.
The scenes reference .ply files of the Stanford bunny
and happy buddha, available from the Stanford 3D scanning
repository:

	http://graphics.stanford.edu/data/3Dscanrep/

Also note that the parsing capabilities of the ray tracer
are extremely limited.

