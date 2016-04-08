Ther are three rendering modes supported in the program. 
1) immediate mode
2) vertex program
3) point convolution using pbuffer and convolution extension.

In the defines.h, uncomment what you want.  The defines.h looks like the following:

//#define _GL_IMMEDIATE
//#define _GL_VERTEXSTREAM
//#define _GL_CONVOLUTION

If you want to use the vertex program rendering, then uncomment _GL_VERTEXSTREAM.  It looks as:

//#define _GL_IMMEDIATE
#define _GL_VERTEXSTREAM
//#define _GL_CONVOLUTION

Only one dataset hipip is included due to file size limitation.  The other datasets can be found at http://www.volvis.org


