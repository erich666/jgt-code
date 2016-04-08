#ifndef __DEFINES_H
#define __DEFINES_H

#define __VERTEX_PROGRAM__

//#define _GL_IMMEDIATE
#define _GL_VERTEXSTREAM
//#define _GL_CONVOLUTION

#ifdef _GL_CONVOLUTION
#define __PBUFFER__
#define __X_RAY_MODEL__
#endif

#endif /* __DEFINES_H */
