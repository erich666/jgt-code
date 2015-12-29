/*--------------------------------------------------------------------------*\
Copyright (c) 2008-2009, Danny Ruijters. All rights reserved.
http://www.dannyruijters.nl/cubicinterpolation/
This file is part of CUDA Cubic B-Spline Interpolation (CI).

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
*  Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
*  Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
*  Neither the name of the copyright holders nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are
those of the authors and should not be interpreted as representing official
policies, either expressed or implied.
\*--------------------------------------------------------------------------*/

// This software contains source code provided by NVIDIA Corporation.
// simpleCubicTexture3D.cpp is derived from the simpleTexture3D.cu example,
// from the CUDA SDK 2.0.


// 3D cubic texture sample
//
// This sample loads a 3D volume from disk and displays slices through it
// using 3D texture lookups. The interpolation can be switched between
// nearest neighbor, linear, pre-filtered simple cubic b-spline, and
// pre-filtered fast cubic b-spline.


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <GL/glew.h>
#if defined(_WIN32)
#include <GL/wglew.h>
#endif

#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <cuda_runtime.h>
#include <cutil.h>
#include <cuda_gl_interop.h>

typedef unsigned int uint;
typedef unsigned char uchar;

const uint width = 512, height = 512;  //output window
uint3 volumeSize;
float w = 0.5;  //texture coordinate in z
GLuint pbo;  //OpenGL pixel buffer object
uint hTimer;
uint filterMethod = 3;
bool animate = true;

extern "C" void render(uchar* output, uint2 imageExtent, uint3 volumeSize, float w, uint filterMethod);
extern "C" void initCuda(const uchar* voxels, uint3 volumeSize);

void computeFPS()
{
	const char* method[] = {"Nearest neighbor", "Linear", "Simple cubic", "Fast cubic", "Non-prefiltered fast cubic"};
	const float updatesPerSec = 6.0f;
	static int counter = 0;
	static int countLimit = 0;
	char str[256];
	
	if (!animate)
	{
		sprintf(str, "%s interpolation", method[filterMethod]);
		glutSetWindowTitle(str);
	}
	else if (counter++ > countLimit)
	{
		CUT_SAFE_CALL(cutStopTimer(hTimer));
		float framerate = 1000.0f * (float)counter / cutGetTimerValue(hTimer);
		sprintf(str, "%s interpolation, Framerate: %3.1f fps", method[filterMethod], framerate);
		glutSetWindowTitle(str);
		CUT_SAFE_CALL(cutResetTimer(hTimer));
		CUT_SAFE_CALL(cutStartTimer(hTimer));
		countLimit = (int)(framerate / updatesPerSec);
		counter = 0;
	}
}

// display results using OpenGL (called by GLUT)
void display()
{
	// map PBO to get CUDA device pointer
    uchar* output;
    CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&output, pbo));
    render(output, make_uint2(width, height), volumeSize, w, filterMethod);  //call the render routine in _kernel.cu
	CUDA_SAFE_CALL(cudaGLUnmapBufferObject(pbo));

#ifndef _NO_DISPLAY
    // display results
    glClear(GL_COLOR_BUFFER_BIT);

    // draw image from PBO
    glDisable(GL_DEPTH_TEST);
    glRasterPos2i(0, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glDrawPixels(width, height, GL_LUMINANCE, GL_UNSIGNED_BYTE, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    glutSwapBuffers();
    glutReportErrors();
#endif

	computeFPS();
}

void idle()
{
    if (animate) {
        w += 0.01f;
        while (w > 1.0f) w--;
        glutPostRedisplay();
    }
}

void keyboard(unsigned char key, int x, int y)
{
    switch(key) {
        case 27:
            exit(0);
            break;
        case '=':
        case '+':
            w += 0.01;
            while (w > 1.0f) w--;
            break;
        case '-':
            w -= 0.01;
            while (w < 0.0f) w++;
            break;
		case 'f':
			filterMethod = (filterMethod + 1) % 5;
			break;
        case ' ':
            animate = !animate;
            break;
        default:
            break;
    }
    glutPostRedisplay();
}

void reshape(int x, int y)
{
    glViewport(0, 0, x, y);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0); 
}

void cleanup()
{
	CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(pbo));    
	glDeleteBuffersARB(1, &pbo);
}

void initOpenGL()
{
	#if defined(_WIN32)
	if (wglSwapIntervalEXT) wglSwapIntervalEXT(GL_FALSE);  //disable vertical synchronization
	#endif

    // create pixel buffer object
    glGenBuffersARB(1, &pbo);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(GLubyte), 0, GL_STREAM_DRAW_ARB);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	CUDA_SAFE_CALL(cudaGLRegisterBufferObject(pbo));
}

int usage(const char* program)
{
	printf("Usage: %s filename voxelsX voxelsY voxelsZ\n", program);
	printf("\tfilename: name of the file containing the raw 8-bit voxel data\n");
	printf("\tvoxelsX: number of voxels in x-direction\n");
	printf("\tvoxelsY: number of voxels in y-direction\n");
	printf("\tvoxelsZ: number of voxels in z-direction\n");
	printf("\texample: %s ../data/bucky.raw 32 32 32\n\n", program);
	return -1;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) 
{
	if (argc < 5) return usage(argv[0]);
	CUT_DEVICE_INIT(argc-4, argv);
	CUT_SAFE_CALL(cutCreateTimer(&hTimer));

    // Obtain the voxel data
	volumeSize = make_uint3(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
	if (volumeSize.x <= 0 || volumeSize.y <= 0 || volumeSize.z <= 0) return usage(argv[0]);
	
	FILE* fp = fopen(argv[1], "rb");
	if (fp == NULL) {
		printf("Could not open file %s\n", argv[1]);
		return usage(argv[0]);
	}

	size_t nrOfVoxels = volumeSize.x * volumeSize.y * volumeSize.z;
	uchar* voxels = new uchar[nrOfVoxels];
	size_t linesRead = fread(voxels, volumeSize.x, volumeSize.y * volumeSize.z, fp);
	fclose(fp);
	if (linesRead * volumeSize.x != nrOfVoxels) {
		delete[] voxels;
		printf("Error: The number of voxels read does not correspond to the number specified!\n");
		return usage(argv[0]);
	}

	initCuda(voxels, volumeSize);  //initialize Cuda with the voxel data
	delete[] voxels;

    printf("Press space to toggle animation\n"
           "Press 'f' to toggle between nearest neighbour, linear, simple cubic and\n"
           "fast cubic texture filtering\n"
           "Press '+' and '-' to change displayed slice\n");

    // initialize GLUT callback functions
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("CUDA 3D texture");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);

    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object")) {
        fprintf(stderr, "Required OpenGL extensions missing.");
        exit(-1);
    }
    initOpenGL();

    atexit(cleanup);

    glutMainLoop();
    return 0;
}
