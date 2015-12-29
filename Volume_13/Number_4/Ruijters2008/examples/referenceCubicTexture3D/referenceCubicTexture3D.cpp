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

// CPU based cubic B-spline pre-filtering and interpolation
// straight-forward implementation: no multi-threading, no SSE2, etc.

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
uchar* output = NULL;  //output buffer
uint3 volumeSize;
float w = 0.5;  //texture coordinate in z
uint hTimer;
uint filterMethod = 0;
uint nrOfThreads = 1;
bool animate = true;

extern "C" void render(uchar* output, uint2 imageExtent, uint3 volumeSize, float w, uint filterMethod, uint nrOfThreads);
extern "C" void prefilter(const uchar* voxels, uint3 volumeSize);

void computeFPS()
{
	const char* method[] = {"Simple CPU", "SSE"};
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
		sprintf(str, "%s interpolation, Framerate: %3.2f fps", method[filterMethod], framerate);
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
    render(output, make_uint2(width, height), volumeSize, w, filterMethod, nrOfThreads);  //call the render routine in _kernel.cpp

#ifndef _NO_DISPLAY
    // display results
    glClear(GL_COLOR_BUFFER_BIT);

    // draw image
    glDisable(GL_DEPTH_TEST);
    glRasterPos2i(0, 0);
    glDrawPixels(width, height, GL_LUMINANCE, GL_UNSIGNED_BYTE, output);

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
			filterMethod = (filterMethod + 1) % 2;
			break;
		case '[':
			nrOfThreads--;
			if (nrOfThreads < 1) nrOfThreads = 1;
			printf("Number of threads: %d\n", nrOfThreads);
			break;
		case ']':
			nrOfThreads++;
			if (nrOfThreads > height) nrOfThreads = height;
			printf("Number of threads: %d\n", nrOfThreads);
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
	delete[] output;
	output = NULL;
}

void initOpenGL()
{
	#if defined(_WIN32)
	if (wglSwapIntervalEXT) wglSwapIntervalEXT(GL_FALSE);  //disable vertical synchronization
	#endif
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
	//CUT_DEVICE_INIT(argc-4, argv);
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

	prefilter(voxels, volumeSize);  //prefilter the voxel data
	delete[] voxels;

    printf("Press space to toggle animation\n"
	       "Press 'f' to toggle between simple CPU interpolation and SSE acceleration.\n"
		   "Press '[' and ']' to de- and increase the number of threads.\n"
           "Press '+' and '-' to change displayed slice\n");

	output = new uchar[width * height];

    // initialize GLUT callback functions
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("CPU 3D texture");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);

    initOpenGL();

    atexit(cleanup);

    glutMainLoop();
    return 0;
}
