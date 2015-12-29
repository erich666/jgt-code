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
float xAngle = 0.0f, yAngle = 0.0f;
float xTranslate = 0.0f, yTranslate = 0.0f;
float zoom = 0.75f;
enum enMouseButton {mouseLeft, mouseMiddle, mouseRight, mouseNone} mouseButton = mouseNone;
int mouseX = 0, mouseY = 0;

GLuint pbo[3];  //OpenGL pixel buffer objects
GLuint renderBufferObj[2];
GLuint fbo = 0;
uint hTimer = 0;
uint filterMethod = 3;


extern "C" void render(float4* output, float3* rayCoords[2], uint2 imageExtent, uint3 volumeSize, uint filterMethod);
extern "C" void initCuda(const uchar* voxels, uint3 volumeSize);


void computeFPS()
{
	const char* method[] = {"Nearest neighbor", "Linear", "Simple cubic", "Fast cubic", "Non-prefiltered fast cubic"};
	const float updatesPerSec = 6.0f;
	static int counter = 0;
	static int countLimit = 0;

	if (counter++ > countLimit)
	{
		CUT_SAFE_CALL(cutStopTimer(hTimer));
		float framerate = 1000.0f * (float)counter / cutGetTimerValue(hTimer);
		char str[256];
		sprintf(str, "%s interpolation, Framerate: %3.1f fps", method[filterMethod], framerate);
		glutSetWindowTitle(str);
		CUT_SAFE_CALL(cutResetTimer(hTimer));
		CUT_SAFE_CALL(cutStartTimer(hTimer));
		countLimit = (int)(framerate / updatesPerSec);
		counter = 0;
	}
}

void drawTextureCoords()
{
	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

	glBegin(GL_QUADS);

	for (int side = 0; side <= 1; side++)
	{
		int move = 2 * side - 1;
		glColor3f(0,  side,  side); glVertex3f(-1,  move, move);
		glColor3f(1,  side,  side); glVertex3f( 1,  move, move);
		glColor3f(1, 1-side, side); glVertex3f( 1, -move, move);
		glColor3f(0, 1-side, side); glVertex3f(-1, -move, move);

		glColor3f(0, side, 1-side); glVertex3f(-1, move, -move);
		glColor3f(1, side, 1-side); glVertex3f( 1, move, -move);
		glColor3f(1, side,  side ); glVertex3f( 1, move,  move);
		glColor3f(0, side,  side ); glVertex3f(-1, move,  move);

		glColor3f(side, 0,  side ); glVertex3f(move, -1,  move);
		glColor3f(side, 1,  side ); glVertex3f(move,  1,  move);
		glColor3f(side, 1, 1-side); glVertex3f(move,  1, -move);
		glColor3f(side, 0, 1-side); glVertex3f(move, -1, -move);
	}

	glEnd();
}

// display results using OpenGL (called by GLUT)
void display()
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glPushMatrix();
	glTranslatef(xTranslate, yTranslate, 0.0f);
	glScalef(zoom, zoom, 0.1f);
	glRotatef(xAngle, 0, -1, 0);
	glRotatef(yAngle, -1, 0, 0);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo);

	// Draw the texture coordinates for the rays
	for (int n=0; n<2; n++)
	{
		glCullFace((n==0) ? GL_BACK : GL_FRONT);
		drawTextureCoords();
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, pbo[n]);
		glReadPixels(0, 0, width, height, GL_RGB, GL_FLOAT, NULL);  //read data into pbo (presently very slow)
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
	}

	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
	glPopMatrix();

	// map PBO to get CUDA device pointer
	float3* rayCoords[2]; 
	for (int n=0; n<2; n++) CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&rayCoords[n], pbo[n]));
	float4* output;
	CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&output, pbo[2]));
	// call the render routine in _kernel.cu
	render(output, rayCoords, make_uint2(width, height), volumeSize, filterMethod);
	for (int n=0; n<3; n++) CUDA_SAFE_CALL(cudaGLUnmapBufferObject(pbo[n]));

	// Display results, draw image from PBO
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glRasterPos2i(-1, -1);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo[2]);
    glDrawPixels(width, height, GL_RGBA, GL_FLOAT, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	computeFPS();
    glutSwapBuffers();
    glutReportErrors();
}

void idle()
{
	glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y)
{
    switch (key)
	{
    case 27:
        exit(0);
        break;
	case 'f':
		filterMethod = (filterMethod + 1) % 5;
		break;
    default:
        break;
    }
    glutPostRedisplay();
}

void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN) mouseButton = (enMouseButton)button;
	else mouseButton = mouseNone;

	mouseX = x;
	mouseY = y;
}

void motion(int x, int y)
{
	switch (mouseButton)
	{
	case mouseLeft:
		xAngle += x - mouseX;
		yAngle += y - mouseY;
		glutPostRedisplay();
		break;
	case mouseMiddle:
		xTranslate += 0.005f * (x - mouseX);
		yTranslate -= 0.005f * (y - mouseY);
		break;
	case mouseRight:
		zoom += 0.01f * (y - mouseY);
		glutPostRedisplay();
		break;
	}

	mouseX = x;
	mouseY = y;
}

void cleanup()
{
	for (int n=0; n<3; n++) CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(pbo[n]));
	glDeleteFramebuffersEXT(1, &fbo);
	glDeleteRenderbuffersEXT(2, renderBufferObj);
	glDeleteBuffers(3, pbo);
}

void initOpenGL()
{
	#if defined(_WIN32)
	//if (wglSwapIntervalEXT) wglSwapIntervalEXT(GL_FALSE);  //disable vertical synchronization
	#endif

	// creat framebuffer and renderbuffer
	glGenRenderbuffersEXT(2, renderBufferObj);
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, renderBufferObj[0]);
	glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_RGB16F_ARB, width, height);
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, renderBufferObj[1]);
	glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT, width, height);
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0);

	glGenFramebuffersEXT(1, &fbo);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo);
	glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_RENDERBUFFER_EXT, renderBufferObj[0]);
	glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, renderBufferObj[1]);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

	// create pixel buffer object
    glGenBuffersARB(3, pbo);
	for (int n=0; n<2; n++)
	{
		glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, pbo[n]);
		glBufferDataARB(GL_PIXEL_PACK_BUFFER_ARB, width*height*sizeof(float3), 0, GL_STREAM_DRAW_ARB);
	}
	glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, 0);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo[2]);
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(float4), 0, GL_STREAM_DRAW_ARB);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	for (int n=0; n<3; n++) CUDA_SAFE_CALL(cudaGLRegisterBufferObject(pbo[n]));
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

	printf("\nPress 'f' to change the interpolation mode.\n");
	printf("Use the mouse to rotate, translate and scale the volume.\n");

    // initialize GLUT callback functions
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("CUDA 3D texture");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
    glutIdleFunc(idle);

    glewInit();
    initOpenGL();

    atexit(cleanup);

    glutMainLoop();
    return 0;
}
