#include <iostream>

#include <GLEW/glew.h>
#include <GL/glu.h>

#include "DepthPeeler.h"
#include "CGLA/Mat4x4f.h"


using namespace CGLA;
using namespace std;

namespace Graphics
{
	namespace
	{

		std::string arbfp1_str = 
		"!!ARBfp1.0\n"
		"TEX result.color.w, fragment.position, texture[0], RECT;\n"
		"MOV result.color.xyz, fragment.color.primary;\n"
		"END\n";
	}

	
	void DepthPeeler::disable_depth_test2()
	{
		// Switch to standard pipeline.
		glDisable(GL_FRAGMENT_PROGRAM_ARB);
		glDisable(GL_ALPHA_TEST);
	}
	
	void DepthPeeler::enable_depth_test2()
	{
		// Bind our depth test fragment program.
		glEnable(GL_FRAGMENT_PROGRAM_ARB);
		glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB,frag_prog);
		glEnable(GL_ALPHA_TEST);

		// bind the depth texture (our second z buffer)
		glBindTexture(GL_TEXTURE_RECTANGLE_NV, ztex);


		// Set up texture matrix to scale window coordinates
		// to texture coordinates
		glMatrixMode(GL_TEXTURE);
		glLoadIdentity();
		glScalef(width, height, 1);
		glTranslatef(.5, .5, 0.5);
		glScalef(.5, .5, .5);
		glMatrixMode(GL_MODELVIEW);
	}


	DepthPeeler::~DepthPeeler()
	{
		glDeleteProgramsARB(1, &frag_prog);
		glDisable(GL_FRAGMENT_PROGRAM_ARB);

		glMatrixMode(GL_TEXTURE);
		glLoadMatrixd(texmat);
		glMatrixMode(GL_MODELVIEW);

 		glDeleteTextures(1, &ztex);
		glPopAttrib();
	}

	void DepthPeeler::read_back_depth()
	{
		// Copy frame buffer depth component to texture
		glBindTexture(GL_TEXTURE_RECTANGLE_NV, ztex);
 		glCopyTexSubImage2D(GL_TEXTURE_RECTANGLE_NV, 0, 0,0,0,0, width, height);
	}


	
	DepthPeeler::DepthPeeler(int _width, int _height): 
		width(_width), height(_height)
	{
		glPushAttrib(GL_ALL_ATTRIB_BITS);
		glEnable(GL_DEPTH_TEST);
		glAlphaFunc(GL_GREATER, 0.0f);

		glGetDoublev(GL_TEXTURE_MATRIX, texmat);

		glGenProgramsARB(1,&frag_prog);
		int errorPos;
		glEnable(GL_FRAGMENT_PROGRAM_ARB);
		glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB,frag_prog);
		glProgramStringARB(GL_FRAGMENT_PROGRAM_ARB, 
											 GL_PROGRAM_FORMAT_ASCII_ARB, 
											 arbfp1_str.length(), 
											 arbfp1_str.data());
		glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &errorPos);
		if(errorPos != -1)
			abort();

		// Create depth texture
		glGenTextures(1, &ztex);
		glBindTexture(GL_TEXTURE_RECTANGLE_NV, ztex);
		unsigned int* zbuf = new unsigned int[ width * height ];
		glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0,GL_DEPTH_COMPONENT24,
								 width, height, 0,
								 GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, zbuf);
		glTexParameteri(GL_TEXTURE_RECTANGLE_NV, 
										GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_RECTANGLE_NV, 
										GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_RECTANGLE_NV, 
										GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_RECTANGLE_NV, 
										GL_TEXTURE_WRAP_T, GL_CLAMP);
 		glTexParameteri(GL_TEXTURE_RECTANGLE_NV, 
 										GL_DEPTH_TEXTURE_MODE, GL_ALPHA);			
 		glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_COMPARE_MODE, 
 										GL_COMPARE_R_TO_TEXTURE);
 		glTexParameteri(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_COMPARE_FUNC,
 										GL_GREATER);
		delete [] zbuf;
	}

}	
