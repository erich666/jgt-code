/************************************************
** framebufferObject.cpp                       **
** ---------------------                       **
**                                             **
** This is the frame-work for general purpose  **
**   initialization of a framebuffer object,   **
**   as specified in the OpenGL extension:     **
**       GL_EXT_FRAMEBUFFER_OBJECT             **
**                                             **
** Since this is an OpenGL extension, not WGL, **
**   it should be much more portable (and      **
**   supposedly) faster than p-buffers and     **
**   render-to-texture.                        **
**                                             **
** Chris Wyman (4/27/2005)                     **
************************************************/

#include <stdio.h>
#include <string.h>
#include "framebufferObject.h"

FrameBuffer::FrameBuffer( char *name )
{
	glGetIntegerv( GL_MAX_COLOR_ATTACHMENTS_EXT, &maxColorBuffers );
	colorIDs = new GLuint[maxColorBuffers];
	depthID = 0;
	stencilID = 0;
	for (int i=0; i<maxColorBuffers; i++)
		colorIDs[i] = 0;
	prevFrameBuf = 0;
	width = height = 0;
	glGenFramebuffersEXT( 1, &ID );

	if (!name)
		sprintf( fbName, "Framebuffer %d", ID );
	else
		strncpy( fbName, name, 79 );
}

FrameBuffer::FrameBuffer( int width, int height, char *name ) : width( width ), height( height )
{
	glGetIntegerv( GL_MAX_COLOR_ATTACHMENTS_EXT, &maxColorBuffers );
	colorIDs = new GLuint[maxColorBuffers];
	depthID = 0;
	stencilID = 0;
	for (int i=0; i<maxColorBuffers; i++)
		colorIDs[i] = 0;
	prevFrameBuf = 0;
	glGenFramebuffersEXT( 1, &ID );

	if (!name)
		sprintf( fbName, "Framebuffer %d", ID );
	else
		strncpy( fbName, name, 79 );
}


FrameBuffer::~FrameBuffer( )
{
	// unbind this buffer, if bound
	GLint tmpFB;
	glGetIntegerv( GL_FRAMEBUFFER_BINDING_EXT, &tmpFB );
	if (tmpFB == ID)
		glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, prevFrameBuf );
	
	// delete the stencil & depth renderbuffers
	if (depthID)
		glDeleteRenderbuffersEXT(1, &depthID);
	if (stencilID)
		glDeleteRenderbuffersEXT(1, &stencilID);

	// delete the framebuffer
	glDeleteFramebuffersEXT( 1, &ID );
	delete [] colorIDs;
}



// check to see if the framebuffer 'fb' is complete (i.e., renderable) 
//    if fb==NULL, then check the currently bound framebuffer          
GLenum FrameBuffer::CheckFramebufferStatus( int printMessage )
{
	GLenum error;
	GLint oldFB = 0;
	glGetIntegerv( GL_FRAMEBUFFER_BINDING_EXT, &oldFB );

	// there may be some other framebuffer currently bound...  if so, save it 
	if ( oldFB != ID )
		glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, ID);
	
	// check the error status of this framebuffer */
	error = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);

	// if error != GL_FRAMEBUFFER_COMPLETE_EXT, there's an error of some sort 
	if (printMessage)
	{
		switch(error)
		{
			case GL_FRAMEBUFFER_COMPLETE_EXT:
				break;
			case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT:
				printf("Error!  %s missing a required image/buffer attachment!\n", fbName);
				break;
			case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT:
				printf("Error!  %s has no images/buffers attached!\n", fbName);
				break;
			case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:
				printf("Error!  %s has mismatched image/buffer dimensions!\n", fbName);
				break;
			case GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT:
				printf("Error!  %s's colorbuffer attachments have different types!\n", fbName);
				break;
			case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT:
				printf("Error!  %s trying to draw to non-attached color buffer!\n", fbName);
				break;
			case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT:
				printf("Error!  %s trying to read from a non-attached color buffer!\n", fbName);
				break;
			case GL_FRAMEBUFFER_UNSUPPORTED_EXT:
				printf("Error!  %s format is not supported by current graphics card/driver!\n", fbName);
				break;
			default:
				printf("*UNKNOWN ERROR* reported from glCheckFramebufferStatusEXT() for %s!\n", fbName);
				break;
		}
	}

	// if this was not the current framebuffer, switch back! 
	if ( oldFB != ID )
		glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, oldFB );

	return error;
}

// attach a texture (colorTexID) to one of the color buffer attachment points 
//    This function is not completely general, as it does not allow specification
//    of which MIPmap level to draw to (it uses the base, level 0).
int FrameBuffer::AttachColorTexture( GLuint colorTexID, int colorBuffer )
{
	// If the colorBuffer value is valid, then bind the texture to the color buffer.
	if (colorBuffer < maxColorBuffers)
	{
		BindBuffer();
		glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT+colorBuffer, 
								   GL_TEXTURE_2D, colorTexID, 0);

		UnbindBuffer();
	}
	else
		return 0;
	colorIDs[colorBuffer] = colorTexID;
	return 1;
}


// attach a texture (depthTexID) to the depth buffer attachment point.
int FrameBuffer::AttachDepthTexture( GLuint depthTexID )
{
	BindBuffer();
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, 
							  GL_TEXTURE_2D, depthTexID, 0);
	depthID = depthTexID;
	UnbindBuffer();
	return 1;
}

// attach a texture (stencilTexID) to the stencil buffer attachment point.
int FrameBuffer::AttachStencilTexture( GLuint stencilTexID )
{
	BindBuffer();
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_STENCIL_ATTACHMENT_EXT, 
							  GL_TEXTURE_2D, stencilTexID, 0);
	stencilID = stencilTexID;
	UnbindBuffer();
	return 1;
}


// attach a renderbuffer (colorBufID) to one of the color buffer attachment points 
int FrameBuffer::AttachColorBuffer( GLuint colorBufID, int colorBuffer )
{
	// If the colorBuffer value is valid, then bind the texture to the color buffer.
	if (colorBuffer < maxColorBuffers)
	{
		BindBuffer();
		glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, colorBufID);
		glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_RGBA, 
		                         width, height);
		glFramebufferRenderbufferEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT+colorBuffer, 
								      GL_RENDERBUFFER_EXT, colorBufID);
		UnbindBuffer();
	}
	else
		return 0;
	colorIDs[colorBuffer] = colorBufID;
	return 1;
}

// attach a renderbuffer (depthBufID) to the depth buffer attachment point.
int FrameBuffer::AttachDepthBuffer( GLuint depthBufID )
{
	BindBuffer();
	//glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, depthBufID);
    //glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT24, 
	//                         width, height);
	glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, 
								   GL_RENDERBUFFER_EXT, depthBufID);
	depthID = depthBufID;
	UnbindBuffer();
	return 1;
}

// attach a renderbuffer (stencilBufID) to the stencil buffer attachment point.
int FrameBuffer::AttachStencilBuffer( GLuint stencilBufID )
{
	BindBuffer();
	//glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, stencilBufID);
    //glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_STENCIL_INDEX8_EXT, 
	 //                        width, height);
	glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_STENCIL_ATTACHMENT_EXT, 
								   GL_RENDERBUFFER_EXT, stencilBufID);
	stencilID = stencilBufID;
	UnbindBuffer();
	return 1;
}


// Bind this framebuffer as the current one.  Store the old one to reattach
//    when we unbind.  Also return the ID of the previous framebuffer.
GLuint FrameBuffer::BindBuffer( void )
{
	GLint tmp;
	glGetIntegerv( GL_FRAMEBUFFER_BINDING_EXT, &tmp );
	prevFrameBuf = tmp;
	glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, ID );
	return prevFrameBuf;
}


// This function unbinds this framebuffer to whatever buffer was attached
//     previously...  If for some reason the binding have changed so we're
//     no longer the current buffer, DO NOT unbind, return 0.  Else, unbind
//     and return 1.
int FrameBuffer::UnbindBuffer( void )
{
	GLint tmpFB;
	glGetIntegerv( GL_FRAMEBUFFER_BINDING_EXT, &tmpFB );
	if (tmpFB != ID) return 0;
	glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, prevFrameBuf );
	prevFrameBuf = 0;
	return 1;
}

void FrameBuffer::DrawToColorMipmapLevel( GLuint colorBuffer, GLuint level )
{
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT,
                              GL_COLOR_ATTACHMENT0_EXT+colorBuffer,
                              GL_TEXTURE_2D, GetColorTextureID( colorBuffer ), level);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, level-1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, level-1);

	glBindTexture( GL_TEXTURE_2D, GetColorTextureID( colorBuffer ) );
	glEnable(GL_TEXTURE_2D);
}

void FrameBuffer::DoneDrawingMipmapLevels( void )
{
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 1000);
}