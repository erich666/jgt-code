/************************************************
** framebufferObject.h                         **
** -------------------                         **
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

#ifndef ___FRAMEBUFFER_OBJECT_H
#define ___FRAMEBUFFER_OBJECT_H

#include <gl/glew.h>

class FrameBuffer
{
private:
	GLuint ID;
	GLuint *colorIDs;
	GLuint depthID;
	GLuint stencilID;
	GLint maxColorBuffers;
	GLuint prevFrameBuf;
	int width, height;
	char fbName[80];

public:
	FrameBuffer( char *name=0 );
	FrameBuffer( int width, int height, char *name=0 );
	~FrameBuffer();

    GLenum CheckFramebufferStatus( int printMessage=0 );

	// Attach textures to various attachment points
	int AttachColorTexture( GLuint colorTexID, int colorBuffer=0 );
	int AttachDepthTexture( GLuint depthTexID );
	int AttachStencilTexture( GLuint stencilTexID );

	// Attach renderbuffers to various attachment points
	//    (note these SHOULD replace textures, but it may not be guaranteed,
	//     so you might want to unbind textures before binding a renderbuffer)
	int AttachColorBuffer( GLuint colorBufID, int colorBuffer=0 );
	int AttachDepthBuffer( GLuint depthBufID );
	int AttachStencilBuffer( GLuint stencilBufID );

	// Functionality for drawing custom mipmap levels.
	void DrawToColorMipmapLevel( GLuint colorBuffer, GLuint level );
	void DoneDrawingMipmapLevels( void );

	// Bind/unbind the current framebuffer.  These functions store the currently
	//    bound framebuffer during a BindBuffer() and rebind it upon an UnbindBuffer()
	GLuint BindBuffer( void );
	int UnbindBuffer( void );

	// Queries to return the texture/renderbuffer ID of the various attachments
	inline GLuint GetColorTextureID( int level=0 ) { return (level < maxColorBuffers && level >= 0 ? colorIDs[level] : -1); }
	inline GLuint GetDepthTextureID( void )      { return depthID; }
	inline GLuint GetStencilTextureID( void )    { return stencilID; }

	inline int GetWidth( void ) { return width; }
	inline int GetHeight( void ) { return height; }
	inline GLuint GetBufferID( void ) { return ID; }
	inline char *GetName( void ) { return fbName; }

	inline void SetSize( int newWidth, int newHeight ) { width = newWidth; height = newHeight; }
};



#endif