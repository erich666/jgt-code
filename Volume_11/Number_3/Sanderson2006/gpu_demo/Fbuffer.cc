//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : Fbuffer.cc
//    Author : Allen R. Sanderson
//    Date   : Jan 17 2006

#include <GL/glew.h>

#include "Fbuffer.h"
#include <iostream>

#include <GL/glu.h>

#include <string>

#ifdef  __GLEW_H__
#define HAVE_GLEW
#endif

static bool mNV_float_buffer = true;
static bool mNV_texture_rectangle = false;

Fbuffer::Fbuffer( int width, int height ) :
  mWidth( width ),
  mHeight( height )
{}

bool
Fbuffer::create ()
{
  // Create the objects
  glGenFramebuffersEXT( 1, &mFB );
//  glGenRenderbuffersEXT( 1, &mRB );

//  glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, mFB );

    // initialize the depth renderbuffer
//  glBindRenderbufferEXT( GL_RENDERBUFFER_EXT, mRB );
//  glRenderbufferStorageEXT( GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT24,
// 			      mWidth, mHeight );

    // attach the renderbuffer to the framebuffer depth buffer.
//  glFramebufferRenderbufferEXT( GL_FRAMEBUFFER_EXT,
// 				  GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT,
// 				  mRB );


//  if (mRenderTex) {
    // create texture object
//     glGenTextures(1, &mTex);
    
//     if(mFormat == GL_FLOAT) {
//       if(mNV_float_buffer) {
// 	mTexTarget = GL_TEXTURE_RECTANGLE_NV;
// 	if(mNumColorBits == 16)
// 	  mTexFormat = GL_FLOAT_RGBA16_NV;
// 	else
// 	  mTexFormat = GL_FLOAT_RGBA32_NV;
//       } else {
// 	mTexTarget = GL_TEXTURE_2D;
//       }
//     } else {
//       mTexTarget = GL_TEXTURE_2D;
//       mTexFormat = GL_RGBA;
//     }

//     glBindTexture(mTexTarget, mTex);
//     glTexParameteri(mTexTarget, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
//     glTexParameteri(mTexTarget, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
//     glTexParameteri(mTexTarget, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
//     glTexParameteri(mTexTarget, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    
//      glTexParameteri(mTexTarget, GL_GENERATE_MIPMAP_SGIS, GL_TRUE);
//      glTexParameteri(mTexTarget, GL_GENERATE_MIPMAP_HINT_SGIS, GL_NICEST);
//      glTexParameteri(mTexTarget, GL_GENERATE_MIPMAP_HINT_SGIS, GL_FASTEST);

//     glTexImage2D(mTexTarget, 0, mTexFormat, mWidth, mHeight, 0,
// 		 GL_RGBA, GL_FLOAT, 0);

    // Attach the texture to the framebuffer color buffer
//     glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,
// 			       mTexTarget, mTex, 0 );
    
//  }

//  glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0 );

  return checkBuffer();
}

bool
Fbuffer::checkBuffer ()
{
  //----------------------
  // Framebuffer Objects initializations
  //----------------------
  GLuint status = glCheckFramebufferStatusEXT( GL_FRAMEBUFFER_EXT );

  switch( status ) {
  case GL_FRAMEBUFFER_COMPLETE_EXT:
//    std::cerr << " GL_FRAMEBUFFER_COMPLETE_EXT \n";
    return true;

  case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT:
    std::cerr << " GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT \n";
    exit(0);

  case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT:
    std::cerr << " GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT \n";
    exit(0);

//   case GL_FRAMEBUFFER_INCOMPLETE_DUPLICATE_ATTACHMENT_EXT:
//     std::cerr << " GL_FRAMEBUFFER_INCOMPLETE_DUPLICATE_ATTACHMENT_EXT \n";
//     exit(0);

  case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:
    std::cerr << " GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT \n";
    exit(0);

  case GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT:
    std::cerr << " GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT \n";
    exit(0);

  case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT:
    std::cerr << " GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT \n";
    exit(0);

  case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT:
    std::cerr << " GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT \n";
    exit(0);

  case GL_FRAMEBUFFER_UNSUPPORTED_EXT:
    std::cerr << " GL_FRAMEBUFFER_UNSUPPORTED_EXT \n";
    exit(0);

  default:
    std::cerr << " GL_FRAMEBUFFER_UNKNOWN " << status << std::endl;
    exit(0);
  }
}

void
Fbuffer::enable ()
{
  glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, mFB );
}

void
Fbuffer::disable ()
{
  glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0 );
}

void
Fbuffer::attach( GLuint texID, GLuint index )
{
  glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT,
			     GL_COLOR_ATTACHMENT0_EXT + index,
			     GL_TEXTURE_RECTANGLE_NV,
			     texID, 0 );
}  
