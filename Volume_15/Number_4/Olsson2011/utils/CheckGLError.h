/****************************************************************************/
/* Copyright (c) 2011, Markus Billeter, Ola Olsson, Erik Sintorn and Ulf Assarsson
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
/****************************************************************************/
#ifndef _CheckGLError_h_
#define _CheckGLError_h_

#include "Assert.h"

/**
 * This macro checks for gl errors using glGetError, it is useful to sprinkle it around the code base, especially
 * when unsure about the correct usage, for example after each call to open gl.
 * When the debugger is atached it will cause a break on the offending line, and also print out the file 
 * and line in a MSVC compatible format on the debug output and console.
 *
 * Note: the macro _cannot_ be used between glBegin/glEnd brackets, as stated by the openGL standard.
 * Note2: be aware that the macro will report any previous errors, since the last call to glGetError.
 *
 * example usage: glClear(GL_COLOR_BUFFER_BIT); CHECK_GL_ERROR(); // this will check for errors in this (and any previous statements)
 */

#ifdef _DEBUG
#define CHECK_GL_ERROR() { checkGLError(__FILE__, __LINE__) && (DBG_BREAK(), 1); }
#else // !_DEBUG
#define CHECK_GL_ERROR() {  }
#endif // _DEBUG
/**
 * Internal function used by macro CHECK_GL_ERROR, use that instead.
 */
bool checkGLError(char *file, int line);


#endif // _CheckGLError_h_
