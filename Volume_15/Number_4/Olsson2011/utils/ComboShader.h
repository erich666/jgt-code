/****************************************************************************/
/* Copyright (c) 2011, Ola Olsson
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
#ifndef _ComboShader_h_
#define _ComboShader_h_

#include "SimpleShader.h"

namespace chag
{

/**
 * Helper to simplify keeping two shaders compiled with and without alpha test enabled. Compiles the given shaders
 * twice, with ENABLE_ALPHA_TEST = 0 and 1 defined. Interface is designed to be mostly compatible with SimpleShader.
 * Setting constants is not really supported, as it's a major hassle with OpenGL (need to bind the shader etc).
 * The only exception is setUniform(int), which must be used to assign samplers to texture stages, this will bind 
 * each of the shaders in turn and set the uniform, only meant to be done once after loading the shader.
 * For other uniforms, it's recomended to use uniform buffers, and bind them once, this makes it a lot easier to switch
 * shader (if all are set up to use the same buffer slots).
 */
class ComboShader
{
public:
	ComboShader(const char *vertexShaderFileName, const char *fragmentShaderFileName, SimpleShader::Context &context);
	~ComboShader();

	bool link();

  void bindAttribLocation(GLint index, GLchar* name);
  void bindFragDataLocation(GLuint location, const char *name);
  bool setUniformBufferSlot(const char *blockName, GLuint slotIndex);
  bool setUniform(const char *varName, int v);
  void begin(bool useAlphaTest);
  void end();

protected:
	SimpleShader *m_opaqueShader;
	SimpleShader *m_alphaTestedShader;
	SimpleShader *m_currentShader;
};


}; // namespace chag


#endif // _ComboShader_h_
