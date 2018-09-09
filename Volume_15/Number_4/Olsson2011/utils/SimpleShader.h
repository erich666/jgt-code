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
#ifndef _SimpleShader_h_
#define _SimpleShader_h_

#include <linmath/float3.h>
#include <linmath/float4x4.h>
#include <linmath/float3x3.h>
#include <linmath/int2.h>

#include <GL/glew.h>
#include <GL/glut.h>

#include <map>
#include <string>

namespace chag
{

/**
 * Very simple glsl shader class, supports vertex and fragment shaders and a handful of operations.
 * The most un-basic feature is that it supports setting #defines, which are supplied in a context
 * that is passed to the compilation.
 */
class SimpleShader
{
public:
  class Context;
  /**
   * Constructs and compiles a shader, the context provides preprocessor definitions.
   */
  SimpleShader(const char *vertexShaderFileName, const char *fragmentShaderFileName, const Context &context);
  /**
   * link the shader, must be done after constructor to complete the shader creation. 
   * But, should not be done before attribute- and fragment data locations are bound,
   * (bindAttribLocation & bindFragDataLocation).
   */
  bool link();
  /**
   * Must be used between ctor and link.
   */
  void bindAttribLocation(GLint index, GLchar* name);
  /**
   * Must be used between ctor and link.
   */
  void bindFragDataLocation(GLuint location, const char *name);

  /**
   * Looks up the index of uniform buffer of the given name in the shader,
   * and binds it to the given buffer index.
   */
  bool setUniformBufferSlot(const char *blockName, GLuint slotIndex);
  /**
   * Binds the texture(2D/2D_MULTISAMPLE) to the given texture stage
   * AND sets the corresponding uniform to the texture stage.
   * Must be called inside a begin/end pair.
   */
  bool setTexture2D(const char *varName, GLuint textureId, GLuint textureStage);
  bool setTexture2DMS(const char *varName, GLuint textureId, GLuint textureStage);
  bool setTextureBuffer(const char *varName, GLuint textureId, GLuint textureStage);

  /**
   * Set uniforms, overloaded for some types.
   * Must be called inside a begin/end pair.
   */
  bool setUniform(const char *varName, GLint v);
  bool setUniform(const char *varName, GLfloat v0, GLfloat v1);
  bool setUniform(const char *varName, const float3 &v);
  bool setUniform(const char *varName, const float4x4 &value);

  /**
   * Before rendering geometry using the shader, call begin(), when done call end().
   */
  void begin();
  void end();

  /**
   * Used to manage preprocessor definitions, is sent to the compile of a shader.
   */
  class Context
  {
  public:
    Context() { };

    /**
     * Set value of preproc def, is created if needed otherwise updated. 
     * Also updates char buffer.
     */
    void setPreprocDef(const char *name, int value);
    void setPreprocDef(const char *name, bool value);

    void updatePreprocDefBuffer();

  
    char m_shaderPreprocDefBuffer[4096];
    typedef std::map<std::string, std::string> StringMap;
    StringMap m_shaderPreprocDefs;
  };

protected:
  GLuint m_shaderProgram;

  bool m_linked;
  bool m_begun;
  bool m_compiled;
};


}; // namespace chag


#endif // _SimpleShader_h_
