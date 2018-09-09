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
#ifndef _GlBufferObject_h_
#define _GlBufferObject_h_

#include <GL/glew.h>
#include <GL/glut.h>
#include <utils/CheckGLError.h>
#include <utils/IntTypes.h>
#include <utils/Assert.h>


template <typename T>
class GlBufferObject
{
public:
  GlBufferObject() : m_id(0), m_elements(0) { } 
  
  ~GlBufferObject()
  {
    if (m_id)
    {
      glDeleteBuffers(1, &m_id);
    }
  }

  void init(size_t elements, const T *hostData = 0, uint32_t dataUpdateKind = GL_DYNAMIC_COPY)
  {
    m_dataUpdateKind = dataUpdateKind;
    ASSERT(!m_id);

    glGenBuffers(1, &m_id);
    CHECK_GL_ERROR();
    m_elements = elements;
    if (elements)
    {
      copyFromHost(hostData, elements);
    }
  }

  void resize(size_t elements)
  {
    copyFromHost(0, elements);
  }

  size_t size() const
  {
    return m_elements;
  }

  operator GLuint() const 
  { 
    ASSERT(m_id > 0);
    return m_id; 
  }

  void loadFromFBO(GLuint srcFbo, GLenum attachment, int width, int height, GLenum format, GLenum type)
  {
    ASSERT(m_elements > 0);
    ASSERT(m_id > 0);
    ASSERT(srcFbo > 0);

    // ensure the FBO is bound.
    glBindFramebuffer(GL_FRAMEBUFFER, srcFbo);
    CHECK_GL_ERROR();
    // indicate the correct buffer to read from
    glReadBuffer(attachment);
    CHECK_GL_ERROR();
    // bind destination buffer.
    glBindBuffer(GL_PIXEL_PACK_BUFFER, m_id);
    CHECK_GL_ERROR();
    // shovel over the data to pbo.
    glReadPixels(0, 0, width, height, format, type, 0);
    CHECK_GL_ERROR();

    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    CHECK_GL_ERROR();
  }

  void loadDepthFromFBO(GLuint srcFbo, GLenum attachment, int width, int height, GLenum format, GLenum type)
  {
    ASSERT(m_elements > 0);
    ASSERT(m_id > 0);
    ASSERT(srcFbo > 0);

    // ensure the FBO is bound.
    glBindFramebuffer(GL_FRAMEBUFFER, srcFbo);
    CHECK_GL_ERROR();
    // indicate the correct buffer to read from
    // glReadBuffer(attachment);
    CHECK_GL_ERROR();
    // bind destination buffer.
    glBindBuffer(GL_PIXEL_PACK_BUFFER, m_id);
    CHECK_GL_ERROR();
    // shovel over the data to pbo.
    glReadPixels(0, 0, width, height, format, type, 0);
    CHECK_GL_ERROR();

    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    CHECK_GL_ERROR();
  }

  void copyFromHost(const T *hostData, size_t elements)
  {
    ASSERT(elements > 0);
    m_elements = elements;
    glBindBuffer(GL_ARRAY_BUFFER, m_id);
    CHECK_GL_ERROR();

    // buffer data
    glBufferData(GL_ARRAY_BUFFER, m_elements * sizeof(T), hostData, m_dataUpdateKind);
    CHECK_GL_ERROR();
    
    // make sure buffer is not bound
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    CHECK_GL_ERROR(); 
  }

  const T *beginMap() const
  {
    ASSERT(m_elements > 0);
    glBindBuffer(GL_ARRAY_BUFFER, m_id);

    const T* result = reinterpret_cast<const T*>(glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY));
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return result;
  }
  const T *beginMapConst() const
  {
    ASSERT(m_elements > 0);
    glBindBuffer(GL_ARRAY_BUFFER, m_id);

    const T* result = reinterpret_cast<const T*>(glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY));
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return result;
  }
  /**
   * NB: Uses GL_WRITE_ONLY flag, so no reading please.
   */
  T *beginMap()
  {
    ASSERT(m_elements > 0);
    glBindBuffer(GL_ARRAY_BUFFER, m_id);

    T* result = reinterpret_cast<T*>(glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY));
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return result;
  }

  void endMap() const
  {
    glBindBuffer(GL_ARRAY_BUFFER, m_id);
    glUnmapBuffer(GL_ARRAY_BUFFER);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
  }

  void bind(GLenum target = GL_ARRAY_BUFFER, uint32_t offset = 0) const
  {
    ASSERT(offset == 0);
    glBindBuffer(target, m_id);
    CHECK_GL_ERROR();
  }

  void bindSlot(GLenum target, uint32_t slot) const
  {
    glBindBufferBase(target, slot, m_id);
    CHECK_GL_ERROR();
  }

  void bindSlotRange(GLenum target, uint32_t slot, uint32_t offset, uint32_t count = 1) const
  {
    CHECK_GL_ERROR();
    ASSERT(offset < m_elements);
    ASSERT(offset + count <= m_elements);
    size_t tmp = sizeof(T) * offset;
    glBindBufferRange(target, slot, m_id, tmp, sizeof(T) * count);
    CHECK_GL_ERROR();
  }

  void unbind(GLenum target = GL_ARRAY_BUFFER) const
  {
    glBindBuffer(target, 0);
  }

private:
  size_t m_elements;
  uint32_t m_id;
  uint32_t m_dataUpdateKind;
};


#endif // _GlBufferObject_h_
