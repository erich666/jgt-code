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
#include "SimpleShader.h"
#include "Assert.h"
#include <stdio.h>
#include <string>
#include <algorithm>
#include <fstream>
#include "CheckGlError.h"
#include "Path.h"

namespace chag
{


static std::string getShaderInfoLog(GLuint obj) 
{
	int logLength = 0;
	glGetShaderiv(obj, GL_INFO_LOG_LENGTH, &logLength);

	std::string log;
	if (logLength > 0) 
  {
	  char *tmpLog = new char[logLength];
	  int charsWritten  = 0;
		glGetShaderInfoLog(obj, logLength, &charsWritten, tmpLog);
		log = tmpLog;
		delete [] tmpLog;
	}

	return log;
}


static char *loadTextFile(const char *fn)
{
	FILE *fp;
	char *content = NULL;

	if (fn != NULL) 
  {
		fp = fopen(fn,"rt");
		if (fp != NULL) 
    {
      fseek(fp, 0, SEEK_END);
      size_t count = ftell(fp);
      rewind(fp);
			if (count > 0) 
      {
				content = new char[count+1];
				count = fread(content,sizeof(char),count,fp);
				content[count] = '\0';
			}
			fclose(fp);
		}
	}
	return content;
}


// scan to end of line, and also return pointer to start of next line
const char *scanToEndl(const char *s, const char *&nextLine)
{
  const char *endl = strpbrk(s, "\n\r");
  nextLine = endl;
  if (endl)
  {
    const char *nextLine = endl + 1;

    // then there may be a crlf or some such combination, so just clobber that!
    if (*nextLine == '\r' && *endl == '\n' || *nextLine == '\n' && *endl == '\r')
    { 
      ++nextLine;
    }
  }
  return endl;
}


/**
 * Supersimplistic include processing, doesn't respect comment blocks or anything, 
 * which could create rather weird effects if care is not taken. Better to make use
 * of some more complete pre-processor implementation, mayhaps boost::wave.
 * Treats <> and "" as the same, and does not support looking in additional directories,
 * though this could easily be added through the context.
 * Not much in the way of error checking, e.g. #include is not verified to be first on a line.
 * Also ought to stick in #line and #file pragmas where suitable...
 */
static std::string processIncludes(const char *inputFile, std::string fileName, const SimpleShader::Context &context)
{
  std::string basePath = getBasePath(fileName);

  std::string result;
  const char *prev = inputFile;
  while (const char *p = strstr(prev, "#include"))
  {
    // append text up to the include.
    result.append(prev, p);
    // stick in a newline for good(?, may be bad) measure.
    result.append("\n");
    // next extract the path:
    if (const char *start = strpbrk(p, "<\""))
    {
      // step past start token.
      ++start;
      // scan to end
      if (const char *end = strpbrk(start, ">\""))
      {
        std::string name(start, end);
        std::string path = basePath + name; 
        // 1. try relative path
        char *s = loadTextFile(path.c_str());
        // 2. try absolute path
        if (!s)
        {
          path = name;
          s = loadTextFile(name.c_str());
        }
        // 3. try extenally defined paths, found in context somehow...
        // 4. recusrively process the include file.
        if (s)
        {
          std::string expandedSource = processIncludes(s, path, context);
          result.append(expandedSource);
          delete [] s;
        }
        else
        {
          static char buf[1024];
          sprintf(buf, "include '%s' not found!\n", name.c_str());
          outputFailure(fileName.c_str(), 0, buf);
        }
        prev = end + 1;
      }
    }
  }
  result.append(prev);
  return result;
}



static bool loadCompileAttachShader(GLuint shaderProgram, GLenum shaderKind, const char *fileName, const SimpleShader::Context &context)
{
  //printf("compiling '%s'\n----------------------------------\n", fileName);

  GLuint shader = glCreateShader(shaderKind);
  //std::ifstream instream("input.cpp");
  //std::string input(std::istreambuf_iterator<char>(instream.rdbuf()), std::istreambuf_iterator<char>());

  const char *s = loadTextFile(fileName);
  char versionString[128] = "";


  // first, we will process includes, this must be done recursively.
  std::string expandedSource = processIncludes(s, fileName, context);
  delete [] s;
  s = expandedSource.c_str();
  const char *source = s;


  // Now, as AMDs glsl compiler insists that the very first thing in a file must absolutely be #version (it it is present)
  // we must splice our defines in between that and the rest.
  if (strncmp("#version", s, sizeof("#version") - 1) == 0)
  {
    const char *sourceStart = 0;
    if (const char *endl = scanToEndl(s, sourceStart))
    {
      source = sourceStart;
      strncpy(versionString, s, sourceStart - s);
      versionString[sourceStart - s + 1] = 0;
    }
  }

  GLint	lengths[3] = 
  {
    (GLint) strlen(versionString),
    (GLint) strlen(context.m_shaderPreprocDefBuffer),
    (GLint) strlen(source),
  };
  const char* sources[3] = 
  {
    versionString,
    context.m_shaderPreprocDefBuffer,
    source,
  };

  //printf("%s", versionString);
  //printf("%s", context.m_shaderPreprocDefBuffer);
  //printf("%s\n----------------------------------------------------------\n", source);

	glShaderSource(shader, 3, sources, lengths);

  glCompileShader(shader);
	int compileOk = 0;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &compileOk);
	if (!compileOk) 
  {
		std::string err = getShaderInfoLog(shader);
    outputFailure(fileName, 0, err.c_str());
    DBG_BREAK();
		return false;
	}
	glAttachShader(shaderProgram, shader);
	glDeleteShader(shader);
  return true;
}



SimpleShader::SimpleShader(const char *vertexShaderFileName, const char *fragmentShaderFileName, const Context &context)
{
	m_shaderProgram = glCreateProgram();

  m_compiled =  loadCompileAttachShader(m_shaderProgram, GL_VERTEX_SHADER, vertexShaderFileName, context) 
             && loadCompileAttachShader(m_shaderProgram, GL_FRAGMENT_SHADER, fragmentShaderFileName, context);

  // debug state
  m_linked = false;
  m_begun = false;
	CHECK_GL_ERROR();
}



bool SimpleShader::link()
{
  ASSERT(!m_linked);
	glLinkProgram(m_shaderProgram);
  GLint linkOk = 0;
  glGetProgramiv(m_shaderProgram, GL_LINK_STATUS, &linkOk);
  if (!linkOk)
	{
	  std::string err = getShaderInfoLog(m_shaderProgram);
    outputFailure("", 0, err.c_str());
    DBG_BREAK();
    return false;
  }
	CHECK_GL_ERROR();
  m_linked = true;
  return true;
}



void SimpleShader::bindAttribLocation(GLint index, GLchar* name)
{
  ASSERT(!m_linked); // this has no effect if it is already linked.
  glBindAttribLocation(m_shaderProgram, index, name);
	CHECK_GL_ERROR();
}



void SimpleShader::bindFragDataLocation(GLuint location, const char *name)
{
  ASSERT(!m_linked); // this has no effect if it is already linked.
  glBindFragDataLocation(m_shaderProgram, location, name);
  CHECK_GL_ERROR();
}



bool SimpleShader::setUniformBufferSlot(const char *blockName, GLuint slotIndex)
{
  ASSERT(m_linked);
  int loc = glGetUniformBlockIndex(m_shaderProgram, blockName);
  CHECK_GL_ERROR();
  if (loc >= 0)
  {
    glUniformBlockBinding(m_shaderProgram, loc, slotIndex);
    CHECK_GL_ERROR();
  }
  return loc >= 0;
}


bool SimpleShader::setTexture2D(const char *varName, GLuint textureId, GLuint textureStage)
{
  ASSERT(m_linked);
  ASSERT(m_begun);
  int loc = glGetUniformLocation(m_shaderProgram, varName);
  if (loc >= 0)
  {
    glActiveTexture(GL_TEXTURE0 + textureStage);
    glBindTexture(GL_TEXTURE_2D, textureId);
    glUniform1i(loc, textureStage);

    CHECK_GL_ERROR();
  }
  return loc >= 0;
}


bool SimpleShader::setTexture2DMS(const char *varName, GLuint textureId, GLuint textureStage)
{
  ASSERT(m_linked);
  ASSERT(m_begun);
  int loc = glGetUniformLocation(m_shaderProgram, varName);
  if (loc >= 0)
  {
    glActiveTexture(GL_TEXTURE0 + textureStage);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, textureId);
    glUniform1i(loc, textureStage);
    CHECK_GL_ERROR();
  }
  return loc >= 0;
}


bool SimpleShader::setTextureBuffer(const char *varName, GLuint textureId, GLuint textureStage)
{
  ASSERT(m_linked);
  ASSERT(m_begun);
  int loc = glGetUniformLocation(m_shaderProgram, varName);
  if (loc >= 0)
  {
    glActiveTexture(GL_TEXTURE0 + textureStage);
    glBindTexture(GL_TEXTURE_BUFFER, textureId);
    glUniform1i(loc, textureStage);
    CHECK_GL_ERROR();
  }
  return loc >= 0;
}



static GLint uniformLocationHelper(GLuint prog, const char *varName)
{
  ASSERT(varName);
  return glGetUniformLocation(prog, varName);
}


bool SimpleShader::setUniform(const char *varName, GLint v0)
{ 
  ASSERT(m_linked);
  ASSERT(m_begun);
  GLint loc = uniformLocationHelper(m_shaderProgram, varName);
  if (loc == -1) 
  {
    return false;
  }

  glUniform1i(loc, v0);
  CHECK_GL_ERROR();
  return true;
}



bool SimpleShader::setUniform(const char *varName, GLfloat v0, GLfloat v1)
{
  ASSERT(m_linked);
  ASSERT(m_begun);
  GLint loc = uniformLocationHelper(m_shaderProgram, varName);
  if (loc == -1) 
  {
    return false;
  }

  glUniform2f(loc, v0, v1);
  CHECK_GL_ERROR();
  return true;
}



bool SimpleShader::setUniform(const char *varName, const float3 &v)
{
  ASSERT(m_linked);
  ASSERT(m_begun);
  GLint loc = uniformLocationHelper(m_shaderProgram, varName);
  if (loc == -1) 
  {
    return false;
  }

  glUniform3fv(loc, 1, &v.x);
  CHECK_GL_ERROR();
  return true;
}


bool SimpleShader::setUniform(const char *varName, const float4x4 &value)
{
  ASSERT(m_linked);
  ASSERT(m_begun);
  GLint loc = uniformLocationHelper(m_shaderProgram, varName);
  if (loc == -1) 
  {
    return false;
  }
  glUniformMatrix4fv(loc, 1, GL_FALSE, &value.c1.x);
  CHECK_GL_ERROR();

  return true;
}



void SimpleShader::begin()
{
  ASSERT(m_linked);
  ASSERT(!m_begun);
  m_begun = true;
  CHECK_GL_ERROR();
  glUseProgram(m_shaderProgram);
  CHECK_GL_ERROR();
}



void SimpleShader::end()
{
  ASSERT(m_linked);
  ASSERT(m_begun);
  m_begun = false;
  CHECK_GL_ERROR();
  glUseProgram(0);
  CHECK_GL_ERROR();
}




void SimpleShader::Context::setPreprocDef(const char *name, int value)
{
  char buf[11];
  sprintf(buf, "%d", value);
  m_shaderPreprocDefs[name] = buf;
  updatePreprocDefBuffer();
}



void SimpleShader::Context::setPreprocDef(const char *name, bool value)
{
  m_shaderPreprocDefs[name] = value ? "1" : "0";
  updatePreprocDefBuffer();
}



void SimpleShader::Context::updatePreprocDefBuffer()
{
  size_t offset = 0;
  // start with a \n to make the version line work, because it got borked...
  offset += sprintf(&m_shaderPreprocDefBuffer[offset], "\n");
  for (StringMap::const_iterator it = m_shaderPreprocDefs.begin(); it != m_shaderPreprocDefs.end(); ++it)
  {
    offset += sprintf(&m_shaderPreprocDefBuffer[offset], "#define %s %s\n", (*it).first.c_str(), (*it).second.c_str());
  }
}


}; // namespace chag


