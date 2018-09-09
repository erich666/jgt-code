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
#include "ComboShader.h"
#include "Assert.h"


namespace chag
{


ComboShader::ComboShader(const char *vertexShaderFileName, const char *fragmentShaderFileName, chag::SimpleShader::Context &context)
{
	context.setPreprocDef("ENABLE_ALPHA_TEST", 0);
	m_opaqueShader = new chag::SimpleShader(vertexShaderFileName, fragmentShaderFileName, context);
	context.setPreprocDef("ENABLE_ALPHA_TEST", 1);
	m_alphaTestedShader = new chag::SimpleShader(vertexShaderFileName, fragmentShaderFileName, context);
	m_currentShader = 0;
}



ComboShader::~ComboShader()
{
	delete m_opaqueShader;
	delete m_alphaTestedShader;
}



bool ComboShader::link()
{
	return m_opaqueShader->link() && m_alphaTestedShader->link();
}



void ComboShader::bindAttribLocation(GLint index, GLchar* name)
{
	m_opaqueShader->bindAttribLocation(index, name);
	m_alphaTestedShader->bindAttribLocation(index, name);
}



void ComboShader::bindFragDataLocation(GLuint location, const char *name)
{
	m_opaqueShader->bindFragDataLocation(location, name);
	m_alphaTestedShader->bindFragDataLocation(location, name);
}



bool ComboShader::setUniformBufferSlot(const char *blockName, GLuint slotIndex)
{
	return m_opaqueShader->setUniformBufferSlot(blockName, slotIndex) 
		&& m_alphaTestedShader->setUniformBufferSlot(blockName, slotIndex);
}



bool ComboShader::setUniform(const char *varName, int v)
{
	m_currentShader->end();
	// this is not meant to be used very much at all...
	m_opaqueShader->begin();
	m_opaqueShader->setUniform(varName, v);
	m_opaqueShader->end();

	m_alphaTestedShader->begin();
	m_alphaTestedShader->setUniform(varName, v);
	m_alphaTestedShader->end();

	m_currentShader->begin();
	return true;
}



void ComboShader::begin(bool useAlphaTest)
{
	ASSERT(!m_currentShader);
	m_currentShader = useAlphaTest ? m_alphaTestedShader : m_opaqueShader;
	m_currentShader->begin();
}



void ComboShader::end()
{
	ASSERT(m_currentShader);
	m_currentShader->end();
	m_currentShader = 0;
}



}; // namespace chag


