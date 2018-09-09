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
#include "GLTimerQuery.h"
#include <algorithm>



GLTimerQuery::GLTimerQuery(void)
{
	glGenQueries(1, &m_id);
	m_currentSample = 0; 
}



GLTimerQuery::~GLTimerQuery(void)
{
	glDeleteQueries(1, &m_id);
}



void GLTimerQuery::start()
{
	glBeginQuery(GL_TIME_ELAPSED, m_id);
}



void GLTimerQuery::stop()
{
	glEndQuery(GL_TIME_ELAPSED);
	glGetQueryObjectui64v(m_id, GL_QUERY_RESULT, &m_timeElapsed[m_currentSample % s_numSamples]);
	++m_currentSample; 
}



float GLTimerQuery::getAvgMs()
{
	float totalMs = 0.0f; 
  int numSamples = std::min<int>(m_currentSample, s_numSamples);
  for (int i = 0; i < numSamples; ++i)
  {
		totalMs += float(double(m_timeElapsed[i]) / 1e6);
	}
	return totalMs / float(numSamples); 
}
