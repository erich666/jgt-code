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
#ifndef _GLTimerQuery_h_
#define _GLTimerQuery_h_


#include <GL/glew.h>
#include <GL/glut.h>


/**
 * Helper class to manage a GL timer query. Intended to be used to repeatedly time a section of 
 * OpenGl calls, for example a frame. 
 * Note: In OpenGl there may only be one query of a specific type (we use GL_TIME_ELAPSED) 
 * active at a time, so class this cannot be used nestedly.
 */
class GLTimerQuery
{
public:
  /**
   */
	GLTimerQuery(void);
  /**
   */
	~GLTimerQuery(void);
  /**
   * Begins the timer query.
   */
	void start(); 
  /**
   * Ends the GL query and retrieves the timer counter, thus a synchronizing operation.
   */
	void stop(); 
  /**
   * Returns the average of the (up to) s_numSamples last frames.
   */
	float getAvgMs(); 
private:
  enum
  {
    s_numSamples = 20
  };
	GLuint m_id;
	GLuint64 m_timeElapsed[s_numSamples]; 
	unsigned int m_currentSample; 
};


#endif // _GLTimerQuery_h_
