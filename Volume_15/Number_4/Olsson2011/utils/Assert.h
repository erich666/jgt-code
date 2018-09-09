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
#ifndef _Assert_h_
#define _Assert_h_

/**
 * Prints the failure to the std error stream, as well as (for msvc) the debug 
 * stream, that appears in the debug output of visual studio. Thus especially
 * useful for reporting programming errors.
 */
void outputFailure(const char *file, const int line, const char *conditionString);

#ifdef _DEBUG

  #define DBG_BREAK() __debugbreak()

  #define ASSERT(_condition_) \
  if (!(_condition_)) \
  { \
    outputFailure(__FILE__, __LINE__, #_condition_); \
    __debugbreak(); \
  }
#else // _DEBUG
  #define DBG_BREAK()
  #define ASSERT(_condition_)
#endif //_DEBUG

#define COMPILE_TIME_ASSERT(_expr_) typedef int _error_thing[(_expr_) != 0];

#endif // _Assert_h_
