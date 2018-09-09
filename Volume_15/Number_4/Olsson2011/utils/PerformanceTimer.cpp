/****************************************************************************/
/* Copyright (c) 2011, Ola Olsson, Markus Billeter
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
#include "PerformanceTimer.h"

#ifdef _WIN32
  #include "Win32ApiWrapper.h"
#elif defined(__linux__)
  #include <time.h>
  #define NSEC_PER_SEC 1000000000llu
#else // 
  #error No implementation for the current platform available
#endif // _WIN32



uint64_t PerformanceTimer::getTickCount()
{
#if defined(_WIN32)
  LARGE_INTEGER i;
  QueryPerformanceCounter(&i);
  return i.QuadPart;
#elif defined(__linux__)
  timespec ts;
  clock_gettime( CLOCK_MONOTONIC, &ts );
  return ts.tv_sec * NSEC_PER_SEC + ts.tv_nsec;
#endif
}


uint64_t PerformanceTimer::initTicksPerSecond()
{
#if defined(_WIN32)
  LARGE_INTEGER hpFrequency;
  QueryPerformanceFrequency(&hpFrequency);
  return  hpFrequency.QuadPart;
#elif defined(__linux__)
  return 1000000000llu;
#endif
}
