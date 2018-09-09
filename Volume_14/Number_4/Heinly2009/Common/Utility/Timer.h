
/*
 *  Copyright 2009, 2010 Grove City College
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef Utility_Timer_h
#define Utility_Timer_h

#if defined(__APPLE__) && defined(__POWERPC__)
#include <ppc_intrinsics.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#elif _WIN32
#include <time.h>
#else
#include <sys/time.h>
#endif

#include <algorithm>

#include <iosfwd>
using std::ostream;

namespace Utility
{

#if defined(USE_CYCLE_TIMER)
#  if defined(__APPLE__) && defined(__POWERPC__)
#    error cycles() not yet implemented
#elif _WIN32
#  error cycles() not yet implemented
#else

  unsigned long long cycles()
  {
    /*
      __asm__ (assembler template 
               : output operands       // optional
               : input operands        // optional
               : clobbered registers   // optional
      );
    */

    unsigned long long temp;
    __asm__ __volatile__ ("cpuid\n\t"
                          "rdtsc\n\t"
                          "leal %0,    %%ecx\n\t"
                          "movl %%eax, (%%ecx)\n\t"
                          "movl %%edx, 4(%%ecx)\n\t"
                          :
                          : "m"(temp)
                          : "%eax", "%ebx", "%ecx", "%edx"
                          );
    return temp;
  }

#endif // defined(__APPLE__) && defined(__POWERPC__)
#endif // defined(USE_CYCLE_TIMER)

  class Timer
  {
  public:
    Timer();

    void  reset();
    float getElapsed() const;
    float getReset();

  private:
#if defined(__APPLE__) && defined(__POWERPC__)
    uint64_t getTime() const;

    float    mach_scale;
    uint64_t start;
#elif _WIN32
    clock_t start;
#else
    timeval start;
#endif // defined(__APPLE__) && defined(__POWERPC__)
  };

  inline Timer::Timer()
  {
#if defined(__APPLE__) && defined(__POWERPC__)
    mach_timebase_info_data_t time_info;
    mach_timebase_info(&time_info);

    // Scales to nanoseconds without 1e-9f
    mach_scale = (1e-9*float(time_info.numer))/
      float(time_info.denom);
#endif // defined(__APPLE__) && defined(__POWERPC__)

    reset();
  }

  inline void Timer::reset()
  {
#if defined(__APPLE__) && defined(__POWERPC__)
    start = getTime();
#elif _WIN32
    start = clock();
#else
    gettimeofday(&start, 0);
#endif // defined(__APPLE__) && defined(__POWERPC__)
  }

  inline float Timer::getElapsed() const
  {
#if defined(__APPLE__) && defined(__POWERPC__)
    uint64_t now = getTime();
    return (float(now - start)*mach_scale);
#elif _WIN32
    return (float(clock() - start)/float(CLOCKS_PER_SEC));
#else
    timeval now;
    gettimeofday(&now, 0);
    return (float(now.tv_sec - start.tv_sec) +
            float(now.tv_usec - start.tv_usec)/1000000.);
#endif // defined(__APPLE__) && defined(__POWERPC__)
  }

  inline float Timer::getReset()
  {
    const float elapsed = getElapsed();
    reset();

    return elapsed;
  }

#if defined(__APPLE__) && defined(__POWERPC__)
  inline uint64_t Timer::getTime() const
  {
    unsigned int tbl, tbu0, tbu1;

    do {
      __asm__ __volatile__ ("mftbu %0" : "=r"(tbu0));
      __asm__ __volatile__ ("mftb %0"  : "=r"(tbl));
      __asm__ __volatile__ ("mftbu %0" : "=r"(tbu1));
    } while (tbu0 != tbu1);

    return ((static_cast<unsigned long long>(tbu0) << 32) | tbl);
  }
#endif // defined(__APPLE__) && defined(__POWERPC__)

  ostream& operator<<(ostream& os, Timer& timer);

} // namespace Utility

#endif // Utility_Timer_h
