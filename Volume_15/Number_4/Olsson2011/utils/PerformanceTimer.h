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
#ifndef _PerformanceTimer_h_
#define _PerformanceTimer_h_

#include "IntTypes.h"
#include "Assert.h"
#include <stdio.h>

/**
 * Used to time some section of code, is reasonably accurate.
 */
class PerformanceTimer
{
public:
  PerformanceTimer() : m_ticksAtStart(0), m_accumulatedTicks(0), m_isRunning(false)
  {
  }

  /**
   * Start the performance timer, after this the getElapsedTime() will return meaningfuil values.
   */
  void start()
  {
    ASSERT(!m_isRunning);
    m_isRunning = true;
    m_ticksAtStart = getTickCount();
  }

  /**
   * After the timer is stopped getElapsedTime() will return the same value until started again.
   */
  void stop()
  {
    ASSERT(m_isRunning);
    m_accumulatedTicks += getTickCount() - m_ticksAtStart;
    m_isRunning = false;
  }

  /**
   * restart is just a shorthand for stop();start();
   */
  void restart() { stop(); start(); }

  /**
   * Get the elapsed time in seconds, if the timer is running it gets the current elapsed time 
   * if it has been stopped, it takes the elapsed time when stop() was called.
   */
  double getElapsedTime()
  {
    uint64_t ticks = getTickCount();
    // SUPERHACK, my computer reports decreasing time at times, need mobo driver update perhaps... weird.
    // - Does not seem to happen after an AMD processor driver install...
    if (ticks < m_ticksAtStart)
    {
    //printf("get: %I64d\n", ticks);
      ticks = m_ticksAtStart;
    }
    return ticksToSeconds((m_isRunning ? ticks - m_ticksAtStart : m_accumulatedTicks));
  }

  /**
   * Get the current tick count, in whatever unit the machine pleases.
   */
  static uint64_t getTickCount();
  /**
   * Returns the number of ticks per second, use for converting the above tick count to something useful.
   */
  static uint64_t getTicksPerSecond()
  {
    static uint64_t ticksPerSecond = initTicksPerSecond();
    return ticksPerSecond;
  }

  /**
   * converts a given number of ticks to seconds.
   */
  static double ticksToSeconds(uint64_t ticks)
  {
    return double(ticks) / double(getTicksPerSecond());
  }

private:
  static uint64_t initTicksPerSecond();
  uint64_t m_ticksAtStart;
  uint64_t m_accumulatedTicks;
  bool m_isRunning;
};

#endif // _PerformanceTimer_h_
