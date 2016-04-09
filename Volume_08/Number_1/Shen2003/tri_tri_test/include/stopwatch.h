/* How to use class CStopwatch

// Create a stopwatch timer (which defaults to the current time).
CStopwatch stopwatch;

// Execute the code I want to profile here.

// Get how much time has elapsed up to now.
__int64 qwElapsedTime = stopwatch.Now();

// qwElapsedTime indicates how long the profiled code
//executed in milliseconds

*/

#ifndef __STOPWATCH_H__
#define __STOPWATCH_H__

#include <afxdisp.h>        // MFC Automation classes

#define MILLISECOND 1000
#define MICROSECOND 1000000

class CStopwatch
{
public:
	CStopwatch()
	{
		QueryPerformanceFrequency(&m_liPerfFreq);
		Start();
	}

	void Start()
	{
		QueryPerformanceCounter(&m_liPerfStart);
	}

	__int64 Now(long PerSecond=MILLISECOND) 	// Returns # of microseconds since Start was called
	{
		QueryPerformanceCounter(&m_liPerfNow);
		return(((m_liPerfNow.QuadPart - m_liPerfStart.QuadPart) * PerSecond)
				/ m_liPerfFreq.QuadPart);
	}

private:
	LARGE_INTEGER m_liPerfFreq;		// Counts per second
	LARGE_INTEGER m_liPerfStart;	// Starting count
	LARGE_INTEGER m_liPerfNow;	// Starting count
};

#endif	//__STOPWATCH_H__