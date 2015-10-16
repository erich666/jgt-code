// -------------------------------------------------------------------------
// File:    cTimer.h 
// Desc:    a simple timer (start-stop watch); uses gettimeofday()
//
// Author:  Tomas Möller
// History: March, 2000 (started)
//          July 2002, rewrote for PCs
// -------------------------------------------------------------------------

#include <windows.h>
//#include <sys/time.h>

#ifndef C_TIMER_H
#define C_TIMER_H

class cTimer
{
protected:
//   struct timeval mStartTime;
	LARGE_INTEGER mStartTime, mFrequency;
	double mTotalTime;
public:
	cTimer();
	void start(void);					// starts the watch
	void stop(void);					// adds the time from start() to an internal time variable
	void reset(void);					// resets the internal time variable
	double getTime(void);				// in seconds 
	void multByFactor(float factor);	// multiply stored by a factor
};

#endif

