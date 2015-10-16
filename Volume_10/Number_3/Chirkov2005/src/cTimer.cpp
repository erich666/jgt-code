// -------------------------------------------------------------------------
// File:    cTimer.h 
// Desc:    a simple timer (start-stop watch); uses gettimeofday()
//
// Author:  Tomas Möller
// History: March, 2000 (started)
//          July 2002, rewrote for PCs
// -------------------------------------------------------------------------

#include "cTimer.h"



cTimer::cTimer()
{
	mTotalTime=0.0;
	QueryPerformanceFrequency(&mFrequency);
}

void cTimer::start(void)
{
	//   void *dummy=NULL;
	//   gettimeofday(&mStartTime,dummy);
	QueryPerformanceCounter(&mStartTime);
}


void cTimer::stop(void)
{
	//   void *dummy=NULL;
	//   struct timeval end_time;
	//   gettimeofday(&end_time,dummy);
	//   mTotalTime+=(((double)end_time.tv_sec*1000000.0+end_time.tv_usec)-((double)mStartTime.tv_sec*1000000.0+mStartTime.tv_usec))/1000000.0;

	LARGE_INTEGER end_time;
	QueryPerformanceCounter(&end_time);
	mTotalTime+=double(end_time.LowPart-mStartTime.LowPart)/mFrequency.LowPart;	
}


void cTimer::reset(void)
{
	mTotalTime=0.0;
}

void cTimer::multByFactor(float factor)	// multiply stored by a factor
{
	mTotalTime*=factor;
}

double cTimer::getTime(void)  // in seconds
{
	return mTotalTime;
}
