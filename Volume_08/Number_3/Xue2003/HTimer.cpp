// HTimer.cpp: implementation of the CHTimer class.
//
//////////////////////////////////////////////////////////////////////

#include "HTimer.h"

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CHTimer::CHTimer()
{
	QueryPerformanceFrequency(&frequency);
}

CHTimer::~CHTimer()
{

}

void CHTimer::Mark()
{
	QueryPerformanceCounter(&base);
}

LARGE_INTEGER CHTimer::Elapse_us()
{
	LARGE_INTEGER now;
	QueryPerformanceCounter(&now);

	LARGE_INTEGER r;
	r.QuadPart = (now.QuadPart-base.QuadPart)*1000000/frequency.QuadPart;

	return r;


}

LARGE_INTEGER CHTimer::Elapse_ms()
{
	LARGE_INTEGER now;
	QueryPerformanceCounter(&now);


	LARGE_INTEGER r;
	r.QuadPart = (now.QuadPart-base.QuadPart)*1000/frequency.QuadPart;

	return r;
}


LARGE_INTEGER CHTimer::Elapse_s()
{
	LARGE_INTEGER now;
	QueryPerformanceCounter(&now);

	LARGE_INTEGER r;
	r.QuadPart = (now.QuadPart-base.QuadPart)/frequency.QuadPart;

	return r;

}
