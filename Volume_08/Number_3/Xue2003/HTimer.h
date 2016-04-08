// HTimer.h: interface for the CHTimer class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_HTIMER_H__843B2886_1758_4007_B8A7_761375376822__INCLUDED_)
#define AFX_HTIMER_H__843B2886_1758_4007_B8A7_761375376822__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include <windows.h>

class CHTimer  
{
public:
	LARGE_INTEGER Elapse_s();
	LARGE_INTEGER Elapse_ms();
	LARGE_INTEGER Elapse_us();
	void Mark();
	CHTimer();
	virtual ~CHTimer();

private:
	LARGE_INTEGER base;
	LARGE_INTEGER frequency;
};

#endif // !defined(AFX_HTIMER_H__843B2886_1758_4007_B8A7_761375376822__INCLUDED_)
