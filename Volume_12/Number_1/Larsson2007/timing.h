

#ifndef _TIMING_H
#define _TIMING_H

class Timing {
	private:
		BOOL pentClk;				
		double pentClkRes;
		
		union {
			DWORD tickStart;
			LARGE_INTEGER clkStart;
		};
	
	public:
		
		Timing(void) {
			LARGE_INTEGER pentClkFreq;
			pentClk = QueryPerformanceFrequency(&pentClkFreq);			
			//pentClk = 0;
			pentClkRes = 1.0f / (double) pentClkFreq.QuadPart;			
		}

		void start(void) {
			if (pentClk) {
				startQueryPerformanceCounter();	// use the Pentium internal clock
				//startAsmGetPentiumCounter();
			} else {
				startGetTickCount(); // use the old bad resolution timer
			}
		}

		// returns time passed in seconds
		double stop(void) {
			double time;

			if (pentClk) {
				time = stopQueryPerformanceCounter();
				//time = stopAsmGetPentiumCounter();
			} else {
				time = stopGetTickCount();
			}
			return time;
		}
		
		void startQueryPerformanceCounter(void) {
			QueryPerformanceCounter(&clkStart);
		}

		double stopQueryPerformanceCounter(void) {
			LARGE_INTEGER clkStop;
			QueryPerformanceCounter(&clkStop);
			return (clkStop.QuadPart - clkStart.QuadPart) * pentClkRes;
		}

		void startGetTickCount(void) {
			tickStart = GetTickCount();
		}

		double stopGetTickCount(void) {			
			return ((double)(GetTickCount() - tickStart)) * 0.001;			
		}
};

#endif

