all: consistencyTest perfTest

# optimization options
#CXXFLAGS += -O3 -march=athlon-xp
#CXXFLAGS += -O3 -march=pentium4 -msse -msse2 -mfpmath=sse
#CXXFLAGS += -g -DNEIGHAND_DEBUG
CXXFLAGS += -O3

# Library is now compatible with the aliasing rule
CXXFLAGS += -Wall # -fno-strict-aliasing is not necessary anymore
CPPFLAGS += -I../src

CXX = g++

lq.o: lq.c lq.h
	$(CXX) $(CXXFLAGS) -c lq.c -o lq.o

consistencyTestWrapXY: consistencyTest.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -DCONSISTENCY_TEST_WRAP=true,true,false consistencyTest.cpp -o consistencyTestWrapXY

consistencyTestWrapAll: consistencyTest.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -DCONSISTENCY_TEST_WRAP=true,true,true consistencyTest.cpp -o consistencyTestWrapAll

consistencyTestWrapNone: consistencyTest.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -DCONSISTENCY_TEST_WRAP=false,false,false consistencyTest.cpp -o consistencyTestWrapNone

consistencyTest: consistencyTestWrapXY consistencyTestWrapAll consistencyTestWrapNone
	./consistencyTestWrapXY
	./consistencyTestWrapAll
	./consistencyTestWrapNone

KDLQConsistencyTest: KDLQConsistencyTest.cpp lq.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I. KDLQConsistencyTest.cpp lq.o -o KDLQConsistencyTest

perfTestWrapXY: perfTest.cpp lq.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I. -DBASE_EXP2=4 -DWRAP_ARGUMENTS=true,true,false perfTest.cpp  lq.o -o perfTestWrapXY

perfTestWrapAll: perfTest.cpp lq.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I. -DBASE_EXP2=4 -DWRAP_ARGUMENTS=true,true,true perfTest.cpp  lq.o -o perfTestWrapAll

perfTestWrapNone: perfTest.cpp lq.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I. -DBASE_EXP2=4 -DWRAP_ARGUMENTS=false,false,false perfTest.cpp  lq.o -o perfTestWrapNone

perfTest: perfTestWrapXY perfTestWrapAll perfTestWrapNone
	./perfTestWrapXY
	./perfTestWrapAll
	./perfTestWrapNone

distknn: distknn.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I. -DBASE_EXP2=4 -DWRAP_ARGUMENTS=false,false,false distknn.cpp -o distknn

clean:
	rm -f  *.o perfTestWrapAll perfTestWrapXY perfTestWrapNone consistencyTestWrapXY consistencyTestWrapAll consistencyTestWrapNone KDLQConsistencyTest distknn


