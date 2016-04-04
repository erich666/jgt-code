.SUFFIXES:	.o .c .cc .C .a


RM = -rm


.cc.o:	
	$(C++) $(CFLAGS) $(CCFLAGS) $(EXTRAFLAGS) $*.cc

.c.o:
	$(CC)  $(CFLAGS) $(EXTRAFLAGS) $*.c


#############################################################
##
##  Enable/disable the lines below to compile on SGI or Linux
##
#############################################################


C++        = g++
CFLAGS     = -c -Iinclude -Wall -Wformat -Wno-switch -Wcast-align -Wpointer-arith \
	     -Wsynth -finline-functions
CCFLAGS    = -fno-for-scope -fno-access-control -O2

#C++	    = CC
#CFLAGS	    = -c -Iinclude -DIS_BIG_ENDIAN
#CCFLAGS    = -O2 -ptused
