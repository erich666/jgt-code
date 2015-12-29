# Simple Makefile for building the test program on Mac OS X
# This should be easy to modify to compile on other systems.
# Written by Petrik Clarberg <petrik@cs.lth.se>, Lund University, 2007-2008.

CXX	:= gcc
LD	:= g++

INCS	:= -F/System/Library/Frameworks
LDFLAGS	:= -framework CoreServices
CXXFLAGS := -DMACOSX -Wall -Wno-unknown-pragmas -Wno-format -O3 -mdynamic-no-pic -fmessage-length=0 -funroll-loops -fvisibility=hidden -fvisibility-inlines-hidden

# -ffast-math -finline-limit=1000  -ftree-vectorize 

OUTPUT	:= simd_mapping
SRCS	:= main.cpp
OBJS	:= $(addsuffix .o,$(basename ${SRCS}))

.PHONY : clean

all: ${OUTPUT}

${OUTPUT} : ${OBJS}
	${LD} -o $@ ${LDFLAGS} ${OBJS}

%.o : %.cpp
	${CXX} -c ${CXXFLAGS} ${INCS} $< -o $@

clean:
	-rm -f ${OBJS} ${OUTPUT}
