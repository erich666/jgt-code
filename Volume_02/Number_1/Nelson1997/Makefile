#
# Makefile for aaline
#

TARGETS	= aaline
OBJS	= aaline.o aadraw.o display_unix.o
CFLAGS	= -g 

CC	= cc

INCS	= -I/usr/openwin/include
LIBS	= -L/usr/openwin/lib -lX11 -lm

all: ${TARGETS}

aaline: ${OBJS}
	${CC} ${CFLAGS} -o $@ ${OBJS} ${LIBS}

clean:
	/bin/rm -f *.o ${TARGETS} *~

.SUFFIXES: 
.SUFFIXES: .o .c

.c.o:
	${CC} ${CFLAGS} ${INCS} -c $<

