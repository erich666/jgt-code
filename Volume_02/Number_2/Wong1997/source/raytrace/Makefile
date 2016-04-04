#
# Makefile for rayshade.
#
# Craig Kolb
#
# $Id: Makefile.SH,v 4.0 91/07/17 14:50:30 kolb Exp Locker: kolb $
#
# Bin directory
#
BINDIR = /uac/gds/ttwong/rayshade.4.0/rayshade 
OPTIMIZE = -O
URTINC = -I/usr/local/urt/include
CCFLAGS =  -DLANGUAGE_C 
URTLIB = /usr/local/urt/lib/librle.a
LDFLAGS = -lm 
CC = gcc
MKDEP = /tmp_mnt/uac/gds1/ttwong/rayshade.4.0/mkdep
YACC = bison -y
CLC = 

LIBRAYDIR = ../libray
LIBSHADEDIR = ../libshade
INCLUDE = -I$(LIBRAYDIR) -I$(LIBSHADEDIR) -I.. $(URTINC)
YFLAGS = -d

LIBRAY = $(LIBRAYDIR)/libray.a
LIBSHADE = $(LIBSHADEDIR)/libshade.a

#
# If you are using LINDA, add -DLINDA
# If you are running 'tsnet'-style LINDA, add -DTSNET
# If you are running on a Multimax, add -DMULTIMAX -DSHAREDMEM
# 

CFLAGS = $(CCFLAGS) $(INCLUDE) $(OPTIMIZE)

#
# C-Linda flags, if appropriate
#CLFLAGS = -linda ts 20000
#

SHELL = /bin/sh

LIBS = $(LIBSHADE) $(LIBRAY) $(URTLIB)

DRIVE_C =	main.c version.c

DRIVE_H =	raytrace.h

DRIVE_O = $(DRIVE_C:.c=.o)

HFILES = $(DRIVE_H)

CFILES = $(DRIVE_C)

SHFILES = Makefile.SH

OBJ = $(DRIVE_O)

DEPENDSRC = $(DRIVE_C)

#
# Change $(CC) below to $(CLC) if using Linda.
#
rayshade: $(OBJ) $(LIBS) raytrace.o
	$(CC) $(OPTIMIZE) -o rayshade raytrace.o $(OBJ) $(LIBS) $(LDFLAGS)

rtpoisson: $(OBJ) $(LIBS) rtpoisson.o
	$(CC) $(OPTIMIZE) -o rtpoisson rtpoisson.o $(OBJ) $(LIBS) $(LDFLAGS)

rtjitter: $(OBJ) $(LIBS) rtjitter.o
	$(CC) $(OPTIMIZE) -o rtjitter rtjitter.o $(OBJ) $(LIBS) $(LDFLAGS)

rtmultijit: $(OBJ) $(LIBS) rtmultijit.o
	$(CC) $(OPTIMIZE) -o rtmultijit rtmultijit.o $(OBJ) $(LIBS) $(LDFLAGS)

rthammersley: $(OBJ) $(LIBS) rthammersley.o
	$(CC) $(OPTIMIZE) -o rthammersley rthammersley.o $(OBJ) $(LIBS) $(LDFLAGS)

rthalton: $(OBJ) $(LIBS) rthalton.o
	$(CC) $(OPTIMIZE) -o rthalton rthalton.o $(OBJ) $(LIBS) $(LDFLAGS)

rtregular: $(OBJ) $(LIBS) rtregular.o
	$(CC) $(OPTIMIZE) -o rtregular rtregular.o $(OBJ) $(LIBS) $(LDFLAGS)

#
# Uncomment the following rule if using Linda.
#
#raytrace.lo: raytrace.cl
#	$(CLC) $(INCLUDE) $(LCFLAGS) -c raytrace.cl

#
# End of configuration section
#
install:	rayshade
		mv rayshade $(BINDIR)/rayshade

clean:
	rm -f $(OBJ) core

realclean:
	rm -f $(OBJ) core y.tab.h
	rm -f *.orig Makefile

lint:
	lint $(CFLAGS) -x $(CFILES) -lm

tags:
	ctags -t $(CFILES)

depend:
	(sed '/^# DO NOT DELETE THIS LINE/q' Makefile && \
	 $(MKDEP) $(DEPENDSRC) | sed 's/: \.\//: /; /\/usr\/include/d' \
	) >Makefile.new
	cp Makefile Makefile.bak
	cp Makefile.new Makefile
	rm -f Makefile.new


# DO NOT DELETE THIS LINE
