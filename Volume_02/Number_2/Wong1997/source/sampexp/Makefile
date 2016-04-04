#
# Makefile for sampexp.
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

DRIVE_C =	sampexp.c version.c poly.c poly_scan.c udpoint.c

DRIVE_H =	poly.h udpoint.h

DRIVE_O = $(DRIVE_C:.c=.o)

HFILES = $(DRIVE_H)

CFILES = $(DRIVE_C)

SHFILES = Makefile.SH

OBJ = $(DRIVE_O)

DEPENDSRC = $(DRIVE_C)

#
# Change $(CC) below to $(CLC) if using Linda.
#
sampexp: $(OBJ) $(LIBS)
	$(CC) $(OPTIMIZE) -o sampexp $(OBJ) $(LIBS) $(LDFLAGS)

#
# Uncomment the following rule if using Linda.
#
#raytrace.lo: raytrace.cl
#	$(CLC) $(INCLUDE) $(LCFLAGS) -c raytrace.cl

#
# End of configuration section
#
install:	sampexp
		mv sampexp $(BINDIR)/sampexp

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
