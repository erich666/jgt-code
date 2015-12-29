.KEEP_STATE:

.SUFFIXES: .o .c

OFLAGS = -O4
IFLAGS = -I. -I$(XHOME)/include

SRC = Main.cpp \
	rd_base.cc \
	rd_turing.cc \
	rd_gray_scott.cc \
	gammp_skeleton.c \
	rand_skeleton.c

HDR = \
	rd_base.h \
	rd_turing.h \
	rd_gray_scott.h \
	gammp.h \
	rand.h

OBJS = Main.o \
	rd_base.o \
	rd_turing.o \
	rd_gray_scott.o \
	gammp.o \
	rand.o

X11LIB = -L$(XHOME)/lib/ -lX11
OGLLIB = -lGLEW -lglui -lglut -lGL 

LIBS = -lm

CFLAGS  = -D$(MACHINE) $(OFLAGS) $(IFLAGS)
CCFLAGS = -D$(MACHINE) $(OFLAGS) $(IFLAGS)

LDFLAGS = $(CXX)

clean :
	-rm -f *.o *.ln *~

.c.o:
	$(CC) -c $(CFLAGS) -o $@ $<

.cc.o:
	$(CXX) -c $(CCFLAGS) -o $@ $<


.cpp.o:
	$(CXX) -c $(CCFLAGS) -o $@ $<

all: $(OBJS) makefile
	if [ $(OS) = Darwin ] ; then \
	  OGLLIBS="-bind_at_load -L$(XHOME)/lib/ $(OGLLIB) -lglu"; \
	else \
	  OGLLIBS="$(OGLLIB)"; \
	fi ; \
	$(LDFLAGS) -o rd_cpu_demo $(OBJS) $${OGLLIBS} $(LIBS)

tarball :
	tar --gzip -cvf rd_cpu_demo.tar.gz README makefile $(SRC) $(HDR)
