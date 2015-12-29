.KEEP_STATE:

.SUFFIXES: .o .c .cg

OFLAGS = -g
IFLAGS = -I. -I$(XHOME)/include

CFLAGS  = $(OFLAGS) $(IFLAGS)
CCFLAGS = $(OFLAGS) $(IFLAGS)

LDFLAGS = $(CXX) $(OFLAGS)

LINTFLAGS = lint $(IFLAGS)

HDR = \
	ReactDiffuse.h \
	Fbuffer.h \
	gammp.h \
	rand.h

SRC = \
	Main.cpp \
	ReactDiffuse.cpp \
	ReactDiffuseImplicit.cpp \
	ReactDiffuseAux.cpp \
	Fbuffer.cc \
	initializer.cg \
	textureRECT.cg \
	turing_explicit.cg \
	turing_explicit_uniform.cg \
	gray_scott_explicit.cg \
	brusselator_explicit.cg \
	turing_implicit_rhs.cg \
	turing_implicit_relax.cg \
	turing_implicit_residuals.cg \
	turing_implicit_theta_rhs.cg \
	implicit_clamp.cg \
	summation.cg \
	gammp_skeleton.c \
	rand_skeleton.c

OBJS = Main.o \
	ReactDiffuse.o \
	ReactDiffuseImplicit.o \
	ReactDiffuseAux.o \
	Fbuffer.o \
	gammp.o \
	rand.o \

CG_OBJS = \
	initializer.o \
	textureRECT.o \
	turing_explicit.o \
	turing_explicit_uniform.o \
	gray_scott_explicit.o \
	brusselator_explicit.o \
	turing_implicit_rhs.o \
	turing_implicit_relax.o \
	turing_implicit_residuals.o \
	turing_implicit_theta_rhs.o \
	implicit_clamp.o \
	summation.o


OGLLIB = -lGLEW -lglui -lglut -lGL 
CGLIB  = -lCgGL -lCg

LIBS = -lpthread -lm

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

.cg.o:
	cgc -profile fp40 -o $@ $<


all: $(OBJS) $(CG_OBJS) makefile
	if [ $(OS) = Darwin ] ; then \
	  CGLIBS="-framework Cg"; \
	  OGLLIBS="-bind_at_load -L$(XHOME)/lib/ $(OGLLIB) -lglu"; \
	else \
	  CGLIBS="$(CGLIB)"; \
	  OGLLIBS="$(OGLLIB)"; \
	fi ; \
	$(LDFLAGS) -o rd_gpu_demo $(OBJS) $${CGLIBS}  $${OGLLIBS} $(LIBS)

tarball :
	tar --gzip -cvf rd_gpu_demo.tar.gz README makefile $(SRC) $(HDR)
