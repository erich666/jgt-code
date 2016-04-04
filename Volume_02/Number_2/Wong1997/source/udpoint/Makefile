
##### MACROS #####
# for Ultrix
#GL_LIB = libMesaGL.a
#GLU_LIB = libMesaGLU.a
#TK_LIB = libMesatk.a
#AUX_LIB = libMesaaux.a
#GLUT_LIB = libglut.a
#CC = gcc
#CPP = g++
#CFLAGS = -pedantic -O2
#MAKELIB = ar ruv
#RANLIB = ranlib
#XLIBS = -lX11 -lXmu -lXt -lXi
#INCDIR = -I/usr/local/mesa/include -I/usr/X11R5/include -I/usr/local/glut/include
#GL_LIBS = -L/usr/X11R5/lib -L/usr/local/mesa/lib -lMesaaux -lMesatk -lMesaGLU -lMesaGL  $(XLIBS)
#LIB_DEP = /usr/local/mesa/lib/$(GL_LIB) /usr/local/mesa/lib/$(GLU_LIB) /usr/local/mesa/lib/$(TK_LIB) /usr/local/mesa/lib/$(AUX_LIB)


## for SunOS
#GL_LIB = libMesaGL.a
#GLU_LIB = libMesaGLU.a
#GLUT_LIB = libglut.a
#TK_LIB = libMesatk.a
#AUX_LIB = libMesaaux.a
#CC = gcc
#CPP = g++
#CFLAGS = -O3 -DSHM -DSUNOS4 
#MAKELIB = ar ruv
#RANLIB = ranlib
#XLIBS = -L/usr/X11R5/lib -lX11 -lXext -lXmu -lXt -lXi
#INCDIR = -I/usr/local/mesa/include -I/usr/X11R5/include -I/usr/local/glut/include
#GL_LIBS = -L/usr/local/mesa/lib -L/usr/local/glut/lib -lglut -lMesaaux -lMesatk -lMesaGLU -lMesaGL  $(XLIBS)  /usr/X11R5/lib/libXt.sa.4.10
#LIB_DEP = /usr/local/mesa/lib/$(GL_LIB) /usr/local/mesa/lib/$(GLU_LIB) /usr/local/mesa/lib/$(TK_LIB) /usr/local/mesa/lib/$(AUX_LIB)


## for Solaris
#GL_LIB = libMesaGL.a
#GLU_LIB = libMesaGLU.a
#GLUT_LIB = libglut.a
#TK_LIB = libMesatk.a
#AUX_LIB = libMesaaux.a
#CC = gcc
#CPP = g++
##CFLAGS = -O3 -DSHM -DSUNOS4
#CFLAGS =
#MAKELIB = ar ruv
#RANLIB = ranlib
#INCDIR = -I/usr/ucbinclude
#LIBS = -L/usr/ucblib -lucb -lsocket
#XLIBS = -L/usr/X11R5/lib -lX11 -lXext -lXmu -lXt -lXi 
#GL_INCDIR = -I/usr/local/mesa/include -I/usr/X11R5/include -I/usr/local/glut/include
#GL_LIBS = -L/usr/local/mesa/lib -L/usr/local/glut/lib -lglut -lMesaaux -lMesatk -lMesaGLU -lMesaGL  $(XLIBS)
#LIB_DEP = /usr/local/mesa/lib/$(GL_LIB) /usr/local/mesa/lib/$(GLU_LIB) /usr/local/mesa/lib/$(TK_LIB) /usr/local/mesa/lib/$(AUX_LIB)


## for SGI
GL_LIB = libGL.a  
GLU_LIB = libGLU.a
GLUT_LIB = libglut.a
TK_LIB = libtk.a
AUX_LIB = libaux.a
CC = cc
CPP = CC
CFLAGS = -O2 -DUNIX
MAKELIB = ar ruv
RANLIB = ranlib
XLIBS = -L/usr/X11R5/lib -lX11 -lXext -lXmu -lXt -lXi
IRIS_INCDIR = -I/uac/gds/ttwong/lib/iristools.solaris/include
IRIS_LIBS = -L/uac/gds/ttwong/lib/iristools.solaris/lib -limage
INCDIR = -I/usr/include
LIBS = -L/usr/lib 
GL_INCDIR = -I/usr/X11R5/include -I/usr/local/glut/include
GL_LIBS =  -L/usr/local/glut/lib -lglut -lGL -lGLU $(XLIBS)
LIB_DEP = /usr/lib/$(GL_LIB) /usr/lib/$(GLU_LIB) /usr/lib/$(TK_LIB) /usr/lib/$(AUX_LIB)


##### RULES #####

.SUFFIXES:
.SUFFIXES: .c .o .cc 


.c.o:
	$(CC) $(INCDIR) $(GL_INCDIR) $(CFLAGS) -c $<

.c: 
	$(CC) $(INCDIR) $(GL_INCDIR) $(LIBS) $(GL_LIBS) $(CFLAG) $< -o $@ -lm

.cc.o:
	$(CPP) $(GL_INCDIR) $(CFLAGS) -c $<



##### TARGETS #####

OBJFILES = vecmath.o trackball.o udpoint.o
HFILES = common.h vecmath.h trackball.h udpoint.h


all:	demo


clean:
	-rm *.o core

realclean:
	-rm $(PROGS)
	-rm *.o *~

targets: $(PROGS)


demo: $(HFILES) $(OBJFILES) sphere.o plane.o
	$(CPP) $(GL_INCDIR) $(CFLAGS) sphere.o $(OBJFILES) $(GL_LIBS) $(LIBS) -o sphere -lm
	$(CPP) $(GL_INCDIR) $(CFLAGS) plane.o  $(OBJFILES) $(GL_LIBS) $(LIBS) -o plane -lm

