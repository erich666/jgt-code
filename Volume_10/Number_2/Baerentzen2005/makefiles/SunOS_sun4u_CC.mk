CXXFLAGS		+= -LANG:std -I/usr/local/glut-3.7/include

ifeq (${TARGET},debug)
CXXFLAGS 		+= -g
else
CXXFLAGS 		+= -O3 -DNDEBUG
endif 

LDFLAGS    	+= -L/usr/X11R6/lib -L/usr/local/glut-3.7/lib/glut

XLIBS      	= -lXt -lXmu -lSM -lX11 
WIN_SYS_LIBS    = ${XLIBS}
GLLIBS     	= -lglut -lGLU -lGL 
GLUTLIBS    = -lglut

AR					= CC -xar -o
DEPFLAGS		= -xM1
INSTALL			= cp
