CXXFLAGS		+= -LANG:std

ifeq (${TARGET},debug)
CXXFLAGS 		+= -g -DOLD_C_HEADERS
else
CXXFLAGS 		+= -O3 -DNDEBUG -DOLD_C_HEADERS
endif 

LDFLAGS    	+= -LANG:std -L/usr/X11R6/lib
GLLIBS     	= -lGLU -lGL 
GLUTLIBS    = -lglut
XLIBS      	= -lXt -lXmu -lSM -lX11 
WIN_SYS_LIBS    = ${XLIBS}


AR			= CC -ar -o
DEPFLAGS		= -M
INSTALL			= cp
