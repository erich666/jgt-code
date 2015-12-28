ifeq (${TARGET},debug)
CXXFLAGS 		+= -g
else
CXXFLAGS 		+= -O3 -DNDEBUG
endif 

LDFLAGS    	+= -L/usr/X11R6/lib

GLLIBS     	= -lglut -lGLU -lGL 
XLIBS      	= -lXt -lXmu -lSM -lX11 

AR					= ar -cr
DEPFLAGS		= -MM
INSTALL			= install -m 0755
