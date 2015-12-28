## Platform specific makefile

## COMPILE OPTIONS

SYSDIR = /System/Library/Frameworks/

CXXFLAGS += 
ifeq (${TARGET},debug)
CXXFLAGS 	+= -g
else
CXXFLAGS 	+= -O3 -DNDEBUG
endif 

## LINK OPTIONS
WIN_SYS_LIBS    = -framework Cocoa -framework ApplicationServices
GLLIBS     	= -framework OpenGL
GLUTLIBS	= -framework GLUT
XLIBS      	= -L/usr/X11R6/lib -lXt -lXmu -lSM -lX11 
ILLIBS		= -framework IL


AR		= libtool -o
DEPFLAGS	= -MM
INSTALL		= install -m 0755
