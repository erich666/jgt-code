ifeq (${TARGET},debug)
CXXFLAGS 	+= -g
else
CXXFLAGS 	+= -O3 -DNDEBUG
endif 

LDFLAGS    	+= -L/usr/X11R6/lib

XLIBS      	=  -lXt -lXmu -lSM -lX11 
WIN_SYS_LIBS 	=
GLLIBS     	= -lGLU -lGL 
GLUTLIBS	= -lglut
ILLIBS		= -lIL -lILU -ljpeg -ltiff -lpng -lmng

AR		= ar -cr
DEPFLAGS	= -MM
INSTALL		= install -m 0755
