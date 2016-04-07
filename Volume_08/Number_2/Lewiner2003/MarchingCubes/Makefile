##############################################################################
#
# Makefile for Marching Cubes examples
#
##############################################################################


###################### default targets ############################

dbg :
	make -k mcd
	make -k lutd
	make -k mcGLd

rel :
	make -k mc
	make -k lut
	make -k mcGL


########################### Programs #################################

CC        = g++ -Wall

########################### Flags #################################

FLAGS = -I.
LIBS  = -L/usr/X11/lib -lm -lz -lGL -lGLU -lglui
ifeq ($(shell uname),Cygwin)
  LIBS += -lopengl32 -lglu32 -lglut32
endif
ifeq ($(shell uname),Darwin)
  LIBS += -framework OpenGL -framework GLUT
endif


ifeq ($(MAKECMDGOALS),mcd)
  OBJDIR  = debug
  CFLAGS  = -pg -g -ggdb $(FLAGS) -DDEBUG
endif
ifeq ($(MAKECMDGOALS),mc)
  OBJDIR  = release
  CFLAGS  = -O3 $(FLAGS)
endif
ifeq ($(MAKECMDGOALS),lutd)
  OBJDIR  = debug_lut
  CFLAGS  = -pg -g -ggdb $(FLAGS) -DDEBUG
endif
ifeq ($(MAKECMDGOALS),lut)
  OBJDIR  = release_lut
  CFLAGS  = -O3 $(FLAGS)
endif
ifeq ($(MAKECMDGOALS),mcGLd)
  OBJDIR  = debug_GL
  CFLAGS  = -pg -g -ggdb $(FLAGS) -DDEBUG
endif
ifeq ($(MAKECMDGOALS),mcGL)
  OBJDIR  = release_GL
  CFLAGS  = -O3 $(FLAGS)
endif

$(OBJDIR) :
	mkdir -p $(OBJDIR)


############################  mc  #################################

MC_OBJECTS = \
  $(OBJDIR)/ply.o \
  $(OBJDIR)/MarchingCubes.o \
  $(OBJDIR)/main.o

mcd : $(OBJDIR) $(MC_OBJECTS)
	$(CC) -pg -g -ggdb -o $@ $(MC_OBJECTS)

mc  : $(OBJDIR) $(MC_OBJECTS)
	$(CC) -O3 -o $@ $(MC_OBJECTS)


###########################  mcGL  ################################

MCGL_OBJECTS = \
  $(OBJDIR)/fparser.o \
  $(OBJDIR)/gl2ps.o \
  $(OBJDIR)/ply.o \
  $(OBJDIR)/MarchingCubes.o \
  $(OBJDIR)/glui_controls.o \
  $(OBJDIR)/glui_draws.o \
  $(OBJDIR)/glui_main.o \
  $(OBJDIR)/glui_cmdline.o \
  $(OBJDIR)/glui_mc.o \
  $(OBJDIR)/glui_mouse.o


mcGLd : $(OBJDIR) $(MCGL_OBJECTS)
	$(CC) -pg -g -ggdb -o $@ $(MCGL_OBJECTS) $(LIBS)

mcGL  : $(OBJDIR) $(MCGL_OBJECTS)
	$(CC) -O3 -o $@ $(MCGL_OBJECTS) $(LIBS)


###########################  lut  ##################################

lutd : LookUpTableTest.cpp LookUpTable.h
	$(CC) $(FLAGS) -pg -g -ggdb -o $@ $< -I. $(LIBS)

lut  : LookUpTableTest.cpp LookUpTable.h
	$(CC) $(FLAGS) -O3 -o $@ $< -I. $(LIBS)


########################### Commands #################################

$(OBJDIR)/%.o : %.cpp
	$(CC) -c $<	-o $@	$(CFLAGS)

$(OBJDIR)/%.o : %.c
	$(CC) -c $<	-o $@	$(CFLAGS)

########################### Tar file #################################

DIRNAME = $(notdir ${PWD})
TARNAME = ../$(shell echo $(DIRNAME) | tr [:upper:] [:lower:]).tgz

$(TARNAME) : README Makefile *.h *.cpp *.c *.csg *.sln *.vcproj *.xcodeproj Doxyfile
	tar -C.. -c -v -z -f $@ $(foreach file,$^,$(DIRNAME)/$(file)) > /dev/null


########################### Documentation #################################

html/index.html : Doxyfile *.h
	rm -Rf html
	doxygen >& /dev/null


########################### Directives #################################

.PHONY: all clean depend doc tar dbg rel cvs

tar : $(TARNAME)

cvs :
	cvs ci -m "backup $(shell date -u '+%A, %d %b %Y, %H:%M')"

depend:
	makedepend -p"\$$(OBJDIR)/" -Y -I. $(wildcard *.cpp) $(wildcard *.c) >& /dev/null
	touch -r Makefile.bak Makefile
	rm -f Makefile.bak

clean:
	rm -f */*.o core *~ gmon.out mc mcd lut lutd mcGL mcGLd

doc  :  html/index.html

all :
	make -k depend
	make -k dbg
	make -k rel
	make tar
	make doc


########################### Dependencies #################################

# DO NOT DELETE

$(OBJDIR)/LookUpTableTest.o: LookUpTable.h
$(OBJDIR)/MarchingCubes.o: MarchingCubes.h ply.h LookUpTable.h
$(OBJDIR)/fparser.o: fparser.h
$(OBJDIR)/glui_cmdline.o: glui_defs.h MarchingCubes.h
$(OBJDIR)/glui_controls.o: glui_defs.h MarchingCubes.h csg.h
$(OBJDIR)/glui_draws.o: glui_defs.h MarchingCubes.h
$(OBJDIR)/glui_main.o: glui_defs.h MarchingCubes.h
$(OBJDIR)/glui_mc.o: gl2ps.h csg.h fparser.h glui_defs.h MarchingCubes.h
$(OBJDIR)/glui_mouse.o: glui_defs.h MarchingCubes.h
$(OBJDIR)/main.o: MarchingCubes.h
$(OBJDIR)/symmetries.o: LookUpTable.h
$(OBJDIR)/gl2ps.o: gl2ps.h
$(OBJDIR)/ply.o: ply.h
