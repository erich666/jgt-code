# You must change the paths to GL and FLTK to match your system's configuration

CC   = g++
GLDIR = /usr/X11R6/lib/
FLTKDIR = /graphics/Fltk
OUTDIR = linux

OBJ = $(OUTDIR)main.o $(OUTDIR)JofGTIBar.o $(OUTDIR)CAMIBar.o $(OUTDIR)OGLObjsCamera.o $(OUTDIR)OGLObjsCamera_Keyboard.o $(OUTDIR)R2Line.o $(OUTDIR)R2Line_seg.o $(OUTDIR)UserInterface.o

LIBS =  -L$(GLDIR) -L$(FLTKDIR) -lfltk -lfltk_gl -lfltk_images -lfltk_forms -lGLU -lGL  -lpng -ljpeg -L$(GLDIR) -L$(FLTKDIR) -lfltk -lfltk_gl -lfltk_images -lfltk_forms -lGLU -lGL  -lpng -ljpeg
INCS =  -I$(FLTKDIR) -I/usr/X11R6/include

BIN  = JofGTIBar.exe
CFLAGS = $(INCS) -ansi -pedantic -W -Wall -Wno-long-long -Wno-unused -s

.PHONY: all all-before all-after clean clean-custom

all: all-before $(BIN) all-after


clean: clean-custom
	rm -f $(OBJ) $(BIN)

$(BIN): $(OBJ)
	$(CC) $(OBJ) -o "JofGTIBar.exe" $(LIBS) $(CFLAGS)

$(OBJ): $(OUTDIR)%.o: %.cpp
	$(CC) -c $(CFLAGS) $< -o $@

