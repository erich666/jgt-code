SHELL = /bin/sh

.SUFFIXES:
.SUFFIXES: .cpp .o .c

include ${SOURCEROOT}/makefiles/definitions.mk

### Include platform specific makefile
include  ${SOURCEROOT}/makefiles/${PLATFORM}.mk

### BUILD is the subdirectory where all builds reside.
BUILD								= .build

### BUILDDIR is the directory where the compilation for a specific 
### platform and target combination takes place.
BUILDDIR						= ./${BUILD}/$(PLATFORM_TARG)

### LIBDIR is the directory where we stick .a files pertaining to 
### the specific combination of platform and target.
LIBDIR							= ${SOURCEROOT}/lib/${PLATFORM_TARG}

### BINDIR is the directory where binary files are put.
BINDIR							= ${SOURCEROOT}/bin

### Append path to compiler flags
CXXFLAGS						+= -I${SOURCEROOT}/Libsrc -I${SOURCEROOT}/include

### Generate list of source files.
sources							= $(shell ls *.cpp)
### Generate list if object files from list of sources.
objects							= $(sources:.cpp=.o)
### Generate list if dependency files from list of sources.
deps								= $(addprefix ${BUILDDIR}/,$(sources:.cpp=.d))

### Set up vpath so that make finds object files in builddir
### hence does not rebuild unneccesarily
VPATH 							= ${BUILDDIR}

### Include dependencies - but not if we are cleaning
ifneq (${MAKECMDGOALS},clean)
include ${deps}
endif

### Make dependencies, but first create BUILDDIR and LIBDIR.
${BUILDDIR}/%.d:%.cpp
	$(shell if \[ \! -d ${BUILD} \] ; then mkdir ${BUILD} ; fi)
	$(shell if \[ \! -d ${BUILDDIR} \] ; then mkdir ${BUILDDIR} ; fi)
	$(shell if \[ \! -d ${LIBDIR} \] ; then mkdir ${LIBDIR}; fi)
	$(CXX) ${DEPFLAGS} ${CXXFLAGS} $< > $@

### Rule for building object files from C++ source files
%.o: %.cpp
	${CXX} -c ${CXXFLAGS} -o ${BUILDDIR}/$@ $<

### Rule for building object files from C source files
### I use a C++ compiler. I think this is best.
%.o: %.c
	${CXX} -c ${CXXFLAGS} -o ${BUILDDIR}/$@ $<


