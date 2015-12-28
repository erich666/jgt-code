### Make sure our default action is to rebuild library
all: lib

### Include definitions and rules common to the library and the
### application makefiles
include ${SOURCEROOT}/makefiles/common.mk

### Name of library is derived from name of present working directory.
LIB = ${LIBDIR}/lib$(notdir $(shell pwd)).a

######################################################################
############# if we are cleaning #####################################
ifeq (${MAKECMDGOALS},clean)

### Clean simply removes the build and library directories
clean:
	rm -fr ${BUILDDIR}
	rm -fr ${LIB}
.PHONY: clean

else
######################################################################
############ Otherwise we are building ###############################

lib: ${LIB}

### Rule for creating the library. 
${LIB}:${objects}
	${AR} ${LIB} $(addprefix ${BUILDDIR}/,$(objects))

endif

