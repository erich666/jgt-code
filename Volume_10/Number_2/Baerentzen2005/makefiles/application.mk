# application.mk - Application makefile. This makefile is included from
# Application directories.
#
#
#

### Make sure app is primary goal
all: app

### Include common rules and defs
include ${SOURCEROOT}/makefiles/common.mk

### Set up load libraries
LOADLIBES  	= $(addprefix -l,${OWN_LIBS}) ${LIBS}

### Where to find libraries
LDFLAGS 		+= -L${LIBDIR} -L${SOURCEROOT}/lib/${PLATFORM}

### Generate a list of object files that are not programs.
not_main_objects = $(filter-out $(addsuffix .o,${PROGRAMS}), ${objects})

######################################################################
############ We are building #########################################
######################################################################
ifeq (${MAKECMDGOALS},clean)

clean:
	rm -fr ${BUILDDIR}

else

# The force target is useful for recompiling libraries. Thus if the appliaction
# or any library it depends on have been changed, we only have to go
# > make force 
# in the directory containing the application. Application is also compiled
# and linked.
force: 
	$(foreach lib, ${OWN_LIBS},	${MAKE} -C ${SOURCEROOT}/Libsrc/${lib};)
	${MAKE} ${PROGRAMS}
	$(foreach prg, ${PROGRAMS},	${INSTALL} ${BUILDDIR}/${prg} ${SOURCEROOT}/bin/${prg};)

# app is default target. Compiles and links all applications.
app:
	${MAKE} ${PROGRAMS}
	$(foreach prg, ${PROGRAMS},	${INSTALL} ${BUILDDIR}/${prg} ${SOURCEROOT}/bin;)

# Do not compile programs directly from source. Always compile object file
%:%.cpp

# Link application. .FORCE ensures that we always link. I haven't found a 
# great way to make compilation depend on the timestamp of the libraries, but
# always linking solves the problem at moderate cost.
%:%.o ${not_main_objects} .FORCE
	${CXX} -o ${BUILDDIR}/$@ ${BUILDDIR}/$@.o \
	${CXXFLAGS} ${LDFLAGS} $(addprefix ${BUILDDIR}/,${not_main_objects}) \
	${LOADLIBES}

.FORCE:

endif

