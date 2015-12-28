SHELL = /bin/sh
########################################################################
#
# Central makefile. Descends all source directories, making recursively.
#
# Targets:
#
#  all       - make all libraries, then all applications
#  cvs       - add all cpp h and Makefile files to cvs
#  lib       - make all libraries
#  app       - make all applications
#  clean     - clean all library directories and app directories.
#  this removes only files pertaining to the current platform and
#	 target (release/debug). This also removes generated libraries.
#  distclean - removes all build directories. 
#  platform  - copies a template to OS_CPU_COMPILER.mk in the makefiles
#			directory
#
########################################################################

override SOURCEROOT		= $(shell pwd)

include makefiles/definitions.mk

LIBSRC 				= ./Libsrc
LIBRARIES 		 =$(dir $(shell find ${LIBSRC} -type f -name 'Makefile'))
APPSRC 				= ./Appsrc
APPLICATIONS 	= $(dir $(shell find ${APPSRC} -type f -name 'Makefile'))

.PHONY: cvs app lib clean distclean platform makefiles

all: lib app

cvs: 
	$(foreach lib, ${LIBRARIES},cd ${lib}; cvs add *.cpp *.h Makefile; cd ${SOURCEROOT};)
	$(foreach app, ${APPLICATIONS},cd ${app}; cvs add *.cpp *.h Makefile; cd ${SOURCEROOT};)
	cvs commit

lib: 
	$(foreach lib, ${LIBRARIES},	${MAKE} -C ${lib};)

app:
	$(foreach app, ${APPLICATIONS},	${MAKE} -C ${app};)

clean:
	$(foreach lib, ${LIBRARIES},	${MAKE} -C ${lib} clean;)
	$(foreach app, ${APPLICATIONS},	${MAKE} -C ${app} clean;)

distclean: clean
	find ./lib -type d \! \( -name 'CVS' -o -name 'lib' \) -print > .remove
	find ./bin -type f -print >> .remove
	find . -name '.build' -type d -print >> .remove
	find . -type f -name '*.o' -o -name '*.d' -o -name 'core' -o -name '*~' -print >> .remove
	${RM} -fr `cat .remove`
	${RM} -f  .remove

platform: makefiles/${PLATFORM}.mk lib/${PLATFORM}

makefiles/${PLATFORM}.mk:
	cp makefiles/PlatformTemplate.mk makefiles/${PLATFORM}.mk

lib/${PLATFORM}:
	mkdir lib/${PLATFORM}
