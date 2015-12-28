SHELL=/bin/sh

### Include the user configuration makefile
include ${SOURCEROOT}/makefiles/config.mk

ifeq ($(strip ${CXX}),)
CXX = $(shell ${SOURCEROOT}/makefiles/findcompiler.sh)
endif

OS =$(subst ${empty} ${empty},_,$(shell uname -s))
CPU =$(subst ${empty} ${empty},_,$(shell uname -m))

empty =
### The platform is determined by OS CPU and compiler 
PLATFORM = ${OS}_${CPU}_${CXX}


### Default target is release, unless debug is given as goal
ifndef TARGET
TARGET = release
endif

### Concatenation of platform and target yields a string used as 
### suffix of makefiles and in the name of the builddir.
PLATFORM_TARG = ${PLATFORM}_$(TARGET)
