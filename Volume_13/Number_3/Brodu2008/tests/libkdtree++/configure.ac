#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_REVISION($Revision: 1.8 $)
AC_COPYRIGHT([Please see the ./COPYING file for license information.])

AC_PREREQ(2.57)

AH_TOP(
#undef LIBKDTREEPP_VERSION
)

##### VERSIONING AND VARIABLES ###############################################

# Making releases:
#     if functions were added, IFACE_AGE = 0
#     if not backward compatible, IFACE_AGE = BIN_AGE = 0
#
m4_define([IFACE_AGE], [0])
m4_define([BIN_AGE],   [0])
m4_define([MD_INIT_NAME], m4_esyscmd([set -- `head -1 README` && echo -n $1]))
m4_define([MD_INIT_VERSION], m4_esyscmd([head -1 debian/changelog | sed -e 's,.*(\([^-]*\)-[0-9]*).*,\1,' | tr -d "\n"]).IFACE_AGE.BIN_AGE)
m4_define([MD_INIT_ADDRESS], m4_esyscmd([grep -i bugreports\ to README | sed -e 's,.*<\([^>]\+\).*,\1,' | tr -d "\n"]))

AC_INIT(MD_INIT_NAME, MD_INIT_VERSION, MD_INIT_ADDRESS)

AC_CONFIG_HEADER([config.h])
AC_CONFIG_SRCDIR([include/kdtree.hpp])
AC_CONFIG_AUX_DIR([.])

AM_INIT_AUTOMAKE
AM_MAINTAINER_MODE

##### VERSIONING #############################################################

MD_VERSION_INFO
MD_LT_VERSION_INFO

PACKAGE=AC_PACKAGE_NAME
VERSION=$MD_VERSION

##### OUTPUT #################################################################

AC_CONFIG_FILES([Makefile])
AC_CONFIG_FILES([pkgconfig/libkdtree++.pc])

AC_OUTPUT

# COPYRIGHT --
#
# This file is part of libkdtree++, a C++ template KD-Tree sorting container.
# libkdtree++ is (c) 2004 Martin F. Krafft <krafft@ailab.ch>
# and distributed under the terms of the Artistic Licence.
# See the ./COPYING file in the source tree root for more information.
#
# THIS PACKAGE IS PROVIDED "AS IS" AND WITHOUT ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTIES
# OF MERCHANTIBILITY AND FITNESS FOR A PARTICULAR PURPOSE.
