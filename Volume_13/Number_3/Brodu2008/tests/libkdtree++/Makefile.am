ACLOCAL_AMFLAGS = -I m4

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = pkgconfig/libkdtree++.pc

pkgincludedir = $(includedir)/kdtree++
pkginclude_HEADERS = \
	include/accessor.hpp \
	include/allocator.hpp \
	include/iterator.hpp \
	include/kdtree.hpp \
	include/node.hpp \
	include/region.hpp

dist_noinst_DATA = \
	debian/changelog \
	debian/compat \
	debian/control \
	debian/copyright \
	debian/docs \
	debian/libkdtree++-dev.dirs \
	debian/libkdtree++-dev.install \
	debian/rules \
	examples/Makefile \
	examples/test_kdtree.cpp \
  doc/index.txt \
	doc/doxygen.rc

.PHONY: clean-local
clean-local:
	rm -f $(wildcard $(PACKAGE)-*.tar.gz)

include Makefile.debian

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
