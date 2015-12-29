AC_DEFUN([MD_VERSION_INFO],
  [
    AH_TOP(
#undef MD_VERSION
    )

    MD_MAJOR_VERSION=`echo "AC_PACKAGE_VERSION" | cut -d. -f1`
    MD_MINOR_VERSION=`echo "AC_PACKAGE_VERSION" | cut -d. -f2`
    MD_MICRO_VERSION=`echo "AC_PACKAGE_VERSION" | cut -d. -f3`
    MD_INTERFACE_AGE=`echo "AC_PACKAGE_VERSION" | cut -d. -f4`
    MD_BINARY_AGE=`echo "AC_PACKAGE_VERSION" | cut -d. -f5`
    MD_VERSION=$MD_MAJOR_VERSION.$MD_MINOR_VERSION.$MD_MICRO_VERSION

    AC_DEFINE_UNQUOTED(MD_VERSION, $MD_VERSION)

    AC_SUBST(MD_MAJOR_VERSION)
    AC_SUBST(MD_MINOR_VERSION)
    AC_SUBST(MD_MICRO_VERSION)
    AC_SUBST(MD_INTERFACE_AGE)
    AC_SUBST(MD_BINARY_AGE)
    AC_SUBST(MD_VERSION)

    MD_MICRO_VERSION_NUM=`echo $MD_MICRO_VERSION | sed 's/[[a-zA-Z]]//g'`
  ])

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
