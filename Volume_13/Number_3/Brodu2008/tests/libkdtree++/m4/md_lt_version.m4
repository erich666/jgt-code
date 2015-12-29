AC_DEFUN([MD_LT_VERSION_INFO],
  [
    LT_RELEASE=$MD_MAJOR_VERSION.$MD_MINOR_VERSION
    LT_CURRENT=$(($MD_MICRO_VERSION_NUM - $MD_INTERFACE_AGE))
    LT_REVISION=$MD_INTERFACE_AGE
    LT_AGE=$(($MD_BINARY_AGE - $MD_INTERFACE_AGE))

    AC_SUBST(LT_RELEASE)
    AC_SUBST(LT_CURRENT)
    AC_SUBST(LT_REVISION)
    AC_SUBST(LT_AGE)
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
