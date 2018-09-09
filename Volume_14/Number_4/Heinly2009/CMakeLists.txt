
################################################################################
#  Copyright 2009, 2010 Grove City College
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#


################################################################################
# tangere distributable

PROJECT(tangere-dist)

CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

SET(BUILD_SHARED_LIBS ON)
SET(CMAKE_VERBOSE_MAKEFILE ON)

SET(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib CACHE INTERNAL "Single output directory for building all libraries")
SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin CACHE INTERNAL "Single output directory for building all executables")

INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}
  )

SUBDIRS(
  Common
  tangere
  )

SET(PASSED_FIRST_CONFIGURE ON CACHE INTERNAL "Already configured once?")
