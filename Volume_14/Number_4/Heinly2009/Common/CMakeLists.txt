
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


#################################################################################
# Common library

SET(LIBRARY_NAME vissim_Common)

SET(COMMON_SOURCES
  FileIO/BBox.h
  FileIO/BBox.cc
  FileIO/BVH.h
  FileIO/BVH.cc
  FileIO/GeometryProcessor.h
  FileIO/GeometryProcessor.cc
  FileIO/Image.t
  FileIO/Mesh.h
  FileIO/Mesh.cc
  FileIO/Options.h
  FileIO/Options.cc
  FileIO/Pixel.h
  FileIO/Pixel.cc
  FileIO/Scene.h
  FileIO/Scene.cc
  )

SET(COMMON_SOURCES
  ${COMMON_SOURCES}
  rtCore/Material.h
  rtCore/RGB.t
  rtCore/RGB.cc
  rtCore/TexCoord.h
  rtCore/Texture.h
  )

SET(COMMON_SOURCES
  ${COMMON_SOURCES}
  Utility/MinMax.t
  Utility/OutputCC.h
  Utility/Timer.h
  Utility/Timer.cc
  )

SET(COMMON_SOURCES
  ${COMMON_SOURCES}
  Constants.h
  Types.h
  )

INCLUDE_DIRECTORIES(
  )

ADD_LIBRARY(${LIBRARY_NAME}
  ${COMMON_SOURCES}
  )
