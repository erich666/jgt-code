
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


###############################################################################
# tangere executable

SET(EXEC_NAME tangere)

SET(TANGERE_SOURCES
  BBox.t
  BVH.t
  Camera.t
  Constants.h
  Context.t
  DielectricShader.t
  Flags.h
  HitRecord.t
  LambertianShader.t
  Light.t
  Material.t
  NdotVShader.t
  Node.t
  Options.h
  Ray.t
  ReflectionShader.t
  Renderer.t
  Scene.t
  SceneScaler.h
  SceneScaler.cc
  Shader.t
  ShaderTable.h
  ShaderTable.cc
  Triangle.t
  Triangle.cc
  TriangleProcessor.t
  )

INCLUDE_DIRECTORIES(
  )

ADD_EXECUTABLE(${EXEC_NAME} 
  ${TANGERE_SOURCES}
  tangere.cc
  )

TARGET_LINK_LIBRARIES(${EXEC_NAME}
  vissim_Common
  )
