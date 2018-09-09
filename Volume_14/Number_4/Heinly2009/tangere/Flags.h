
/*
 *  Copyright 2009, 2010 Grove City College
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef tangere_Flags_h
#define tangere_Flags_h

#define MAX_DEPTH 10
// #define RENDER_ALL

#define USE_32_BIT_SHADERS
// #define USE_NDOTV_SHADER

// #define USE_BENCHMARK_MODE
#define USE_BVH
#define USE_D_BITS
#define USE_INVDIR
//#define USE_GEOMETRY_LIGHTS
#ifndef USE_GEOMETRY_LIGHTS
#  define OFFSET_EYE_LIGHT
#endif
// #define USE_HARDCODED_PATH

#endif // tangere_Flags_h
