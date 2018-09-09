#version 330 compatibility
/****************************************************************************/
/* Copyright (c) 2011, Ola Olsson, Ulf Assarsson
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
/****************************************************************************/
#include "globals.glsl"
#include "tiledShading.glsl"
#include "srgb.glsl"

#if NUM_MSAA_SAMPLES == 0
uniform sampler2D diffuseTex;
uniform sampler2D specularShininessTex;
uniform sampler2D ambientTex;
uniform sampler2D normalTex;
uniform sampler2D depthTex;
#else // NUM_MSAA_SAMPLES != 0
uniform sampler2DMS diffuseTex;
uniform sampler2DMS specularShininessTex;
uniform sampler2DMS ambientTex;
uniform sampler2DMS normalTex;
uniform sampler2DMS depthTex;
#endif // NUM_MSAA_SAMPLES == 0


out vec4 resultColor;

vec3 unProject(vec2 fragmentPos, float fragmentDepth)
{
  vec4 pt = inverseProjectionMatrix * vec4(fragmentPos.x * 2.0 - 1.0, fragmentPos.y * 2.0 - 1.0, 2.0 * fragmentDepth - 1.0, 1.0);
  return vec3(pt.x, pt.y, pt.z) / pt.w;
}

#if NUM_MSAA_SAMPLES == 0

vec3 fetchPosition(vec2 p)
{
  vec2 fragmentPos = vec2(p.x * invFbSize.x, p.y * invFbSize.y);
  float d = texelFetch(depthTex, ivec2(p), 0).x;
  return unProject(fragmentPos, d);
}

#else // NUM_MSAA_SAMPLES != 0

vec3 fetchPosition(vec2 p, int sample)
{
  vec2 fragmentPos = vec2(p.x * invFbSize.x, p.y * invFbSize.y);
  float d = texelFetch(depthTex, ivec2(p), sample).x;
  return unProject(fragmentPos, d);
}

#endif // NUM_MSAA_SAMPLES == 0


void main()
{
#if NUM_MSAA_SAMPLES == 0
  vec3 diffuse = texelFetch(diffuseTex, ivec2(gl_FragCoord.xy), 0).xyz;
  vec4 specularShininess = texelFetch(specularShininessTex, ivec2(gl_FragCoord.xy), 0); 
  vec3 position = fetchPosition(gl_FragCoord.xy); 
  vec3 normal = texelFetch(normalTex, ivec2(gl_FragCoord.xy), 0).xyz; 
  vec3 viewDir = -normalize(position);

  vec3 lighting = evalTiledLighting(diffuse, specularShininess.xyz, specularShininess.w, position, normal, viewDir);
  resultColor = vec4(toSrgb(lighting + texelFetch(ambientTex, ivec2(gl_FragCoord.xy), 0).xyz), 1.0);

	//resultColor = vec4(mod(vec3(-position.z / 200.0, -position.z / 50.0, -position.z / 7.0), 1.0), 1.0);
#else // NUM_MSAA_SAMPLES != 0
  vec3 color = vec3(0.0, 0.0, 0.0);
  
#if 0
  // This, perfectly reasonable, version does not work on nvidia hw/drivers, triggers a linker error.
  for (int sampleIndex = 0; sampleIndex < NUM_MSAA_SAMPLES; ++sampleIndex)
  {
    vec3 diffuse = texelFetch(diffuseTex, ivec2(gl_FragCoord.xy), sampleIndex).xyz;
    vec4 specularShininess = texelFetch(specularShininessTex, ivec2(gl_FragCoord.xy), sampleIndex); 
    vec3 position = fetchPosition(gl_FragCoord.xy, sampleIndex); 
    vec3 normal = texelFetch(normalTex, ivec2(gl_FragCoord.xy), sampleIndex).xyz; 
    vec3 viewDir = -normalize(position);
    color += evalTiledLighting(diffuse, specularShininess.xyz, specularShininess.w, position, normal, viewDir)
          + texelFetch(ambientTex, ivec2(gl_FragCoord.xy), sampleIndex).xyz;
  }
#else
  // So as a workaround, we manually unroll the loop for up to 16 samples.
#define LOOP_BODY(sampleIndex)\
  if (sampleIndex < NUM_MSAA_SAMPLES) \
  { \
    vec3 diffuse = texelFetch(diffuseTex, ivec2(gl_FragCoord.xy), sampleIndex).xyz; \
    vec4 specularShininess = texelFetch(specularShininessTex, ivec2(gl_FragCoord.xy), sampleIndex); \
    vec3 position = fetchPosition(gl_FragCoord.xy, sampleIndex); \
    vec3 normal = texelFetch(normalTex, ivec2(gl_FragCoord.xy), sampleIndex).xyz; \
    vec3 viewDir = -normalize(position);  \
    color += evalTiledLighting(diffuse, specularShininess.xyz, specularShininess.w, position, normal, viewDir)  \
          + texelFetch(ambientTex, ivec2(gl_FragCoord.xy), sampleIndex).xyz;  \
  }

  LOOP_BODY(0);
  LOOP_BODY(1);
  LOOP_BODY(2);
  LOOP_BODY(3);
  LOOP_BODY(4);
  LOOP_BODY(5);
  LOOP_BODY(6);
  LOOP_BODY(7);
  LOOP_BODY(8);
  LOOP_BODY(9);
  LOOP_BODY(10);
  LOOP_BODY(11);
  LOOP_BODY(12);
  LOOP_BODY(13);
  LOOP_BODY(14);
  LOOP_BODY(15);

#undef LOOP_BODY
#endif //
  resultColor = vec4(toSrgb(color / float(NUM_MSAA_SAMPLES)), 1.0);
#endif //NUM_MSAA_SAMPLES == 0
}
