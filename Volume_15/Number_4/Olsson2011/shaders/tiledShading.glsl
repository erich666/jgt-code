#ifndef _TILED_SHADING_GLSL_
#define _TILED_SHADING_GLSL_
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
uniform LightGrid
{
  ivec4 lightGridCountOffsets[LIGHT_GRID_MAX_DIM_X * LIGHT_GRID_MAX_DIM_Y];
};

uniform isamplerBuffer tileLightIndexListsTex;

uniform LightPositionsRanges
{
  vec4 g_light_position_range[NUM_POSSIBLE_LIGHTS];
};
uniform LightColors
{
  vec4 g_light_color[NUM_POSSIBLE_LIGHTS];
};


vec3 doLight(vec3 position, vec3 normal, vec3 diffuse, vec3 specular, float shininess, vec3 viewDir, vec3 lightPos, vec3 lightColor, float range)
{
  vec3 lightDir = vec3(lightPos) - position;
  float dist = length(lightDir);
  lightDir = normalize(lightDir);
  float inner = 0.0;

  float ndotL = max(dot(normal, lightDir),0.0);
  float att = max(1.0 - max(0.0, (dist - inner) / (range - inner)), 0.0);

	vec3 fresnelSpec = specular + (vec3(1.0) - specular) * pow(clamp(1.0 + dot(-viewDir, normal), 0.0, 1.0), 5.0);
	vec3 h = normalize(lightDir + viewDir);

	float normalizationFactor = ((shininess + 2.0) / 8.0);

	vec3 spec = fresnelSpec * pow(max(0, dot(h, normal)), shininess) * normalizationFactor;

  return att * ndotL * lightColor * (diffuse + spec);
}


vec3 doLight(vec3 position, vec3 normal, vec3 diffuse, vec3 specular, float shininess, vec3 viewDir, int lightIndex)
{
  vec3 lightPos = g_light_position_range[lightIndex].xyz;
  float lightRange = g_light_position_range[lightIndex].w;
  vec3 lightColor = g_light_color[lightIndex].xyz;

  return doLight(position, normal, diffuse, specular, shininess, viewDir, lightPos, lightColor, lightRange);
}


// computes tiled lighting for the current fragment, using the built in gl_FragCoord to determine the correct tile.
vec3 evalTiledLighting(in vec3 diffuse, in vec3 specular, in float shininess, in vec3 position, in vec3 normal, in vec3 viewDir)
{
  ivec2 l = ivec2(int(gl_FragCoord.x) / LIGHT_GRID_TILE_DIM_X, int(gl_FragCoord.y) / LIGHT_GRID_TILE_DIM_Y);
  int lightCount = lightGridCountOffsets[l.x + l.y * LIGHT_GRID_MAX_DIM_X].x;
  int lightOffset = lightGridCountOffsets[l.x + l.y * LIGHT_GRID_MAX_DIM_X].y;

  vec3 color = vec3(0.0, 0.0, 0.0);

  for (int i = 0; i < lightCount; ++i)
  {
    int lightIndex = texelFetch(tileLightIndexListsTex, lightOffset + i).x; 
    color += doLight(position, normal, diffuse, specular, shininess, viewDir, lightIndex);
  }
  
  return color;
}


#endif // _TILED_SHADING_GLSL_
