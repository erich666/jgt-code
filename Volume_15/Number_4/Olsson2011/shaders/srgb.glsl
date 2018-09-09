#ifndef _SRGB_GLSL_
#define _SRGB_GLSL_

vec3 toSrgb(vec3 color)
{
  return pow(color, vec3(1.0 / 2.2));
}

#endif // _SRGB_GLSL_
