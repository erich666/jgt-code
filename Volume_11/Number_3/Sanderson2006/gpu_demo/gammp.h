#ifndef GAMMP_H
#define GAMMP_H

#ifdef __cplusplus
extern "C" {
#endif

float gammln(float xx);
void gser(float *gamser, float a, float x, float *gln);
void gcf(float *gammcf, float a, float x, float *gln);
float gammp(float a, float x);
void nrerror(char error_text[]);


#ifdef __cplusplus
}
#endif

#endif
