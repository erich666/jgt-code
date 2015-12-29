/***************************************************************************
                          misc.h  -  description
                             -------------------
    copyright            : (C) 2005 by MOUSSA
    email                : mmousa@liris.cnrs.fr
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef MISC_H
#define MISC_H

#include "config.h"

#include <gsl/gsl_math.h>
#include <gsl/gsl_monte.h>
#include <gsl/gsl_monte_plain.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>


//////////////////////////////////////////////////////////////////////////////
template<class T> void Add_vectors(T *vec1, T *vec2, T *result, int size);
template<class T> void Add_vectors(T *vec1, T *vec2, T *result, int size){
  for(int i=0; i<size; i++){
    result[i] = vec1[i]+vec2[i];
  }
}
//////////////////////////////////////////////////////////////////////////////
template<class T> void Mult_vector_const(T *vec, T c, T *result, int size);
template<class T> void Mult_vector_const(T *vec, T c, T *result, int size){
  for(int i=0; i<size; i++){
    result[i] = vec[i]*c;
  }
}
//////////////////////////////////////////////////////////////////////////////
template<class T>void assign_value(T *vec, T c, int size);
template<class T>void assign_value(T *vec, T c, int size){
  for(int i = 0; i<size; i++) vec[i] = c;
}
//////////////////////////////////////////////////////////////////////////////
template<class T>void copy_vectors(T *src, T *dest, int size);
template<class T>void copy_vectors(T *src, T *dest, int size){
  for(int i = 0; i<size; i++){
    dest[i] = src[i];
  }
}
//////////////////////////////////////////////////////////////////////////////
template<class T> int is_overlaping(T *vec1, T *vec2, int size);
template<class T> int is_overlaping(T *vec1, T *vec2, int size){
  for(int i = 0; i<size; i++){
    if((fabs(vec1[i])>1e-6)&&(fabs(vec2[i])>1e-6)) return true;
  }
  return false;
}
//////////////////////////////////////////////////////////////////////////////
template<class T> void print_vector(T *vec, int size);
template<class T> void print_vector(T *vec, int size){
  for(int i = 0; i<size; i++){
    if(vec[i]>0.0) std::cout<<vec[i]<<std::endl;
  }
}
//////////////////////////////////////////////////////////////////////////////
template<class T> bool AllOnes(T *vec, int size);
template<class T> bool AllOnes(T *vec, int size){
  for(int i = 0; i < size; i++)
    if(fabs(vec[i]) < 1e-6) return false;
  return true;
}
//////////////////////////////////////////////////////////////////////////////
int gettoken(char[30], FILE *);
bool FileName(char[30], char *);
void theta_phi_bounds(Triangle_3 tr, double *mintheta, double *maxtheta, double *minphi, double *maxphi);
void SphericalHarmonic(int, int, double, double, gsl_complex *);
#endif
