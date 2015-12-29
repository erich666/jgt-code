/***************************************************************************
                          misc.cpp  -  description
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

#include "misc.h"
#include "config.h"
#include <CGAL/intersections.h>

#define min(A, B)	(A<B ? A : B)
#define max(A, B)	(A>B ? A : B)


int gettoken(char buffer[30], FILE *f){
  int i=0;
  char ch;
  buffer[i]=0;
  while(fread(&ch,sizeof(char),sizeof(char),f)!=0){
    if((ch!=10)&&(ch!=13)&&(ch!=32)&&(ch!='\t')&&(ch!='&')&&(ch!='(')&&(ch!=')'))
      buffer[i++]=ch;
    else if(i!=0){
      buffer[i]=0;
      return i; //return the length of the token
    }
  }
  if(feof(f)&&(i!=0)){
    buffer[i]=0;
    return i;
  }
  return 0; //error in reading or end of file
}

bool FileName(char fname[30], char *str){
  int i=0;
  if(str==NULL)  return false;
  while((str[i]!='.')&&(str[i]!=0)){
    fname[i] = str[i];
    i++;
  }
  fname[i]=0;
  if(str[i]==0) return false;
  return true;
}

void theta_phi_bounds(Triangle_3 tr, double *mintheta, double *maxtheta, double *minphi, double *maxphi){
  double th0, th1, th2, phi0, phi1, phi2;
  double r0, r1, r2, s0, s1, s2;
  
  r0 = sqrt(pow(tr[0].x(),2) + pow(tr[0].y(),2) + pow(tr[0].z(),2)); 
  r1 = sqrt(pow(tr[1].x(),2) + pow(tr[1].y(),2) + pow(tr[1].z(),2)); 
  r2 = sqrt(pow(tr[2].x(),2) + pow(tr[2].y(),2) + pow(tr[2].z(),2)); 
  
  s0 = sqrt(pow(tr[0].x(),2) + pow(tr[0].y(),2));
  s1 = sqrt(pow(tr[1].x(),2) + pow(tr[1].y(),2));
  s2 = sqrt(pow(tr[2].x(),2) + pow(tr[2].y(),2));
  
  th0 = atan2(s0, tr[0].z()); 
  th1 = atan2(s1, tr[1].z()); 
  th2 = atan2(s2, tr[2].z());
  
  phi0 = atan2(tr[0].y(), tr[0].x()); 
  if(tr[0].y() < 0.0) phi0 += 2.0*M_PI; 
  if(fabs(phi0-2.0*M_PI)<1e-6) phi0 = 0.0;
  
  phi1 = atan2(tr[1].y(), tr[1].x()); 
  if(tr[1].y() < 0.0) phi1 += 2.0*M_PI; 
  if(fabs(phi1-2.0*M_PI)<1e-6) phi1 = 0.0;
  
  phi2 = atan2(tr[2].y(), tr[2].x()); 
  if(tr[2].y() < 0.0) phi2 += 2.0*M_PI; 
  if(fabs(phi2-2.0*M_PI)<1e-6) phi2 = 0.0;
  
  *mintheta = min(th0, th1);  *mintheta= min(*mintheta, th2);
  *maxtheta = max(th0, th1);  *maxtheta= max(*maxtheta, th2);
  *minphi = min(phi0, phi1); *minphi = min(*minphi, phi2);
  *maxphi = max(phi0, phi1); *maxphi = max(*maxphi, phi2);
  
  Plane_3 h(Point_3(1.0,0.0,0.0),Point_3(0.0,0.0,1.0),Point_3(-1.0,0.0,0.0));
 
  if(!((h.oriented_side(tr[0])==h.oriented_side(tr[1]))&&(h.oriented_side(tr[1])==h.oriented_side(tr[2])))){
    //the triangle pass through the great circle phi = 0
    //if maxphi < M_PI/2 we must not reverse the phi bounds
    if(*maxphi>M_PI){
      double temp;
      temp = *minphi;
      *minphi = *maxphi;
      *maxphi = temp;
    }
  }  
}


void SphericalHarmonic(int l, int m, double theta, double phi, gsl_complex *res){
  double plm;
  plm = gsl_sf_legendre_sphPlm (l, abs(m), cos(theta));
  if(m==0){
    GSL_SET_COMPLEX(res, plm, 0.0); 
  }
  if(m>0){
    GSL_SET_COMPLEX(res, plm*cos(m*phi), plm*sin(m*phi)); 
  }
  if(m<0){
    GSL_SET_COMPLEX(res, pow(-1,m)*plm*cos(m*phi), pow(-1,m)*plm*sin(m*phi));
  }
}

