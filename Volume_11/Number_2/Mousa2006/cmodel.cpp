/***************************************************************************
                          cmodel.cpp  -  description
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

#include "cmodel.h"
#include "misc.h"

CModel::~CModel(){
  p_no = 0;
  f_no = 0;
  delete [] plist;
  delete [] flist;
}

CModel::CModel(){
  p_no = 0;
  f_no = 0;
  plist = NULL;
  flist = NULL;
}

CModel::CModel(const CModel & m){
  printf("need a copy constructor\n");
}

int CModel::Facets_no(){
	return f_no;
}
int CModel::Vertices_no(){
	return p_no;
}

void CModel::LoadModel(char * fname){
  int i;
  double x,y,z;
  char buffer[30];
  FILE *fp1 = fopen(fname,"r");

  gettoken(buffer, fp1);
  gettoken(buffer, fp1);  p_no = atoi(buffer);
  gettoken(buffer, fp1);  f_no = atoi(buffer);
  gettoken(buffer, fp1);
  if(plist!=NULL) delete [] plist;
  if(flist!=NULL) delete [] flist;
  plist = new Point_3[p_no];
  flist = new int[3*f_no];

  for(i = 0; i < p_no; i++){
    gettoken(buffer, fp1); x = atof(buffer);
    gettoken(buffer, fp1); y = atof(buffer);
    gettoken(buffer, fp1); z = atof(buffer);
    plist[i] = Point_3(x,y,z);
  }
  for(i = 0; i < f_no; i++){
    gettoken(buffer, fp1); // read the number 3
    gettoken(buffer, fp1); flist[3*i+0] = atoi(buffer);
    gettoken(buffer, fp1); flist[3*i+1] = atoi(buffer);
    gettoken(buffer, fp1); flist[3*i+2] = atoi(buffer);
  }
}

void CModel::SaveModel(char *fname){
  FILE *fp1 = fopen(fname,"w");
  fprintf(fp1, "OFF \n%d %d 0 \n", p_no, f_no);
  for(int i=0; i<p_no; i++)
    fprintf(fp1, "%f %f %f\n", plist[i].x(),plist[i].y(),plist[i].z());
  for(int i=0; i<f_no; i++)
    fprintf(fp1, "3 %d %d %d\n", flist[3*i+0],flist[3*i+1],flist[3*i+2]);
  fclose(fp1);
}

void CModel::FacetHarmonics(int fh, int bw, double r, gsl_complex *coeff){
  int i, j, sign;
  gsl_complex err;
  
  Triangle_3 tr;
  tr = Triangle_3(plist[flist[fh*3+0]],plist[flist[fh*3+1]],plist[flist[fh*3+2]]);
  Tetrahedron_3 tetr;
  tetr = Tetrahedron_3(Point_3(0.0, 0.0, 0.0),plist[flist[fh*3+0]],plist[flist[fh*3+1]],plist[flist[fh*3+2]]);
  if(tetr.is_degenerate()) return;
  sign = (tetr.volume()<0 ? -1 : 1);
  
  for(i=0; i<bw; i++){
    for(j=0; j<=i; j++){
      Integrate(i, j, r, tetr, &coeff[i*bw+j], &err);
      coeff[i*bw+j] = gsl_complex_mul_real(coeff[i*bw+j], (double)sign);
    }
  }
}

void CModel::AllFacetHarmonics(int bw, double r, gsl_complex *coeff){
  for(int i=0; i<bw*bw; i++) GSL_SET_COMPLEX(&coeff[i], 0.0, 0.0);
  
  gsl_complex *temp = new gsl_complex[bw*bw];
  
  for(int i=0; i<f_no; i++){
    FacetHarmonics(i, bw, r, temp);
    for(int j=0; j<bw*bw; j++) coeff[j] = gsl_complex_add(coeff[j],temp[j]);
  }
	
  delete [] temp;
}

void CModel::Normalize(){
  double xmax, xmin, ymax, ymin, zmax, zmin;
  double x, y, z;
  double xcentre = 0.0, ycentre = 0.0, zcentre = 0.0;
  double scale = 0.0;

  xmax = plist[0].x();  xmin = plist[0].x();
  ymax = plist[0].y();  ymin = plist[0].y();
  zmax = plist[0].z();  zmin = plist[0].z();

  for(int i =0; i < p_no; ++i){
    xcentre += plist[i].x(); ycentre += plist[i].y(); zcentre += plist[i].z();
    if(xmax < plist[i].x()) xmax = plist[i].x();     if(xmin > plist[i].x()) xmin = plist[i].x();
    if(ymax < plist[i].y()) ymax = plist[i].y();     if(ymin > plist[i].y()) ymin = plist[i].y();
    if(zmax < plist[i].z()) zmax = plist[i].z();     if(zmin > plist[i].z()) zmin = plist[i].z();
  }

  xcentre /= p_no;  ycentre /= p_no;  zcentre /= p_no;

  if(scale<fabs(xmax-xmin)) scale = fabs(xmax-xmin);
  if(scale<fabs(ymax-ymin)) scale = fabs(ymax-ymin);
  if(scale<fabs(zmax-zmin)) scale = fabs(zmax-zmin);
  scale = 2.0/scale;
  
  for(int i =0; i < p_no; ++i){
    plist[i] = Point_3(plist[i].x()-xcentre, plist[i].y()-ycentre, plist[i].z()-zcentre);
    x = plist[i].x()*scale ;
    y = plist[i].y()*scale ;
    z = plist[i].z()*scale ;
    plist[i] = Point_3(x, y, z);
  }
}
