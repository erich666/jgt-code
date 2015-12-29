/***************************************************************************
                          main.cpp  -  description
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

#include <CGAL/IO/Geomview_stream.h>


#include "cmodel.h"
#include "misc.h"

int main(int argc, char **argv){
  if(argc < 4){
    printf("usage: %s file.off bw n\n", argv[0]);
		printf("bw : the required bandwidth\n");
		printf("n : the number of intersecting sphere\n");
    exit(0);
  }
  int bw=atoi(argv[2]), n=atoi(argv[3]);
  double r[n], *data = new double[4*bw*bw];
  CModel m;
  m.LoadModel(argv[1]);
  m.Normalize();
  //m.SaveModel("normalized.off");
  printf("File loaded: %d vertices & %d facets\n", m.Vertices_no(), m.Facets_no());
  
   char fname[30];
   FileName(fname,argv[1]);
   
   

//part of numerical integration
  int indicator=1;
  gsl_complex *coeff = new gsl_complex[bw*bw];
  gsl_complex c, zz, *temp=new gsl_complex;

  for(int i=0; i<n; i++) r[i] = (i+1)/1.0/n;
  double x, y, z;
  for(int k=n-1; k>-1; k--){
     m.AllFacetHarmonics(bw, r[k], coeff);
		 /***************************************************************************/
	   // coeff is the required coefficients of the SHT of the intersecting sphere k
		 /***************************************************************************/
	 /*
		 the spherical harmonics coeff is of the form
		 coeff[0][0]      0      0 ... 0
		 coeff[1][0] coeff[1][1] 0 ...0
		 ...
		 coeff[bw-1][0]...coeff[bw-1][bw-1]
	 */
		 /*
		 coeff[l][-m] = pow(-1,m)*coeff[l][m]
		 */ 	 
		 
		 
		 printf("sphere %d is done.\n", k);
		 printf("coef(0,0)  = %f + i%fi\n", coeff[0].dat[0], coeff[0].dat[1]);
		 printf("coef(1,-1) = %f + i%fi\n", pow(-1, -1)* coeff[2].dat[0], pow(-1, -1)* coeff[2].dat[1]);
		 printf("coef(1,0)  = %f + i%fi\n", coeff[1].dat[0], coeff[1].dat[1]);
		 printf("coef(1,1)  = %f + i%fi\n", coeff[2].dat[0], coeff[2].dat[1]);
  }
  delete [] data;
  delete [] coeff;
  
  return 0;
}
