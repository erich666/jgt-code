/***************************************************************************
                          cmodel.h  -  description
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

#ifndef CMODELFILE
#define CMODELFILE

#include "config.h"
#include "integration.h"


class CModel {
  public:
    /**
    *Create an empty model: 0 vertices and 0 facets
    */
    CModel();
    /**Destroy the model*/
    ~CModel();
    /**
    *A copy constructor
    */
    CModel(const CModel & m);
    /**
    *Load the specified off file 
    */
    void LoadModel(char * fname);
    /**
    *Save the current model in off formate
    */
    void SaveModel(char *fname);
    /**
    *Normalize the current model
    */
    void Normalize();
    /**
    *Calculate the spherical harmonic coefficiets for a certain face using \f$ C^r_{l,m}=\int\int_T Y_l^m(\theta,\varphi)\sin(\theta)d\theta d\varphi \f$
    */
    void FacetHarmonics(int fh, int bw, double r, gsl_complex *coeff);
    /**
    *Calculate the spherical harmonic coefficiets for all faces
    */
    void AllFacetHarmonics(int bw, double r, gsl_complex *coeff);
    int Facets_no();
    int Vertices_no();
  private:
    /**
    *The number of points
    */ 
    int p_no;
    /**
    *The number of facets
    */
    int f_no;
    /**
    *The list of points
    */
    Point_3 *plist;
    /**
    *The list of facets
    */
    int *flist;
};



#endif
