/***************************************************************************
                          integration.h  -  description
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

#ifndef INTEGRATIONFILE
#define INTEGRATIONFILE

#include "config.h"
#include "misc.h"

#include <gsl/gsl_math.h>
#include <gsl/gsl_monte.h>
#include <gsl/gsl_monte_plain.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>

void Integrate(int , int , double , Tetrahedron_3 , gsl_complex *, gsl_complex *); 

#endif
