/***************************************************************************
                          integration.cpp  -  description
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

#include "integration.h"

typedef struct{
  Tetrahedron_3 H;
  double radius;
  int l;
  int m;
} Parameters;

/**
*Evaluation of the real part \f$Y_l^m(\theta, \varphi)sin(\theta)cos(m\varphi)\f$
*/
double gr (double *k, size_t dim, void *params)
{
  Parameters param = *((Parameters *) params);
  //check if the point is inside the tetrahedron or not
  Point_3 p = Point_3(param.radius*cos(k[0])*sin(k[1]),param.radius*sin(k[0])*sin(k[1]), param.radius*cos(k[1]));
  if(param.H.has_on_bounded_side(p)){
    double A;
    if(param.m==0)
      A = gsl_sf_legendre_sphPlm (param.l, param.m, cos(k[1]))*sin(k[1]);
    else
      A = gsl_sf_legendre_sphPlm (param.l, param.m, cos(k[1]))*cos(param.m*k[0])*sin(k[1]);
    return A;
  }else
    return 0.0;
}
/**
*Evaluation of the imaginary part \f$Y_l^m(\theta, \varphi)sin(\theta)sin(m\varphi)\f$
*/
double gi (double *k, size_t dim, void *params)
{
  Parameters param = *((Parameters *) params);
  //check if the point is inside the tetrahedron or not
  Point_3 p = Point_3(param.radius*cos(k[0])*sin(k[1]),param.radius*sin(k[0])*sin(k[1]), param.radius*cos(k[1]));
  if(param.H.has_on_bounded_side(p)){
    double A;
    if(param.m==0)
      A = 0.0;
    else
      A = gsl_sf_legendre_sphPlm (param.l, param.m, cos(k[1]))*sin(-param.m*k[0])*sin(k[1]);
    return A;
  }else
    return 0.0;  
}
/**
*Integrate the \f$\int\int_{T}Y_l^m(\theta, \varphi)e^{-im\varphi}\f$
*/

void Integrate(int l, int m, double radius, Tetrahedron_3  H, gsl_complex *result, gsl_complex *error)
{
  double rres, rerr, ires, ierr;
  
  double mintheta, maxtheta, minphi, maxphi;
  theta_phi_bounds(Triangle_3(H[1],H[2],H[3]), &mintheta, &maxtheta, &minphi, &maxphi);
  double xl[2] = { 0.0, 0.0 };
  double xu[2] = { 2.0*M_PI, M_PI};

  const gsl_rng_type *T;
  gsl_rng *r;
  Parameters param = {H, radius, l, m};
  gsl_monte_function Gr = { &gr, 2, &param };
  gsl_monte_function Gi = { &gi, 2, &param };

  size_t calls = 1000;

  gsl_rng_env_setup ();

  T = gsl_rng_default;
  r = gsl_rng_alloc (T);

  gsl_monte_plain_state *s = gsl_monte_plain_alloc (2);
  gsl_monte_plain_integrate (&Gr, xl, xu, 2, calls, r, s, &rres, &rerr);
  gsl_monte_plain_init (s);
  gsl_monte_plain_integrate (&Gi, xl, xu, 2, calls, r, s, &ires, &ierr);
  gsl_monte_plain_free (s);
  
  GSL_SET_COMPLEX(result, rres, ires);
  GSL_SET_COMPLEX(error, rerr, ierr);
  gsl_rng_free (r);
}
