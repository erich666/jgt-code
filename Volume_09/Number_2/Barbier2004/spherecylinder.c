// Cylinder
// Changelog 01.12.17

#include "shapes.h"

/*!
  \class SphereCylinder shapes.h
  \brief This class implements a simple data-structure to define 
  a simple rounded cylinder characterized by its end vertices and radius.
*/

/*!
  \brief Creates a generic rounded cylinder given vertices and radius.
  \param a, b End vertices of the axis.
  \param r Radius of the cylinder.
*/
SphereCylinder::SphereCylinder(const Vector& a,const Vector& b,const double& r):Axis(a,b)
{
  SphereCylinder::r[0]=r;
  SphereCylinder::r[1]=r*r;
  c=(a+b)*0.5;
  h=0.5*length;
}

/*!
  \brief Computes the distance between a point in space and the cylinder. 
  
  Cost: 9 * 11 + 1 sqrt() 1 fabs() 2 ?
  
  Timings on a 2.4MHz PIV report 3.670 for 100.000.000 queries.
*/ 
double SphereCylinder::R(const Vector& p) const
{
  Vector n=p-c;     		       // Cost: 3 +
  double y=axis*n;   			// Cost: 3 * 2 +
  y=fabs(y);     		       // Cost: 1 fabs() 
  double yy=y*y;     			// Cost: 1 *

  double xx=n*n-yy;  			// Cost: 3 * 3 +
  
  double e=0.0; 
  
  // Cylinder
  if (y<h)				// Cost: 1 ?
  {
    if (xx>r[1])    		       // Cost: 1 ?
    {
      double x=sqrt(xx);   		// Cost: 1 sqrt()
      x-=r[0];     		       // Cost: 1 +
      e=x*x;     		       // Cost: 1 *
    }
    //else
    //{
    //  e=0.0;
    //}
  }				       // Overall cost: 8 * 9 + 1 sqrt() 1 fabs() 2 ?
  // Either end of cylinder
  else
  {
    y-=h;			       // Cost: 1 +	
    yy=y*y;			       // Cost: 1 *
    e=xx+yy;			       // Cost: 1 +	
    // Outside sphere 
    if (e>r[1])    		       // Cost: 1 ?
    {
      e=sqrt(e);		       // Cost: 1 sqrt()
      e-=r[0];		      	       // Cost: 1 +
      e*=e;		               // Cost: 1 *
    }				       // Overall cost: 9 * 11 + 1 sqrt() 1 fabs() 2 ?
    //else
    //{
    //  e=0.0;
    //}	
  }

  return e;
}

/*!
  \brief Computes the pre-processing equations.
  \param o, d Ray origin and direction (which should be normalized).
*/
void SphereCylinder::Set(const Vector& o,const Vector& d)
{
  Vector pa=c-o;

  double dx=d*axis;
  double pax=pa*axis;
  double dpa=d*pa;

  quadric[2]=1.0-dx*dx;
  quadric[1]=2.0*(dx*pax-dpa);
  quadric[0]=pa*pa-pax*pax;
  linear[1]=dx;
  linear[0]=-pax;
}

/*!
  \brief Compute the distance between a points on a line and a cylinder-sphere.
  
  The member function SphereCylinder::Set() should be called for pre-processing.
  \param t Parameter of the point on the line.
*/
double SphereCylinder::R(const double& t) const
{
  double y=linear[1]*t+linear[0];
  double xx=(quadric[2]*t+quadric[1])*t+quadric[0];
  
  y=fabs(y);     		       // Cost: 1 fabs() 
  
  double e=0.0; 
  
  // Cylinder
  if (y<h)				// Cost: 1 ?
  {
    if (xx>r[1])    		       // Cost: 1 ?
    {
      double x=sqrt(xx);   		// Cost: 1 sqrt()
      x-=r[0];     		       // Cost: 1 +
      e=x*x;     		       // Cost: 1 *
    }
    //else
    //{
    //  e=0.0;
    //}
  }				       // Overall cost: 8 * 9 + 1 sqrt() 1 fabs() 2 ?
  // Either end of cylinder
  else
  {
    y-=h;			       // Cost: 1 +	
    double yy=y*y;			       // Cost: 1 *
    e=xx+yy;			       // Cost: 1 +	
    // Outside sphere 
    if (e>r[1])    		       // Cost: 1 ?
    {
      e=sqrt(e);		       // Cost: 1 sqrt()
      e-=r[0];		      	       // Cost: 1 +
      e*=e;		               // Cost: 1 *
    }				       // Overall cost: 9 * 11 + 1 sqrt() 1 fabs() 2 ?
    //else
    //{
    //  e=0.0;
    //}	
  }

  return e;
}

