// Cone
// Changelog 03.08.17

#include "shapes.h"

/*!
  \class SphereCone shapes.h
  \brief This class implements a rounded cone primitive. 
*/

/*!
  \brief Create a rounded cone skeletal element. 
  
  This function assumes that the first radius ra is greater than rb.
   
  \param a, b Vertices of the cone-sphere.
  \param ra, rb Radii at the end vertices of the cone-sphere.
*/
SphereCone::SphereCone(const Vector& a,const Vector& b,const double& ra,const double& rb):Axis(a,b)
{
  SphereCone::ra=ra;
  SphereCone::rb=rb;
  SphereCone::rrb=rb*rb;
  SphereCone::rra=ra*ra;
    
  double rab=ra-rb;
  double s=sqrt(length*length-rab*rab);
  
  SphereCone::ha=ra*rab/length;
  SphereCone::hb=rb*rab/length;
  
  SphereCone::hra=ra*s/length;
  SphereCone::hrb=rb*s/length;
   
  SphereCone::side=Vector(-ha/ra,hra/ra,0.0);
    
  conelength=length*hra/ra;
}

/*!
  \brief Computes the distance between a point in space and the cone. 

  The overall computational cost in the worst case is 12 <B>+</B>, 
  11 <B>*</B>, 2 <B>sqrt()</B> and 5 <B>?</B>.

  Timings on a 2.4MHz PIV report 23.380 seconds for 100.000.000 queries 
  without sphere and cylinder optimizations. Turning both SPHERE and CYLINDER
  optimizations reduce computational time to 13.020 seconds. Turning SPHERE
  optimization only yields 13.110 seconds.
*/ 
#define SPHERE
#define CYLINDER

double SphereCone::R(const Vector& p) const
{
  Vector n=p-a;    			       // Cost: 3 +
  double y=n*axis; 				// Cost: 3 * 2 +
  double nn=n*n;   				// Cost: 3 * 2 +
  
  double e=0.0;
    
#ifdef SPHERE
  // Speed-up large sphere check (could be disabled)
  if (y<ha)        				// Cost: 1 ?
  {
    // Sphere check
    if (nn>rra)    				// Cost: 1 ?
    {
      e=sqrt(nn);  				// Cost: 1 sqrt()
      e-=ra;       				// Cost: 1 + 
      e*=e;        				// Cost: 1 *
    }              				// Overall: 2 ? 7 * 8 + 1 sqrt()
    //else
    //{
    //  e=0.0;
    //}
  }
  else
#endif
  {
    // Compute squared distance to axis
    double yy=y*y;     				// Cost: 1 *
    double xx=nn-yy;   				// Cost: 1 +
    
#ifdef CYLINDER
    // Speed-up cylinder check (could be disabled)
    if (xx<rrb)        				// Cost: 1 ?
    {
      // Check if on the left part of the cone-sphere
      if (y>length+hb)     				// Cost: 1 + 1 ?
      {
        y-=length;         				// Cost: 1 +
	yy=y*y;        				// Cost: 1 *
	nn=xx+yy;      				// Cost: 1 +
	// Sphere check
	if (nn>rrb)    				// Cost: 1 ?
	{
          e=sqrt(nn);  				// Cost: 1 sqrt()
          e-=rb;       				// Cost: 1 +
          e*=e;        				// Cost: 1 *
        }					// Overall: 4 ? 9 * 12 + 1 sqrt()
        //else
        //{
        //  e=0.0;
        //}
      }
      // Otherwise, inside cylinder part of the cone
      //else
      //{
      //  e=0.0;
      //}
    }
    else
    #endif

    // The last complex cases need a different coordinate system
    {
      double x=sqrt(xx); 			// Cost: 1 sqrt()

      // Rotate
      double ry=x*side[0]+y*side[1]; 		// Cost: 2 * 1+
      
      // Sphere check
      if (ry<0.0)        			// Cost: 1 ?
      {
        if (nn>rra)      			// Cost: 1 ?
        {
          e=sqrt(nn);    			// Cost: 1 sqrt()
          e-=ra;         			// Cost: 1 +
          e*=e;          			// Cost: 1 *
        }					// Overall: 4 ? 10 * 10 + 2 sqrt()
        //else
        //{
        //  e=0.0;
        //}
      }
      // Other sphere check
      else if (ry>conelength) 			// Cost: 1 ?
      {
        //cout<<"Small sphere"<<endl;
	// Compute the squared distance to the other end vertex
	y-=length;            			// Cost: 1 +
	yy=y*y;           			// Cost: 1 *
	nn=xx+yy;         			// Cost: 1 +

	// Sphere check
	if (nn>rrb)       			// Cost: 1 ?
	{
          e=sqrt(nn);     			// Cost: 1 sqrt()
          e-=rb;          			// Cost: 1 +
          e*=e;           			// Cost: 1 *
        }
        //else
        //{
        //  e=0.0;
        //}        					 // Cost: 5 ? 11 * 12 + 2 sqrt()
      }
      else
      {
        double rx=x*side[1]-y*side[0]; // Cost: 2 * 1 +
        if (rx>ra)   				// Cost: 1 ?
        {
	  rx-=ra;     				// Cost: 1 +
	  e=rx*rx;    				// Cost: 1 *
	}
	// Otherwise, inside cone
        //else
        //{
        //  e=0.0;
        //}                                             // Cost: 5 ? 12 * 11 + 1 sqrt()
      }
    }
  }  
  return e;
}

/*!
  \brief Computes the pre-processing equations.
  \param o, d Ray origin and direction (which should be normalized).
*/
void SphereCone::Set(const Vector& o,const Vector& d)
{
  Vector pa=a-o;

  double dx=d*axis;
  double pax=pa*axis;
  double dpa=d*pa;

  // Quadric stores the squared distance to vertex a
  quadric[2]=1.0;
  quadric[1]=-2.0*dpa;
  quadric[0]=pa*pa;
  
  linear[1]=dx;
  linear[0]=-pax;
}

/*!
  \brief Compute the distance between a points on a line and a cone-sphere.
  
  The member function SphereCone::Set() should be called for pre-processing.
  \param t Parameter of the point on the line.
*/
double SphereCone::R(const double& t) const
{
  double y=linear[1]*t+linear[0];
  double nn=(quadric[2]*t+quadric[1])*t+quadric[0];
  
  double e=0.0;
    
#ifdef SPHERE
  // Speed-up large sphere check (could be disabled)
  if (y<ha)        				// Cost: 1 ?
  {
    // Sphere check
    if (nn>rra)    				// Cost: 1 ?
    {
      e=sqrt(nn);  				// Cost: 1 sqrt()
      e-=ra;       				// Cost: 1 + 
      e*=e;        				// Cost: 1 *
    }              				// Overall: 2 ? 7 * 8 + 1 sqrt()
    //else
    //{
    //  e=0.0;
    //}
  }
  else
#endif
  {
    // Compute squared distance to axis
    double yy=y*y;     				// Cost: 1 *
    double xx=nn-yy;   				// Cost: 1 +
    
#ifdef CYLINDER
    // Speed-up cylinder check (could be disabled)
    if (xx<rrb)        				// Cost: 1 ?
    {
      // Check if on the left part of the cone-sphere
      if (y>length+hb)     				// Cost: 1 + 1 ?
      {
        y-=length;         				// Cost: 1 +
	yy=y*y;        				// Cost: 1 *
	nn=xx+yy;      				// Cost: 1 +
	// Sphere check
	if (nn>rrb)    				// Cost: 1 ?
	{
          e=sqrt(nn);  				// Cost: 1 sqrt()
          e-=rb;       				// Cost: 1 +
          e*=e;        				// Cost: 1 *
        }					// Overall: 4 ? 9 * 12 + 1 sqrt()
        //else
        //{
        //  e=0.0;
        //}
      }
      // Otherwise, inside cylinder part of the cone
      //else
      //{
      //  e=0.0;
      //}
    }
    else
    #endif

    // The last complex cases need a different coordinate system
    {
      double x=sqrt(xx); 			// Cost: 1 sqrt()

      // Rotate
      double ry=x*side[0]+y*side[1]; 		// Cost: 2 * 1+
      
      // Sphere check
      if (ry<0.0)        			// Cost: 1 ?
      {
        if (nn>rra)      			// Cost: 1 ?
        {
          e=sqrt(nn);    			// Cost: 1 sqrt()
          e-=ra;         			// Cost: 1 +
          e*=e;          			// Cost: 1 *
        }					// Overall: 4 ? 10 * 10 + 2 sqrt()
        //else
        //{
        //  e=0.0;
        //}
      }
      // Other sphere check
      else if (ry>conelength) 			// Cost: 1 ?
      {
        //cout<<"Small sphere"<<endl;
	// Compute the squared distance to the other end vertex
	y-=length;            			// Cost: 1 +
	yy=y*y;           			// Cost: 1 *
	nn=xx+yy;         			// Cost: 1 +

	// Sphere check
	if (nn>rrb)       			// Cost: 1 ?
	{
          e=sqrt(nn);     			// Cost: 1 sqrt()
          e-=rb;          			// Cost: 1 +
          e*=e;           			// Cost: 1 *
        }
        //else
        //{
        //  e=0.0;
        //}        					 // Cost: 5 ? 11 * 12 + 2 sqrt()
      }
      else
      {
        double rx=x*side[1]-y*side[0]; // Cost: 2 * 1 +
        if (rx>ra)   				// Cost: 1 ?
        {
	  rx-=ra;     				// Cost: 1 +
	  e=rx*rx;    				// Cost: 1 *
	}
	// Otherwise, inside cone
        //else
        //{
        //  e=0.0;
        //}                                             // Cost: 5 ? 12 * 11 + 1 sqrt()
      }
    }
  }  
  return e;
}
