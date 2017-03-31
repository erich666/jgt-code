// Cone
// Changelog 03.08.17

#include "shapes.h"

/*!
  \class Cone shapes.h
  \brief This class implements a truncated cone primitive. 
*/

/*!
  \brief Create a cone skeletal element. 
  
  This function assumes that the first radius ra is greater than rb.
   
  \param a, b End vertices of the cone.
  \param ra, rb Radii at the end vertices of the cone.
*/
Cone::Cone(const Vector& a,const Vector& b,const double& ra,const double& rb):Axis(a,b)
{
  Cone::ra=ra;
  Cone::rb=rb;
  Cone::rrb=rb*rb;
  Cone::rra=ra*ra;
    
  // Compute the length of side of cone, i.e. its slant height
  Cone::conelength=sqrt((rb-ra)*(rb-ra)+length*length);

  // Line segment
  Cone::side=Vector(rb-ra,length,0.0);
  Cone::side/=conelength;
}


/*!
  \brief Compute the distance between a point in space and a cone.
*/
double Cone::R(const Vector& p) const
{
  // Compute revolution coordinates 
  Vector n=p-a;			        // Cost: 3 +
  double y=n*axis;   			 // Cost: 3 * 2 + 
  double yy=y*y;    			 // Cost: 1 *
  
  // Squared radial distance to axis: postpone square root evaluation only when needed
  double xx=n*n-yy;  		 	 // Cost: 3 * 3 +
				        // Overall cost: 8 + 7 *
  double e=0.0;

  // Distance to large cap
  if (y<0.0)  			        // Cost: 1 ?
  {
    // Disk : distance to plane cap
    if (xx<rra) 			// Cost: 1 ?
    { 
      e=yy; 
    }
    // Distance to plane circle
    else
    {
      double x=sqrt(xx)-ra;
      e=x*x+yy; 			// Cost: 1 + 1 *
    }					// Cost in worst case: 10 + 8 * 1 sqrt() 3 ?
  }
  // Small cylinder test (optimization)
  else if (xx<rrb)        			// Cost: 1 ?
  {
    if (y>length) 			 // Cost: 1 ? 
    {
      e=y-length;      			// Cost: 1 -
      e*=e;            			// Cost: 1 *
    }					// Total cost in worst case : 9 + 8 * 3 ? 
    // Inside cone
    //else
    //{
    //  e=0.0;
    //}
  }
  else
  {
    // Evaluate radial distance to axis
    double x=sqrt(xx);  		// Cost: 1 sqrt()  
    
    // Change frame, so that point is now on large cap
    x-=ra;   			       // Cost: 1 -
    
    // Distance to large cap
    if (y<0.0)  			// Cost: 1 ?
    {
      // Disk : distance to plane cap
      if (x<0.0) 			// Cost: 1 ?
      { 
        e=yy; 
      }
      // Distance to plane circle
      else
      {
        e=x*x+yy; 			// Cost: 1 + 1 *
      }					// Cost in worst case: 10 + 8 * 1 sqrt() 3 ?
    }
    else
    {
      // Compute coordinates in the new rotated frame
      // Postpone some computation that may not be needed in the following case 
      double ry=x*side[0]+y*side[1]; 	 // Cost : 1 + 2 *
      if (ry<0.0) 			// Cost: 1 ?
      {
        e=x*x+yy; 			// Cost: 1+ 1 *
      }
      else 
      {
	double rx=x*side[1]-y*side[0];   // Cost : 1 + 2 *
        if (ry>conelength)              // Cost: 1 ?
        {
          ry-=conelength;               // Cost: 1 +
          e=rx*rx+ry*ry;                // Cost: 1 + 2 *
        }
        else
        {
          if (rx>0.0)  			// Cost: 1 ?
          {
            e=rx*rx;  			// Cost: 1 *
          }
          //else
          //{
          //  e=0.0;
          //}
        }
      }					// Cost in worst case: 14 + 14 * 1 sqrt() 4 ?
    }
  }
  return e;
}

/*!
  \brief Computes the pre-processing equations.
  \param o, d Ray origin and direction (which should be normalized).
*/
void Cone::Set(const Vector& o,const Vector& d)
{
  Vector pa=a-o;

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
  \brief Compute the distance between a points on a line and a cone.
  
  The member function Cone::Set() should be called for pre-processing.
  \param t Parameter of the point on the line.
*/
double Cone::R(const double& t) const
{
  double y=linear[1]*t+linear[0];
  double xx=(quadric[2]*t+quadric[1])*t+quadric[0];

  double yy=y*y;    			 // Cost: 1 *
				        // Overall cost: 8 + 7 *
  double e=0.0;

  // Distance to large cap
  if (y<0.0)  			        // Cost: 1 ?
  {
    // Disk : distance to plane cap
    if (xx<rra) 			// Cost: 1 ?
    { 
      e=yy; 
    }
    // Distance to plane circle
    else
    {
      double x=sqrt(xx)-ra;
      e=x*x+yy; 			// Cost: 1 + 1 *
    }					// Cost in worst case: 10 + 8 * 1 sqrt() 3 ?
  }
  // Small cylinder test (optimization)
  else if (xx<rrb)        			// Cost: 1 ?
  {
    if (y>length) 			 // Cost: 1 ? 
    {
      e=y-length;      			// Cost: 1 -
      e*=e;            			// Cost: 1 *
    }					// Total cost in worst case : 9 + 8 * 3 ? 
    // Inside cone
    //else
    //{
    //  e=0.0;
    //}
  }
  else
  {
    // Evaluate radial distance to axis
    double x=sqrt(xx);  		// Cost: 1 sqrt()  
    
    // Change frame, so that point is now on large cap
    x-=ra;   			       // Cost: 1 -
    
    // Distance to large cap
    if (y<0.0)  			// Cost: 1 ?
    {
      // Disk : distance to plane cap
      if (x<0.0) 			// Cost: 1 ?
      { 
        e=yy; 
      }
      // Distance to plane circle
      else
      {
        e=x*x+yy; 			// Cost: 1 + 1 *
      }					// Cost in worst case: 10 + 8 * 1 sqrt() 3 ?
    }
    else
    {
      // Compute coordinates in the new rotated frame
      // Postpone some computation that may not be needed in the following case 
      double ry=x*side[0]+y*side[1]; 	 // Cost : 1 + 2 *
      if (ry<0.0) 			// Cost: 1 ?
      {
        e=x*x+yy; 			// Cost: 1+ 1 *
      }
      else 
      {
	double rx=x*side[1]-y*side[0];   // Cost : 1 + 2 *
        if (ry>conelength)              // Cost: 1 ?
        {
          ry-=conelength;               // Cost: 1 +
          e=rx*rx+ry*ry;                // Cost: 1 + 2 *
        }
        else
        {
          if (rx>0.0)  			// Cost: 1 ?
          {
            e=rx*rx;  			// Cost: 1 *
          }
          //else
          //{
          //  e=0.0;
          //}
        }
      }					// Cost in worst case: 14 + 14 * 1 sqrt() 4 ?
    }
  }
  return e;
}
