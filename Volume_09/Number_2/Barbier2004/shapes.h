// Axis
// Changelog 02.02.24

#ifndef __Shapes__
#define __Shapes__

#include "vector.h"

// Axis class
class Axis
{
protected:
  Vector a,b;   //!< End vertices of the axis.
  Vector axis;  //!< Normalized axis vector.
  double length; //!< Length of the axis.
  double quadric[3]; //!< Quadric equation of the squared distance to the axis.
  double linear[2];  //!< Linear equation of the distance along the axis.
public:
  Axis(const Vector&,const Vector&);
};

// Cone class
class Cone:public Axis
{
protected:
  double ra,rb,rrb,rra;  //!< Radius of the cone at the end vertices of the axis.
  double conelength; //!< Side length of the cone
  Vector side;       //!< Side vector of the cone.
public:
  Cone(const Vector&,const Vector&,const double&,const double&);
  double R(const Vector&) const;
  void Set(const Vector&,const Vector&);
  double R(const double&) const;
};

// Cone class
class SphereCone:public Axis
{
protected:
  double ra,rb,rrb,rra;  //!< Radius and squared radius values of the cone-sphere.
  double ha,hb,hrb,hra;  //!< Internal parameters of the cone joining the spheres.
  double conelength; //!< Side length of the cone
  Vector side;       //!< Side vector of the cone.
public:
  SphereCone(const Vector&,const Vector&,const double&,const double&);
  double R(const Vector&) const;
  void Set(const Vector&,const Vector&);
  double R(const double&) const;
};

// Cylinder class
class Cylinder:public Axis
{
protected:
  double r[2]; //!< Radius and squared radius.
  Vector c;   //!< Center.
  double h;    //!< Half length.
public:
  Cylinder(const Vector&,const Vector&,const double&);
  double R(const Vector&) const;
  void Set(const Vector&,const Vector&);
  double R(const double&) const;
};

// Cylinder class
class SphereCylinder:public Axis
{
protected:
  double r[2]; //!< Radius and squared radius.
  Vector c;   //!< Center of the cylinder.
  double h;    //!< Half length.
public:
  SphereCylinder(const Vector&,const Vector&,const double&);
  double R(const Vector&) const;
  void Set(const Vector&,const Vector&);
  double R(const double&) const;
};

#endif
