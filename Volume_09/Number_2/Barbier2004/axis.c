// Axis
// Changelog 01.12.17

#include "shapes.h"

/*!
  \class Axis shapes.h
  \brief This class implements a minimal data-structure to define 
  a simple axis characterized by its end vertices.
*/

/*!
  \brief Creates a generic axis given end vertices.
  \param a, b End vertices of the axis.
*/
Axis::Axis(const Vector& a,const Vector& b)
{
  Axis::a=a;
  Axis::b=b; 
  axis=b-a;
  length=Norm(axis);
  axis/=length;
}

