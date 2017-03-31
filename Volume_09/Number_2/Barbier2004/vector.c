// Vector  
// Changelog 00.10.16

// Self include
#include "vector.h"

/*!
  \class Vector vector.h
  \brief This class implements a vector structure of three doubles. 
  
  Most binary operators have been overloaded as expected. 
  Destructive operators, such as addition and subtraction 
  have been implemented. Destructive operators += and -= 
  behave as one could expect. 
  
  Operators *= and /= behave in a specific way however, 
  scaling vector coordinates by the coordinates of the 
  argument vector.
  
  The cross product of two vectors is defined by the operator /.
*/

/*!
  \brief Check if three vectors are coplanar.
  
  Simply compute the cross product of a and b, and the dot product with c.
  Compare the result with a given tolerance.
*/
int Coplanar(const Vector& a,const Vector& b,const Vector& c,const double& epsilon)
{
  double s=fabs((a/b)*c)/(Norm(a)*Norm(b)*Norm(c));
  return (s<epsilon);
}

/*!
  \brief Normalize a vector, computing the inverse of its norm and scaling 
  the components. 
  
  This function does not check if the vector is null,
  which might resulting in floating point errors.
*/
void Normalize(Vector& u)
{
  u*=1.0/Norm(u);
}

/*!
  \brief Returns the positive sine of two vectors. Basically computes the 
  cross product of the vectors and normalizes the result.
*/
double Sine(const Vector& u,const Vector& v)
{
  return Norm(u/v)/sqrt((u*u)*(v*v));
}

/*!
  \brief Returns the positive cosine of two vectors. Basically computes the 
  dot product of the normalized vectors.
*/
double Cosine(const Vector& u,const Vector& v)
{
  return (u*v)/sqrt((u*u)*(v*v));
}

/*!
  \brief Returns alignment boolean. 
  Basically computes the cosine of the two vectors, and checks for unity.
*/
int Aligned(const Vector& u,const Vector& v)
{
  double c=Cosine(u,v);
  c*=c;
  return (c>(1.0-0.0001));
}

/*!
  \brief Swap two vectors.
*/
void Swap(Vector& a,Vector& b)
{
  Vector t=a;
  a=b;
  b=t;
}


/*!
  \brief Checks if four points are coplanar.
*/
int Coplanar(const Vector& t,const Vector& u,const Vector& v,const Vector& w)
{
  return Coplanar(u-t,v-t,w-t);
}

/*!
  \brief Returns a new vector orthogonal to the argument vector.
*/
Vector Orthogonal(const Vector& u)
{  
  Vector a=Abs(u);
  int i=0;
  int j=1;
  if (a[0]>a[1])
  {
    if (a[2]>a[1])
    {
      j=2;
    }
  }
  else
  {
    i=1;
    j=2;
    if (a[0]>a[2])
    {
      j=0;
    }
  }
  a=Vector(0.0);
  a[i]=u[j];
  a[j]=-u[i];
  return a;
}

/*!
  \brief Overloaded output-stream operator.
*/
ostream& operator << (ostream& s, const Vector& u)
{
  s<<'('<<u.x<<','<<u.y<<','<<u.z<<')';
  return s;
}
