// Vector  
// Changelog 02.07.04

#ifndef __Vector__
#define __Vector__

#include <math.h>
#include <iostream.h>

// Inline extrema
inline int min(const int a, const int b)
{
  return (a<b?a:b);
}

inline int max(const int a, const int b)
{
  return (a>b?a:b);
}

inline double min(const double& a, const double& b)
{
  return (a<b?a:b);
}
  
inline double max(const double& a, const double& b)
{
  return (a>b?a:b);
}

inline double min(const double& a, const double& b, const double& c)
{
  return (a<b)?((a<c)?a:c):((b<c)?b:c);
}
  
inline double max(const double& a, const double& b, const double& c)
{
  return (a>b)?((a>c)?a:c):((b>c)?b:c);
}
// Forward-declare some other classes.
// ... none

// Class
class Vector
{
protected:
  double x,y,z;
public:
  //! Empty constructor.
  Vector() { }
  //! Create a vector with the same coordinates.
  Vector(const double& a) { x=y=z=a; }
  //! Create a vector with argument coordinates.
  Vector(const double& a, const double& b, const double& c) { x=a; y=b; z=c; }
  //! Create a vector with argument coordinates in an array
  Vector(const double coords[3]) { x=coords[0]; y=coords[1]; z=coords[2]; }

  //! Access to the i<SUP>th</SUP> coordinate of vector
  double& operator[] (int i);
  
  //! Access to the i<SUP>th</SUP> coordinate of vector
  double operator[] (int i) const;
  
  // Unary operators
  Vector operator+ () const;
  Vector operator- () const;
  
  // Assignment operators
  Vector& operator+= (const Vector&);
  Vector& operator-= (const Vector&);
  Vector& operator*= (const Vector&);
  Vector& operator/= (const Vector&);
  Vector& operator*= (double);
  Vector& operator/= (double);
  
  // Binary operators
  friend int operator> (const Vector&, const Vector&);
  friend int operator< (const Vector&, const Vector&);

  friend int operator>= (const Vector&, const Vector&);
  friend int operator<= (const Vector&, const Vector&);
  
  // Binary operators
  friend Vector operator+ (const Vector&, const Vector&);
  friend Vector operator- (const Vector&, const Vector&);
  
  friend double operator* (const Vector&, const Vector&);
  
  friend Vector operator* (const Vector&, double);
  friend Vector operator* (double, const Vector&);
  friend Vector operator/ (const Vector&, double);
  
  friend Vector operator/ (const Vector&, const Vector&);
    
  // Boolean functions
  friend int operator==(const Vector&,const Vector&);
  friend int operator!=(const Vector&,const Vector&);
  
  // Norm
  friend double Norm(const Vector&);
  friend double NormInfinity(const Vector&);
  friend void Normalize(Vector&);
  friend Vector Normalized(const Vector&);
  
  // High level functions
  friend double Sine(const Vector&,const Vector&);
  friend double Cosine(const Vector&,const Vector&);
  
  // Compare functions
  friend Vector min(const Vector&,const Vector&);
  friend Vector max(const Vector&,const Vector&);
  
  // Abs
  friend Vector Abs(const Vector&);
  
  // Orthogonal
  friend Vector Orthogonal(const Vector&);
  
  // Swap
  friend void Swap(Vector&,Vector&);

  friend int Aligned(const Vector&,const Vector&);
  friend int Coplanar(const Vector&,const Vector&,const Vector&,const double& =1.0e-6);
  friend int Coplanar(const Vector&,const Vector&,const Vector&,const Vector&);

  friend ostream& operator<<(ostream&,const Vector&);
};

//! Gets the i<SUP>th</SUP> coordinate of vector.
inline double& Vector::operator[] (int i) 
{
  if (i == 0)    return x;
  else if (i == 1) return y;
  else	     return z;
}
  
//! Returns the i<SUP>th</SUP> coordinate of vector.
inline double Vector::operator[] (int i) const 
{
  if (i == 0)    return x;
  else if (i == 1) return y;
  else	     return z;
}
 
 // Unary operators

inline Vector Vector::operator+ () const
{
  return *this;
}

inline Vector Vector::operator- () const
{
  return Vector(-x,-y,-z);
}

// Assignment unary operators
//! Destructive addition.
inline Vector& Vector::operator+= (const Vector& u)
{
  x+=u.x; y+=u.y; z+=u.z;
  return *this;
}

//! Destructive subtraction.
inline Vector& Vector::operator-= (const Vector& u)
{
  x-=u.x; y-=u.y; z-=u.z;
  return *this;
}

//! Destructive scalar multiply.
inline Vector& Vector::operator*= (double a)
{
  x*=a; y*=a; z*=a;
  return *this;
}

//! Destructive division by a scalar.
inline Vector& Vector::operator/= (double a)
{
  x/=a; y/=a; z/=a;
  return *this;
}

//! Destructively scale a vector by another vector.
inline Vector& Vector::operator*= (const Vector& u)
{
  x*=u.x; y*=u.y; z*=u.z;
  return *this;
}

//! Destructively divide the components of a vector by another vector.
inline Vector& Vector::operator/= (const Vector& u)
{
  x/=u.x; y/=u.y; z/=u.z;
  return *this;
}

//! Compare two vectors.
inline int operator> (const Vector& u, const Vector& v)
{
  return ((u.x>v.x) && (u.y>v.y) && (u.z>v.z));
}

//! Compare two vectors.
inline int operator< (const Vector& u, const Vector& v)
{
    return ((u.x<v.x) && (u.y<v.y) && (u.z<v.z));
}

//! Overloaded
inline int operator>= (const Vector& u, const Vector& v)
{
  return ((u.x>=v.x) && (u.y>=v.y) && (u.z>=v.z));
}

//! Overloaded
inline int operator<= (const Vector& u, const Vector& v)
{
    return ((u.x<=v.x) && (u.y<=v.y) && (u.z<=v.z));
}

//! Adds up two vectors.
inline Vector operator+ (const Vector& u, const Vector& v)
{
  return Vector(u.x+v.x,u.y+v.y,u.z+v.z);
}

inline Vector operator- (const Vector& u, const Vector& v)
{
  return Vector(u.x-v.x,u.y-v.y,u.z-v.z);
}

//! Scalar product.
inline double operator* (const Vector& u, const Vector& v)
{
  return (u.x*v.x+u.y*v.y+u.z*v.z);
}

//! Right multiply by a scalar.
inline Vector operator* (const Vector& u,double a)
{
  return Vector(u.x*a,u.y*a,u.z*a);
}

//! Left multiply by a scalar.
inline Vector operator* (double a, const Vector& v)
{
  return v*a;
}

//! Cross product.
inline Vector operator/ (const Vector& u, const Vector& v)
{
  return Vector(u.y*v.z-u.z*v.y,u.z*v.x-u.x*v.z,u.x*v.y-u.y*v.x);
}

inline Vector operator/ (const Vector& u, double a)
{
  return Vector(u.x/a,u.y/a,u.z/a);
}

// Boolean functions

inline int operator== (const Vector& u,const Vector& v)
{
  return ((u.x==v.x)&&(u.y==v.y)&&(u.z==v.z));
}

inline int operator!= (const Vector& u,const Vector& v)
{
  return (!(u==v));
}

/*!
  \brief Compute the Euclidean norm of a vector.
*/
inline double Norm(const Vector& u)
{
  return sqrt(u.x*u.x+u.y*u.y+u.z*u.z);
}

/*!
  \brief Return a Normalized a vector, computing the inverse of its 
  norm and scaling the components. This function does not check if 
  the vector is null, which might resulting in floating point errors.
*/
inline Vector Normalized(const Vector& u)
{
  return u*(1.0/Norm(u));
}

/*!
  \brief Compute the infinity norm of a vector.
*/
inline double NormInfinity(const Vector& u)
{
  return max(fabs(u.x),fabs(u.y),fabs(u.z));
}

/*!
  \brief Computes the absolute value of a vector.
*/
inline Vector Abs(const Vector& u)
{
  return Vector(u[0]>0.0?u[0]:-u[0],u[1]>0.0?u[1]:-u[1],u[2]>0.0?u[2]:-u[2]);
}


/*!
  \brief Return a new vector with coordinates set to the minimum coordinates
  of the two argument vectors.
*/
inline Vector min(const Vector& a,const Vector& b)
{
  return Vector(a[0]<b[0]?a[0]:b[0],a[1]<b[1]?a[1]:b[1],a[2]<b[2]?a[2]:b[2]);
}

/*!
  \brief Return a new vector with coordinates set to the maximum coordinates
  of the two argument vectors.
*/
inline Vector max(const Vector& a,const Vector& b)
{
  return Vector(a[0]>b[0]?a[0]:b[0],a[1]>b[1]?a[1]:b[1],a[2]>b[2]?a[2]:b[2]);
}

#endif

