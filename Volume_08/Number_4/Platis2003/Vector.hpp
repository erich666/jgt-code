////////////////////////////////////////////////////////////////////////////////
//
//  Source code for the paper
//
//  "Fast Ray--Tetrahedron Intersection"
//
//  (c) Nikos Platis, Theoharis Theoharis 2003
//
//  Department of Informatics and Telecommunications,
//  University of Athens, Greece
//  {nplatis|theotheo}@di.uoa.gr
//
////////////////////////////////////////////////////////////////////////////////

#ifndef _Vector_hpp_
#define _Vector_hpp_

#include "Util.h"

#include <cmath>
#include <iostream>


// Vector //////////////////////////////////////////////////////////////////////

class Vector  
{
public:
    double x;
    double y;
    double z;

    Vector();
    Vector(double xx, double yy, double zz);

    double Measure2() const;
    double Measure() const;
    void MakeUnitVector();

    // Some operators for vectors
    Vector& operator+=(const Vector& vec);        // Vector addition
    Vector& operator-=(const Vector& vec);        // Vector subtraction
    bool      operator==(const Vector& vec) const;  // Equality test
};


// Vector negation
Vector operator-(const Vector& vec);

// Vector addition
Vector operator+(const Vector& vec1, const Vector& vec2);

// Vector subtraction
Vector operator-(const Vector& vec1, const Vector& vec2);

// Dot product
double operator*(const Vector& vec1, const Vector& vec2);

// Scalar product
Vector operator*(double c, const Vector& vec);

// Cross product
Vector operator^(const Vector& vec1, const Vector& vec2);


// Iut/Output of vectors
std::istream& operator>>(std::istream& istr, Vector& vec);
std::ostream& operator<<(std::ostream& ostr, const Vector& vec);



// Inline functions ////////////////////////////////////////////////////////////

inline Vector::Vector()
    : x(0), y(0), z(0)
{
}


inline Vector::Vector(double xx, double yy, double zz)
    : x(xx), y(yy), z(zz)
{
}


inline double Vector::Measure2() const
{
    return (x*x + y*y + z*z);
}


inline double Vector::Measure() const
{
    return sqrt(Measure2());
}


inline void Vector::MakeUnitVector()
{
    double measure = Measure();
    x /= measure;
    y /= measure;
    z /= measure;
}


inline Vector& Vector::operator+=(const Vector& vec)
{
    x += vec.x;
    y += vec.y;
    z += vec.z;

    return *this;
}


inline Vector& Vector::operator-=(const Vector& vec)
{
    x -= vec.x;
    y -= vec.y;
    z -= vec.z;

    return *this;
}


inline bool Vector::operator==(const Vector& vec) const
{
    return ( (x == vec.x)  &&  (y == vec.y)  &&  (z == vec.z) );
}


inline Vector operator-(const Vector& vec)
{
    return Vector(-vec.x, -vec.y, -vec.z);
}


inline Vector operator+(const Vector& vec1, const Vector& vec2)
{
    return Vector(vec1.x+vec2.x, vec1.y+vec2.y, vec1.z+vec2.z);
}


inline Vector operator-(const Vector& vec1, const Vector& vec2)
{
    return Vector(vec1.x-vec2.x, vec1.y-vec2.y, vec1.z-vec2.z);
}


inline double operator*(const Vector& vec1, const Vector& vec2)
{
    return (vec1.x*vec2.x + vec1.y*vec2.y + vec1.z*vec2.z);
}


inline Vector operator*(double c, const Vector& vec)
{
    return Vector(c*vec.x, c*vec.y, c*vec.z);
}


inline Vector operator^(const Vector& vec1, const Vector& vec2)
{
    return Vector(vec1.y*vec2.z - vec2.y*vec1.z,
                    vec1.z*vec2.x - vec2.z*vec1.x,
                    vec1.x*vec2.y - vec2.x*vec1.y);
}


inline std::istream& operator>>(std::istream& istr, Vector& vec)
{
    istr >> vec.x >> vec.y >> vec.z;
    return istr;
}


inline std::ostream& operator<<(std::ostream& ostr, const Vector& vec)
{
    ostr << vec.x << " " << vec.y << " " << vec.z;
    return ostr;
}


#endif   // _Vector_hpp_
