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

#ifndef _Pluecker_hpp_
#define _Pluecker_hpp_

#include "Vector.hpp"


// Pluecker ////////////////////////////////////////////////////////////////////

class Pluecker
{
private:
    Vector d;   // direction
    Vector c;   // cross

public:
    Pluecker()
    { };
    
    // Pluecker coordinates of a ray specified by points orig and dest.
    Pluecker(const Vector& orig, const Vector& dest)
        : d(dest-orig), c(dest^orig)
    { };

    // Pluecker coordinates of a ray specified by point orig and
    // direction dir. The bool parameter serves only to discriminate this
    // from the previous constructor
    Pluecker(const Vector& orig, const Vector& dir, bool haveDir)
        : d(dir), c(dir^orig)
    { };

    // Permuted inner product operator
    friend double operator*(const Pluecker& pl1, const Pluecker& pl2);
};


inline double operator*(const Pluecker& pl1, const Pluecker& pl2)
{
    return (pl1.d * pl2.c + pl2.d * pl1.c);
}


#endif   // _Pluecker_hpp_
