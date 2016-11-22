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

#ifndef _Util_h_
#define _Util_h_


#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288419716939937510582097494459230
#endif

#include <cstdlib>


const double Zero = 1e-6;


inline int Sign(const double x)
{
    return (x>Zero ? 1 : (x<-Zero ? -1 : 0));
}


#endif  // _Util_h_
