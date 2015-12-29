/***************************************************************************
                          config.h  -  description
                             -------------------
    copyright            : (C) 2005 by MOUSSA
    email                : mmousa@liris.cnrs.fr
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

/**
Configuration for CGAL package
*/


#ifndef CONFIGFILE
#define CONFIGFILE

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Static_filters.h>

#include <CGAL/Polyhedron_3.h>
#include <CGAL/intersections.h>
#include <iostream>

/**Number Type (Arithmetic chosen)*/
typedef double                          NT; 
/**Coordinate type*/
typedef CGAL::Simple_cartesian<NT>      Rep;
/**The main kernel*/ 
typedef CGAL::Static_filters<Rep>       Kernel;
typedef Kernel::Point_3                 Point_3;
typedef Kernel::Triangle_3              Triangle_3;
typedef Kernel::Tetrahedron_3           Tetrahedron_3;
typedef Kernel::Line_3                  Line_3;
typedef Kernel::Segment_3               Segment_3; 
typedef Kernel::Plane_3                 Plane_3;

#endif
