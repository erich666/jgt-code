/*
 * FastArea.c++
 *
 * From the paper:
 *
 *      Daniel Sunday
 *      "Fast Polygon Area and Newell Normal Computation"
 *      journal of graphics tools, 7(2):9-13, 2002
 *
 */

// assume vertex coordinates are in arrays x[], y[], and z[]

// with room to duplicate the first two vertices at the end


// return the signed area of a 2D polygon

inline double 
findArea(int n, double *x, double *y)             // 2D polygon

{
    // guarantee the first two vertices are also at array end

    x[n] = x[0];    y[n] = y[0];
    x[n+1] = x[1];  y[n+1] = y[1];

    double sum = 0.0;
    double *xptr = x+1, *ylow = y, *yhigh = y+2;
    for (int i=1; i <= n; i++) {
        sum += (*xptr++) * ( (*yhigh++) - (*ylow++) );
    }
    return (sum / 2.0);
}

// return the signed area of a 3D planar polygon (given normal vector)

double 
findArea3D(int n, double *x, double *y, double *z,  // 3D planar polygon

           double nx, double ny, double nz)         // and plane normal

{
    // select largest normal coordinate to ignore for projection

    double ax = (nx>0 ? nx : -nx);	// abs nx

    double ay = (ny>0 ? ny : -ny);	// abs ny

    double az = (nz>0 ? nz : -nz);	// abs nz

    double len = sqrt(nx*nx + ny*ny + nz*nz); // length of normal

    
    if (ax > ay) {
        if (ax > az)			       // ignore x-coord

            return findArea(n, y, z) * (len / nx);
    }
    else if (ay > az)			       // ignore y-coord

        return findArea(n, z, x) * (len / ny);

    return findArea(n, x, y) * (len / nz); // ignore z-coord

}

// output the approximate unit normal of a 3D nearly planar polygon

// return the area of the polygon

double 
findNormal3D(int n, double *x, double *y, double *z, // 3D polygon

             double *nx, double *ny, double *nz) // output unit normal

{
    // get the Newell normal

    double nwx = findArea(n, y, z);
    double nwy = findArea(n, z, x);
    double nwz = findArea(n, x, y);

    // get length of the Newell normal

    double nlen = sqrt( nwx*nwx + nwy*nwy + nwz*nwz );
    // compute the unit normal

    *nx = nwx / nlen;
    *ny = nwy / nlen;
    *nz = nwz / nlen;

    return nlen;    // area of polygon = length of Newell normal

}
