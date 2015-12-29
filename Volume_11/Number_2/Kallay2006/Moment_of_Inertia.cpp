/**********************************************************************
Add the contribution of a triangle to the mass properties.
Call this method for each one of the mesh's triangles.  
The order of vertices must be such that their determinant is positive
when the origin lies on the inner side of this facet's plane.
**********************************************************************/
    void AddTriangleContribution(
        double x1, double y1, double z1,    // Triangle's vertex 1
        double x2, double y2, double z2,    // Triangle's vertex 2
        double x3, double y3, double z3)    // Triangle's vertex 3
    {
        // Signed volume of this tetrahedron.
        double v = x1*y2*z3 + y1*z2*x3 + x2*y3*z1 -
                  (x3*y2*z1 + x2*y1*z3 + y3*z2*x1);
        
        // Contribution to the mass
        _m += v;

        // Contribution to the centroid
        double x4 = x1 + x2 + x3;           _Cx += (v * x4);
        double y4 = y1 + y2 + y3;           _Cy += (v * y4);
        double z4 = z1 + z2 + z3;           _Cz += (v * z4);

        // Contribution to moment of inertia monomials
        _xx += v * (x1*x1 + x2*x2 + x3*x3 + x4*x4);
        _yy += v * (y1*y1 + y2*y2 + y3*y3 + y4*y4);
        _zz += v * (z1*z1 + z2*z2 + z3*z3 + z4*z4);
        _yx += v * (y1*x1 + y2*x2 + y3*x3 + y4*x4);
        _zx += v * (z1*x1 + z2*x2 + z3*x3 + z4*x4);
        _zy += v * (z1*y1 + z2*y2 + z3*y3 + z4*y4);        
    }

 
/**********************************************************************
This method is called to obtain the results.
This call modifies the internal data; calling it again will return 
incorrect results.
**********************************************************************/
    void GetResults(
        double & m,                               // Total mass
        double & Cx,  double & Cy,  double & Cz,  // Centroid
        double & Ixx, double & Iyy, double & Izz, // Moment of inertia       
                                                  // diagonal entries
        double & Iyx, double & Izx, double & Izy) // Moment of inertia 
                                                  // mixed entries
    {
        // Centroid.  
        // The case _m = 0 needs to be addressed here.
        double r = 1.0 / (4 * _m);
        Cx = _Cx * r;
        Cy = _Cy * r;
        Cz = _Cz * r;

        // Mass
        m = _m / 6;

        // Moment of inertia about the centroid.
        r = 1.0 / 120;
        Iyx = _yx * r - m * Cy*Cx;
        Izx = _zx * r - m * Cz*Cx;
        Izy = _zy * r - m * Cz*Cy;

        _xx = _xx * r - m * Cx*Cx;
        _yy = _yy * r - m * Cy*Cy;
        _zz = _zz * r - m * Cz*Cz;

        Ixx = _yy + _zz;
        Iyy = _zz + _xx;
        Izz = _xx + _yy;
    }
    
    // Data members
private:
    double _m;                              // Mass
    double _Cx, _Cy, _Cz;                   // Centroid
    double _xx, _yy, _zz, _yx, _zx, _zy;    // Moment of inertia tensor
};
