
This package contains the C/C++ source code implementing the algorithm 
described in the JGT paper "Computing a View Frustum to Maximize an Object's Image Area", by Kok-Lim Low and Adrian Ilie.

You are free to use and distribute the code. There is no guarantee that 
the code will work correctly as described. The authors are not 
responsible for any consequences as a result of its use.


The essential files are:

  * kl_camcalib.h
  * kl_camcalib.cpp
  * kl_convexhull2d.h
  * kl_convexhull2d.cpp
  * kl_minquad.h
  * kl_minquad.cpp
  * kl_optifrustum.h
  * kl_optifrustum.cpp


Also essential are the followings files from Numerical Recipes in C. 
They are not included in this package.

  * nrutil.h
  * nrutil.c
  * dsvdcmp.c
  * dpythag.c


The only routines the user will call are inside "kl_optifrustum.h".
The function kl_BoundingSphere() is used to compute a bounding sphere.
This bounding sphere (or any other bounding sphere) is then passed to
the function kl_OptiFrustum() to compute the optimized frustum.
kl_OptiFrustum() uses the OpenGL API, so OpenGL must be initialized 
before calling kl_OptiFrustum().

The user can try the test program included in this package.
When the program is running, press 'M' to cycle through the different 
views of the model.

3-25-2003
Kok-Lim Low
