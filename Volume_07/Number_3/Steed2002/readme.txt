Example Program

The following archive contains C++ source code using OpenGL as well as the relevant Visual C++ project files. Compiling this example requires Rademacher’s GLUI library. The code has been tested on Win32 and IRIX. pseudo_cursor1.0.zip (24K zip archive)

A pre-compiled Windows executable is available in the following archive: pseudo_cursor_bin1.0.zip (95K zip archive)

Follow-up

Since the original paper was written for JGT, Everitt and Kilgard published their paper “Practical and Robust Stenciled Shadow Volumes for Hardware-Accelerated Rendering”. Their z-fail approach was adopted in the above example program as it simplified the code which otherwise had to deal with capping the shadow volumes on the near clipping plane since the infinite slab would often intersect the view plane window. To use the z-fail approach the “infinite slabs” are now large enough that they extend beyond the scene, but are not clipped by the far clipping plane.