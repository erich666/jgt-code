Readme for C++ code accompanying the paper:
"Exact Evaluation of Catmull-Clark Subdivision Surfaces Near B-spline
Boundaries"

The code consists of two parts: a shared library (libsubdeval.so) and 
a small test program in directory 'test' to show how to use it.  

Compile the shared library using 'make', then change to the 'test' 
directory and type 'make' again.

The test program reads a mesh, samples the limit surface at a specified 
number of UV locations, then saves (p, dPdU, dPdV) to an output file.  At 
each limit point (p) vectors dPdU and dPdV are drawn as short line 
segments.

The 'test' directory also contains a sample input/output OBJ pair.  
To reproduce the output, change to the 'test' directory and run:
./testsubd cross.obj out.obj 10

out.obj should be identical to cross_samples.obj

This code has been tested on Linux, gcc version 3.3.2
Please report any compilation problems to
lacewell@cs.utah.edu

