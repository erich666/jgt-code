Example code for "Efficient Generatino of Poisson-Disk Sampling
Patterns," Thouis R. Jones, JGT vol. 11, No. 2, pp. 27-36

This code is distributed under the terms of the LGPL.

This code requires GTS (http://gts.sourceforge.net) to build.

Usage:
fast_delaunay num_pts exclusion_radius > point_list.txt

This will insert points into the [0-1]x[0-1] square, separated by at
least exclusion_radius from each other, until either num_pts points
have been added, or there is no more free space to insert a point.


