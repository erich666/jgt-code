// voronoi.h - 

// Example code for "Efficient Generatino of Poisson-Disk Sampling
// Patterns," Thouis R. Jones, JGT vol. 11, No. 2, pp. 27-36
// 
// Copyright 2004-2006, Thouis R. Jones
// This code is distributed under the terms of the LGPL.

double compute_voronoi_area(GtsVertex *v, double exclude_radius);
GtsVertex *new_vertex_in_voronoi(GtsVertex *v, double exclude_radius);
