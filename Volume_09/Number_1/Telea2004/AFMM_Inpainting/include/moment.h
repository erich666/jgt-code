#ifndef MOMENT_H
#define MOMENT_H

#include "field.h"

//   This file contains several functions for computing moments of several
//   orders on fields. Also, several other norms are provided here.




//Computes the order-0 moment of a given scalar field inp.
//The moment's x and y components are returned in two fields out_m0 and out_m1.
//The last parameter specifies the radius of the ball used for integration,
//in domain-cells.

void computeM0(const FIELD<float>& inp, 
	       FIELD<float>& out_m0,
     	       FIELD<float>& out_m1,
	       int=5);

//Computes the order-1 moment of a given scalar field inp.
//The moment's [0,0],[0,1], and [1,1] components are returned in three fields 
//m00,m01, and m11. The component [1,0] is not computed being identical to [0,1].
//The last parameter specifies the radius of the ball used for integration,
//in domain-cells.

void computeM1(const FIELD<float>& in_field, 
	       const FIELD<float>& in_m0,
	       const FIELD<float>& in_m1,	   
	       FIELD<float>& out_m00,
     	       FIELD<float>& out_m01,
               FIELD<float>& out_m11,
	       int=5);



//Computes the order-0 moment of the graph of a given scalar field inp.
//This is equivalent to the barycenters, in 3D, of all infinitesimal surface
//discs on the graph surface. The graph is given here as z=f(x,y) where 
//f is given by the field inp.

void computeSurfM0(const FIELD<float>& inp, 
	           FIELD<float>& out_m0,
     	           FIELD<float>& out_m1,
		   FIELD<float>& out_m2, 
	           int=5);

//Same function as above, but subsampling the field 'inp' on
//a finer grid of size 'step'. 
//Integration-radius D is now also a real number.
void computeSurfM0(const FIELD<float>& inp, 
	           FIELD<float>& out_m0,
     	           FIELD<float>& out_m1,
		   FIELD<float>& out_m2, 
	           float step=0.5,float D=5);



//Computes the average of the input signal as follows. In each point of the output out, 
//we compute the integral over a ball of the input, divided by the ball area.
//The radius of the ball can be provided. 
void computeT0(const FIELD<float>& in_field,
	       FIELD<float>& out,
	       int=5);

//Compute another kind of norm. in_field and in_t0 are the input field and its
//average respectively. The output, a symmetric 2x2 matrix, is returned in
//the out_t00..out_t11 parameters.
void computeT1(const FIELD<float>& in_field,
               const FIELD<float>& in_t0,
	       FIELD<float>& out_t00,
               FIELD<float>& out_t01,
               FIELD<float>& out_t11,
	       int=5);

//Computes the eigenvalues and eigenvectors of a symmetric 2x2 matrix field.
//The matrix is given as in00..in11. The largest eigenvalue is output in
//out_w0, the smallest in out_w1. 
void computeEVW(const FIELD<float>& in00,
	        const FIELD<float>& in01,
		const FIELD<float>& in11,
		FIELD<float>& out_w0,
		FIELD<float>& out_w1,
		VFIELD<float>& out_v0,
		VFIELD<float>& out_v1);


#endif


