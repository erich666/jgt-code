#include <iostream>

#include "Dual.h"

//
// Compute derivative of exp(-x*x-y*y) with respect to x and y
// at (x,y)=(0,1)
// We write our function as a 'function object'
// or functor.
// http://www.sgi.com/tech/stl/functors.html
//
class F {
public:
    template<class X> X operator()(const X &x) const {
	return exp(-x*x);
    }
};

//
// Compute (d/dx)^n exp(-x^2) at x=0
// Results for n = 0, 1, 2, 3 should be
// 1, 0, -2, 0
//
// Note that this is not the most efficient way to compute
// higher derivatives. A better approach would be something
// like the Taylor series support in the FADBAD++
// library: http://www.imm.dtu.dk/nag/proj_km/fadbad/
// But for second derivatives it is acceptable.
//
int main() {
    F f;

    std::cout << f(0.0) << std::endl;
    std::cout << Derivative(f)(0.0) << std::endl;
    std::cout << Derivative(Derivative(f))(0.0) << std::endl;
    std::cout << Derivative(Derivative(Derivative(f)))(0.0) << std::endl;

    //
    // Higher derivatives require a more complex implementation of
    // type conversion from base types.
    //

    return 0;
}
