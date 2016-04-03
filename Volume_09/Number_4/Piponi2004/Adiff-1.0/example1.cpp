#include <iostream>

#include "Dual.h"

//
// Compute derivative of exp(-x*x-y*y) with respect to x and y
// at (x,y)=(1,1)
//
template<class X> X f(const X &x,const X &y) {
    return (X(2.0)*x*x+y)*exp(-x*x-y*y);
}

int main() {
    typedef Dual<double,2> D;

    D d0 = D::d(0);
    D d1 = D::d(1);

    //
    // Results should be:
    // -2e^-2 and -5e^-2
    // See section 5 of the paper for an explanation of d0 and d1
    //
    std::cout << "Derivative wrt x = " << f(D(1.0)+d0,D(1.0)+d1).im(0) << std::endl;
    std::cout << "Derivative wrt y = " << f(D(1.0)+d0,D(1.0)+d1).im(1) << std::endl;

    return 0;
}
