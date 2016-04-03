#include <algorithm>
#include <functional>
#include <cmath>

using namespace std;

//
// Important note: This is not industrial strength software.
// It is a demonstration of the ideas in the paper
// "Automatic Differentiation, C++ Templates and Photogrammetry" by D.Piponi
// Dual<X,1> corresponds to Dual<X> in the paper.
//
// This code (c) D Piponi 2004 (d.piponi (at) sigfpe.com) http://www.sigfpe.com
//
template<class X,int N> class Dual {
	//
	// In a real implementation real and imag wouldn't be public.
	// Note that this implementation repeatedly makes copies
	// of the array imag. A better implementation might use
	// copy-on-write for efficiency.
	// Additionally there are great benefits to be gained from
	// representing imag sparsely.
	//
public:
    X real;
    X imag[N];

    Dual() { }
    Dual(const X &x) : real(x) {
	fill(imag,imag+N,X(0));
    }

    static Dual<X,N> d(int j = 0) {
	Dual<X,N> a;
	a.real = X(0);
	for (int i = 0; i<N; ++i) {
	    a.imag[i] = i==j ? X(1) : X(0);
	}
	return a;
    }

    X re() const {
	return real;
    }

    X im(int i = 0) const {
	return imag[i];
    }
};

template<class X,int N>
ostream &operator<<(ostream &out,const Dual<X,N> &a) {
    out << a.real << "[";
    for (int i = 0; i<N; ++i) {
	out << a.imag[i];
	if (i<N-1) {
	    out << ' ';
	}
    }
    return out << ']';
}

template<class X,int N>
Dual<X,N> operator+(const Dual<X,N> &a,const Dual<X,N> &b) {
    Dual<X,N> c;
    c.real = a.real+b.real;
    transform(a.imag,a.imag+N,b.imag,c.imag,plus<X>());
    return c;
}

template<class X,int N>
Dual<X,N> operator-(const Dual<X,N> &a,const Dual<X,N> &b) {
    Dual<X,N> c;
    c.real = a.real-b.real;
    transform(a.imag,a.imag+N,b.imag,c.imag,minus<X>());
    return c;
}

template<class X,int N>
Dual<X,N> operator-(const Dual<X,N> &a) {
    Dual<X,N> b;
    b.real = -a.real;
    transform(a.imag,a.imag+N,b.imag,negate<X>());
    return b;
}

template<class X,int N>
Dual<X,N> operator*(const Dual<X,N> &a,const Dual<X,N> &b) {
    Dual<X,N> c;
    c.real = a.real*b.real;
    for (int i = 0; i<N; ++i) {
	c.imag[i] = a.real*b.imag[i]+a.imag[i]*b.real;
    }
    return c;
}

//
// Transcendental functions.
// Note that sin and cos may be implemented more
// efficiently than by simply copying the method
// for exp. In particular sin and cos should
// be replaced by a function that computes sin
// and cos simultaneously to save redundant
// computations of these functions.
//
template<class X,int N>
Dual<X,N> exp(const Dual<X,N> &a) {
    Dual<X,N> b;
    X d = exp(a.real);
    b.real = d;
    transform(a.imag,a.imag+N,b.imag,bind1st(multiplies<X>(),d));
    return b;
}

//
// Derivatives implemented as function objects allowing
// repeated differentiation.
// See example2.C
//
template<class F>
class _Derivative {
    const F &f;
public:
    _Derivative(const F &f0) : f(f0) { }
    template<class X>
    X operator()(const X &x) const {
	typedef Dual<X,1> D;
	//
	// See section 4 of the paper for an explanation
	// of this line.
	//
	return f(D(x)+D::d()).im();
    }
};

template<class F>
_Derivative<F> Derivative(const F &f) {
    return _Derivative<F>(f);
}
