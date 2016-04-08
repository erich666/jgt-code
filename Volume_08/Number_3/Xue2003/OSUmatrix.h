// vector and matrix class and operations
// Created by Matt Camuto
// Modified by R. Wenger: 07-09-99
// Modified by K. Boner: 08-20-99
//Modified by R. Machiraju: 09/29/200


#ifndef _MATRIX_H
#define _MATRIX_H

/* 3d and 4d vector and matrix classes.
   Supports vector and matrix addition and subtraction, 
   matrix multiplication, vector dot products, cross products, normalization,
   etc.
*/

#include <iostream>
using namespace std;


typedef float SCALAR;

class VECTOR3 {

  private :

// const int dimens = 3;
//   SCALAR vec[dimens];
// Modified by RMachiraju for reasons of ANSI compliance

   SCALAR vec[3];

  public :
    VECTOR3()                                    // constructor
      { Zero(); };
    VECTOR3(const SCALAR x0, const SCALAR x1, const SCALAR x2)// constructor
      { vec[0] = x0; vec[1] = x1; vec[2] = x2; };
    int Dimension() const
      { return(3); };
    SCALAR & operator [](const int i)           // index i'th element
      { return(vec[i]); };
    SCALAR operator ()(const int i) const        // return i'th element
      { return(vec[i]); };
    const VECTOR3 & operator =(const VECTOR3 & v0)  // copy vector v0
      { vec[0] = v0(0); vec[1] = v0(1); vec[2] = v0(2); 
        return(*this); };

    void Zero()                                  // make zero vector
      { vec[0] = vec[1] = vec[2] = 0.0; };
    void Normalize();                            // normalize vector

};

class VECTOR4 {

  private :

// Not used for ANSI-compliance. Modified by R. Machiraju

//    const int dimens = 4;
//    SCALAR vec[dimens];
 
      SCALAR vec[4];

  public :
    VECTOR4()                                    // constructor
      { Zero(); };
    VECTOR4(const SCALAR x0, const SCALAR x1, 
	    const SCALAR x2, const SCALAR x3)    // constructor
      { vec[0] = x0; vec[1] = x1; vec[2] = x2; vec[3] = x3; };
    int Dimension() const
      { return(4); };
//      { return(dimens); };
    SCALAR & operator [](const int i)            // index i'th element
      { return(vec[i]); };
    SCALAR operator ()(const int i) const        // return i'th element
      { return(vec[i]); };
    const VECTOR4 & operator =(const VECTOR4 & v0)  // copy vector v0
      { vec[0] = v0(0); vec[1] = v0(1); vec[2] = v0(2); vec[3] = v0(3);
        return(*this); };

    void Zero()                                  // make zero vector
      { vec[0] = vec[1] = vec[2] = vec[3] = 0.0; };
    void Normalize();                            // normalize vector
};

class MATRIX3 {

  private :
//    const int dimens = 3;
//    VECTOR3 mat[dimens];       // a vector represents each matrix row

    VECTOR3 mat[3];       	// a vector represents each matrix row


  public :
    MATRIX3()                                    // constructor
      { Identity(); };
    MATRIX3(const VECTOR3 & v0, const VECTOR3 & v1, const VECTOR3 & v2)
      { mat[0] = v0; mat[1] = v1; mat[2] = v2; };  // constructor
    int Dimension() const
//      { return(dimens); };
      { return(3); };
    VECTOR3 & operator [](const int i)           // index row i
      { return(mat[i]); };
    // Note: reference row i, column j of MATRIX3 m0 as m0[i][j] (not m0[i,j])
    VECTOR3 operator()(const int i) const        // return row i
      { return(mat[i]); };
    SCALAR operator ()(const int i, const int j) const   
      { return(mat[i](j)); };                    // return element (i,j)
    MATRIX3 & operator =(const MATRIX3 & m0)     // copy matrix m0
      { mat[0] = m0(0); mat[1] = m0(1); mat[2] = m0(2); 
        return(*this); };
    void Identity();                             // set to identity
};


class MATRIX4 {

  private :
//    const int dimens = 4;
//    VECTOR4 mat[dimens];       // a vector represents each matrix row

    VECTOR4 mat[4];       // a vector represents each matrix row

  public :
    MATRIX4()                                    // constructor
      { Identity(); };
    MATRIX4(const VECTOR4 & v0, const VECTOR4 & v1, 
	    const VECTOR4 & v2, const VECTOR4 & v3) // constructor
      { mat[0] = v0; mat[1] = v1; mat[2] = v2; mat[3] = v3; };  
    int Dimension() const
      { return(4); };
//      { return(dimens); };
    VECTOR4 & operator [](int i)                 // index row i
      { return(mat[i]); };
    // Note: reference row i, column j of MATRIX4 m0 as m0[i][j] (not m0[i,j])
    VECTOR4 operator()(const int i) const        // return row i
      { return(mat[i]); };
    SCALAR operator ()(const int i, const int j) const
      { return(mat[i](j)); };                    // return element (i,j)
    MATRIX4 & operator =(const MATRIX4 & m0)     // copy matrix m0
      { mat[0] = m0(0); mat[1] = m0(1); mat[2] = m0(2); mat[3] = m0(3);
        return(*this); };

    void Identity();                             // set to identity
};


//************************
// VECTOR3 operations
//************************

inline SCALAR dot(const VECTOR3 & v0, const VECTOR3 & v1)   
// return dot product of v0 and v1
{ return(v0(0)*v1(0) + v0(1)*v1(1) + v0(2)*v1(2)); };

inline VECTOR3 cross(const VECTOR3 & v0, const VECTOR3 & v1)
// return cross product of v0 and v1
{ return(VECTOR3(v0(1)*v1(2) - v0(2)*v1(1), 
		 v0(2)*v1(0) - v0(0)*v1(2),
		 v0(0)*v1(1) - v0(1)*v1(0))); };

inline VECTOR3 operator +(const VECTOR3 & v0, const VECTOR3 & v1)
// return v0 + v1
{ return(VECTOR3(v0(0) + v1(0), v0(1) + v1(1), v0(2) + v1(2))); };

inline VECTOR3 operator -(const VECTOR3 & v0, const VECTOR3 & v1)
// return v0 - v1
{ return(VECTOR3(v0(0) - v1(0), v0(1) - v1(1), v0(2) - v1(2))); };

inline VECTOR3 operator *(SCALAR x0, const VECTOR3 & v0)
// return x0*v0
{ return(VECTOR3(x0*v0(0), x0*v0(1), x0*v0(2))); };

inline VECTOR3 operator *(const VECTOR3 & v0, SCALAR x0)
// return v0*x0 (= x0*v0)
{ return(x0*v0); };

//************************
// VECTOR4 operations
//************************

inline SCALAR dot(const VECTOR4 & v0, const VECTOR4 & v1)   
// return dot product of v0 and v1
{ return(v0(0)*v1(0) + v0(1)*v1(1) + v0(2)*v1(2) + v0(3)*v1(3)); };

inline VECTOR4 operator +(const VECTOR4 & v0, const VECTOR4 & v1)
// return v0 + v1
{ return(VECTOR4(v0(0)+v1(0), v0(1)+v1(1), v0(2)+v1(2), v0(3)+v1(3))); };

inline VECTOR4 operator -(const VECTOR4 & v0, const VECTOR4 & v1)
// return v0 - v1
{ return(VECTOR4(v0(0)-v1(0), v0(1)-v1(1), v0(2)-v1(2), v0(3)-v1(3))); };

inline VECTOR4 operator *(SCALAR x0, const VECTOR4 & v0)
// return x0*v0
{ return(VECTOR4(x0*v0(0), x0*v0(1), x0*v0(2), x0*v0(3))); };

inline VECTOR4 operator *(const VECTOR4 & v0, SCALAR x0)
// return v0*x0 (= x0*v0)
{ return(x0*v0); };

//************************
// MATRIX3 operations
//************************

MATRIX3 operator +(const MATRIX3 & m0, const MATRIX3 & m1); // return m0 + m1
MATRIX3 operator -(const MATRIX3 & m0, const MATRIX3 & m1); // return m0 - m1
MATRIX3 operator *(const MATRIX3 & m0, const MATRIX3 & m1); // return m0 * m1
MATRIX3 operator *(const SCALAR x0, const MATRIX3 & m0);    // return x0 * m0
MATRIX3 operator *(const MATRIX3 & m0, const SCALAR x0);    // return m0 * x0
VECTOR3 operator *(const MATRIX3 & m0, const VECTOR3 & v0); // return m0 * v0
VECTOR3 operator *(const VECTOR3 & v0, const MATRIX3 & m0); // return v0 * m0

//************************
// MATRIX4 operations
//************************

MATRIX4 operator +(const MATRIX4 & m0, const MATRIX4 & m1); // return m0 + m1
MATRIX4 operator -(const MATRIX4 & m0, const MATRIX4 & m1); // return m0 - m1
MATRIX4 operator *(const MATRIX4 & m0, const MATRIX4 & m1); // return m0 * m1
MATRIX4 operator *(const SCALAR x0, const MATRIX4 & m0);    // return x0 * m0
MATRIX4 operator *(const MATRIX4 & m0, const SCALAR x0);    // return m0 * x0
VECTOR4 operator *(const MATRIX4 & m0, const VECTOR4 & v0); // return m0 * v0
VECTOR4 operator *(const VECTOR4 & v0, const MATRIX4 & m0); // return v0 * m0

MATRIX4 inverse(const MATRIX4 & m);  // return inverse of m; return 0 matrix if
                                     // m is singular

//******************************
// output procedures, operators
//******************************

ostream & operator<<(ostream & os, const VECTOR3 & v);  // output vector
ostream & operator<<(ostream & os, const VECTOR4 & v);  // output vector
void print_matrix(const MATRIX3 & mat,                  // print matrix
		  const int indent = 0,
		  const int cout_width = 6, const int precision = 1);
void print_matrix(const MATRIX4 & mat,                  // print matrix
		  const int indent = 0,
		  const int cout_width = 6, const int precision = 1);

#endif // _MATRIX_H
