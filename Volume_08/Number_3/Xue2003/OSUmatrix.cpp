// vector and matrix class and operations
// Created by Matt Camuto
// Modified by R. Wenger: 07-09-99

#include <iomanip>
#include <math.h>
#include "OSUmatrix.h"

void VECTOR3::Normalize()
// normalize vector
// Assumes (pre-condition): vector != (0,0,0)
{
  float norm = sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);

  for (int i = 0; i < Dimension(); i++)
    vec[i] = vec[i]/norm;
}

void VECTOR4::Normalize()
// normalize vector
// Assumes (pre-condition): vector != (0,0,0,0)
{
  float norm = sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);

  for (int i = 0; i < Dimension(); i++)
    vec[i] = vec[i]/norm;
}

void MATRIX3::Identity()
// set matrix to identity matrix
{
  for (int i = 0; i < Dimension(); i++) {
    mat[i].Zero();
    mat[i][i] = 1.0;
  };
}

void MATRIX4::Identity()
// set matrix to identity matrix
{
  for (int i = 0; i < Dimension(); i++) {
    mat[i].Zero();
    mat[i][i] = 1.0;
  };
}

//************************
// MATRIX3 operations
//************************

MATRIX3 operator +(const MATRIX3 & m0, const MATRIX3 & m1)
// return m0 + m1
{
  MATRIX3 result;

  result[0] = m0(0) + m1(0);
  result[1] = m0(1) + m1(1);
  result[2] = m0(2) + m1(2);

  return(result);
}

MATRIX3 operator -(const MATRIX3 & m0, const MATRIX3 & m1)
// return m0 - m1
{
  MATRIX3 result;

  result[0] = m0(0) - m1(0);
  result[1] = m0(1) - m1(1);
  result[2] = m0(2) - m1(2);

  return(result);
}

MATRIX3 operator *(const MATRIX3 & m0, const MATRIX3 & m1)
// return m0 * m1
{
  MATRIX3 result;

  for (int i = 0; i < m0.Dimension(); i++)
    for (int j = 0; j < m0.Dimension(); j++) {
      result[i][j] = 0;
      for (int k = 0; k < m0.Dimension(); k++)
	result[i][j] += m0(i,k) * m1(k,j);
    };

  return(result);
}

MATRIX3 operator *(const SCALAR x0, const MATRIX3 & m0)
// return x0*m0
{
  MATRIX3 result;

  result[0] = x0*m0(0);
  result[1] = x0*m0(1);
  result[2] = x0*m0(2);

  return(result);
}

MATRIX3 operator *(const MATRIX3 & m0, const SCALAR x0)
// return m0*x0
{ return(x0*m0); };

VECTOR3 operator *(const MATRIX3 & m0, const VECTOR3 & v0)
// return m0 * v0
{
  VECTOR3 result;

  result[0] = dot(m0(0),v0);
  result[1] = dot(m0(1),v0);
  result[2] = dot(m0(2),v0);

  return(result);
}

VECTOR3 operator *(const VECTOR3 & v0, const MATRIX3 & m0)
// return v0 * m0
{
  VECTOR3 result;

  result[0] = v0(0)*m0(0,0) + v0(1)*m0(1,0) + v0(2)*m0(2,0);
  result[1] = v0(0)*m0(0,1) + v0(1)*m0(1,1) + v0(2)*m0(2,1);
  result[2] = v0(0)*m0(0,2) + v0(1)*m0(1,2) + v0(2)*m0(2,2);

  return(result);
}

//************************
// MATRIX4 operations
//************************

MATRIX4 operator +(const MATRIX4 & m0, const MATRIX4 & m1)
// return m0 + m1
{
  MATRIX4 result;

  result[0] = m0(0) + m1(0);
  result[1] = m0(1) + m1(1);
  result[2] = m0(2) + m1(2);
  result[3] = m0(3) + m1(3);

  return(result);
}

MATRIX4 operator -(const MATRIX4 & m0, const MATRIX4 & m1)
// return m0 - m1
{
  MATRIX4 result;

  result[0] = m0(0) - m1(0);
  result[1] = m0(1) - m1(1);
  result[2] = m0(2) - m1(2);
  result[3] = m0(3) - m1(3);

  return(result);
}

MATRIX4 operator *(const MATRIX4 & m0, const MATRIX4 & m1)
// return m0 * m1
{
  MATRIX4 result;

  for (int i = 0; i < m0.Dimension(); i++)
    for (int j = 0; j < m0.Dimension(); j++) {
      result[i][j] = 0;
      for (int k = 0; k < m0.Dimension(); k++)
	result[i][j] += m0(i,k) * m1(k,j);
    };

  return(result);
}

MATRIX4 operator *(const SCALAR x0, const MATRIX4 & m0)
// return x0*m0
{
  MATRIX4 result;

  result[0] = x0*m0(0);
  result[1] = x0*m0(1);
  result[2] = x0*m0(2);
  result[3] = x0*m0(3);

  return(result);
}

MATRIX4 operator *(const MATRIX4 & m0, const SCALAR x0)
// return m0*x0
{ return(x0*m0); };

VECTOR4 operator *(const MATRIX4 & m0, const VECTOR4 & v0)
// return m0 * v0
{
  VECTOR4 result;

  result[0] = dot(m0(0),v0);
  result[1] = dot(m0(1),v0);
  result[2] = dot(m0(2),v0);
  result[3] = dot(m0(3),v0);

  return(result);
}

VECTOR4 operator *(const VECTOR4 & v0, const MATRIX4 & m0)
// return v0 * m0
{
  VECTOR4 result;

  result[0] = v0(0)*m0(0,0) + v0(1)*m0(1,0) + v0(2)*m0(2,0);
  result[1] = v0(0)*m0(0,1) + v0(1)*m0(1,1) + v0(2)*m0(2,1);
  result[2] = v0(0)*m0(0,2) + v0(1)*m0(1,2) + v0(2)*m0(2,2);
  result[3] = v0(0)*m0(0,3) + v0(1)*m0(1,3) + v0(2)*m0(2,3);

  return(result);
}

//Code was taken from the original 'edge' library written by dave ebert.
MATRIX4 inverse(const MATRIX4 & m) {
  register int lp,i,j,k;
  static double wrk[4][8];
  static double a, b;
  MATRIX4 result;
  
  for( i=0; i<4; i++ )	/* Set up matrices */
    {
      for( j=0; j<4; j++ )
	{
	  wrk[i][j]=(double)m(i,j);
	  wrk[i][j+4]=0.0;

          result[i][j] = 0.0;
	}
      wrk[i][i+4]=1.0;
    }
  
  for( lp=0; lp<4; lp++ )	/* Loop over all rows */
    {
      a=0.0;
      j=(-1);
      for( i=lp; i<4; i++ )	/* Find largest non-zero element */
	{
	  b=wrk[i][lp];
	  if( b< 0.0 )
	    b=(-b);
	  if( b>a )
	    {
	      a=b;
	      j=i;
	    }
	}
      if( j!=lp )			/* If not on diagonal, put it there */
	{
	  if( j<0 )		/* Singular if none found */
	    return(result);
	  else			/* Exchange rows from lp to end */
	    for( k=lp; k<8; k++ )
	      {
		a=wrk[j][k];
		wrk[j][k]=wrk[lp][k];
		wrk[lp][k]=a;
	      }
	}
      a=wrk[lp][lp];		/* Normalize working row */
      for( i=lp; i<8; i++ )
	wrk[lp][i]/=a;
      
      for( i=lp+1; i<8; i++ )  /* Adjust rest of work space */
	{
	  b=wrk[lp][i];
	  for( j=0; j<4; j++ )	/* One column at a time */
	    if( j!=lp )
	      wrk[j][i]-=wrk[j][lp]*b;
	}
    }

  for( i=0; i<4; i++ )	/* Return result matrix */
    for( j=0; j<4; j++ )
      result[i][j]=(float)wrk[i][j+4];
  return(result);
}

//*******************************
// output procedures/operators
//*******************************

ostream & operator<<(ostream & os, const VECTOR3 & v)
// output vector
{
  os << "( " << v(0) << " " << v(1) << " " << v(2) << " )";
  return(os);
}

ostream & operator<<(ostream & os, const VECTOR4 & v)
// output vector
{
  os << "( " << v(0) << " " << v(1) << " " << v(2) << " " << v(3) << " )";
  return(os);
}

void print_matrix(const MATRIX3 & mat,
		  const int indent, const int cout_width, const int precision)
// routine to print out matrix.  Useful for debugging.
//   call at the beginning of a new line for proper formatting
// mat = matrix
// indent = # of indented spaces
// cout_width = output field width
// precision = output precision
{
  int i, j;
  cout.precision(precision);
  cout.flags(ios::fixed);
  for (i = 0; i < mat.Dimension(); i++) {
    for (j = 0; j < indent; j++)
      cout << " ";
    cout << "( ";
    for (j = 0; j < mat.Dimension(); j++)
      cout << setw(cout_width) << mat(i,j) << " ";
    cout << ")" << endl;
  }
}

