#ifndef __ARITHSQMAT_H
#define __ARITHSQMAT_H

#include "ArithMat.h"

namespace CGLA {

	/** Template for square matrices.
			Some functions like trace and determinant work only on
			square matrices. To express this in the class hierarchy,
			ArithSqMat was created. ArithSqMat is derived from ArithMat
			and contains a few extra facilities applicable only to
			square matrices.
	*/
	template <class VT, class MT, int ROWS>
	class ArithSqMat: public ArithMat<VT,VT,MT,ROWS> 
	{ 
	public:
		/// The type of a matrix element
		typedef typename VT::ScalarType ScalarType;

	protected:

		/// Construct 0 matrix
		ArithSqMat() {}

		/// Construct matrix where all values are equal to constructor argument.
		explicit ArithSqMat(ScalarType _a):
			ArithMat<VT,VT,MT,ROWS>(_a) {}

		/// Construct 2x2 Matrix from two vectors
		ArithSqMat(VT _a, VT _b): 
			ArithMat<VT,VT,MT,ROWS>(_a,_b) {}

		/// Construct 3x3 Matrix from three vectors
		ArithSqMat(VT _a, VT _b, VT _c): 
			ArithMat<VT,VT,MT,ROWS>(_a,_b,_c) {}

		/// Construct 4x4 Matrix from four vectors
		ArithSqMat(VT _a, VT _b, VT _c, VT _d): 
			ArithMat<VT,VT,MT,ROWS>(_a,_b,_c,_d) {}

		/// Construct matrix from array of values.
		explicit ArithSqMat(const ScalarType* sa) {set(sa);}
		
	public:

		/** Assignment multiplication of matrices. 
				This function is not very efficient. This because we need a temporary
				matrix anyway, so it can't really be made efficient. */
		const MT& operator*=(const MT& m2)
			{
				(*this) = (*this) * m2;
				return static_cast<const MT&>(*this);
			}

		
	};

	/** Multiply two matrices derived from same type, 
			producing a new of same type */
	template <class VT, class MT, int ROWS>
	inline MT operator*(const ArithSqMat<VT,MT,ROWS>& m1,
											const ArithSqMat<VT,MT,ROWS>& m2) 
	{
		MT n;
		for(int i=0;i<ROWS;i++)
			for(int j=0;j<ROWS;j++)
				for(int k=0;k<ROWS;k++)
					n[i][j] += m1[i][k] * m2[k][j]; 
		return n;
	}

	/** Multiply two matrices derived from same type, 
			producing a new of same type */
	template <class VT, class MT, int ROWS>
	inline MT transpose(const ArithSqMat<VT,MT,ROWS>& m) 
	{
		MT m_new;
		for(int i=0;i<MT::get_v_dim();i++)
			for(int j=0;j<MT::get_h_dim();j++)
				m_new[i][j] = m[j][i];
		return m_new;
	}



	/// Compute trace. Works only for sq. matrices.
	template <class VT, class MT, int ROWS>
	inline typename MT::ScalarType trace(const ArithSqMat<VT,MT,ROWS>& M)
	{
		typename ArithSqMat<VT,MT,ROWS>::ScalarType s=0;
		for(int i=0;i<ROWS;i++)
			s += M[i][i];
		return s;
	}

}
#endif
