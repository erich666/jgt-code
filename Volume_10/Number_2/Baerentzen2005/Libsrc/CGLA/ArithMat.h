#ifndef __ARITHMAT_H
#define __ARITHMAT_H

#include <vector>
#include <iostream>
#include <algorithm>

namespace CGLA {

	/** Basic class template for matrices.
		
	In this template a matrix is defined as an array of vectors. This may
	not in all cases be the most efficient but it has the advantage that 
	it is possible to use the double subscripting notation:
		
	T x = m[i][j]

	This template should be used through inheritance just like the 
	vector template */
	template <class VVT, class HVT, class MT, int ROWS>
	class ArithMat
	{ 
#define for_all_i(expr) for(int i=0;i<ROWS;i++) {expr;}

	public:

		/// The type of a matrix element
		typedef typename HVT::ScalarType ScalarType;

	protected:

		/// The actual contents of the matrix.
		HVT data[ROWS];

	protected:

		/// Construct 0 matrix
		ArithMat() 
		{
			for_all_i(data[i]=HVT(ScalarType(0)));
		}

		/// Construct a matrix where all entries are the same.
		explicit ArithMat(ScalarType x)
		{
			for_all_i(data[i] = HVT(x));
		}

		/// Construct a matrix where all rows are the same.
		explicit ArithMat(HVT _a)
		{
			for_all_i(data[i] = _a);
		}


		/// Construct a matrix with two rows.
		ArithMat(HVT _a, HVT _b)
		{
			assert(ROWS==2);
			data[0] = _a;
			data[1] = _b;
		}

		/// Construct a matrix with three rows.
		ArithMat(HVT _a, HVT _b, HVT _c)
		{
			assert(ROWS==3);
			data[0] = _a;
			data[1] = _b;
			data[2] = _c;
		}

		/// Construct a matrix with four rows.
		ArithMat(HVT _a, HVT _b, HVT _c, HVT _d)
		{
			assert(ROWS==4);
			data[0] = _a;
			data[1] = _b;
			data[2] = _c;
			data[3] = _d;
		}
		
	public:

		/// Get vertical dimension of matrix 
		static int get_v_dim() {return VVT::get_dim();}

		/// Get horizontal dimension of matrix
		static int get_h_dim() {return HVT::get_dim();}


		/** Get const pointer to data array.
				This function may be useful when interfacing with some other API 
				such as OpenGL (TM). */
		const ScalarType* get() const 
		{
			return data[0].get();
		}

		/** Get pointer to data array.
				This function may be useful when interfacing with some other API 
				such as OpenGL (TM). */
		ScalarType* get()
		{
			return data[0].get();
		}

		/** Set values by passing an array to the matrix.
				The values should be ordered like [[row][row]...[row]] */
		void set(const ScalarType* sa) 
		{
			memcpy(get(), sa, sizeof(ScalarType)*get_h_dim()*get_v_dim());
		}

		/// Construct a matrix from an array of scalar values.
		explicit ArithMat(const ScalarType* sa) 
		{
			set(sa);
		}

		/// Assign the rows of a 2D matrix.
		void set(HVT _a, HVT _b)
		{
			assert(ROWS==2);
			data[0] = _a;
			data[1] = _b;
		}

		/// Assign the rows of a 3D matrix.
		void set(HVT _a, HVT _b, HVT _c)
		{
			assert(ROWS==3);
			data[0] = _a;
			data[1] = _b;
			data[2] = _c;
		}

		/// Assign the rows of a 4D matrix.
		void set(HVT _a, HVT _b, HVT _c, HVT _d)
		{
			assert(ROWS==4);
			data[0] = _a;
			data[1] = _b;
			data[2] = _c;
			data[3] = _d;
		}


		//----------------------------------------------------------------------
		// index operators
		//----------------------------------------------------------------------

		/// Const index operator. Returns i'th row of matrix.
		const HVT& operator [] ( int i ) const
		{
			assert(i<ROWS);
			return data[i];
		}

		/// Non-const index operator. Returns i'th row of matrix.
		HVT& operator [] ( int i ) 
		{
			assert(i<ROWS);
			return data[i];
		}

		//----------------------------------------------------------------------

		/// Equality operator. 
		bool operator==(const MT& v) const 
		{
			for_all_i(if (data[i] != v[i]) return false)
				return true;
		}

		/// Inequality operator.
		bool operator!=(const MT& v) const 
		{
			return !(*this==v);
		}

		//----------------------------------------------------------------------

		/// Multiply scalar onto matrix. All entries are multiplied by scalar.
		const MT operator * (ScalarType k) const
		{
			MT v_new;
			for_all_i(v_new[i] = data[i] * k);
			return v_new;
		}

		/// Divide all entries in matrix by scalar.
		const MT operator / (ScalarType k) const
		{
			MT v_new;
			for_all_i(v_new[i] = data[i] / k);
			return v_new;      
		}

		/// Assignment multiplication of matrix by scalar.
		const MT& operator *=(ScalarType k) 
			{
				for_all_i(data[i] *= k); 
				return static_cast<const MT&>(*this);
			}

		/// Assignment division of matrix by scalar.
		const MT& operator /=(ScalarType k) 
			{ 
				for_all_i(data[i] /= k); 
				return static_cast<const MT&>(*this);
			}

		//----------------------------------------------------------------------

		/// Add two matrices. 
		const MT operator + (const MT& m1) const
		{
			MT v_new;
			for_all_i(v_new[i] = data[i] + m1[i]);
			return v_new;
		}

		/// Subtract two matrices.
		const MT operator - (const MT& m1) const
		{
			MT v_new;
			for_all_i(v_new[i] = data[i] - v1[i]);
			return v_new;
		}

		/// Assigment addition of matrices.
		const MT& operator +=(const MT& v) 
			{
				for_all_i(data[i] += v[i]); 
				return static_cast<const MT&>(*this);
			}

		/// Assigment subtraction of matrices.
		const MT& operator -=(const MT& v) 
			{
				for_all_i(data[i] -= v[i]); 
				return static_cast<const MT&>(*this);
			}

		//----------------------------------------------------------------------

		/// Negate matrix.
		const MT operator - () const
		{
			MT v_new;
			for_all_i(v_new[i] = - data[i]);
			return v_new;
		}

#undef for_all_i  
 
	};

	/// Multiply scalar onto matrix
	template <class VVT, class HVT, class MT, int ROWS>
	inline const MT operator * (double k, const ArithMat<VVT,HVT,MT,ROWS>& v) 
	{
		return v * k;
	}

	/// Multiply scalar onto matrix
	template <class VVT, class HVT, class MT, int ROWS>
	inline const MT operator * (float k, const ArithMat<VVT,HVT,MT,ROWS>& v) 
	{
		return v * k;
	}

	/// Multiply scalar onto matrix
	template <class VVT, class HVT, class MT, int ROWS>
	inline const MT operator * (int k, const ArithMat<VVT,HVT,MT,ROWS>& v) 
	{
		return v * k;
	}

	/// Multiply vector onto matrix 
	template <class VVT, class HVT, class MT, int ROWS>
	inline VVT operator*(const ArithMat<VVT,HVT,MT,ROWS>& m,const HVT& v) 
	{
		VVT v2;
		for(int i=0;i<ROWS;i++) v2[i] = dot(m[i], v);
		return v2;
	}


#ifndef WIN32
	/** Multiply two arbitrary matrices. 
			In principle, this function could return a matrix, but in general
			the new matrix will be of a type that is different from either of
			the two matrices that are multiplied together. We do not want to 
			return an ArithMat - so it seems best to let the return value be
			a reference arg.
		
			This template can only be instantiated if the dimensions of the
			matrices match -- i.e. if the multiplication can actually be
			carried out. This is more type safe than the win32 version below.
	*/

	template <class VVT, class HVT, 
						class HV1T, class VV2T,
						class MT1, class MT2, class MT,
						int ROWS1, int ROWS2>
	inline void mul(const ArithMat<VVT,HV1T,MT1,ROWS1>& m1,
									const ArithMat<VV2T,HVT,MT2,ROWS2>& m2,
									ArithMat<VVT,HVT,MT,ROWS1>& m)
	{
		int cols = ArithMat<VVT,HVT,MT,ROWS1>::get_h_dim();
		for(int i=0;i<ROWS1;i++)
			for(int j=0;j<cols;j++)
				for(int k=0;k<ROWS2;k++)
					m[i][j] += m1[i][k] * m2[k][j]; 
	}


	/** Transpose. See the discussion on mul if you are curious as to why
			I don't simply return the transpose. */
	template <class VVT, class HVT, class M1T, class M2T, int ROWS, int COLS>
	inline void transpose(const ArithMat<VVT,HVT,M1T,ROWS>& m,
												ArithMat<HVT,VVT,M2T,COLS>& m_new)
	{
		for(int i=0;i<M2T::get_v_dim();i++)
			for(int j=0;j<M2T::get_h_dim();j++)
				m_new[i][j] = m[j][i];
	}

#else

	//----------------- win32 -------------------------------
	// Visual studio is not good at deducing the args. to these template functions.
	// This means that you can call the two functions below with 
	// matrices of wrong dimension.

	template <class M1, class M2, class M>
	inline void mul(const M1& m1, const M2& m2, M& m)
	{
		int cols = M::get_h_dim();
		int rows1 = M1::get_v_dim();
		int rows2 = M2::get_v_dim();

		for(int i=0;i<rows1;i++)
			for(int j=0;j<cols;j++)
				for(int k=0;k<rows2;k++)
					m[i][j] += m1[i][k] * m2[k][j];
	}


	/** Transpose. See the discussion on mul if you are curious as to why
			I don't simply return the transpose. */
	template <class M1, class M2>
	inline void transpose(const M1& m1, M2& m2)
	{
		for(int i=0;i<M2::get_v_dim();i++)
			for(int j=0;j<M2::get_h_dim();j++)
				m2[i][j] = m1[j][i];
	}

#endif


	/** Put to operator */
	template <class VVT, class HVT, class MT, int ROWS>
	inline std::ostream& 
	operator<<(std::ostream&os, const ArithMat<VVT,HVT,MT,ROWS>& m)
	{
		os << "[\n";
		for(int i=0;i<ROWS;i++) os << "  " << m[i] << "\n";
		os << "]\n";
		return os;
	}

	/** Get from operator */
	template <class VVT, class HVT, class MT, int ROWS>
	inline std::istream& operator>>(std::istream&is, 
																	const ArithMat<VVT,HVT,MT,ROWS>& m)
	{
		for(int i=0;i<ROWS;i++) is>>m[i];
		return is;
	}


}
#endif
