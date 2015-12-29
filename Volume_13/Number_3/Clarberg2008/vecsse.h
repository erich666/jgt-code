/* 
 * Contains simple wrapper and vector classes using SSE and unions to provide
 * access to the individual elements. Many of the functions are provided solely
 * for convenience, and should not be used in performance critical code.
 *
 * Written by Petrik Clarberg <petrik@cs.lth.se>, Lund University, 2007-2008.
 * This code is released as public domain for use free of charge for any
 * purpose, but without any kind of warranty.
 */
#ifndef __VECSSE_H__
#define __VECSSE_H__

#include <cmath>
#include <iostream>
#include <iomanip>

#include "vec.h"


// Include the highest level of SSEx header enabled.
#if defined(__SSE2__)
#include <emmintrin.h>		// SSE2
#elif defined(__SSE__)
#include <xmmintrin.h>		// SSE
#endif

// Portable align: gcc uses __attribute__
#ifndef __WIN32__
#define __align16 __attribute__((aligned(16)))
#else
#define __align16 __declspec(align(16) )
#endif

/// Returns true if the pointer p is aligned to a 16-byte boundary.
inline bool is_aligned16(void* p) { return ((int)p&0xF)==0; }


namespace mapping 
{
#ifdef __SSE__

	// Define a multiply-add (MADD) instruction for convenience, as it is not 
	// supported by SSE/SSE2. Returns a+b*c, where a,b,c are packed floats.
	#define _mm_madd_ps(a,b,c)		_mm_add_ps(a,_mm_mul_ps((b),(c)))

	
	//  ------------------------------------------------------------------------
	/// Wrapper for __m128 providing easy access to the four float elements,
	/// and some useful functions for initialization etc.
	//  ------------------------------------------------------------------------
	class f32_4
	{
	public:
		union
		{
			__m128 m;				///< Access as a single 128-bit type.
			float f[4];				///< Direct access as four floats f3...f0.
			unsigned int bits[4];	///< Direct access to bitwise representation.
		};
		
	public:
		f32_4() {}
		~f32_4() {}
		
		/// Initialize from __m128 type.
		inline f32_4(const __m128& _m) { _mm_store_ps(f,_m); }
		
		/// Overload assignment operator from __m128.
		inline f32_4& operator= (const __m128& _m) { _mm_store_ps(f,_m); return *this; }
		
		/// Overload cast to __m128 operator.
		inline operator __m128() const { return m; }
		
		/// Returns the float at position i=[0,3].
		inline float at(int i) const { return f[i]; }
		
		/// Initialize all four floats to the value s.
		inline explicit f32_4(float s) 
		{ 
			m = _mm_load_ss(&s);
			m = _mm_shuffle_ps(m, m, 0x00);
		}
		
		/// Explicit initialization from unsigned int (useful for bit masks).
		inline explicit f32_4(unsigned int u)
		{
			bits[0] = u;
			m = _mm_shuffle_ps(m, m, 0x00);
		}
		
		/// Initialize the four elements with the values f3...f0 (f0 at LSB).
		inline f32_4(float f3, float f2, float f1, float f0)
		{
			f[0]=f0; f[1]=f1; f[2]=f2; f[3]=f3;
		}
					
		/// Writes the four elements f3...f0 to an output stream.
		friend std::ostream& operator<< (std::ostream& os, const f32_4& rhs)
		{
			using namespace std;
			os << setprecision(6);
			os << setw(14) << rhs.f[3] << setw(14) << rhs.f[2] << setw(14) << rhs.f[1] << setw(14) << rhs.f[0];
			return os;
		}		
	}; // class f32_4
	
	
	
	//  ------------------------------------------------------------------------
	/// Vector class for 4x 2D vectors stored as two __m128 values, providing
	/// easy access to both the __m128 values (x,y) and their individual floats
	/// (fx[0..3],fy[0..3]).
	//  ------------------------------------------------------------------------
	class vec2f_4
	{
	public:
		union
		{
			__m128 x;			///< Access to the x-coord as a 128-bit XMM reg.
			float fx[4];		///< Direct access to its four floats fx3...fx0.
		};
		union
		{
			__m128 y;			///< Access to the y-coord as a 128-bit XMM reg.
			float fy[4];		///< Direct access to its four floats fy3...fy0.
		};
		
	public:
		vec2f_4() {}
		~vec2f_4() {}

		/// Sets the four 2D vectors to the elements of the vec2f v.
		inline explicit vec2f_4(const vec2f& v)
		{
			x = _mm_load_ss(&v.x);
			y = _mm_load_ss(&v.y);
			x = _mm_shuffle_ps(x, x, 0x00);
			y = _mm_shuffle_ps(y, y, 0x00);
		}

		/// Sets the (x,y) elements to the given values.
		inline vec2f_4(const __m128& _x, const __m128& _y)
		{
			_mm_store_ps(fx,_x);
			_mm_store_ps(fy,_y);
		}
			
		/// Returns the vec2f at position i=[0,3].
		inline vec2f at(int i) const
		{
			return vec2f(fx[i],fy[i]);
		}
			
		/// Sets the 2D vector at position i=[0,3] to v.
		inline void setAt(int i, const vec2f& v)
		{
			fx[i] = v.x;
			fy[i] = v.y;
		}
			
		/// Writes the four vectors' elements to an output stream.
		friend std::ostream& operator<< (std::ostream& os, const vec2f_4& rhs)
		{
			using namespace std;
			os << setprecision(6);
			os << setw(14) << rhs.fx[0] << setw(14) << rhs.fy[0] << endl;
			os << setw(14) << rhs.fx[1] << setw(14) << rhs.fy[1] << endl;
			os << setw(14) << rhs.fx[2] << setw(14) << rhs.fy[2] << endl;
			os << setw(14) << rhs.fx[3] << setw(14) << rhs.fy[3];
			return os;
		}
	}; // class vec2f_4
	
	
	
	//  ------------------------------------------------------------------------
	/// Vector class for 4x 3D vectors stored as three __m128 values, providing
	/// easy access to both the __m128 values (x,y,z) and their individual 
	/// floats (fx[0..3],fy[0..3],fz[0..3]).
	//  ------------------------------------------------------------------------
	class vec3f_4
	{
	public:
		union
		{
			__m128 x;			///< Access to the x-coord as a 128-bit XMM reg.
			float fx[4];		///< Direct access to its four floats fx3...fx0.
		};
		union
		{
			__m128 y;			///< Access to the y-coord as a 128-bit XMM reg.
			float fy[4];		///< Direct access to its four floats fy3...fy0.
		};
		union
		{
			__m128 z;			///< Access to the z-coord as a 128-bit XMM reg.
			float fz[4];		///< Direct access to its four floats fz3...fz0.
		};
		
	public:
		vec3f_4() {}
		~vec3f_4() {}
		
		/// Sets the four 3D vectors to the elements of the vec3f v.
		inline explicit vec3f_4(const vec3f& v)
		{
			x = _mm_load_ss(&v.x);
			y = _mm_load_ss(&v.y);
			z = _mm_load_ss(&v.z);
			x = _mm_shuffle_ps(x, x, 0x00);
			y = _mm_shuffle_ps(y, y, 0x00);
			z = _mm_shuffle_ps(z, z, 0x00);
		}
		
		/// Sets the (x,y,z) elements to the given values.
		inline vec3f_4(const __m128& _x, const __m128& _y, const __m128& _z)
		{
			_mm_store_ps(fx,_x);
			_mm_store_ps(fy,_y);
			_mm_store_ps(fz,_z);
		}
		
		/// Returns the vec3f at position i=[0,3].
		inline vec3f at(int i) const
		{
			return vec3f(fx[i],fy[i],fz[i]);
		}
		
		/// Sets the 3D vector at position i=[0,3] to v.
		inline void setAt(int i, const vec3f& v)
		{
			fx[i] = v.x;
			fy[i] = v.y;
			fz[i] = v.z;
		}
		
		/// Writes the four vectors' elements to an output stream.
		friend std::ostream& operator<< (std::ostream& os, const vec3f_4& rhs)
		{
			using namespace std;
			os << setprecision(6);
			os << setw(14) << rhs.fx[0] << setw(14) << rhs.fy[0] << setw(14) << rhs.fz[0] << endl;
			os << setw(14) << rhs.fx[1] << setw(14) << rhs.fy[1] << setw(14) << rhs.fz[0] << endl;
			os << setw(14) << rhs.fx[2] << setw(14) << rhs.fy[2] << setw(14) << rhs.fz[0] << endl;
			os << setw(14) << rhs.fx[3] << setw(14) << rhs.fy[3] << setw(14) << rhs.fz[0];
			return os;
		}
	}; // class vec3f_4
		
#endif // __SSE__


#ifdef __SSE2__
	
	//  ------------------------------------------------------------------------
	/// Wrapper for __m128i providing easy access to the four int32 elements,
	/// and some useful functions for initialization etc.
	//  ------------------------------------------------------------------------
	class int32_4
	{
	public:
		union
		{
			__m128i m;		///< Access as a single 128-bit integer type.
			int w[4];		///< Direct access as four int32 words w3...w0.
		};
		
	public:
		int32_4() {}
		~int32_4() {}
		
		/// Initialize from __m128i type.
		inline int32_4(const __m128i& _m) { _mm_store_si128(&m,_m); }
	
		/// Overload assignment operator from __m128i.
		inline int32_4& operator= (const __m128i& _m) { _mm_store_si128(&m,_m); return *this; }
		
		/// Overload cast to __m128i operator.
		inline operator __m128i() const { return m; }
		
		/// Returns the integer at position i=[0,3].
		inline int at(int i) const { return w[i]; }
		
		/// Initialize all four integers to the value _w.
		inline explicit int32_4(int _w) 
		{ 
			w[0] = _w;
			m = _mm_shuffle_epi32(m, 0x00);
		}
		
		/// Initialize the four elements with the values w3...w0 (w0 at LSB).
		inline int32_4(int w3, int w2, int w1, int w0)
		{
			w[0]=w0; w[1]=w1; w[2]=w2; w[3]=w3;
		}
		
		/// Writes the four integer elements to an output stream.
		/// The order is w3...w0, where w0 is the lowest dword (LSB).
		friend std::ostream& operator<< (std::ostream& os, const int32_4& rhs)
		{
			using namespace std;
			os << setw(14) << rhs.w[3] << setw(14) << rhs.w[2] << setw(14) << rhs.w[1] << setw(14) << rhs.w[0];
			return os;
		}
	}; // class int32_4
	
	

	//  ------------------------------------------------------------------------
	/// Wrapper for __m128i providing easy access as eight int16 elements,
	/// and some useful functions for initialization etc.
	//  ------------------------------------------------------------------------
	class int16_8
	{
	public:
		union
		{
			__m128i m;		///< Access as a single 128-bit integer type.
			short w[8];		///< Direct access as eight int16 words w3...w0.
		};
		
	public:
		int16_8() {}
		~int16_8() {}
		
		/// Initialize from __m128i type.
		inline int16_8(const __m128i& _m) { _mm_store_si128(&m,_m); }
		
		/// Overload assignment operator from __m128i.
		inline int16_8& operator= (const __m128i& _m) { _mm_store_si128(&m,_m); return *this; }
		
		/// Overload cast to __m128i operator.
		inline operator __m128i() const { return m; }
		
		/// Returns the word at position i=[0,7].
		inline short at(int i) const { return w[i]; }
		
		/// Initialize all eight words to the value _w.
		inline explicit int16_8(int _w) 
		{
			w[1] = w[0] = _w;
			m = _mm_shuffle_epi32(m, 0x00);
		}
		
		/// Initialize the eight words with the values w7...w0 (w0 at LSB).
		inline int16_8(short w7, short w6, short w5, short w4, 
					   short w3, short w2, short w1, short w0)
		{
			w[0]=w0; w[1]=w1; w[2]=w2; w[3]=w3;
			w[4]=w4; w[5]=w5; w[6]=w6; w[7]=w7;
		}
		
		/// Writes the eight words to an output stream.
		/// The order is w7...w0, where w0 is the lowest word (LSB).
		friend std::ostream& operator<< (std::ostream& os, const int16_8& rhs)
		{
			using namespace std;
			os << setw(9) << rhs.w[7] << setw(9) << rhs.w[6] << setw(9) << rhs.w[5] << setw(9) << rhs.w[4];
			os << setw(9) << rhs.w[3] << setw(9) << rhs.w[2] << setw(9) << rhs.w[1] << setw(9) << rhs.w[0];
			return os;
		}
	}; // class int16_8
	
#endif // __SSE2__
	
} // namespace mapping
#endif // __VECSSE_H__
