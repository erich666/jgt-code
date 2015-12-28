#ifndef BITMASK_H
#define BITMASK_H

#include "Common/CommonDefs.h"
#include "Vec3i.h"

namespace CGLA
{
	const int MASKS[33] = 
		{
			0x00000000,
			0x00000001,
			0x00000003,
			0x00000007,
			0x0000000f,
			0x0000001f,
			0x0000003f,
			0x0000007f,
			0x000000ff,
			0x000001ff,
			0x000003ff,
			0x000007ff,
			0x00000fff,
			0x00001fff,
			0x00003fff,
			0x00007fff,
			0x0000ffff,
			0x0001ffff,
			0x0003ffff,
			0x0007ffff,
			0x000fffff,
			0x001fffff,
			0x003fffff,
			0x007fffff,
			0x00ffffff,
			0x01ffffff,
			0x03ffffff,
			0x07ffffff,
			0x0fffffff,
			0x1fffffff,
			0x3fffffff,
			0x7fffffff,
			0xffffffff
		};

	/** The BitMask class is mostly a utility class.
			The main purpose is to be able to extract a set of bits from
			an integer. For instance this can be useful if we traverse
			some tree structure and the integer is the index. */
	class BitMask
	{
		int fb, lb, bdiff;
		int msk;

	public:

		/** Mask _fb-_lb+1 bits beginning from _fb. First bit is 0.
				Say _fb=_lb=0. In this case, mask 1 bit namely 0.*/
		BitMask(int _fb, int _lb):
			fb(_fb), lb(_lb), 
			bdiff(lb-fb+1), msk(MASKS[bdiff]<<fb)
		{}
	
		/// first bit is 0 mask num bits.
		BitMask(int num):
			fb(0),lb(CMN::two_to_what_power(num)-1), 
			bdiff(lb-fb+1), msk(MASKS[bdiff]<<fb)
		{}
	
		/// Mask everything.
		BitMask():
			fb(0), lb(15),
			bdiff(lb-fb+1), msk(MASKS[bdiff]<<fb)
		{}

		/// get number of first bit in mask
		int first_bit() const {return fb;} 

		/// get number of last bit in mask
 		int last_bit() const {return lb;}

		/// Return number of masked bits
		int no_bits() const {return bdiff;}

		/// Mask a number
		int mask(int var) const {return msk&var;}
		
		/** Mask a number and shift back so the first bit inside
				the mask becomes bit 0. */
		int mask_shift(int var) const {return (msk&var)>>fb;}

		/** Mask a vector by masking each coordinate. */
		Vec3i mask(const Vec3i& v) const 
		{
			return Vec3i(mask(v[0]),mask(v[1]),mask(v[2]));
		}
  
		/** Mask each coord of a vector and shift */
		Vec3i maskshift(const Vec3i& v) const 
		{
			return Vec3i(mask_shift(v[0]),mask_shift(v[1]),mask_shift(v[2]));
		}
  
	};

}
#endif
