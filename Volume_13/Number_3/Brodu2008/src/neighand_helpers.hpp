/*
    Neighand: Neighborhood Handling library

    The goal of this project is to find 3D neighbors efficiently.
    Read the documentation contained in the article "Query Sphere
    Indexing for Neighborhood Requests" for details.

    This file defines helper utilities:
    - A function for performing fast floating-point modulo arithmetic.
    - A function for performing fast floating-point truncation to -infinity.

    It also includes the helpers that are specialized for cyclic or
    non-cyclic worlds (or cyclic only in some dimensions).

    Nicolas Brodu, 2006/7
    Code released according to the GNU LGPL, v2 or above.
*/

#ifndef NEIGHAND_HELPERS_HPP
#define NEIGHAND_HELPERS_HPP

// Bit-hacking to get a fast remainder function when the divisor is a power of two
// With neither mul or div: Only a float add, a float sub, and integer shift & adds
// For non-power-of-two divisors, you'd need to scale to such a number and scale
// back afterwards. That's two more float muls, but still faster than a div.
// Check the tttf wrap helper for details.
//
// This routine relies on IEEE754 float representation on 4 bytes.
// This is faster than system scalnbf (let alone remainder) because:
// - We don't care for infinities
// - We don't care for subnormals (flush to 0)
// - Branchless
// - Inline in very few ops
// As for the rest of this project, -fno-strict-aliasing is necessary for correct results
// Note: Unlike remainder the integer multiple of 2^exp2 is not rounded to nearest even number,
//       but consistently toward the same sign infinity.
//       ex with exp2=3 => 8.0f: Number    fastRem    remainderf
//                               12        -4         -4
//                               4         -4         4
//                               -4        4          -4
//                               -12       4          4
//                               -20       4          -4
//                               -28       4          4
// Normal numbers other than midpoints give the same results with both functions.
// For our use, computing d^2 with d the wrapped up distance, the sign is cancelled out anyway.
// For other usage, the fastRem may even be better as it's more consistent.
template <int exp2> NEIGHAND_INLINE FloatType fastExp2Rem(FloatConverter x) NEIGHAND_ALWAYS_INLINE;
template <int exp2> NEIGHAND_INLINE FloatType fastExp2Rem(FloatConverter x) {

    // First compute the x/2^exp2 division, without dividing of course!
    FloatConverter xOver2exp2(
        // subtracts exp2 from exponent part, no influence on mantissa for our exp2 range
        (x.i - (exp2<<23))
        // mask with 0x0000000 if result would be subnormal, 0xFFFFFFFF if normal
        // >> is guaranteed to insert 0 on left with uint32_t
        // generate the bitmask from the sign of the exponent subtraction result
        & ((uint32_t((x.i & 0x7f800000) - (exp2<<23)) >> 31) - 1)
        );

    // +- 0.5f depending on the sign of x
    FloatConverter half( (x.i & 0x80000000) | 0x3F000000 );

    // Now get the remainder
    return
        // remainder is x - n*y, with y = 2^exp2 and n integer from the value just computed
        // This just needs a float sub and an add, everything else was done bitwise
        x.f - (
            // conversion from float to signed int
            (int32_t)(
            // previous value as float
            xOver2exp2.f
            // C truncates to 0, need to add/subtract 0.5 depending on sign for remainder
            + half.f
            )
            // multiply back by 2^exp2
            << exp2
        );
}

// floor(x) is int(x) if x>0, or int(x)-1 if x<0, so in any case int(x)-signbit, >> inserts 0 for unsigned numbers
// Works only for numbers with absolute value below 2^23.
NEIGHAND_INLINE int32_t fastFloorInt(FloatConverter f) NEIGHAND_ALWAYS_INLINE;
NEIGHAND_INLINE int32_t fastFloorInt(FloatConverter f) {
    return int32_t(f.f) - uint32_t(f.i >> 31);
}

// For convenience only, technically fastFloorInt would be enough
NEIGHAND_INLINE FloatType fastFloor(FloatConverter f) NEIGHAND_ALWAYS_INLINE;
NEIGHAND_INLINE FloatType fastFloor(FloatConverter f) {
    return int32_t(int32_t(f.f) - uint32_t(f.i >> 31));
}


template <typename UserObject, int exp2divx, int exp2divy, int exp2divz, bool wrapX, bool wrapY, bool wrapZ, bool layerZ, class _Allocator >
struct WrapHelper {};

// Specializations
#include "neighand_wraphelper_tttf.hpp"
#include "neighand_wraphelper_ffff.hpp"
#include "neighand_wraphelper_ttff.hpp"


#endif
