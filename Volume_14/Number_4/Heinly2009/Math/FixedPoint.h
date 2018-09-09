
/*
 *  Copyright 2009, 2010 Grove City College
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef Math_FixedPoint_h
#define Math_FixedPoint_h

#include <climits>

#include <Common/Types.h>

////////////////////////////////////////////////////////////////////////////////
// Constant values

#define POW_2_31 0x80000000
#define POW_2_16 0x00010000

////////////////////////////////////////////////////////////////////////////////
// Shift down and include rounding

#define SHIFT_DOWN_ROUND(a, shift) ((((a) >> ((shift) - 1)) + 1) >> 1)

////////////////////////////////////////////////////////////////////////////////
// Perform multiplication with a shift

#define MUL(a, b, shift) SHIFT_DOWN_ROUND(static_cast<int64_t>(a) * static_cast<int64_t>(b), (shift))
#define MUL16(a, b) static_cast<int>(MUL(a, b, 16))
#define MUL31(a, b) static_cast<int>(MUL(a, b, 31))
#define MUL16_64(a, b) MUL(a, b, 16)
#define MUL31_64(a, b) MUL(a, b, 31)

////////////////////////////////////////////////////////////////////////////////
// Perform division with a shift

#define DIV(a, b, shift) ((static_cast<int64_t>(a) << (shift)) / static_cast<int64_t>(b))
#define DIV16(a, b) static_cast<int>(DIV(a, b, 16))
#define DIV31(a, b) static_cast<int>(DIV(a, b, 31))
#define DIV16_64(a, b) DIV(a, b, 16)
#define DIV31_64(a, b) DIV(a, b, 31)

////////////////////////////////////////////////////////////////////////////////
// Conversions

#define FIXED31_TO_FLOAT(a) (static_cast<float>(a) / POW_2_31)
// Cast to double so that there are enough bits in the mantissa to represent
// a 32-bit number (to which we are converting)
#define FLOAT_TO_FIXED31(a) static_cast<int>(static_cast<double>(a) * INT_MAX)

#define FIXED16_TO_FLOAT(a) (static_cast<float>(a) / POW_2_16)
#define FLOAT_TO_FIXED16(a) static_cast<int>((a) * POW_2_16 + 0.5f)

#endif // Math_FixedPoint_h
