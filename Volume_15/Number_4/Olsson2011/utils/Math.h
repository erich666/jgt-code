/****************************************************************************/
/* Copyright (c) 2011, Ola Olsson
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
/****************************************************************************/
#ifndef _chag_Math_h_
#define _chag_Math_h_

#include "IntTypes.h"


namespace chag
{
const float g_pi = 3.1415926535897932384626433832795f;

  
template <typename T>
T min(const T &a, const T &b)
{
  return a < b ? a : b;
}



template <typename T>
T max(const T &a, const T &b)
{
  return a > b ? a : b;
}



template <typename T>
void swap(T &a, T &b)
{
  T tmp(a);
  a = b;
  b = tmp;
}



template <typename T>
const T clamp(const T &value, const T &lowerInc, const T &upperInc)
{
  return min(max(value, lowerInc), upperInc);
}



template <typename T>
const T square(T a)
{
  return a * a;
}



template <typename T>
inline T getNearestHigherMultiple(const T count, const T multipleOf)
{
  return ((count + multipleOf - T(1)) / multipleOf) * multipleOf;
}



}; // namespace chag

#endif // _chag_Math_h_
