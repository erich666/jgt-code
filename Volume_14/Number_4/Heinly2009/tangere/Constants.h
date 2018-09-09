
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

#ifndef tangere_Constants_h
#define tangere_Constants_h

#include <Common/Types.h>

#include <Math/FixedPoint.h>

#include <tangere/Flags.h>

namespace tangere
{

#if defined(USE_D_BITS)
  static const uint cBits       = 15;
  static const uint cdDiff      = 6;
  static const uint cdDiffShift = cdDiff - 2;
  static const uint dBits       = cBits - cdDiff;
#else
  static const uint cBits       = 12;
  static const uint dBits       = cBits;
#endif // defined(USE_D_BITS)
  static const uint sideShift   = 29;
  static const uint M           = 32;
  static const uint mMinusOne   = M - 1;

  template<typename T>
  struct Constants
  {
    static const T kd;
    static const T ka;
  };

  ///////////////////////////////////////////////////////////////////////////////
  // Helper function + specializations for Constants<int>

  template<typename T>
  const T kdHelper()
  {
    return T(0.6f);
  }

  template<typename T>
  const T kaHelper()
  {
    return T(0.4f);
  }

  template<>
  inline const int kdHelper<int>()
  {
    return FLOAT_TO_FIXED16(kdHelper<float>());
  }

  template<>
  inline const int kaHelper<int>()
  {
    return FLOAT_TO_FIXED16(kaHelper<float>());
  }

  ///////////////////////////////////////////////////////////////////////////////
  // Static member definitions

  template<typename T>
  const T Constants<T>::kd(kdHelper<T>());

  template<typename T>
  const T Constants<T>::ka(kaHelper<T>());

} // namespace tangere

#endif // tangere_Constants_h
