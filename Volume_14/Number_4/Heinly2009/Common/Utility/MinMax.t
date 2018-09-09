
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

#ifndef Common_MinMax_t
#define Common_MinMax_t

namespace Utility
{

  //////////////////////////////////////////////////////////////////////////////
  // Min/Max functions

  template<typename T>
  inline T Min(const T& x0, const T& x1)
  {
    return (x0 < x1 ? x0 : x1);
  }

  template<typename T>
  inline T Min(const T& x0, const T& x1, const T& x2)
  {
    return Min(x0, Min(x1, x2));
  }

  template<typename T>
  inline T Min(const T& x0, const T& x1, const T& x2, const T& x3)
  {
    return Min(Min(x0, x1), Min(x2, x3));
  }

  template<typename T>
  inline T Max(const T& x0, const T& x1)
  {
    return (x0 > x1 ? x0 : x1);
  }

  template<typename T>
  inline T Max(const T& x0, const T& x1, const T& x2)
  {
    return Max(x0, Max(x1, x2));
  }

  template<typename T>
  inline T Max(const T& x0, const T& x1, const T& x2, const T& x3)
  {
    return Max(Max(x0, x1), Max(x2, x3));
  }

} // namespace Utility

#endif // Common_MinMax_t
