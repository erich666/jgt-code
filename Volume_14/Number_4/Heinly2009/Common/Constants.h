
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

#ifndef Common_Constants_h
#define Common_Constants_h

#include <cmath>

#ifdef WIN32
#  define M_PI 3.14159265358979323846f
#endif

static const float Pi               = M_PI;
static const float TwoPi            = 2.f*Pi;
static const float OneOverPi        = 1.f/Pi;
static const float DegreesToRadians = Pi/180.f;

template<typename T>
struct Constants
{
  static const T Epsilon;
};

/////////////////////////////////////////////////////////////////////////////////
// Helper function + specialization for Constants<int>

template<typename T>
const T EpsilonHelper()
{
  return T(1.e-3f);
}

template<>
inline const int EpsilonHelper<int>()
{
  return 6;

  // NOTE(cpg) - required for cbox_luxjr.txt under Linux/g++-4.3.3
  // return 10;
}

/////////////////////////////////////////////////////////////////////////////////
// Static member definitions

template<typename T>
const T Constants<T>::Epsilon(EpsilonHelper<T>());

#endif // Common_Constants_h
