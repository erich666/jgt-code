
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

#ifndef tangere_Material_t
#define tangere_Material_t

#include <vector>
using std::vector;

#include <Common/rtCore/RGB.t>

namespace tangere
{
  using rtCore::RGB;

  template<typename T>
  class RenderContext;

  template<typename T>
  struct Material
  {
    inline Material(const RGB<T>& d_,
                    const RGB<T>& s_,
                    const RGB<T>& e_,
                    const T&      ka_,
                    const T&      kd_,
                    const T&      exp_,
                    const T&      r0_,
                          bool    reflective_ = false,
                          bool    tr_ = false) :
      diffuse(d_),
      specular(s_),
      emissive(e_),
      ka(ka_),
      kd(kd_),
      exp(exp_),
      r0(r0_),
      reflective(reflective_),
      transparent(tr_)
    {
      // no-op
    }

    inline Material() : 
      diffuse(0,0,0),
      specular(0,0,0),
      emissive(0,0,0),
      ka(0),
      kd(0),
      exp(100),
      r0(0),
      reflective(false),
      transparent(false)
    {
      // no-op
    }

    inline ~Material()
    {
      // no-op
    }

    RGB<T> diffuse;
    RGB<T> specular;
    RGB<T> emissive;
    T      ka;
    T      kd;
    T      exp;
    T      r0;
    bool   reflective;
    bool   transparent;
  };

} // namespace tangere

#endif // tangere_Material_t
