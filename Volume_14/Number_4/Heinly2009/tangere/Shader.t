
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

#ifndef tangere_Shader_t
#define tangere_Shader_t

namespace rtCore
{

  template<typename T>
  class RGB;

} // namespace FileIO

namespace tangere
{

  ///////////////////////////////////////////////////////////////////////////////
  // Using declarations

  using rtCore::RGB;

  ///////////////////////////////////////////////////////////////////////////////
  // Forward declarations

  template<typename T>
  class HitRecord;

  template<typename T>
  struct Material;

  template<typename T>
  class Ray;

  template<typename T>
  class RenderContext;

  ///////////////////////////////////////////////////////////////////////////////
  // Class template definition

  template<typename T>
  struct Shader
  {
    enum Model { NdotV = 0, Lambertian, Reflection, Dielectric, nshaders };

    virtual RGB<T> shade(const RenderContext<T>& rc,
                         const Ray<T>&           ray,
                         const HitRecord<T>&     hit,
                         const Material<T>&      material,
                               int               depth) const = 0;
  };

} // namespace tangere

#endif // tangere_Shader_t
