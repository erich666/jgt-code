
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

#include <Common/Types.h>

#include <tangere/DielectricShader.t>
#include <tangere/LambertianShader.t>
#include <tangere/NdotVShader.t>
#include <tangere/ReflectionShader.t>
#include <tangere/Shader.t>
#include <tangere/ShaderTable.h>

namespace tangere
{

  ShaderTable::ShaderTable()
  {
#ifdef USE_NDOTV_SHADER
    shadersF.resize(Shader<float>::nshaders);
    shadersF[Shader<float>::NdotV]      = new NdotVShader<float>();
    shadersF[Shader<float>::Lambertian] = new NdotVShader<float>();
    shadersF[Shader<float>::Reflection] = new NdotVShader<float>();
    shadersF[Shader<float>::Dielectric] = new NdotVShader<float>();


    shadersI.resize(Shader<int>::nshaders);
    shadersI[Shader<int>::NdotV]      = new NdotVShader<int>();
    shadersI[Shader<int>::Lambertian] = new NdotVShader<int>();
    shadersI[Shader<int>::Reflection] = new NdotVShader<int>();
    shadersI[Shader<int>::Dielectric] = new NdotVShader<int>();
#else
    shadersF.resize(Shader<float>::nshaders);
    shadersF[Shader<float>::NdotV]      = new NdotVShader<float>();
    shadersF[Shader<float>::Lambertian] = new LambertianShader<float>();
    shadersF[Shader<float>::Reflection] = new ReflectionShader<float>();
    shadersF[Shader<float>::Dielectric] = new DielectricShader<float>();


    shadersI.resize(Shader<int>::nshaders);
    shadersI[Shader<int>::NdotV]      = new NdotVShader<int>();
    shadersI[Shader<int>::Lambertian] = new LambertianShader<int>();
    shadersI[Shader<int>::Reflection] = new ReflectionShader<int>();
    shadersI[Shader<int>::Dielectric] = new DielectricShader<int>();
#endif // USE_NDOTV_SHADER
  }

  ShaderTable::~ShaderTable()
  {
    for (uint i = 0; i < shadersF.size(); ++i)
      delete shadersF[i];

    for (uint i = 0; i < shadersI.size(); ++i)
      delete shadersI[i];
  }

} // namespace tangere
