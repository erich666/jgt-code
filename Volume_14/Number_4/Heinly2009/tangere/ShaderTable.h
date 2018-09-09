
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

#ifndef tangere_ShaderTable_h
#define tangere_ShaderTable_h

#include <vector>
using std::vector;

#include <tangere/Shader.t>

namespace tangere
{

  class ShaderTable
  {
  public:
    ShaderTable();
    ~ShaderTable();

  // XXX(cpg) - why won't a template member (with specialization) work here?
  Shader<float>* getShader(Shader<float>::Model) const;
  Shader<int>*   getShader(Shader<int>::Model)   const;

  private:
    vector<Shader<float>* > shadersF;
    vector<Shader<int>*   > shadersI;
  };
  
  inline Shader<float>* ShaderTable::getShader(Shader<float>::Model i) const
  {
    return shadersF[i];
  }

  inline Shader<int>* ShaderTable::getShader(Shader<int>::Model i) const
  {
    return shadersI[i];
  }

  static ShaderTable stable;

} // namespace tangere

#endif // tangere_ShaderTable_h
