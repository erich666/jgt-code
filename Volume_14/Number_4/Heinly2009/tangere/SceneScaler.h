
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

#ifndef intrt_SceneScaler_h
#define intrt_SceneScaler_h

#include <Math/Math.h>

namespace tangere
{

  class SceneScaler 
  {
  public:
    SceneScaler();
    ~SceneScaler();

    void extend(const MathF::Point&);
    void computeScale();

    MathI::Point scale(const MathF::Point&) const;

    void printStats() const;

  private:
    MathF::Point max;
    MathF::Point min;
    float        maxEdgeLength;
    float        scalingFactor;
  };

} // namespace tangere

#endif // tangere_SceneScaler_h
