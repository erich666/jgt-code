
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

#include <Common/Utility/MinMax.t>
#include <Common/Utility/OutputCC.h>

#include <Math/Math.h>

#include <tangere/Constants.h>
#include <tangere/SceneScaler.h>

namespace tangere
{
  ///////////////////////////////////////////////////////////////////////////////
  // Using declarations

  using Utility::Max;
  using Utility::Min;

  ///////////////////////////////////////////////////////////////////////////////
  // Member function definitions

  SceneScaler::SceneScaler() :
    min(Math::Point<float>::Max), 
    max(Math::Point<float>::Min),
    scalingFactor(0.f)
  {
    // no-op
  }

  SceneScaler::~SceneScaler() 
  {
    // no-op
  }

  void SceneScaler::extend(const MathF::Point& p)
  {
    min = Min(min, p);
    max = Max(max, p);
  }

  void SceneScaler::computeScale() 
  {
    const MathF::Vector d = max - min;

    maxEdgeLength = (d[0] > d[1] ? d[0] : d[1]);
    maxEdgeLength = (maxEdgeLength > d[2] ? maxEdgeLength : d[2]);

    scalingFactor = powf(2.f, float(sideShift)) / maxEdgeLength;
  }

  MathI::Point SceneScaler::scale(const MathF::Point& p) const
  {
    const MathF::Vector scaled = scalingFactor*(p - min);
    return MathI::Point(int(scaled[0] + 0.5f),
                        int(scaled[1] + 0.5f),
                        int(scaled[2] + 0.5f));
  }

  void SceneScaler::printStats() const
  {
    if (scalingFactor != 0.f)
    {
      Output("Scene stats:" << endl);
      Output("  Maximum side =  " << maxEdgeLength << endl);
      Output("  Scene scale  =  " << scalingFactor << endl);
      Output(endl);
    }
  }

} // namespace tangere
