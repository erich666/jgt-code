
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

#include <tangere/Triangle.t>

namespace tangere
{

        int   Triangle<int>::edgeBias         =  0;
        float Triangle<int>::maxEdge          = -FLT_MAX;
        float Triangle<int>::minEdge          =  FLT_MAX;
        float Triangle<int>::maxEdgeComponent = -FLT_MAX;
        float Triangle<int>::edgeScale        =  0;
  const uint  Triangle<int>::mod3[5]          =  { 0, 1, 2, 0, 1 };

} // namespace tangere
