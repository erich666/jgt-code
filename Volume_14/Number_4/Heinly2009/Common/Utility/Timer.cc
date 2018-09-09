
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

#include <iomanip>
using std::setprecision;
using std::setw;

#include <iostream>
using std::ios;

#include <Common/Utility/Timer.h>

namespace Utility
{

  ostream& operator<<(ostream& out, Timer& timer)
  {
    double seconds = timer.getElapsed();
    int    minutes = 0;
    int    hours   = 0;
    if (seconds > 60)
    {
      minutes  = int(seconds/60.f);
      seconds -= 60.f*minutes;
      
      hours   = minutes/60;
      minutes = minutes%60;
    }

    char fill = out.fill('0');
    out << setw(2) << hours << ":";
    out << setw(2) << minutes << ":";

    out.setf(ios::fixed);
    int precision = out.precision(2);

    out << setw(5) << seconds;

    out.precision(precision);
    out.unsetf(ios::fixed);
    out.fill(fill);

    return out;
  }

} // namespace Utility
