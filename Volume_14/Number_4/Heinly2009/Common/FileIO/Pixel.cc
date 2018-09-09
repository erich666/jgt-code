
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

#include <fstream>
using std::ifstream;
using std::ofstream;

#include <iostream>
using std::istream;
using std::ostream;

#include <Common/FileIO/Pixel.h>

namespace FileIO
{

  istream& operator>>(istream& in, Pixel& p)
  {
    char junk;
    in >> junk;
    in >> p.e[0] >> p.e[1] >> p.e[2];
    in >> junk;

    return in;
  }

  ostream& operator<<(ostream& out, const Pixel& p)
  {
    out << '(' << p.e[0] << ' ' << p.e[1] << ' ' << p.e[2] << ')';
    return out;
  }

  ifstream& operator>>(ifstream& in, Pixel& p)
  {
    in >> p.e[0] >> p.e[1] >> p.e[2];
    return in;
  }

  ofstream& operator<<(ofstream& out, const Pixel& p)
  {
    out << p.e[0] << p.e[1] << p.e[2];
    return out;
  }

} // namespace FileIO
