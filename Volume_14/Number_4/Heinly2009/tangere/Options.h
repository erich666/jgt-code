
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

#ifndef tangere_Options_h
#define tangere_Options_h

#include <string>
using std::string;

#include <Common/FileIO/Options.h>
#include <Common/rtCore/RGB.t>
#include <Common/Types.h>

namespace tangere
{

  struct Options : public FileIO::Options
  {
    Options();

    // File I/O options
    bool   iInput;
    string iWrite;
    bool   overwrite;
  };

  inline Options::Options() :
    FileIO::Options(),
    iInput(false),
    iWrite(""),
    overwrite(true)
  {
    // no-op
  }

} // namespace tangere

#endif // tangere_Options_h
