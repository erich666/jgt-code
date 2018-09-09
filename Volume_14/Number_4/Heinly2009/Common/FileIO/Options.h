
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

#ifndef Common_FileIO_Options_h
#define Common_FileIO_Options_h

#include <string>
using std::string;

#include <Common/rtCore/RGB.t>
#include <Common/Types.h>

namespace FileIO
{

  using rtCoreF::RGB;

  struct Options
  {
    Options();
    virtual ~Options();

    // NOTE(cpg) - be careful not to unnecessarily bloat this structure;
    //             think carefully before adding a new option...

    // File I/O parameters
    string path;
    string fname;
    string bname;

    // Ground/sky color
    RGB grnd;
    RGB sky;

    // Image parameters
    uint nspp;
    uint xres;
    uint yres;

    // BVH leaf creation threshold
    uint thold;
  };

} // namespace FileIO

#endif // Common_FileIO_Options_h
