
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

#ifndef Common_FileIO_GeometryProcessor_h
#define Common_FileIO_GeometryProcessor_h

#include <vector>
using std::vector;

#include <Common/FileIO/BVH.h>
#include <Common/FileIO/Mesh.h>

namespace FileIO
{
  class GeometryProcessor
  {
  public:
    GeometryProcessor();
    virtual ~GeometryProcessor();

    virtual const vector<FileIO::BVH::Object*>& operator()
      (
        const vector<FileIO::Mesh::Triangle*>&,
        const vector<FileIO::Mesh::Vertex>&,
        const vector<FileIO::Mesh::Material>&
      ) = 0;

  protected:
    vector<FileIO::BVH::Object*> objects;
  };

} // namespace FileIO

#endif // Common_FileIO_GeometryProcessor_h
