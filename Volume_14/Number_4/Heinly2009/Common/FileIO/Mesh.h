
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

#ifndef Common_FileIO_Mesh_h
#define Common_FileIO_Mesh_h

#include <iosfwd>
using std::ostream;

#include <fstream>
using std::ifstream;

#include <map>
using std::map;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include <Common/FileIO/BBox.h>
#include <Common/rtCore/RGB.t>
#include <Common/rtCore/TexCoord.h>
#include <Common/Types.h>

#include <Math/Math.h>

namespace FileIO
{

  using MathF::Point;
  using MathF::Vector;

  using rtCoreF::RGB;
  using rtCoreF::TexCoord;

  class Mesh
  {
  public:
    static const int undefined;

    struct Material
    {
      static const float ExponentScale;
      static const float DefaultExponent;
      static const float DefaultR0;

      RGB diffuse;
      RGB emissive;
      RGB specular;

      float exp;
      float r0;
      float tr;
    };

    struct Triangle
    {
      Triangle(uint, uint, uint, uint);

      uint vID[3];
      uint mID;
    };

    struct Vertex
    {
      Point    p;
      Vector   n;
      TexCoord t;
    };

    Mesh(const string& fileName);
    Mesh(ifstream& fin);
    Mesh();
    ~Mesh();

    void load(const string& fileName);

    const vector<Material>&  getMaterials() const;
    const vector<Triangle*>& getTriangles() const;
    const vector<Vertex>&    getVertices()  const;
    const BBox&              getBounds()    const;

  private:
    void loadML( const string&);
    void loadML(       ifstream&);
    void loadOBJ(const string&);
    void loadMTL(      vector<Material>&,
                       map<string, int>&,
                       string&,
                 const string&);
    void loadIW( const string&);

    vector<Material>  materials;
    vector<Triangle*> triangles;
    vector<Vertex>    vertices;

    BBox bbox;
    uint nbytes;
  };

  ostream& operator<<(ostream&, const Mesh::Vertex&);

  inline const vector<Mesh::Material>& Mesh::getMaterials() const
  {
    return materials;
  }

  inline const vector<Mesh::Triangle*>& Mesh::getTriangles() const
  {
    return triangles;
  }

  inline const vector<Mesh::Vertex>& Mesh::getVertices() const
  {
    return vertices;
  }

  inline const BBox& Mesh::getBounds() const
  {
    return bbox;
  }

} // namespace FileIO

#endif // Common_FileIO_Mesh_h
