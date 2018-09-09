
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

#include <cfloat>
#include <cstring>

#include <iostream>
using std::ostream;

#include <fstream>
using std::ifstream;

#include <utility>
using std::pair;

#include <Common/FileIO/Mesh.h>
#include <Common/Utility/OutputCC.h>

#include <Math/Math.h>

namespace FileIO
{

  using MathF::Point;
  using MathF::Vector;

  using rtCoreF::RGB;
  using rtCoreF::TexCoord;

  struct Sphere
  {
    Point center;
    Point radius;
    uint  material_id;
  };

  //////////////////////////////////////////////////////////////////////////////
  // Utility functions

  void matchDirectory(string& fname, const string& full)
  {
    string result;
#if defined(WIN32)
    size_t lastSlash = full.rfind('\\');
    if (lastSlash == string::npos)
      result = ".\\";
#else
    size_t lastSlash = full.rfind('/');
    if (lastSlash == string::npos)
      result = "./";
#endif // defined(WIN32)
    else
      result = full.substr(0, lastSlash + 1);

    result += fname;
    fname   = result;
  }

  ostream& operator<<(ostream& out, const Mesh::Vertex& v)
  {
    out << v.p << ' ' << v.n << ' ' << v.t;
    return out;
  }


  //////////////////////////////////////////////////////////////////////////////
  // Mesh member definitions

  const float Mesh::Material::ExponentScale   =  10.f;
  const float Mesh::Material::DefaultExponent = 100.f;
  const float Mesh::Material::DefaultR0       =   0.04f;

  const int Mesh::undefined = 0x7ffffff0;

  Mesh::Triangle::Triangle(uint v0i, uint v1i, uint v2i, uint mID_) :
    mID(mID_)
  {
    vID[0] = v0i;
    vID[1] = v1i;
    vID[2] = v2i;
  }

  Mesh::Mesh() :
    nbytes(0)
  {
    // no-op
  }

  Mesh::Mesh(const string& filename) :
    nbytes(0)
  {
    load(filename);
  }

  Mesh::Mesh(ifstream& fin)
  {
    loadML(fin);
  }

  Mesh::~Mesh()
  {
    for (uint i = 0; i < triangles.size(); ++i)
    {
      if (triangles[i] != NULL)
        delete triangles[i];
    }
  }

  void Mesh::load(const string& filename)
  {
    string ext = filename.substr(filename.length()-3);
    if (ext == "obj")
      loadOBJ(filename);
    else if (ext == "txt")
      loadML(filename);
    else if (ext == ".iw")
      loadIW(filename);
    else
      FatalError("Unrecognized mesh format:  \"" << ext << "\"" << endl);
  }

  ///////////////////////////////////////////////////////////////////////////////
  // Load MiniLight geometry file

  void Mesh::loadML(const string& filename)
  {
    ifstream fin(filename.c_str());
    if (!fin.is_open())
      FatalError("Failed to open \"" << filename << "\" for reading" << endl);

    // Parse scene file
    string junk;
    fin >> junk;

    // Parse image settings
    uint niters, xres, yres;
    fin >> niters >> xres >> yres;

    // Parse camera parameters
    Vector eye, dir;
    float hfov;
    fin >> eye >> dir >> hfov;

    // Parse scene parameters
    float skyEmission, grndEmission;
    fin >> skyEmission >> grndEmission;

    loadML(fin);
  }

  void Mesh::loadML(ifstream& fin)
  {
    // Parse geometry
    while (!fin.eof())
    {
      // Read vertex data
      Vertex v0, v1, v2;
      fin >> v0.p >> v1.p >> v2.p ;

      // Read material data
      RGB diffuse, emissive;
      fin >> diffuse >> emissive;

      // Construct edges, face normal
      const Vector e0 = v1.p - v0.p;
      const Vector e1 = v2.p - v0.p;
            Vector n  = Cross(e0, e1);

      // NOTE(jsh) - this causes errors for scenes with small triangles
      //             because simply continuing will result in a null pointer
      /*
      // Skip degenerate triangles
      if (n.length() <= Epsilon)
        continue;
      */

      n.normalize();

      // Finish construction
      v0.n = n;
      v1.n = n;
      v2.n = n;

      diffuse.clamp(0.f, 1.f);
      emissive.clamp(0.f, FLT_MAX);

      Material m = { diffuse, emissive, RGB(), 0.f, 0.f, 0.f };

      // Add data
      vertices.push_back(v0);
      vertices.push_back(v1);
      vertices.push_back(v2);
      materials.push_back(m);

      // Construct face
      const uint nverts = vertices.size();
      const uint nmats  = materials.size();

      // Add face
      triangles.push_back(new Triangle(nverts - 3,
                                       nverts - 2,
                                       nverts - 1,
                                       nmats - 1));

      // Update bounds
      bbox.extend(v0.p);
      bbox.extend(v1.p);
      bbox.extend(v2.p);
    }

    fin.close();
  }

  ///////////////////////////////////////////////////////////////////////////////
  // Helper class

  typedef pair<int, int>      intPair2;
  typedef pair<int, intPair2> intPair3;
  typedef pair<int, intPair3> intPair4;

  struct Tuple4
  {
    Tuple4(int, int, int, int);

    intPair4 data;
  };

  Tuple4::Tuple4(int a, int b, int c, int d) :
    data(a, intPair3(a, intPair2(c, d)))
  {
    // no-op
  }

  bool operator<(const Tuple4& t1, const Tuple4& t2)
  {
    return (t1.data < t2.data);
  }

  ///////////////////////////////////////////////////////////////////////////////
  // Load OBJ geometry file

  void Mesh::loadOBJ(const string& filename)
  {
    // Open OBJ file
    FILE* fp = fopen(filename.c_str(), "rt");
    if (!fp)
      FatalError("Failed to open \"" << filename << "\" for reading\n");

    char line[1024];
    char token[1024];

    // Per-vertex data
    vector<Point>    in_p;
    vector<TexCoord> in_t;
    vector<Vector>   in_n;

    vector<int>      clean_p;
    vector<int>      clean_t;
    vector<int>      clean_n;
    map<Tuple4, int> clean_map;

    // Per-face data
    vector<int> vertex0, vertex1, vertex2;
    vector<int> mID;

    map<string, int> mID_map;
    vector<Material> clean_m;

    // Spheres
    vector<Sphere> spheres;

    // Parse OBJ file
    uint currentMaterial = 0;
    uint currentGroup    = 0;

    while (true)
    {
      float x, y, z, r;

      if (!fgets(line, 1024, fp))
        break;

      // NOTE(cpg) - Leading whitespace on all sscanf format strings tells the
      //             function to discard all leading whitespace on the input
      //             strings.
      if (sscanf(line, " v %f %f %f", &x, &y, &z) == 3)
      {
        in_p.push_back(Point(x, y, z));
      }
      else if (sscanf(line, " vt %f %f", &x, &y) == 2)
      {
        in_t.push_back(TexCoord(x, y, 0.f));
      }
      else if (sscanf(line, " vn %f %f %f", &x, &y, &z) == 3)
      {
        in_n.push_back(Vector(x, y, z));
      }
      else if (sscanf(line, " mtllib %s", token) == 1)
      {
        string tokenS(token);
        string filenameS(filename);
        loadMTL(clean_m, mID_map, tokenS, filenameS);
      }
      else if (sscanf(line, " usemtl %s", token) == 1)
      {
        currentMaterial = mID_map[token];
      }
      else if (sscanf(line, " s %d", &currentGroup) == 1)
      {
        // no-op: currentGroup set in sscanf
      }      
      else if (sscanf(line, " sphere %f %f %f %f", &x, &y, &z, &r) == 4)
      {
        Sphere newSphere;
        newSphere.center = Point(x, y, z);
        newSphere.radius = Point(r, r, r);
        newSphere.material_id = currentMaterial;
        spheres.push_back(newSphere);
      }
      else
      {
        int v0 = undefined + 1;
        int v1 = undefined + 1;
        int v2 = undefined + 1;
        int v3 = undefined + 1;

        int vt0 = undefined + 1;
        int vt1 = undefined + 1;
        int vt2 = undefined + 1;
        int vt3 = undefined + 1;

        int vn0 = undefined + 1;
        int vn1 = undefined + 1;
        int vn2 = undefined + 1;
        int vn3 = undefined + 1;

        if (sscanf(line, " f %d/%d/%d %d/%d/%d %d/%d/%d %d/%d/%d",
                   &v0, &vt0, &vn0,
                   &v1, &vt1, &vn1,
                   &v2, &vt2, &vn2,
                   &v3, &vt3, &vn3) >= 9 ||
            sscanf(line, " f %d//%d %d//%d %d//%d %d//%d",
                   &v0, &vn0,
                   &v1, &vn1,
                   &v2, &vn2,
                   &v3, &vn3) >= 6 ||
            sscanf(line, " f %d/%d %d/%d %d/%d %d/%d",
                   &v0, &vt0,
                   &v1, &vt1,
                   &v2, &vt2,
                   &v3, &vt3) >= 6 ||
            sscanf(line, " f %d %d %d %d",
                   &v0,
                   &v1,
                   &v2,
                   &v3) >= 3)
        {
          const int poffset = in_p.size();
          const int toffset = in_t.size();
          const int noffset = in_n.size();

          v0  += (v0 < 0 ? poffset : -1);
          v1  += (v1 < 0 ? poffset : -1);
          v2  += (v2 < 0 ? poffset : -1);
          v3  += (v3 < 0 ? poffset : -1);

          vt0 += (vt0 < 0 ? toffset : -1);
          vt1 += (vt1 < 0 ? toffset : -1);
          vt2 += (vt2 < 0 ? toffset : -1);
          vt3 += (vt3 < 0 ? toffset : -1);

          vn0 += (vn0 < 0 ? noffset : -1);
          vn1 += (vn1 < 0 ? noffset : -1);
          vn2 += (vn2 < 0 ? noffset : -1);
          vn3 += (vn3 < 0 ? noffset : -1);

          // Add new vertices (as necessary)
          int smoothingGroup = (currentGroup > 0 ?
                                currentGroup :
                                -1 - (int)vertex0.size());

          Tuple4 id0(v0, vt0, vn0, smoothingGroup);
          if (clean_map.find(id0) == clean_map.end())
          {
            clean_map[id0] = clean_p.size();
            clean_p.push_back(v0);
            clean_t.push_back(vt0);
            clean_n.push_back(vn0);
          }

          Tuple4 id1(v1, vt1, vn1, smoothingGroup);
          if (clean_map.find(id1) == clean_map.end())
          {
            clean_map[id1] = clean_p.size();
            clean_p.push_back(v1);
            clean_t.push_back(vt1);
            clean_n.push_back(vn1);
          }

          Tuple4 id2(v2, vt2, vn2, smoothingGroup);
          if (clean_map.find(id2) == clean_map.end())
          {
            clean_map[id2] = clean_p.size();
            clean_p.push_back(v2);
            clean_t.push_back(vt2);
            clean_n.push_back(vn2);
          }

          vertex0.push_back(clean_map[id0]);
          vertex1.push_back(clean_map[id1]);
          vertex2.push_back(clean_map[id2]);
          mID.push_back(currentMaterial);

          if (v3 != undefined)
          {
            Tuple4 id3(v3, vt3, vn3, smoothingGroup);
            if (clean_map.find(id3) == clean_map.end())
            {
              clean_map[id3] = clean_p.size();
              clean_p.push_back(v3);
              clean_t.push_back(vt3);
              clean_n.push_back(vn3);
            }

            vertex0.push_back(clean_map[id0]);
            vertex1.push_back(clean_map[id2]);
            vertex2.push_back(clean_map[id3]);
            mID.push_back(currentMaterial);
          }
        }
      }
    }

    fclose(fp);

    // Make sure all texture coordinates positive
    const uint ntex = in_t.size();
    if (ntex > 0)
    {
      TexCoord tmin = in_t[0];
      for (uint i = 1; i < ntex; ++i)
      {
        tmin[0] = (tmin[0] < in_t[i][0] ? tmin[0] : in_t[i][0]);
        tmin[1] = (tmin[1] < in_t[i][1] ? tmin[0] : in_t[i][1]);
        tmin[2] = 0.f;
      }

      TexCoord correction;
      correction[0] = (tmin[0] < 0.f ? -floorf(tmin[0]) : 0.f);
      correction[1] = (tmin[1] < 0.f ? -floorf(tmin[1]) : 0.f);
      correction[2] = 0.f;

      for (uint i = 0; i < ntex; ++i)
        in_t[i] += correction;
    }

    // Add spheres
    for (uint i = 0; i < spheres.size(); ++i)
    {
      in_p.push_back(spheres[i].center);
      in_p.push_back(spheres[i].radius);

      clean_p.push_back(in_p.size() - 2);
      clean_p.push_back(in_p.size() - 1);

      vertex0.push_back(undefined);
      vertex1.push_back(clean_p.size() - 2);
      vertex2.push_back(clean_p.size() - 1);

      mID.push_back(spheres[i].material_id);

      clean_t.push_back(undefined);
      clean_t.push_back(undefined);

      clean_n.push_back(undefined);
      clean_n.push_back(undefined);
    }

    // Populate mesh
    const uint nverts = clean_p.size();
    const uint ntris  = vertex0.size();
    const uint nmats  = clean_m.size();

    // Geometry data
    const float vtotal = float(nverts*sizeof(Vertex));
    const float ftotal = float(ntris*sizeof(Triangle));
    const float gtotal = vtotal + ftotal;

    /*
    Output("Geometry data:" << gtotal/1024.f << endl);
    Output("  # vertices  = " << nverts << ", "
           << vtotal/1024.f << " KB" << endl);
    Output("  # faces     = " << ntris << ", "
           << ftotal/1024.f << " KB" << endl);
    */

    // Texture data
    const float ttotal = float(nmats*sizeof(Material));
    
    /*
    Output("Texture data:  " ttotal/1024.f << endl);
    Output("  # materials = " << nmats << endl);
    */

    nbytes += size_t(gtotal + ttotal);

    // Fill in per-vertex data
    vertices.resize(nverts);
    for (uint i = 0; i < nverts; ++i)
    {
      const uint& pidx = clean_p[i];
      const uint& tidx = clean_t[i];
      const uint& nidx = clean_n[i];

      const Point& vp = in_p[pidx];

      TexCoord vt;
      vt[0] = (tidx == undefined ? 0.f : in_t[tidx][0]);
      vt[1] = (tidx == undefined ? 0.f : in_t[tidx][1]);
      vt[2] = 0.f;

      Vector vn;
      vn[0] = (nidx == undefined ? 0.f : in_n[nidx][0]);
      vn[1] = (nidx == undefined ? 0.f : in_n[nidx][1]);
      vn[2] = (nidx == undefined ? 0.f : in_n[nidx][2]);

      Vertex vertex = { vp, vn.normal(), vt };
      vertices[i] = vertex;
    }

    // Fill in face data
    triangles.resize(ntris);
    for (uint i = 0; i < ntris; ++i)
    {
      // Interpolate normals
      const uint& v0i = vertex0[i];
      const uint& v1i = vertex1[i];
      const uint& v2i = vertex2[i];
    
      if (v0i == undefined)
      {
        triangles[i] = new Triangle(v0i, v1i, v2i, mID[i]);
        continue;
      }

      const Point& v0 = vertices[v0i].p;
      const Vector e0 = vertices[v1i].p - v0;
      const Vector e1 = vertices[v2i].p - v0;
            Vector n  = Cross(e0, e1);

      // NOTE(jsh) - this causes errors for scenes with small triangles
      /*
      // Skip degenerate triangles
      if (n.length() <= Epsilon)
        continue;
      */

      if (clean_n[v0i] == undefined)
        vertices[v0i].n += n;

      if (clean_n[v1i] == undefined)
        vertices[v1i].n += n;

      if (clean_n[v2i] == undefined)
        vertices[v2i].n += n;

      // Add face
      triangles[i] = new Triangle(v0i, v1i, v2i, mID[i]);

      /*
      Output("face[" << i << "]:" << endl);
      Output("  v0 = " << vertices[v0i] << endl);
      Output("  v1 = " << vertices[v1i] << endl);
      Output("  v2 = " << vertices[v2i] << endl);
      */

      // Update bounds
      bbox.extend(v0);
      bbox.extend(vertices[v1i].p);
      bbox.extend(vertices[v2i].p);
    }

    // Renormalize
    for (uint i = 0; i < nverts; ++i)
      vertices[i].n.normalize();

    // Fill in material data
    materials.resize(nmats);
    for (uint i = 0; i < nmats; ++i)
    {
      Material& m = clean_m[i];

      m.diffuse.clamp(0.f, 1.f);
      m.emissive.clamp(0.f, FLT_MAX);
      m.specular.clamp(0.f, 1.f);

      // Add material
      materials[i] = m;
    }
  }

  ///////////////////////////////////////////////////////////////////////////////
  // Load OBJ material library

  void Mesh::loadMTL(      vector<Material>& clean_m,
                           map<string, int>& mID_map,
                           string&           fnameMTL,
                     const string&           filename)
  {
    char line[1024];
    char token[1024];

    float x, y, z;

    // Open MTL library
    matchDirectory(fnameMTL, filename);
    FILE* fp = fopen(fnameMTL.c_str(), "rt");
    if (!fp)
      FatalError("Failed to open \"" << fnameMTL << "\" for reading\n");

    // Parse MTL library
    while (true)
    {
      if (!fgets(line, 1024, fp))
        break;

      if (sscanf(line, " newmtl %s", token) == 1)
      {
        Material material = { RGB::Zero, RGB::Zero, RGB::Zero, 0.f, 0.f, 0.f };
        clean_m.push_back(material);
        mID_map[token] = clean_m.size()-1;
      }
      else if (sscanf(line, " Kd %f %f %f", &x, &y, &z) == 3)
      {
        clean_m.back().diffuse = RGB(x, y, z);
      }
      else if (sscanf(line, " map_Kd %s", token) == 1)
      {
        // XXX(cpg) - not yet implemented
        /*
        // Two forms of map_Kd
        float uscale = 1.f;
        float vscale = 1.f;
        if (sscanf(line, "map_Kd -s %f %f %s",
                   &uscale, &vscale, token) == 3)
        {
          // no-op: uscale, vscale, token set in sscanf
        }
        else if (sscanf(line, "map_Kd %s", token) == 1)
        {
          // no-op:  token set in sscanf
        }

        matchDirectory(token, filename);

        Material& mat(clean_m.back());
        mat.filename[Material::Diffuse] = token;
        mat.uscale[Material::Diffuse]   = (uscale < 1.f ? 1.f/uscale : uscale);
        mat.vscale[Material::Diffuse]   = (vscale < 1.f ? 1.f/vscale : vscale);
        */
      }
      else if (sscanf(line, " Ks %f %f %f", &x, &y, &z) == 3)
      {
        clean_m.back().specular = RGB(x, y, z);
      }
      else if (sscanf(line, " map_Ks %s", token) == 1)
      {
        // XXX(cpg) - not yet implemented
        /*
        matchDirectory(token, filename);

        Material& mat(clean_m.back());
        mat.filename[Material::Specular] = token;
        */
      }
      else if (sscanf(line, " Ke %f %f %f", &x, &y, &z) == 3)
      {
        clean_m.back().emissive = RGB(x, y, z);
      }
      else if (sscanf(line, " map_Ke %s", token) == 1)
      {
        // XXX(cpg) - not yet implemented
        /*
        matchDirectory(token, filename);

        Material& mat(clean_m.back());
        mat.filename[Material::Emissive] = token;
        */
      }
      else if (sscanf(line, " Ns %f", &x) == 1)
      {
        clean_m.back().exp = x*Material::ExponentScale;
        // XXX(cpg) - not yet implemented
        /*
        Material& mat = clean_m.back();
        mat.model     = clean_m.back().model;
        if (x > 0.f)
        {
          mat.model = ShadingModel::CoupledModel;
          type[ShadingModel::CoupledModel] = true;
        }
        */
      }
      else if (sscanf(line, " R0 %f", &x) == 1)
      {
        clean_m.back().r0 = x;
      }
      else if (sscanf(line, " Tr %f", &x) == 1)
      {
        if (x > 0.f)
        {
          clean_m.back().tr = x;
          // XXX(cpg) - not yet implemented      
          //clean_m.back().model = ShadingModel::DielectricModel;
          //type[ShadingModel::DielectricModel] = true;
        }
        
      }
    }

    fclose(fp);
  }

  ///////////////////////////////////////////////////////////////////////////////
  // Load IW geometry file

  void Mesh::loadIW(const string& filename)
  {
    // Open IW file
    FILE* fp = fopen(filename.c_str(), "rt");
    if (!fp)
      FatalError("Failed to open \"" << filename << "\" for reading\n");

    char line[1024];
    char token[1024];
    char type[10];

    uint  nverts   = 0;
    uint  nnormals = 0;
    uint  ntris    = 0;
    uint  nmats    = 0;
    char* rc;

    // Parse IW file
    while (true)
    {
      if (!fgets(line, 1024, fp))
        break;

      // NOTE(cpg) - Leading whitespace on all sscanf format strings tells the
      //             function to discard all leading whitespace on the input
      //             strings.
      if (sscanf(line, " vertices: %d", &nverts) == 1)
      {
        if (nnormals > 0 && nverts != nnormals)
        {
          FatalError("Number of vertices (" << nverts
                           << ") does not match number of normals ("
                           << nnormals << ")\n");
        }
        
        if (vertices.size() != nverts)
        {
          vertices.resize(nverts);
          nbytes += nverts*sizeof(Vertex);
        }

        float x, y, z;
        for (uint i = 0; i < nverts; ++i)
        {
          rc = fgets(line, 1024, fp);
          sscanf(line, "%f %f %f", &x, &y, &z);

          // Add vertex position
          vertices[i].p = Point(x, y, z);
        }
      }
      else if (sscanf(line, " vtxnormals: %d", &nnormals) == 1)
      {
        if (nverts > 0 && nnormals != nverts)
        {
          FatalError("Number of normals (" << nnormals
                           << ") does not match number of vertices ("
                           << nverts << ")\n");
        }

        if (vertices.size() != nverts)
        {
          vertices.resize(nverts);
          nbytes += nnormals*sizeof(Vertex);
        }

        float x, y, z;
        for (uint i = 0; i < nverts; ++i)
        {
          rc = fgets(line, 1024, fp);
          sscanf(line, "%f %f %f", &x, &y, &z);

          // Add vertex normal
          vertices[i].n = Vector(x, y, z).normal();
        }
      }
      else if (sscanf(line, " triangles: %d", &ntris) == 1)
      {
        triangles.resize(ntris);
        nbytes += ntris*sizeof(Triangle);

        uint v0i, v1i, v2i, mID;
        for (uint i = 0; i < ntris; ++i)
        {
          rc = fgets(line, 1024, fp);
          sscanf(line, "%d %d %d %d", &v0i, &v1i, &v2i, &mID);

          // Add face
          triangles[i] = new Triangle(v0i, v1i, v2i, mID);

          // Update bounds
          bbox.extend(vertices[v0i].p);
          bbox.extend(vertices[v1i].p);
          bbox.extend(vertices[v2i].p);
        }
      }
      else if (sscanf(line, " shaders %d", &nmats) == 1)
      {
        map<string, int> mID_map;
        vector<Material> clean_m;
        vector<int>      actual_mid(nmats);

        uint mID;
        float r, g, b, exp, r0;
        for (uint i = 0; i < nmats; ++i)
        {
          rc = fgets(line, 1024, fp);

          // Build material string
          if (sscanf(line, "shader %d %s (%f,%f,%f)",
                     &mID, type, &r, &g, &b) == 5)
            sprintf(token, "%s (%f,%f,%f)", type, r, g, b);
          else if (sscanf(line, "shader %d %s (%f,%f,%f) %f %f",
                          &mID, type, &r, &g, &b, &exp, &r0) == 6)
            sprintf(token, "%s (%f,%f,%f) %f %f", type, r, g, b, exp, r0);

          if (mID < 0 || mID > nmats)
            FatalError("Invalid material id (" << mID << ")\n");

          // Search for material string, adding new materials as necessary
          if (mID_map.find(token) == mID_map.end())
          {
            // Add a new material
            Material material = { RGB::Zero, RGB::Zero, RGB::Zero, 0.f, 0.f, 0.f };
            if (strcmp(type, "diffuse") == 0)
            {
              material.diffuse = RGB(r, g, b);
            }
            else if (strcmp(type, "emissive") == 0)
            {
              material.emissive = RGB(r, g, b);
            }
            else if (strcmp(type, "coupled") == 0)
            {
              material.specular = RGB(r, g, b);
              // XXX(cpg) - not yet (fully) implemented
            }
            else
            {
              FatalError("Invalid shading model (" << type << ")\n");
            }

            clean_m.push_back(material);
            mID_map[token] = clean_m.size()-1;
          }

          actual_mid[mID] = mID_map[token];
        }

        nmats   = clean_m.size();
        nbytes += nmats*sizeof(Material);

        // Fill in material data
        for (uint i = 0; i < nmats; ++i)
        {
          Material& m = clean_m[i];

          m.diffuse.clamp(0.f, 1.f);
          m.emissive.clamp(0.f, FLT_MAX);
          m.specular.clamp(0.f, 1.f);

          materials.push_back(m);
        }

        // Update material ids
        for (uint i = 0; i < ntris; ++i)
        {
          const int mid = triangles[i]->mID;
          triangles[i]->mID  = actual_mid[mid];
        }
      }
    }

    fclose(fp);
  }

} // namespace FileIO
