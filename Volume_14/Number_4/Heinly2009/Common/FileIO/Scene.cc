
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

#include <vector>
using std::vector;

#include <Common/FileIO/BVH.h>
#include <Common/FileIO/Scene.h>
#include <Common/Utility/OutputCC.h>

namespace FileIO
{

  Scene::Scene(const Options* opt)
  {
    const string& fname = opt->fname;
    const string& path  = opt->path;

    ifstream fin(fname.c_str());
    if (!fin.is_open())
      FatalError("Failed to open \"" << fname << "\" for reading" << endl);

    /////////////////////////////////////////////////////////////////////////////
    // Parse scene file

    string viewFile;
    string geoFile;

    char token;
    fin >> token;

    if (token == '#')
    {
      // Parse format
      string format;
      fin >> format;

      // Parse image settings
      fin >> image.numSamples >> image.xRes >> image.yRes;

      if (format == "MiniLight")
      {
        viewFile = fname;
        geoFile  = fname;

        // Parse camera parameters
        Vector dir;
        fin >> view.eye >> dir;
        if (!(fin >> view.fov))
        {
          view.fov = 45.0f;
        }

        view.lookat = view.eye + dir;
        view.up     = Vector(0.0f, 1.0f, 0.0f);

        // Parse scene parameters
        fin >> sky >> ground;

        // Read geometry
        mesh = new Mesh(fin);
        fin.close();
      }
      else if (format == "luxjr")
      {
        // Parse scene parameters
        fin >> sky >> ground;
        fin >> viewFile >> geoFile;
        fin.close();
        fin.clear();

        viewFile = path + viewFile;
        geoFile  = path + geoFile;

#if defined(WIN32)
        for (uint i = 0; i < viewFile.size(); ++i)
        {
          if (viewFile[i] == '/')
            viewFile[i] = '\\';
        }

        for (uint i = 0; i < geoFile.size(); ++i)
        {
          if (geoFile[i] == '/')
            geoFile[i] = '\\';
        }
#endif // defined(WIN32)

        // Open view file
        fin.open(viewFile.c_str());
        if (!fin.is_open())
          FatalError("Failed to open \"" << viewFile << "\" for reading");

        // Parse camera parameters
        string junk;
        fin >> junk >> view.eye;
        fin >> junk >> view.lookat;
        fin >> junk >> view.up;
        if (!(fin >> junk >> view.fov))
        {
          view.fov = 45.0f;
        }
        fin.close();
        fin.clear();

        // Read geometry
        mesh = new Mesh(geoFile);
      }
      else
      {
        FatalError("Unrecognized scene format:  \"" << format << "\"" << endl);
      }
    }
    else
    {
      FatalError("Invalid scene file:  expecting \"# <format>\", found \""
                 << token << "\" instead" << endl);
    }

    // Validate mesh
    const uint ntris = mesh->getTriangles().size();
    if (ntris <= 0)
      FatalError("Degenerate mesh:  # triangles = " << ntris << endl);

    /////////////////////////////////////////////////////////////////////////////
    // Process command line overrides

    // Image parameters
    image.numSamples = (opt->nspp > 0 ? opt->nspp : image.numSamples);
    image.xRes       = (opt->xres > 0 ? opt->xres : image.xRes      );
    image.yRes       = (opt->yres > 0 ? opt->yres : image.yRes      );

    // Background colors
    ground = (opt->grnd != rtCore::RGB<float>::Zero ? opt->grnd : ground);
    sky    = (opt->sky  != rtCore::RGB<float>::Zero ? opt->sky  : sky   );

    // BVH leaf creation threshold
    thold = (opt->thold > 0 ? opt->thold : BVH::defaultThreshold);

    // Output configuration
    Output("path  = " << path << endl);
    Output("fname = " << fname << endl);
    Output("view  = " << viewFile << endl);
    Output("mesh  = " << geoFile << endl);
  }

  Scene::~Scene()
  {
    delete mesh;
  }

} // namespace FileIO
