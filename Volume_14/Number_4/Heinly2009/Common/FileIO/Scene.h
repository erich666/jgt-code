
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

#ifndef Common_FileIO_Scene_h
#define Common_FileIO_Scene_h

#include <string>
using std::string;

#include <Common/FileIO/Mesh.h>
#include <Common/FileIO/Options.h>
#include <Common/Types.h>

#include <Math/Math.h>

namespace FileIO
{

  ///////////////////////////////////////////////////////////////////////////////
  // Using declarations

  using MathF::Point;
  using MathF::Vector;

  using rtCoreF::RGB;

  ///////////////////////////////////////////////////////////////////////////////
  // Class definition

  class Scene
  {
  public:
    struct ImageParameters
    {
      uint xRes;
      uint yRes;
      uint numSamples;
    };

    struct ViewParameters
    {
      Point  eye;
      Point  lookat;
      Vector up;
      float  fov;
    };

    Scene(const Options*);
    ~Scene();

    const Mesh*            getMesh()            const;
    const ImageParameters& getImageParameters() const;
    const ViewParameters&  getViewParameters()  const;
    const RGB&             getSky()             const;
    const RGB&             getGround()          const;
          uint             getThreshold()       const;

  private:
    Mesh*           mesh;
    ImageParameters image;
    ViewParameters  view;
    RGB             sky;
    RGB             ground;
    uint            thold;
  };

  inline const Mesh* Scene::getMesh() const
  {
    return mesh;
  }

  inline const Scene::ImageParameters& Scene::getImageParameters() const
  {
    return image;
  }

  inline const Scene::ViewParameters& Scene::getViewParameters() const
  {
    return view;
  }

  inline const RGB& Scene::getSky() const
  {
    return sky;
  }

  inline const RGB& Scene::getGround() const
  {
    return ground;
  }

  inline uint Scene::getThreshold() const
  {
    return thold;
  }

} // namespace FileIO

#endif // Common_FileIO_Scene_h
