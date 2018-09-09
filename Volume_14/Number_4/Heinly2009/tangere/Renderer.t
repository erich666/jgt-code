
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

#ifndef tangere_Renderer_t
#define tangere_Renderer_t

#ifdef USE_CHUD
#include <CHUD/CHUD.h>
#endif // USE_CHUD

#ifdef USE_SATURN
#include <Saturn.h>
#endif // USE_SATURN

#include <string>
using std::string;

#include <Common/FileIO/BVH.h>
#include <Common/FileIO/Image.t>
#include <Common/FileIO/Mesh.h>
#include <Common/FileIO/Scene.h>
#include <Common/Utility/OutputCC.h>
#include <Common/Utility/Timer.h>
#include <Common/Types.h>

#include <Math/Math.h>

#include <tangere/BVH.t>
#include <tangere/Camera.t>
#include <tangere/Context.t>
#include <tangere/Flags.h>
#include <tangere/HitRecord.t>
#include <tangere/Material.t>
#include <tangere/Options.h>
#include <tangere/Ray.t>
#include <tangere/Scene.t>
#include <tangere/Shader.t>

namespace tangere
{

  ///////////////////////////////////////////////////////////////////////////////
  // Class template definitions

  template<typename T>
  class Renderer
  {
  public:
    inline Renderer(const Scene<T>* scene_, const Options& opt) :
      scene(scene_)
    {
      bname = opt.bname + "_" + (typeid(T).name())[0];
    }

    inline ~Renderer()
    {
      // no-op
    }

    void render()
    {
#ifdef USE_CHUD
      chudStartRemotePerfMonitor(bname.c_str());
#endif // USE_CHUD

#ifdef USE_SATURN
      startSaturn();
#endif // USE_SATURN

      Timer timer;

      const Camera<T>*      camera = scene->getCamera();
      const BVH<T>*         bvh    = scene->getBVH();
            Image<RGB<T> >* image  = scene->getImage();
      const uint            xRes   = image->w();
      const uint            yRes   = image->h();

      const RenderContext<T> rc(scene);

      const float xMin = -1.f + 1.f/xRes;
      const float dx   = 2.f/xRes;

      const float yMin = -1.f + 1.f/yRes;
      const float dy   = 2.f/yRes;

      Output(endl);
#if defined(USE_BENCHMARK_MODE)
      Output("Benchmarking:  ");
#else
      Output("Progress:       0%" << flush);
#endif // defined(USE_BENCHMARK)

      float y = yMin;
      for (uint j = 0; j < yRes; ++j)
      {
#if !defined(USE_BENCHMARK_MODE)
        Output("\b\b\b");
        Output(setw(2) << (100*j/yRes) << "%" << flush);
#endif // !defined(USE_BENCHMARK_MODE)

        float x = xMin;
        for (uint i = 0; i < xRes; ++i)
        {
          Ray<T> ray = camera->generate(x, y);
          HitRecord<T> hit;
          bvh->intersect(hit, rc, ray);
          if (hit.getTriangle())
          {
            const Shader<T>*   shader   = hit.getShader();
            const uint&        mID      = hit.getMaterialID();
            const Material<T>& material = bvh->getMaterials()[mID];
            image->set(i, j, shader->shade(rc, ray, hit, material, 0));
          }
          else
            image->set(i, j, scene->getBackground(ray.dir()));

          x += dx;
        }

        y += dy;
      }

#if defined(USE_BENCHMARK_MODE)
      Output("done" << endl);
#else
      Output("\b\b\b" << "100%" << endl);
#endif // defined(USE_BENCHMARK_MODE)

      Output("Render time:   " << timer.getElapsed() << endl);

#ifdef USE_CHUD
      chudStopRemotePerfMonitor();
#endif // USE_CHUD

#ifdef USE_SATURN
      stopSaturn();
#endif // USE_SATURN

      image->writePPM(bname, 0 /* iteration */, true /* overwrite */);
      image->writeBMP(bname, 0 /* iteration */, true /* overwrite */);
      Output("Wrote images:  \"" << bname << ".{bmp,ppm}\"" << endl);
    }

  private:
          string    bname;
    const Scene<T>* scene;
  };

} // namespace tangere

namespace tangereF
{

  typedef tangere::Renderer<float> Renderer;

} // namespace tangereF

namespace tangereI
{

  typedef tangere::Renderer<int> Renderer;

} // namespace tangereI

#endif // tangere_Renderer_t
