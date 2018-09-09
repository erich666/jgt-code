
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

#ifndef tangere_Context_t
#define tangere_Context_t

namespace tangere
{
  template<typename T>
  class Scene;

  template<typename T>
  class RenderContext
  {
  public:
    RenderContext(const Scene<T>* scene_) :
      scene(scene_)
    {
      // no-op
    }

    RenderContext() :
      scene(0)
    {
      // no-op
    }

    ~RenderContext()
    {
      // no-op
    }
    
    const Scene<T>* getScene() const
    {
      return scene;
    }

  private:
    const Scene<T>* scene;
  };

} // namespace tangere

#endif // tangere_Context_h
