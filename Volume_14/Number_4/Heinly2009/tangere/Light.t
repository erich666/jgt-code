
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

#ifndef tangere_Light_t
#define tangere_Light_t

#include <Common/rtCore/RGB.t>

#include <Math/Math.h>

#include <tangere/SceneScaler.h>

namespace tangere
{

  using Math::Point;
  using Math::Vector;

  using rtCore::RGB;

  template<typename T>
  class RenderContext;

  template<typename T>
  class Light
  {
  public:
    inline Light(const Point<float>& position_,
                 const SceneScaler&  /* unused */,
                 const RGB<T>&       color_) :
      position(position_), color(color_)
    {
      // no-op
    }

    inline Light()
    {
      // no-op
    }

    inline ~Light()
    {
      // no-op
    }

    inline T getLight(       RGB<T>&           lcolor,
                             Vector<T>&        ldir, 
                       const RenderContext<T>& rc,
                       const Point<T>&         hp) const
    {
      lcolor        = color;
      Vector<T> dir = position - hp;
      T len         = dir.normalize();
      ldir          = dir;

      return len;
    }

    inline void     preprocess()        { /* no-op */      }
    inline Point<T> getPosition() const { return position; }
    inline RGB<T>   getColor()    const { return color;    }

  private:
    RGB<T>   color;
    Point<T> position;
  };

  //////////////////////////////////////////////////////////////////////////////
  // Template specializations - Light<int>

  template<>
  inline Light<int>::Light(const Point<float>& position_,
                           const SceneScaler&  scaler,
                           const RGB<int>&     color_) :
    color(color_)
  {
    position = scaler.scale(position_);
  }

} // namespace tangere

#endif // tangere_Light_h
