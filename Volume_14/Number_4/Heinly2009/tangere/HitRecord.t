
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

#ifndef tangere_HitRecord_t
#define tangere_HitRecord_t

#include <limits>

#include <Common/Types.h>

#include <Math/Math.h>

namespace tangere
{

  using Math::Vector;

  template<typename T>
  struct Shader;

  template<typename T>
  class Triangle;

  template<typename T>
  class HitRecord
  {
  public:
    inline HitRecord(const T&           tMin_     = numeric_limits<T>::max(),
                     const Triangle<T>* triangle_ = 0,
                     const Vector<T>&   bary_     = Vector<T>(0, 0, 0),
                     const Shader<T>*   shader_   = 0,
                           int          mID_      = -1) :
      tMin(tMin_), triangle(triangle_), bary(bary_), shader(shader_), mID(mID_)
    {
      // no-op
    }

    inline ~HitRecord()
    {
      // no-op
    }

    inline const T&           getMinT()       const { return tMin;     }
    inline const Triangle<T>* getTriangle()   const { return triangle; }
    inline const Shader<T>*   getShader()     const { return shader;   }
    inline const Vector<T>&   getBary()       const { return bary;     }
    inline       int          getMaterialID() const { return mID;      }

    inline bool hit(const T&           tMin_,
                    const Triangle<T>* triangle_,
                    const Vector<T>&   bary_,
                          int          mID_)
    {
      if (tMin_ > ::Constants<T>::Epsilon && tMin_ < tMin)
      {
        tMin     = tMin_;
        triangle = triangle_;
        bary     = bary_;
        shader   = triangle_->getShader();
        mID      = mID_;
        return true;
      }

      return false;
    }

  private:
          T            tMin;
    const Triangle<T>* triangle;
          Vector<T>    bary;
    const Shader<T>*   shader;
          int          mID;
  };

} // namespace tangere

#endif // tangere_HitRecordF_h
