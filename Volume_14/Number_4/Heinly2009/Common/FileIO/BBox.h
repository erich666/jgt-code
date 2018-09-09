
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

#ifndef Common_FileIO_BBox_h
#define Common_FileIO_BBox_h

#include <iostream>
using std::istream;
using std::ostream;

#include <Math/Math.h>

namespace FileIO
{
  
  using Utility::Max;
  using Utility::Min;

  using MathF::Point;
  using MathF::Vector;

  class BBox
  {
  public:
    BBox(const Point& p0, const Point& p1, const Point& p2);
    BBox();

    ~BBox();
    
    void reset();

    void extend(const Point& p);
    void extend(const BBox&  b);

    Point  center()    const;
    Vector diagonal()  const;
    float  computeSA() const;

    Point getMin() const;
    Point getMax() const;

    friend istream& operator>>(istream& in,        BBox& b);
    friend ostream& operator<<(ostream& out, const BBox& b);

  private:
    Point min;
    Point max;
  };

  inline BBox::BBox()
  {
    reset();
  }

  inline BBox::BBox(const Point& p0, const Point& p1, const Point& p2)
  {
    reset();

    extend(p0);
    extend(p1);
    extend(p2);
  }

  inline BBox::~BBox()
  {
    // no-op
  }

  inline void BBox::reset()
  {
    min = Point::Max;
    max = Point::Min;
  }

  inline void BBox::extend(const Point& p)
  {
    min = Min(p, min);
    max = Max(p, max);
  }

  inline void BBox::extend(const BBox& b)
  {
    min = Min(b.min, min);
    max = Max(b.max, max);
  }

  inline Point BBox::center() const
  {
    return (min + max)/2;
  }

  inline Vector BBox::diagonal() const
  {
    return (max - min);
  }

  inline float BBox::computeSA() const
  {
    float x = max[0] - min[0];
    float y = max[1] - min[1];
    float z = max[2] - min[2];

    return 2.f*(x*y + y*z + x*z);
  }

  inline Point BBox::getMin() const
  {
    return min;
  }

  inline Point BBox::getMax() const
  {
    return max;
  }

} // namespace FileIO

#endif // Common_FileIO_BBox_h
