
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

#ifndef Common_FileIO_Pixel_h
#define Common_FileIO_Pixel_h

#include <iosfwd>
using std::ifstream;
using std::ofstream;
using std::istream;
using std::ostream;

#include <Common/Types.h>

#include <Math/Math.h>

namespace FileIO
{

  class Pixel
  {
  public:
    Pixel(uchar, uchar, uchar);
    Pixel();

    uchar& operator[](uint);
    uchar  operator[](uint) const;

    Pixel rgb() const;
    Pixel bgr() const;

    friend istream& operator>>(istream&,       Pixel&);
    friend ostream& operator<<(ostream&, const Pixel&);

    friend ifstream& operator>>(ifstream&,       Pixel&);
    friend ofstream& operator<<(ofstream&, const Pixel&);

  private:
    uchar e[3];
  };

  inline Pixel::Pixel(uchar r, uchar g, uchar b)
  {
    e[0] = r;
    e[1] = g;
    e[2] = b;
  }

  inline Pixel::Pixel()
  {
    e[0] = 0;
    e[1] = 0;
    e[2] = 0;
  }

  inline Pixel Pixel::rgb() const
  {
    return *this;
  }

  inline Pixel Pixel::bgr() const
  {
    return Pixel(e[2], e[1], e[0]);
  }

  inline uchar& Pixel::operator[](uint i)
  {
    return e[i];
  }

  inline uchar Pixel::operator[](uint i) const
  {
    return e[i];
  }

} // namespace FileIO

#endif // Common_FileIO_Pixel_h
