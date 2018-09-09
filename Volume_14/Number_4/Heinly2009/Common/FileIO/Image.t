
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

#ifndef Common_FileIO_Image_h
#define Common_FileIO_Image_h

#include <fstream>
using std::ios;
using std::ofstream;

#include <iomanip>
using std::setfill;
using std::setw;

#include <sstream>
using std::ostringstream;

#include <string>
using std::string;

#include <Common/FileIO/Pixel.h>
#include <Common/rtCore/RGB.t>
#include <Common/Utility/OutputCC.h>
#include <Common/Types.h>

#include <Math/Math.h>

namespace FileIO
{

  using MathF::Vector;

  template<typename T, const bool TONEMAP = false>
  class Image
  {
  public:
    static const float  GammaCorrection;
    static const float  LuminanceMax;
    static const Vector LuminanceRGB;

    Image(uint xres_, uint yres_) :
      xres(xres_), yres(yres_)
    {
      xres = (xres < 1 ? 1 : xres);
      yres = (yres < 1 ? 1 : yres);

      pixels = new T[xres*yres];
    }

    ~Image()
    {
      delete [] pixels;
    }

    uint  w()      const { return xres; }
    uint  h()      const { return yres; }
    float aspect() const { return float(xres)/float(yres); }

    void set(uint x, uint y, const T& p)
    {
      pixels[y*xres+x] = p;
    }

    void add(uint x, uint y, const T& p)
    {
      pixels[y*xres+x] += p;
    }


    void writePPM(const string& basename,
                        uint    iteration,
                        bool    overwrite) const
    {
      const float ispp  = 1.f/float(iteration + 1);
      const float scale = toneMap(ispp);

      ostringstream fname;
      fname << basename;
      if (!overwrite)
        fname << setfill('0') << setw(3) << iteration;
      fname << ".ppm";

      ofstream out(fname.str().c_str());
      if (!out.is_open())
        FatalError("Failed to open \"" << fname << " for writing" << endl);

      out << "P6" << endl;
      out << xres << ' ' << yres << endl;
      out << "255" << endl;

      for (int y = yres-1; y >= 0; --y)
        for (uint x = 0; x < xres; ++x)
          out << gammaCorrect(ispp*scale*Vector(pixels[y*xres+x]));

      out.close();
    }

    void writeBMP(const string& basename,
                        uint    iteration,
                        bool    overwrite) const
    {
      const float ispp  = 1.f/float(iteration + 1);
      const float scale = toneMap(ispp);

      ostringstream fname;
      fname << basename;
      if (!overwrite)
        fname << setfill('0') << setw(3) << iteration;
      fname << ".bmp";

      ofstream out(fname.str().c_str(), ios::binary);
      if (!out.is_open())
        FatalError("Failed to open \"" << fname << " for writing" << endl);

      // Write BMP header
      char data[3] = { 'B', 'M', 0 };
      out.write(data, 2);

      uint  widthSpacer = (4 - ((xres * BMP_BYTES_PER_COLOR) % 4)) % 4;
      ulong dataSize    = xres*yres*static_cast<ulong>(BMP_BYTES_PER_COLOR) +
        widthSpacer * yres;
      ulong fileSize = BMP_HEADER_SIZE + dataSize;

      writeLE(out, fileSize,                           4);
      writeLE(out, 0,                                  4);
      writeLE(out, BMP_HEADER_SIZE,                    4);
      writeLE(out, BMP_PARTIAL_HEADER_SIZE,            4);
      writeLE(out, xres,                               4);
      writeLE(out, yres,                               4);
      writeLE(out, BMP_NUM_OF_PLANES,                  2);
      writeLE(out, BMP_BYTES_PER_COLOR*BITS_PER_BYTE,  2);
      writeLE(out, BMP_COMPRESSION,                    4);
      writeLE(out, dataSize,                           4);
      writeLE(out, 0,                                 16);

      // Write pixel data
      for (uint y = 0; y < yres; ++y)
      {
        for (uint x = 0; x < xres; ++x)
        {
          const Pixel  p = gammaCorrect(ispp*scale*Vector(pixels[y*xres+x]));
          out << p.bgr();
        }

        // Align rows to four bytes
        for (uint i = 0; i < widthSpacer; ++i)
          out << static_cast<uchar>(0);
      }

      out.close();
    }

  private:

    ////////////////////////////////////////////////////////////////////////////
    // BMP support

    static const int BITS_PER_BYTE           = 8;
    static const int MAX_BYTE_VALUE          = 256;
    static const int BMP_HEADER_SIZE         = 54;
    static const int BMP_BYTES_PER_COLOR     = 3;
    static const int BMP_PARTIAL_HEADER_SIZE = 40;
    static const int BMP_NUM_OF_PLANES       = 1;
    static const int BMP_COMPRESSION         = 0;

    void writeLE(ostream& out, ulong num, uint nbytes) const
    {
      char* data = new char[nbytes];
      for (uint i = 0; i < nbytes; ++i)
      {
        data[i] = static_cast<char>(num % MAX_BYTE_VALUE);
        num /= MAX_BYTE_VALUE;
      }

      out.write(data, nbytes);
      
      delete [] data;
    }

    float toneMap(float /* unused */) const
    {
      return 1.f;
    }

    Pixel gammaCorrect(const Vector& v) const
    {
      float value;
      Pixel rgb;
      for (uint i = 0; i < 3; ++i)
      {
        value   = powf((v[i] < 0.f ? 0.f : v[i]), GammaCorrection);
        value   = floorf(255.f*value + 0.5f);
        rgb[i]  = static_cast<uchar>(value > 255.f ? 255.f : value);
      }

      return rgb;
    }

    uint xres;
    uint yres;
    T*   pixels;
  };

  //////////////////////////////////////////////////////////////////////////////
  // Static constant members

  template<typename T, const bool TONEMAP>
  const float Image<T, TONEMAP>::GammaCorrection = 0.45f;

  template<typename T, const bool TONEMAP>
  const float Image<T, TONEMAP>::LuminanceMax = 200.f;

  template<typename T, const bool TONEMAP>
  const Vector Image<T, TONEMAP>::LuminanceRGB = Vector(0.2126f, 0.7152f, 0.0722f);

  ///////////////////////////////////////////////////////////////////////////////
  // Template specialization - rtCore::RGB<float>, TONEMAP = true

  //////////////////////////////////////////////////////////////////////////////
  // Ward's tonemapper
  //
  //   "A Contrast-Based Scalefactor for Luminance Display"
  //   Greg Ward, Graphics Gems 4, Morgan Kaufmann 1994

  // XXX(cpg) - why does this template specilaization have to be inline'd
  //            with g++ 4.2.1 on Mac (and maybe elsewhere)?
  template<>
  inline float Image<rtCoreF::RGB, true>::toneMap(float ispp) const
  {
    float sum = 0.f;
    for (uint i = 0; i < xres*yres; ++i)
    {
      const float Y = ispp*Dot(Vector(pixels[i]), LuminanceRGB);
      sum += log10f(Y > 1e-4f ? Y : 1e-4f);
    }

    const float logMeanLuminance = powf(10.f, sum/float(xres*yres));

    const float a = 1.219f + powf(0.25f*LuminanceMax, 0.4f);
    const float b = 1.219f + powf(logMeanLuminance,   0.4f);

    return powf(a/b, 2.5f)/LuminanceMax;
  }

} // namespace FileIO

#endif // Common_FileIO_Image_h
