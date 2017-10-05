//-----------------------------------------------------------------------------
// File: lut.cpp
// Desc: class to handle transfer functions
// Copyright (C) 2005, Joao Comba, Fabio Bernardon, UFRGS-Brasil
//-----------------------------------------------------------------------------
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
//-----------------------------------------------------------------------------
#include "../Header/lut.h"
#include <stdio.h>
#include <vector>

using namespace std;

inline double clamp_hi (double x, double hi) { return x > hi ? hi : x; }

//-----------------------------------------------------------------------------
// LUT1D::readTransferFunction
// Load the tetrahedral mesh into the internal data structure
//-----------------------------------------------------------------------------
void LUT1D::readTransferFunction(char *filename) {

  // aux vars
  int size;
  float r, g, b, a;
  FILE *fd = fopen(filename, "r");

  fscanf (fd, "%d", &size);
  for (int i = 0; i < size; i++) {

    fscanf (fd, "%f %f %f %f", &r, &g, &b, &a);
    _tf[i] = vec4(r, g, b, a);
  }

  fclose(fd);
}


//-----------------------------------------------------------------------------
// LUT1D::readTransferFunction2
// Load the tetrahedral mesh into the internal data structure
//-----------------------------------------------------------------------------
void LUT1D::readTransferFunction2(char *filename, int dummystr1, int dummystr2,
                                  double dl) {

  // aux vars
  float fff[512];
  int size = 128;
  FILE *fd = fopen(filename, "rb");

  fread(&fff, sizeof(float), 512, fd);
  fclose(fd);

  for (int i=0; i<size; i++) {

    vec4 v(fff[4*i], fff[4*i+1], fff[4*i+2], fff[4*i+3]);
    _tf[i][0] = v[0];
    _tf[i][1] = v[1];
    _tf[i][2] = v[2];
    _tf[i][3] = v[3];
  }
}


//-----------------------------------------------------------------------------
// Name: LUT3D::ComputeIncLookupTableVol ()
// Desc: Compute Pre-Integrated LUT incrementally
//-----------------------------------------------------------------------------
void LUT3D::ComputeIncLookupTableVol (int cur, int prev, int first, double l,
                                      double dl) {

  // aux vars
  int n = sizeTF;

  for(int i = 0; i < sizeTF; i++) {

    for(int j = 0; j < sizeTF; j++) {

      double sf = (2.0 * j + 1.0)/(2.0 * n);
      double sb = (2.0 * i + 1.0)/(2.0 * n);
      double sp = ((l-dl)*sf + (dl*sb)) / l;

      int k = (int)(sp*n - 0.5);
      double beta = sp*n - (k+0.5);
      vec4 c_first;
      vec4 c_prev;
      if (k == n-1) {

        c_first = get(k, j, first);
        c_prev = get(i, k, prev);
      }
      else {

        c_first =	get(k, j, first) * (1.0 - beta) + get(k+1, j, first) * beta;
        c_prev =	get(i, k, prev) * (1.0 - beta) + get(i, k+1, prev) * beta;
      }

      set(i, j, cur, c_first + c_prev * (1.0 - c_first[3]));
    }
  }
}


//-----------------------------------------------------------------------------
// Name: LUT3D::ComputeExactLookupTableVol ()
// Desc: Compute Pre-Integrated LUT
//-----------------------------------------------------------------------------
void LUT3D::ComputeExactLookupTableVol (vec4* tf, int lut, double dl) {

  // aux vars
  int n = sizeTF;
  const int M = 8; // supersampling factor
  for (int i=0; i<n-1; i++) {

    for (int j=0; j<n-1; j++) {

      if (i == j) {

        vec4 c = tf[i];
        c[3] = 1.0f - pow((double)(1.0f - c[3]), (double)dl);
        c[0] *= c[3];
        c[1] *= c[3];
        c[2] *= c[3];
        set(i, j, lut, c);
      }
      else if (i > j) {

        vec4 c(0.0, 0.0, 0.0, 0.0);
        for (int k=j; k<i; k++) {

          float beta = 0.0;
          float dbeta = 1.0/M;
          float dgamma = 1.0f/(float)(i-j);
          vec4 c0 = tf[k];
          vec4 c1 = tf[k+1];
          c0[3] = 1.0f - pow((double)(1.0f - c0[3]), (double)(dl*dbeta*dgamma));
          c0[0] *= c0[3];
          c0[1] *= c0[3];
          c0[2] *= c0[3];
          c1[3] = 1.0f - pow((double)(1.0f - c1[3]), (double)(dl*dbeta*dgamma));
          c1[0] *= c1[3];
          c1[1] *= c1[3];
          c1[2] *= c1[3];
          for (int m=0; m<M; m++, beta+=dbeta) {

            vec4 ck = c0*(1-beta) + c1*beta;
            c += ck*(1.0 - c[3]);
          }
        }
        set(i, j, lut, c);//[i][j][lut] = c;
      }
      else {

        vec4 c(0.0, 0.0, 0.0, 0.0);

        for (int k=j; k>i; k--) {

          float beta = 0.0;
          float dbeta = 1.0/M;
          float dgamma = 1.0f/(float)(j-i);
          vec4 c0 = tf[k];
          vec4 c1 = tf[k-1];
          c0[3] = 1.0f - pow((double)(1.0f - c0[3]), (double)(dl*dbeta*dgamma));
          c0[0] *= c0[3];
          c0[1] *= c0[3];
          c0[2] *= c0[3];
          c1[3] = 1.0f - pow((double)(1.0f - c1[3]), (double)(dl*dbeta*dgamma));
          c1[0] *= c1[3];
          c1[1] *= c1[3];
          c1[2] *= c1[3];
          for (int m=0; m<M; m++, beta+=dbeta) {

            vec4 ck = c0*(1-beta) + c1*beta;
            c += ck*(1.0 - c[3]);
          }
        }
        set(i, j, lut, c);//[i][j][lut] = c;
      }
    }
  }
}


//-----------------------------------------------------------------------------
// Name: LUT3D::printSliceLUT3D()
// Desc: print Pre-Integrated LUT slice
//-----------------------------------------------------------------------------
void LUT3D::printSliceLUT3D(int k) {

  // aux vars
  char filename[100];
  sprintf (filename, "pre-int%d.txt", k);
  FILE *fd = fopen (filename, "w");

  for (int i=0; i<sizeTF; i++) {

    for (int j=0; j<sizeTF; j++) {

      fprintf(fd, "(%f,%f,%f,%f)", get(i, j, k)[0], get(i, j, k)[1],
        get(i, j, k)[2], 
        get(i, j, k)[3]);
    }
    fprintf(fd,"\n");
  }
  fclose(fd);
}


//-----------------------------------------------------------------------------
// Name: LUT3D::ComputeFixedLUT3D()
// Desc: Compute Pre-Integrated LUT
//-----------------------------------------------------------------------------
void LUT3D::ComputeFixedLUT3D() {

  for(int i = 0; i < sizeTF; i++)
    for(int j = 0; j < sizeTF; j++)
      for (int k=0; k < sizeTF; k++)
        set(i, j, k, vec4(1.0, 0.0, 1.0, 0.05));
}


//-----------------------------------------------------------------------------
// Name: LUT3D::ComputeLUT3D()
// Desc: Compute Pre-Integrated LUT
//-----------------------------------------------------------------------------
void LUT3D::ComputeLUT3D(vec4 *tf, double maxEdgeLength) {
  double dl = maxEdgeLength/(double)(sizeTF - 1);
  double l = dl;
  int printslices = 0;

  for(int i = 0; i < sizeTF; i++) {
    for(int j = 0; j < sizeTF; j++) {
      for(int k = 0; k < sizeTF; k++) {
        set(i, j, k, vec4(0.0, 0.0, 0.0, 0.0)); 
      }
    }
  }

  ComputeExactLookupTableVol(tf, 1, l);

  if (printslices) printSliceLUT3D(1);

  for(int i = 2; i < sizeTF; i++) {
    if (i%10==0) printf("i ");
    l+=dl;
    ComputeIncLookupTableVol(i, i-1, 1, l, dl);
    if (printslices) printSliceLUT3D(i);
  }
}


//-----------------------------------------------------------------------------
// Name: LUT2D::ComputeLUT2DFixed()
// Desc: Compute Pre-Integrated LUT
//-----------------------------------------------------------------------------
void LUT2D::ComputeLUT2DFixed() {

  for(int i=0; i < sizeTF; i++)
    for(int j=0; j < sizeTF; j++)
      _lutTF2D[i][j] = vec4(0.0, 1.0, 1.0, 0.25);
}


//-----------------------------------------------------------------------------
// Name: LUT2D::ComputeLUT2D()
// Desc: Compute Pre-Integrated LUT based on a 3D lut slice
//-----------------------------------------------------------------------------
void LUT2D::ComputeLUT2D(LUT3D *lut3D, int slice) {

  for(int i=0; i < sizeTF; i++)
    for(int j=0; j < sizeTF; j++)
      _lutTF2D[i][j] = lut3D->get(i, j, slice);
}


//-----------------------------------------------------------------------------
// Name: LUT3D::ComputeLookupTableVol()
// Desc: compute 2D slice preintegrated lookup table -- ignore self-attenuation 
//-----------------------------------------------------------------------------
void LUT3D::ComputeLookupTableVol (vec4 *tf, int lastDim, double dl) {

  vector<vec4> mIntTable;
  vector<vec4> mDiagTable;
  mIntTable.resize(sizeTF);
  mDiagTable.resize(sizeTF);

  // pre-integration -- independent of interpolation (nearest or linear)
  vec4 c = tf[0];
  double t = 1.0 - c[3];

  // cap t, so that tau does not go to infinity
  // when converted back to alpha, it will be rounded to 1.0 due to the finite
  // resolution of the table
  double tau = -log(t < 0.5/(double)sizeTF ? 0.5/(double)sizeTF : t);
  vec4 color(c[0]*tau, c[1]*tau, c[2]*tau, tau);
  mIntTable[0] = color*0.5;
  set(0, 0, lastDim, color*dl);
  vec4 color_prev(color);
  for (int i=1; i<sizeTF; i++) {

    c = tf[i];
    t = 1.0 - c[3];
    tau = -log(t < 0.5/(double)sizeTF ? 0.5/(double)sizeTF : t);
    //color.set(c.x*tau, c.y*tau, c.z*tau, tau);
    color = vec4(c[0]*tau, c[1]*tau, c[2]*tau, tau);
    mDiagTable[i] = color;
    mIntTable[i] = mIntTable[i-1] + (color + color_prev)*0.5;
    color_prev = color;
  }

  // simple update ignores self-attenuation within a ray segment
  // this means that the 2D pre-integration table is symmetric about the diagonal
  for (int i=0; i<sizeTF; i++) {

    for (int j=0; j<sizeTF; j++) {

      // when the two scalars are equal, the formulas in the paper
      // are not valid, but we get the following from the integral equations
      // we also consider the elementary ray length (dl) here
      vec4 val;
      if (i == j)
        val = mDiagTable[i] * dl;
      else
        val = (mIntTable[j] - mIntTable[i]) / (j - i) * dl;
      // clamp -- this is important, otherwise the values will overflow
      val[0] = clamp_hi(val[0], 1.0f);
      val[1] = clamp_hi(val[1], 1.0f);
      val[2] = clamp_hi(val[2], 1.0f);
      // compute opacity
      val[3] = 1.0f - exp(-val[3]);
      set(i, j, lastDim, vec4(val));
    }
  }
}



//--------------------------------------------------------------------------
// Build a pre-integrated lookup table using a user specified transfer
// function.
//--------------------------------------------------------------------------
void LUT1D::BuildTransferFunction(char *colorFilename, char *alphaFilename,
                                  double maxScalar, double minScalar,
                                  double maxEdgeLength) {

  cout << "Creating transfer function... " << flush;

  int mSize = sizeTF;

  double range = maxScalar - minScalar;
  vector<vec4> colors;
  vector<float> colorValues;

  if (colorFilename) {

    // Read colormap file
    ifstream cfile(colorFilename);
    if (!cfile) {

      cerr << "Error: Can't open input file " << colorFilename << "!\n";
      exit(-1);
    }

    char *tmp = new char[20];
    float val, red, green, blue;
    cfile >> tmp;
    while (cfile >> val >> red >> green >> blue) {

      vec4 rgb(red, green, blue, 1.0);
      colors.push_back(rgb);
      colorValues.push_back(val);
    }
    cfile.close();
    delete tmp;
  }
  else {

    // Default rainbow colormap
    vec4 red(1.0, 0.0, 0.0, 1.0);
    vec4 yellow(1.0, 1.0, 0.0, 1.0);
    vec4 green(0.0, 1.0, 0.0, 1.0);
    vec4 cyan(0.0, 1.0, 1.0, 1.0);
    vec4 blue(0.0, 0.0, 1.0, 1.0);
    vec4 magenta(1.0, 0.0, 1.0, 1.0);
    colors.push_back(red);
    colors.push_back(yellow);
    colors.push_back(green);
    colors.push_back(cyan);
    colors.push_back(blue);
    colors.push_back(magenta);
    colorValues.push_back((float)(minScalar-range/100.0));
    colorValues.push_back((float)(1.0*(maxScalar-minScalar)/5.0 + minScalar));
    colorValues.push_back((float)(2.0*(maxScalar-minScalar)/5.0 + minScalar));
    colorValues.push_back((float)(3.0*(maxScalar-minScalar)/5.0 + minScalar));
    colorValues.push_back((float)(4.0*(maxScalar-minScalar)/5.0 + minScalar));
    colorValues.push_back((float)(maxScalar+range/100.0));
  }

  vector<float> alphas;
  vector<float> alphaValues;
  if (alphaFilename) {

    // Read opacitymap file
    char *tmp = new char[20];
    float a, val;
    ifstream afile(alphaFilename);
    if (!afile) {

      cerr << "Error: Can't open input file " << alphaFilename << "!\n" << endl;
      exit(-1);
    }
    afile >> tmp;
    while (afile >> val >> a) {

      alphas.push_back(a);
      alphaValues.push_back(val);
    }
    afile.close();
    delete tmp;
  }
  else {

    // Default opacity map
    alphas.push_back(0.2f);
    alphas.push_back(0.2f);
    alphaValues.push_back((float)(minScalar-range/100.0));
    alphaValues.push_back((float)(maxScalar+range/100.0));
  }

  // Create Transfer Function by sampling color and opacity ranges
  double r = minScalar;
  double dr = range/(double)(mSize-1);
  for(int i = 0; i < mSize; i++) {

    // Sample color
    float prev = colorValues[0];
    float next = prev;
    int csize = colorValues.size();
    int index = 0;
    for (index = 1; index < csize; index++) {

      next = colorValues[index];
      if (prev < r && next >= r)
        break;
      prev = next;
    }

    float interpNext = (float)((r - prev)/(next-prev));
    float interpPrev = (float)(1.0 - interpNext);
    vec4 nextRGB = colors[index];
    vec4 prevRGB = colors[index-1];

    _tf[i][0] = nextRGB[0]*interpNext + prevRGB[0]*interpPrev;
    _tf[i][1] = nextRGB[1]*interpNext + prevRGB[1]*interpPrev;
    _tf[i][2] = nextRGB[2]*interpNext + prevRGB[2]*interpPrev;

    // Sample opacity
    prev = alphaValues[0];
    next = prev;
    int asize = alphaValues.size();
    index = 0;
    for (index = 1; index < asize; index++) {

      next = alphaValues[index];
      if (prev < r && next >= r)
        break;
      prev = next;
    }

    interpNext = (float)((r - prev)/(next-prev));
    interpPrev = (float)(1.0 - interpNext);
    float nextAlpha = alphas[index];
    float prevAlpha = alphas[index-1];

    _tf[i][3] = nextAlpha*interpNext + prevAlpha*interpPrev;

    r += dr;
  }
  cout << "done!\n" << flush;
}