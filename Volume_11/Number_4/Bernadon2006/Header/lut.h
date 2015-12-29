//-----------------------------------------------------------------------------
// File: lut.h
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
#ifndef __LUT_H__
#define __LUT_H__

#ifdef WIN32
#  include <windows.h>
#endif

#include <math.h>
#include "algebra3.h"
#define sizeTF 128

//-----------------------------------------------------------------------------
// LUT3D
//-----------------------------------------------------------------------------
class LUT3D {

private:
	vec4* _lutTF3D;

public:

	LUT3D(int i, int j, int k) {

    _lutTF3D = new vec4[i*j*k];
  }

  LUT3D() { _lutTF3D = NULL ;}
	~LUT3D() { if (_lutTF3D != NULL) delete _lutTF3D; } 

  void ComputeFixedLUT3D();
	void ComputeLUT3D(vec4 *tf, double maxEdgeLength);

	void set(int i, int j, int k, vec4& c) {

		_lutTF3D[i*sizeTF*sizeTF+j*sizeTF+k] = c; 
	}

  vec4 get(int i, int j, int k) {

    return _lutTF3D[i*sizeTF*sizeTF+j*sizeTF+k];
  }

	vec4 get2(int i, int j, int k) {

    return _lutTF3D[(sizeTF-j-1)*sizeTF*sizeTF+i*sizeTF+k];
  }

	void printSliceLUT3D(int k);
	void ComputeExactLookupTableVol(vec4* tf, int lut, double dl);
	void ComputeLookupTableVol(vec4 *tf, int lastDim, double dl);
  void ComputeIncLookupTableVol(int cur, int prev, int first, double l,
    double dl);
};


//-----------------------------------------------------------------------------
// LUT2D
//-----------------------------------------------------------------------------
class LUT2D {

public:
	vec4 _lutTF2D[sizeTF][sizeTF];

	LUT2D() {}
	void ComputeLUT2DFixed();
	void ComputeLUT2D(LUT3D *lut3D, int slice);
};


//-----------------------------------------------------------------------------
// LUT1D
//-----------------------------------------------------------------------------
class LUT1D {

public:
	vec4 _tf[sizeTF];

	LUT1D() {}
  void readTransferFunction(char *filename);
	void readTransferFunction2(char *filename, int dummystr1, int dummystr2,
    double dl);

  void BuildTransferFunction(char *colorFilename, char *alphaFilename,
                             double maxScalar, double minScalar, double maxEdgeLength);
}; 

#endif
