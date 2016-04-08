/* volume.h
 *
 *  $Id: volume.h,v 1.2 2001/07/15 04:57:59 xue Exp $
 *  $Log: volume.h,v $
 *  Revision 1.2  2001/07/15 04:57:59  xue
 *  Modify the format to store voxel location
 *
 *  Revision 1.1  2001/07/14 00:41:28  xue
 *  complete basic functions for splatter
 *
 *  Revision 1.1  2001/07/08 21:49:33  xue
 *  Initial version
 *
 */

#ifndef __VOLUME_H
#define __VOLUME_H

#include <vector>
using namespace std;

#include <math.h>
#include "Datatype.h"
#include "OSUmatrix.h"


typedef struct _VOXEL {
  REAL x, y, z;
  REAL r,g,b,a; 
}VOXEL;

class CVolume
{
 public:
  CVolume();
  CVolume(INT cx, INT cy, INT cz);
  ~CVolume();
  virtual void SetVoxel(INT i, REAL x, REAL y, REAL z, REAL r, REAL g, REAL b, REAL a);
  virtual void SetVoxel(INT i, REAL x, REAL y, REAL z, REAL intens);
  void SetVolume2World(const float sx, const float sy, const float sz,
                       const float rx, const float ry, const float rz,
                       const float tx, const float ty, const float tz);

  void SetMaterial(REAL ka, REAL kd, REAL ks, REAL kn);
  void SetColor(REAL r, REAL g, REAL b);
  void Z_MinMax(REAL *min, REAL *max);
  virtual void ReadVol(char* filename);
  void Transform(MATRIX4  mat);
  void Clear();

 public:
  MATRIX4 volume2world;
  INT voxel_num;
  VOXEL *voxel; 
  INT nx, ny, nz;             // the sample points along x, y, z
  INT max_nx, max_ny, max_nz; // the max sample points along x, y, z
  REAL ka, kd, ks, kn;				// material for volume
  REAL red, green, blue, alpha;      // color for volume

  REAL center[3];		// center for volume;
};

class CCube : public CVolume {
 public:
  CCube(INT cx);
  
 private:
  void Initialize();
};

class CRectangle : public CVolume {
 public:
  CRectangle(INT cx, INT cy, INT cz);
  
 private:
  void Initialize();
};

class CSphere : public CVolume {
 public:
  CSphere(INT cx);
  
 private:
  void Initialize();
};

class CDensSphere : public CVolume {
 public:
  CDensSphere(INT cx);

 private:
  void Initialize();
};


class CTorus : public CVolume {
 public:
  CTorus(INT cx, INT cy, INT cz);
  
 private:
  void Initialize();

 private:
  REAL R;
  REAL r;
};

struct HIPIPH_LUT {
	float r, g, b, a;
	float k1,k2,k3,k4,k5;
};

class CHipiph : public CVolume {
 public:
  CHipiph(char* filename);
  
 private:
  void Initialize(char* filename);
  void LoadLut();
	
  HIPIPH_LUT lut[256];
};

class CFuel : public CVolume {
 public:
  CFuel(char* filename);
  virtual void ReadVol(char* filename);
  
 private:
  void Initialize(char* filename);
  void LoadLut();
	
  HIPIPH_LUT lut[256];
};

class CHydrogen : public CVolume {
 public:
  CHydrogen(char* filename);
  virtual void ReadVol(char* filename);
  
 private:
  void Initialize(char* filename);
};

class CCThead : public CVolume {
 public:
  CCThead(char* filename);
  virtual void ReadVol(char* filename);
  
 private:
  void Initialize(char* filename);
  void CCThead::ReadVol2(char* filename);

};

class CFoot : public CVolume {
 public:
  CFoot(char* filename);
  virtual void ReadVol(char* filename);
  
 private:
  void Initialize(char* filename);

};

class CFoot2 : public CVolume {
 public:
  CFoot2(char* filename);
  virtual void ReadVol(char* filename);
  
 private:
  void Initialize(char* filename);

};

class CSkull : public CVolume {
 public:
  CSkull(char* filename);
  virtual void ReadVol(char* filename);
  
 private:
  void Initialize(char* filename);

};

class CUncbrain : public CVolume {
 public:
  CUncbrain(char* filename);
  virtual void ReadVol(char* filename);
  virtual void SetVoxel(INT i, REAL x, REAL y, REAL z, REAL r, REAL g, REAL b, REAL a);
  
 private:
  void Initialize(char* filename);

};

#endif /* __VOLUME_H */
