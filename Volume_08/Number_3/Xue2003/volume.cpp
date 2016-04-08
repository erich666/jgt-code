/* volume.C
 *
 *  $Id: volume.C,v 1.2 2001/07/15 04:58:00 xue Exp $
 *  $Log: volume.C,v $
 *  Revision 1.2  2001/07/15 04:58:00  xue
 *  Modify the format to store voxel location
 *
 *
 */

#include <stdio.h>
#include <fcntl.h>
#include <io.h>
#include <fstream>
#include <assert.h>
#include "defines.h"
#include "OSUmatrix.h"
#include "volume.h"

#define max(a,b)  ((a)>(b)?(a):(b))
#define min(a,b)  ((a)<(b)?(a):(b))

#define SWAP_BYTE(w)  { WORD lo = w>>8; WORD hi = (w&0x00ff)<<8; w = hi|lo; }

#define RGB2GRAY(r, g, b)	(0.30*(r)+0.59*(g)+0.11*(b))

class MyHistogram{
public:
	MyHistogram(int num);
	~MyHistogram();
	void Save();
	void Histogram(int index);
	void Init(int num);

private:
	int entries;
	int *histogram;
};

MyHistogram::MyHistogram(int num)
{
	Init(num);
}

MyHistogram::~MyHistogram()
{
	//if (histogram) delete [] histogram;
}

void MyHistogram::Histogram(int index)
{
	histogram[index] += 1;
}

void MyHistogram::Init(int num)
{
	entries = num;
	histogram = new int[entries];
	memset(histogram, 0, sizeof(int)*entries);
}

void MyHistogram::Save()
{
	ofstream outf("out.his");
	for (int i=0; i<entries; i++){
		outf << i << "\t" << histogram[i] << endl;
	}
}

//********************
// CVolume
//********************

CVolume::CVolume()
{
  max_nx = max_ny = max_nz = 256;
  nx = ny = nz = 0;
  voxel_num = 0;
  if (voxel_num>0)
    voxel = new VOXEL[voxel_num];
  else
    voxel = NULL; 

  center[0] = center[1] = center[2] = 0.5;

  SetVolume2World(1,1,1,0,0,0,0,0,0);  
}

CVolume::CVolume(INT cx, INT cy, INT cz)
{
  max_nx = max_ny = max_nz = 128;
  nx = cx; ny = cy; nz = cz;
  voxel_num = nx*ny*nz;
  if (voxel_num>0)
    voxel = new VOXEL[voxel_num];
  else
    voxel = NULL; 

  center[0] = center[1] = center[2] = 0.5;

  SetVolume2World(1,1,1,0,0,0,0,0,0);  
}

CVolume::~CVolume()
{
  if (voxel!=NULL)
    delete [] voxel;
}

void CVolume::Clear()
{
  if (voxel!=NULL)
    delete [] voxel;
}

void CVolume::SetMaterial(const float ambient, const float diffuse,
			  const float specular, const float specular_exp)  
{
  ka = ambient;  kd = diffuse;  ks = specular;  kn = specular_exp;
}

void CVolume::SetColor(REAL r, REAL g, REAL b)
{
	red = r;  green = g;  blue = b;
}

void CVolume::SetVolume2World(const float sx, const float sy, const float sz,
                              const float rx, const float ry, const float rz,
                              const float tx, const float ty, const float tz)
// sx, sy, sz = x,y,z scaling factors
// rx, ry, rz = (CCW) rotation around x,y,z axes in degrees
// tx, ty, tz = x,y,z translation distances
{
  MATRIX4 scale_mat;
  MATRIX4 rot_x_mat, rot_y_mat, rot_z_mat;
  MATRIX4 translation_mat;

  // scaling
  scale_mat[0][0] = sx;
  scale_mat[1][1] = sy;
  scale_mat[2][2] = sz;

  // rotating
  float rad_x = rx*3.14159/180;
  rot_x_mat[1][1] = cos(rad_x);
  rot_x_mat[1][2] = -sin(rad_x);
  rot_x_mat[2][1] = sin(rad_x);
  rot_x_mat[2][2] = cos(rad_x);

  float rad_y = ry*3.14159/180;
  rot_y_mat[0][0] = cos(rad_y);
  rot_y_mat[0][2] = sin(rad_y);
  rot_y_mat[2][0] = -sin(rad_y);
  rot_y_mat[2][2] = cos(rad_y);

  float rad_z = rz*3.14159/180;
  rot_z_mat[0][0] = cos(rad_z);
  rot_z_mat[0][1] = -sin(rad_z);
  rot_z_mat[1][0] = sin(rad_z);
  rot_z_mat[1][1] = cos(rad_z);

  // translating
  translation_mat[0][3] = tx;
  translation_mat[1][3] = ty;
  translation_mat[2][3] = tz;


  volume2world = translation_mat * rot_z_mat * rot_y_mat * rot_x_mat * scale_mat;
}


void CVolume::SetVoxel(INT i, REAL x, REAL y, REAL z, REAL r, REAL g, REAL b, REAL a)
{
  assert (i<voxel_num);

  int m = max(nx, max(ny, nz));
  //int m = min(nx, min(ny, nz));

  voxel[i].x = m==1 ? x : x/(m-1.0);
  voxel[i].y = m==1 ? y : y/(m-1.0);
  voxel[i].z = m==1 ? z : z/(m-1.0);
  voxel[i].r = r;
  voxel[i].g = g;
  voxel[i].b = b;
  voxel[i].a = a;
  

  /* for testing time/voxel
  int m = nx; //max(nx, max(ny, nz));
  //int m = min(nx, min(ny, nz));
  voxel[i].x = m==1 ? x : x/(m-1.0);
  voxel[i].y = m==1 ? y : y/(m-1.0);

  int n = max(nx, nz);
  voxel[i].z = m==1 ? z : z/(n-1.0);
  voxel[i].intensity = intens;
*/
}


void CVolume::SetVoxel(INT i, REAL x, REAL y, REAL z, REAL intens)
{
  assert (i<voxel_num);

  //int m = max(nx, max(ny, nz));
  //int m = min(nx, min(ny, nz));
	int m = max(nx, ny);
	
  voxel[i].x = m==1 ? x : x/(m-1.0);
  voxel[i].y = m==1 ? y : y/(m-1.0);
  voxel[i].z = m==1 ? z : z/(m-1.0);
  voxel[i].r = intens;
  voxel[i].g = intens;
  voxel[i].b = intens;
  voxel[i].a = intens;
  

  /* for testing time/voxel
  int m = nx; //max(nx, max(ny, nz));
  //int m = min(nx, min(ny, nz));
  voxel[i].x = m==1 ? x : x/(m-1.0);
  voxel[i].y = m==1 ? y : y/(m-1.0);

  int n = max(nx, nz);
  voxel[i].z = m==1 ? z : z/(n-1.0);
  voxel[i].intensity = intens;
*/
}
/*
void CVolume::SetVoxel(INT i, REAL x, REAL y, REAL z, REAL intens)
{
  assert (i<voxel_num);

  int m = max(nx, max(ny, nz));
  //int m = min(nx, min(ny, nz));
  voxel[i].x = m==1 ? x : x/(m-1.0);
  voxel[i].y = m==1 ? y : y/(m-1.0);
  voxel[i].z = m==1 ? z : z/(m-1.0);
  voxel[i].r = intens;
  voxel[i].g = intens;
  voxel[i].b = intens;
  voxel[i].a = intens;
  

}
*/

void CVolume::Z_MinMax(REAL *min, REAL *max)
{
  if (voxel!=NULL){
    REAL mn = voxel[0].z;
    REAL mx = voxel[0].z;

    for (int i=0; i<voxel_num; i++){
      REAL z = voxel[i].z;
      if (z < mn)  mn = z;
      if (z > mx)  mx = z;
    }

    *min = mn;
    *max = mx;
  }
}

void CVolume::Transform(MATRIX4  mat)
{
  for (int i=0; i<voxel_num; i++){
    VECTOR4 vox(voxel[i].x, voxel[i].y, voxel[i].z, 1);
    vox = mat *volume2world * vox;
    SetVoxel(i, vox[0], vox[1], vox[2], voxel[i].r, voxel[i].g, voxel[i].b, voxel[i].a);
    }
}
/*
void CVolume::ReadVol(char* filename)
{  
  FILE *fp;
  INT cx, cy, cz;

  fp = fopen(filename, "r");
  if (fp == NULL){
    fprintf(stderr, "could not open/read %s to output\n", filename);
    return;
  }
  
  fscanf(fp, "%d %d %d\n", &cx, &cy, &cz);
  nx = cx>max_nx ? max_nx : cx;
  ny = cy>max_ny ? max_ny : cy;
  nz = cz>max_nz ? max_nz : cz;

  voxel_num = nx*ny*nz;

  voxel = new VOXEL[voxel_num];
  
  BYTE *buffer = new BYTE[cx*cy];

  INT index = 0;
  if (voxel!=NULL){
    BYTE intensity;
    for(INT i=0; i<nz; i++){
	  fread(buffer, 1, cx*cy, fp);
	  printf("read %d frame...\n", i);
      for(INT j=0; j<ny; j++){
		for(INT k=0; k<nx; k++){
		  INT index = i*nx*ny + j*ny + k;
		  intensity = buffer[j*cx+k];
		  SetVoxel(index, k, j, i, intensity, intensity?1:0);
		}
	  }
    }
  }
  delete [] buffer;
  fclose(fp);
}
*/

void CVolume::ReadVol(char* filename)
{  
  FILE *fp;

  fp = fopen(filename, "rt");
  if (fp == NULL){
    fprintf(stderr, "could not open/read %s to output\n", filename);
    return;
  }
  
  fscanf(fp, "%d %d %d\n", &nx, &ny, &nz);

  _setmode(_fileno(fp), _O_BINARY);

  voxel_num = nx*ny*nz;

  voxel = new VOXEL[voxel_num];
  
  BYTE mmin=255, mmax=0;
  BYTE *buffer = new BYTE[nx*ny];

  INT index = 0;
  if (voxel!=NULL){
    BYTE intensity;
    for(INT i=0; i<nz; i++){
	  fread(buffer, 1, nx*ny, fp);
	  printf("read %d frame...\n", i);
      for(INT j=0; j<ny; j++){
		for(INT k=0; k<nx; k++){
		  INT index = i*nx*ny + j*ny + k;
		  intensity = buffer[j*nx+k];
						
		  if (intensity > mmax)  mmax = intensity;
		  if (intensity < mmin)  mmin = intensity;

		  SetVoxel(index, k, j, i, intensity);
		}
	  }
    }
  }

  cout << "range: " << (int)mmin <<" - " << (int)mmax<<endl;

  delete [] buffer;
  fclose(fp);
}

//********************
// CCube
//********************

// member functions for class CCube
CCube::CCube(INT cx):CVolume(cx, cx, cx)
{
  Initialize();
} 

void CCube::Initialize()
{
  assert (voxel_num>0);
  for (int i=0; i<nz; i++)
    for (int j=0; j<ny; j++)
      for (int k=0; k<nx; k++){
	//SetVoxel(i*ny*nx+j*nx+k, k, j, i, 255*0.005);
	SetVoxel(i*ny*nx+j*nx+k, k, j, i, 255);
      }
}


//********************
// CRectangel
//********************

// member functions for class CCube
CRectangle::CRectangle(INT cx, INT cy, INT cz):CVolume(cx, cy, cz)
{
  Initialize();
} 

void CRectangle::Initialize()
{
  assert (voxel_num>0);
  for (int i=0; i<nz; i++)
    for (int j=0; j<ny; j++)
      for (int k=0; k<nx; k++){
	SetVoxel(i*ny*nx+j*nx+k, k, j, i, 255);
      }
}

//********************
// CSphere
//********************

// member fuctions for class CSphere
CSphere::CSphere(INT cx): CVolume(cx, cx, cx)
{
  Initialize();
}

void CSphere::Initialize()
{
  INT intensity;
  assert (voxel_num>0);
  REAL c = ((REAL)nx-1)/2.0;
  for (int i=0; i<nx; i++)
    for (int j=0; j<nx; j++)
      for (int k=0; k<nx; k++){
	REAL rr = (k-c)*(k-c) + (j-c)*(j-c) + (i-c)*(i-c);
	INT index = i*nx*nx + j*nx + k;
	if (rr<c*c)
       intensity = 255;
	else
		intensity = 0;
	SetVoxel(index, k, j, i, intensity);
 }
}

//********************
// CDensSphere
//********************

// member fuctions for class CDensSphere
CDensSphere::CDensSphere(INT cx): CVolume(cx, cx, cx)
{
  Initialize();
}

void CDensSphere::Initialize()
{
  assert (voxel_num>0);
  REAL c = ((REAL)nx-1)/2.0;
  REAL r_in  = 0.0*c;
  REAL r_out = 0.9*c;
  for (int i=0; i<nx; i++)
    for (int j=0; j<nx; j++)
      for (int k=0; k<nx; k++){
        REAL rr = (k-c)*(k-c) + (j-c)*(j-c) + (i-c)*(i-c);
        REAL rr_in  = r_in*r_in;
        REAL rr_out = r_out*r_out;
        if (rr<rr_in)
          SetVoxel(i*nx*nx+j*nx+k, k, j, i, 1.0);
        else if (rr<rr_out){
          SetVoxel(i*nx*nx+j*nx+k, k, j, i, (rr_out-rr)/(rr_out-rr_in));
        }
        else
          SetVoxel(i*nx*nx+j*nx+k, k, j, i, 0);
      }
}

//********************
// CTorus
//********************

// member functions for class CTorus
CTorus::CTorus(INT cx, INT cy, INT cz): CVolume(cx, cy, cz)
{
  r = (nz-1)/2.0;
  R = (nx-1)/2.0 - r;
  
  Initialize();
}

void CTorus::Initialize()
{
  INT num=0;
  assert (voxel_num>0);
  REAL cx = (nx-1)/2.0;
  REAL cy = (ny-1)/2.0;
  REAL cz = (nz-1)/2.0;
  for (int i=0; i<nz; i++)
    for (int j=0; j<ny; j++)
      for (int k=0; k<nx; k++){
	REAL x = k-cx;
	REAL y = j-cy;
	REAL z = i-cz;
	REAL rr = R - sqrt((x*x+y*y));
	REAL v = sqrt(rr*rr + z*z );
	if (v<=r){
	  SetVoxel(num, k, j, i, 1.0);
          num ++;
        }
      }
  voxel_num = num;
}


//********************
// CHipiph
//********************

// member functions for class CHipiph
CHipiph::CHipiph(char* filename): CVolume(64,64,64)
{
  Initialize(filename);
}

void CHipiph::Initialize(char* filename)
{  

  ifstream fin(filename);  
  if (!fin) {
    cerr << "Unable to open/read file " << filename << endl;
	return;
  }

//  LoadLut();

  voxel_num = nx*ny*nz;

  voxel = new VOXEL[voxel_num];

  float mmin=1000, mmax=-1000;


  float range_min = -0.55618;
  float range_max =  0.58127;

  int entries = 1024;
  //MyHistogram histo(entries);

  unsigned int index;
  for (int i=0; i<nz; i++)
	  for (int j=0; j<ny; j++)
		  for (int k=0; k<nx; k++){
			  index = i*nx*ny + j*nx +k;
			  float intens;
			  fin >> intens;
			  
			  if (intens>0) intens = 0;
			  intens = (intens)/(range_min);
			  //intens = fabs(intens);
			  int m = int(entries*intens);

			  //histo.Histogram(m);

			  float r=0, g=0, b=0, a=0;

			  float f0, f1, f2, f3, h, h1, M;
			  f0 = 2;
			  f1 = 10;
			  f2 = 100;
			  f3 = 192;
			  M = 2048.0;
			  h1 = 0.02;
			  h = 0.05;

			  float mm = 255*intens;

			  if (m<f0){				  
			  }
			  else if (m<f1){
				  r = 1.0;
				  g = mm/f1;
				  b = 0.0;
				  a = h1+mm/M;
			  }
			  else if (m<f2){
				  r = 0.0;//1.0 -(mm-f1)/(f2-f1);
				  g = 1.0;
				  b = 0.0;
				  a = h1+mm/M;
			  }
			  else if (m<f3) {
				  r = 0.0;
				  g = 1 - (mm-f2)/(entries-f2);
				  b = (mm-f2)/(f3-f2);
				  a = h+mm/M;
			  }
			  else{
				  r = 0.0;
				  g = 0.5;
				  b = 1.0;
				  a = h+mm/M;
			  }
			  r = r*255;
			  g = g*255;
			  b = b*255;
			  a = a*255;

#ifdef __X_RAY_MODEL__
			  float tmp = RGB2GRAY(r, g, b);
			  tmp = tmp/2.0;
			  r = tmp/2.0;
			  g = tmp;
			  b = tmp;
#endif

			  SetVoxel(index, k, j, i, r, g, b, a);
			  //SetVoxel(index, k, j, i, m*100);
		  }

	cout <<mmin<< "\t" << mmax<<endl;
	//histo.Save();
}

void CHipiph::LoadLut()
{
	ifstream fin("..\\project\\vol_data\\hipip.lut");  
	if (!fin) {
		cerr << "Unable to open/read file " << "..\\project\\vol_data\\hipip.lut" << endl;
		return;
	}

	int start, end;
	fin >> start >> end;

	for (int i=0; i<end; i++){
		fin >> lut[i].r >> lut[i].g >> lut[i].b >> lut[i].a >>lut[i].k1 >> lut[i].k2 >> lut[i].k3 >> lut[i].k4 >> lut[i].k5;
	}
}

//********************
// CFuel
//********************

// member functions for class CFuel
CFuel::CFuel(char* filename) 
{
  Initialize(filename);
}

void CFuel::Initialize(char* filename)
{  
	ReadVol(filename);
}

void CFuel::ReadVol(char* filename)
{  
  FILE *fp;

  fp = fopen(filename, "r");
  if (fp == NULL){
    fprintf(stderr, "could not open/read %s to output\n", filename);
    return;
  }

  LoadLut();

  nx = ny = nz = 64;

  voxel_num = nx*ny*nz;

  voxel = new VOXEL[voxel_num];
  
  BYTE *buffer = new BYTE[voxel_num];

  fread(buffer, 1, voxel_num, fp);


//  MyHistogram histo(256);

  for(INT i=0; i<nz; i++)
    for(INT j=0; j<ny; j++)
	  for(INT k=0; k<nx; k++){
		  INT index = i*nx*ny + j*ny + k;
			INT m = buffer[index];

//			  histo.Histogram(m);

			  float r=0, g=0, b=0, a=0;

			  float f0,f1, f2, f3, h, M;
			  f0 = 1;
			  f1 = 20;
			  f2 = 64;
			  f3 = 128;
			  M = 2048.0;
			  h = 0.1;

			  float mm = m;

			  if (m<f0){				  
			  }
			  else if (m<f1){
				  r = 1.0;
				  g = 1.0;
				  b = 1.0;
				  a = h+mm/M;
			  }
			  else if (m<f2){
				  r = (mm-f1)/(f2-f1);
				  g = (mm-f1)/(f2-f1);;
				  b = 0.0;
				  a = h+mm/M;
			  }
			  else if (m<f3) {
				  r = (mm-f2)/(f3-f2);
				  g = 1 - (mm-f2)/(255-f2);
				  b = (mm-f2)/(f3-f2);
				  a = h+mm/M;
			  }
			  else{
				  r = 1.0;
				  g = 0.0;
				  b = 1 - (mm-f3)/(255-f3);
				  a = h+mm/M;
			  }

/*

			  r = lut[m].r;
			  g = lut[m].g;
			  b = lut[m].b;
			  a = lut[m].a;
*/
/*
			  if (r>0)  a = a + 0.01;
			  else if (g>0) a = a - 0.3;
			  else if (b>0) a = a - 0.5;
*/
			  r = r*255;
			  g = g*255;
			  b = b*255;
			  a = a*255;

			  SetVoxel(index, k, j, i, r, g, b, a);
		}

  delete [] buffer;
  fclose(fp);

//  histo.Save();
}

void CFuel::LoadLut()
{
	ifstream fin("..\\vol_data\\fuel.lut");  
	if (!fin) {
		cerr << "Unable to open/read file " << "..\\vol_data\\fuel.lut" << endl;
		return;
	}

	int end=256;

	for (int i=0; i<end; i++){
		fin >> lut[i].r >> lut[i].g >> lut[i].b >> lut[i].a >>lut[i].k1 >> lut[i].k2 >> lut[i].k3;
	}
}

//********************
// CHydrogen
//********************

// member functions for class CFuel
CHydrogen::CHydrogen(char* filename) 
{
  Initialize(filename);
}

void CHydrogen::Initialize(char* filename)
{  
	ReadVol(filename);
}

void CHydrogen::ReadVol(char* filename)
{  
  FILE *fp;

  fp = fopen(filename, "r");
  if (fp == NULL){
    fprintf(stderr, "could not open/read %s to output\n", filename);
    return;
  }

//  LoadLut();

  nx = ny = nz = 128;

  voxel_num = nx*ny*nz;

  voxel = new VOXEL[voxel_num];
  
  BYTE *buffer = new BYTE[voxel_num];

  fread(buffer, 1, voxel_num, fp);

  for(INT i=0; i<nz; i++)
    for(INT j=0; j<ny; j++)
	  for(INT k=0; k<nx; k++){
		  INT index = i*nx*ny + j*ny + k;
			INT m = buffer[index];

			  float r=0, g=0, b=0, a=0;

			  if (m<1){
				  r=g=b=0;
				  a = 1;
			  }
			  else if (m<15) {
			  r = 1;
			  g = 1;
			  b = 1;
			  a = 0.00395;
			  }
			  else if (m<80){
				  r = 0;
				  g = 1;
				  b = 0;
				  a = 0.004;
			  }
			  else{
 				  r = 1;
				  g = 0;
				  b = 1;
				  a = 0.2;
			  }

			  
/*
			  if (r>0)  a = a + 0.01;
			  else if (g>0) a = a - 0.3;
			  else if (b>0) a = a - 0.5;
*/
			  r = r*a*255;
			  g = g*a*255;
			  b = b*a*255;
			  a = a*255;

			  SetVoxel(index, k, j, i, r, g, b, a);
		}

  delete [] buffer;
  fclose(fp);
}


//********************
// CCThead
//********************

// member functions for class CCThead
CCThead::CCThead(char* filename)
{

  Initialize(filename);
}

void CCThead::Initialize(char* filename)
{  
  char* t= strrchr(filename, '.');

  if (t){
	  if (strcmp(t,".vol")==0){
		  ReadVol2(filename);
	  }
	  else if (strcmp(t, ".raw")==0){
		  ReadVol(filename);
	  }
  }
  else
	  cout <<"error input file name" <<endl;
}

void CCThead::ReadVol(char* filename)
{  
  FILE *fp;

  fp = fopen(filename, "r");
  if (fp == NULL){
    fprintf(stderr, "could not open/read %s to output\n", filename);
    return;
  }

  nx = ny = 256;
  nz = 64;

  voxel_num = nx*ny*nz;

  voxel = new VOXEL[voxel_num];
  
  unsigned short *buffer = new unsigned short[voxel_num];

  fread(buffer, 1, voxel_num, fp);

//  MyHistogram histo(256);

  for(INT i=0; i<nz; i++)
    for(INT j=0; j<ny; j++)
	  for(INT k=0; k<nx; k++){
		  INT index = i*nx*ny + j*nx + k;
			unsigned short m = buffer[index];

			  //histo.Histogram(m);

#if defined (_WIN32)
			unsigned short m1, m2;
			m1 = m>>8;
			m2 = (m&0x00ff)<<8;

			m  = m2;// + m2;
#endif

			float r, g, b, a;

			r = m/65535.0;
			g = m/65535.0;
			b = m/65535.0;
			a = m/65535.0;

			  r = r*255;
			  g = g*255;
			  b = b*255;
			  a = a*255;

			  SetVoxel(index, k, j, i, r, g, b, a);
		}

  delete [] buffer;
  fclose(fp);

//  histo.Save();
}


void CCThead::ReadVol2(char* filename)
{  
  FILE *fp;

  fp = fopen(filename, "r");
  if (fp == NULL){
    fprintf(stderr, "could not open/read %s to output\n", filename);
    return;
  }


  fscanf(fp, "%d %d %d\n", &nx, &ny, &nz);
  _setmode(_fileno(fp), _O_BINARY);

  voxel_num = nx*ny*nz;

  voxel = new VOXEL[voxel_num];
  
  BYTE *buffer = new BYTE[voxel_num];

  fread(buffer, 1, voxel_num, fp);

//  MyHistogram histo(256);

  for(INT i=0; i<nz; i++)
    for(INT j=0; j<ny; j++)
	  for(INT k=0; k<nx; k++){
		  INT index = i*nx*ny + j*ny + k;
			BYTE m = buffer[index];

			  //histo.Histogram(m);

			float r, g, b, a;
			r = g = b = a = 0.0;

			  if (m<30)
				r = g = b = a = 0.0;
			  else{
				r = g = b = m; 
				a = m/4.0;
			  }
			  
			  SetVoxel(index, k, j, i, r, g, b, a);
		}

  delete [] buffer;
  fclose(fp);

//  histo.Save();
}

//********************
// CFoot
//********************

// member functions for class CCThead
CFoot::CFoot(char* filename)
{
  Initialize(filename);
}

void CFoot::Initialize(char* filename)
{  
	ReadVol(filename);
}

void CFoot::ReadVol(char* filename)
{  
  FILE *fp;

  fp = fopen(filename, "rt");
  if (fp == NULL){
    fprintf(stderr, "could not open/read %s to output\n", filename);
    return;
  }
  
  fscanf(fp, "%d %d %d\n", &nx, &ny, &nz);

  _setmode(_fileno(fp), _O_BINARY);

  voxel_num = nx*ny*nz;

  voxel = new VOXEL[voxel_num];
  
  BYTE mmin=255, mmax=0;
  BYTE *buffer = new BYTE[nx*ny];

  INT index = 0;
  if (voxel!=NULL){
    BYTE intensity;
    for(INT i=0; i<nz; i++){
	  fread(buffer, 1, nx*ny, fp);
	  printf("read %d frame...\n", i);
      for(INT j=0; j<ny; j++){
		for(INT k=0; k<nx; k++){
		  INT index = i*nx*ny + j*ny + k;
		  intensity = buffer[j*nx+k];
						

			  int m = intensity;

			  float r=0, g=0, b=0, a=0;

			  float f1, f2, f3, h, M;
			  f1 = 50;
			  f2 = 100;
			  f3 = 192;
			  M = 4096.0;
			  h = 0.006;

			  float mm = m;

			  if (m==0){				  
			  }
			  else if (m<f1){
				  r = .8;
				  g = .4;
				  b = .4;
				  a = 0.006; 
			  }
			  else if (m<f2){
				  r = .8;
				  g = 0.4;
				  b = 0.4;
				  a = 0.006;
			  }
			  else if (m<f3) {
				  r = .4+(m-f2)/(255-f2);
				  g = .2+(m-f2)/(255-f2);
				  b = .2+(m-f2)/(255-f2);
				  a = h+mm/M;
			  }
			  else{
				  r = .4+(m-f3)/(255-f2);
				  g = .2+(m-f3)/(255-f2);
				  b = .2+(m-f3)/(255-f2);
				  a = h+mm/M;
			  }

			  r = r*255;
			  g = g*255;
			  b = b*255;
			  a = a*255;

			  SetVoxel(index, k, j, i, r, g, b, a);

		}
	  }
    }
  }

  cout << "range: " << (int)mmin <<" - " << (int)mmax<<endl;

  delete [] buffer;
  fclose(fp);

}


//********************
// CFoot2
//********************

// member functions for class CCThead
CFoot2::CFoot2(char* filename)
{
  Initialize(filename);
}

void CFoot2::Initialize(char* filename)
{  
	ReadVol(filename);
}

void CFoot2::ReadVol(char* filename)
{  
  FILE *fp;

  fp = fopen(filename, "rt");
  if (fp == NULL){
    fprintf(stderr, "could not open/read %s to output\n", filename);
    return;
  }
  
  fscanf(fp, "%d %d %d\n", &nx, &ny, &nz);

  _setmode(_fileno(fp), _O_BINARY);

  voxel_num = nx*ny*nz;

  voxel = new VOXEL[voxel_num];
  
  BYTE mmin=255, mmax=0;
  BYTE *buffer = new BYTE[nx*ny];

  INT index = 0;
  if (voxel!=NULL){
    BYTE intensity;
    for(INT i=0; i<nz; i++){
	  fread(buffer, 1, nx*ny, fp);
	  printf("read %d frame...\n", i);
      for(INT j=0; j<ny; j++){
		for(INT k=0; k<nx; k++){
		  INT index = i*nx*ny + j*ny + k;
		  intensity = buffer[j*nx+k];
						

			  int m = intensity;

			  float r=0, g=0, b=0, a=0;

			  float f0, f1, f2, f3, h, M;
			  f0 = 12;
			  f1 = 60;
			  f2 = 75;
			  f3 = 192;
			  M = 4096.0;
			  //h = 0.006;		// for x-ray model
			  h = 0.1;			// for low-albedoe model

			  float mm = m;

			  if (m<f0){				  
			  }
			  else if (m<f1){
				  r = .8;
				  g = .4;
				  b = .4;
				  a = 0.01; 
			  }
			  else if (m<f2){
				  r = .8;
				  g = 0.4;
				  b = 0.4;
				  a = 0.0095;
			  }
			  else if (m<f3) {
				  r = .6+(m-f2)/(255-f2);
				  g = .4+(m-f2)/(255-f2);
				  b = .4+(m-f2)/(255-f2);
				  a = h+mm/M;
			  }
			  else{
				  r = .6+(m-f3)/(255-f2);
				  g = .4+(m-f3)/(255-f2);
				  b = .4+(m-f3)/(255-f2);
				  a = h+mm/M;
			  }

			  r = r*255;
			  g = g*255;
			  b = b*255;
			  a = a*255;

			  SetVoxel(index, k, j, i, r, g, b, a);

		}
	  }
    }
  }

  cout << "range: " << (int)mmin <<" - " << (int)mmax<<endl;

  delete [] buffer;
  fclose(fp);

}

//********************
// CSkull
//********************

// member functions for class CSkull
CSkull::CSkull(char* filename)
{
  Initialize(filename);
}

void CSkull::Initialize(char* filename)
{  
	ReadVol(filename);
}

void CSkull::ReadVol(char* filename)
{  
  FILE *fp;

  fp = fopen(filename, "rt");
  if (fp == NULL){
    fprintf(stderr, "could not open/read %s to output\n", filename);
    return;
  }
  
  fscanf(fp, "%d %d %d\n", &nx, &ny, &nz);

  _setmode(_fileno(fp), _O_BINARY);

  voxel_num = nx*ny*nz;

  voxel = new VOXEL[voxel_num];
  
  BYTE mmin=255, mmax=0;
  BYTE *buffer = new BYTE[nx*ny];

  INT index = 0;
  if (voxel!=NULL){
    BYTE intensity;
    for(INT i=0; i<nz; i++){
	  fread(buffer, 1, nx*ny, fp);
	  printf("read %d frame...\n", i);
      for(INT j=0; j<ny; j++){
		for(INT k=0; k<nx; k++){
		  INT index = i*nx*ny + j*ny + k;
		  intensity = buffer[j*nx+k];
						

			  int m = intensity;

			  float r=0, g=0, b=0, a=0;

			  float f1, f2, f3, h, M;
			  f1 = 50;
			  f2 = 100;
			  f3 = 192;
			  M = 4096.0;
			  h = 0.006;

			  float mm = m;

			  if (m==0){				  
			  }
			  else if (m<f1){
				  r = .8;
				  g = .4;
				  b = .4;
				  a = 0.006; 
			  }
			  else if (m<f2){
				  r = .8;
				  g = 0.4;
				  b = 0.4;
				  a = 0.006;
			  }
			  else if (m<f3) {
				  r = .4+(m-f2)/(255-f2);
				  g = .2+(m-f2)/(255-f2);
				  b = .2+(m-f2)/(255-f2);
				  a = h+mm/M;
			  }
			  else{
				  r = .4+(m-f3)/(255-f2);
				  g = .2+(m-f3)/(255-f2);
				  b = .2+(m-f3)/(255-f2);
				  a = h+mm/M;
			  }

			  r = r*255;
			  g = g*255;
			  b = b*255;
			  a = a*255;

			  SetVoxel(index, k, j, i, r, g, b, a);

		}
	  }
    }
  }

  cout << "range: " << (int)mmin <<" - " << (int)mmax<<endl;

  delete [] buffer;
  fclose(fp);

}


//********************
// CUncbrain
//********************

// member functions for class CUncbrain
CUncbrain::CUncbrain(char* filename)
{
  Initialize(filename);
}

void CUncbrain::Initialize(char* filename)
{  
	ReadVol(filename);

	  int m = max(nx, max(ny, nz));

	  center[0] = nx/(1.0*m);
	  center[1] = ny/(1.0*m);
	  center[2] = nz/(1.0*m);
}

void CUncbrain::ReadVol(char* filename)
{  
  FILE *fp;

  fp = fopen(filename, "rt");
  if (fp == NULL){
    fprintf(stderr, "could not open/read %s to output\n", filename);
    return;
  }
  
  fscanf(fp, "%d %d %d\n", &nx, &ny, &nz);

  _setmode(_fileno(fp), _O_BINARY);

  voxel_num = nx*ny*nz;

  voxel = new VOXEL[voxel_num];
  
  BYTE mmin=255, mmax=0;
  BYTE *buffer = new BYTE[nx*ny];

  INT index = 0;
  if (voxel!=NULL){
    BYTE intensity;
    for(INT i=0; i<nz; i++){
	  fread(buffer, 1, nx*ny, fp);
	  printf("read %d frame...\n", i);
      for(INT j=0; j<ny; j++){
		for(INT k=0; k<nx; k++){
		  INT index = i*nx*ny + j*ny + k;
		  intensity = buffer[j*nx+k];
						

			  int m = intensity;

			  float r=0, g=0, b=0, a=0;

			  float f0, f1, f2, f3, f4,h, M;
			  f0 = 6;
			  f1 = 30;
			  f2 = 90;
			  f3 = 102;
			  f4 = 120;
			  M = 4096.0;
			  h = 0.02;

			  float mm = m;

			  if (m<f0){
				  r=g=b=a=0.0;
			  }
			  else if (m<f1){
				  a = mm/255.0; 
				  r = 1*a;
				  g = 1*a;
				  b = 1*a;
			  }
			  else if (m<f2){
				  a = mm/255; 
				  a = a/4;
				  r = .2*a;
				  g = 1*a;
				  b = .4*a;
			  }
			  else if (m<f3) {
				  a = mm/255; 
				  a = a/4.0;
				  r = a*(.8+(m-f2)/(255-f2));
				  g = a*(.8+(m-f2)/(255-f2));
				  b = a*(.2+(m-f2)/(255-f2));

			  }
			  else if (m<f4) {
				  a = mm/255; 
				  a = a/4.0;
				  r = a*(.8+(m-f2)/(255-f2));
				  g = a*(.8+(m-f2)/(255-f2));
				  b = a*(.2+(m-f2)/(255-f2));

			  }
			  else{
				  a = mm/255;
				  a = a/4.0;
				  r = a*(.8+(m-f3)/(255-f2));
				  g = a*(.8+(m-f3)/(255-f2));
				  b = a*(.2+(m-f3)/(255-f2));
			  }

			  r = r*255;
			  g = g*255;
			  b = b*255;
			  a = a*255;

			  if (mm<6)
				  r = g = b = a = 0.;
			  else{
				  r = g = b = mm;
				  a = mm/4.0;
			  }

			  SetVoxel(index, k, j, i, r, g, b, a);

		}
	  }
    }
  }

  cout << "range: " << (int)mmin <<" - " << (int)mmax<<endl;

  delete [] buffer;
  fclose(fp);

}

void CUncbrain::SetVoxel(INT i, REAL x, REAL y, REAL z, REAL r, REAL g, REAL b, REAL a)
{
  assert (i<voxel_num);

  int m = max(nx, max(ny, nz));
  //int m = min(nx, min(ny, nz));

  voxel[i].x = m==1 ? x : 2*x/(m-1.0);
  voxel[i].y = m==1 ? y : 2*y/(m-1.0);
  voxel[i].z = m==1 ? z : 2*z/(m-1.0);
  voxel[i].r = r;
  voxel[i].g = g;
  voxel[i].b = b;
  voxel[i].a = a;
}
