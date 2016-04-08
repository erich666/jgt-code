/* viewer.h
 *
 *  $Id: viewer.h,v 1.2 2001/07/14 00:41:27 xue Exp $
 *  $Log: viewer.h,v $
 *  Revision 1.2  2001/07/14 00:41:27  xue
 *  complete basic functions for splatter
 *
 *  Revision 1.1  2001/07/08 21:49:34  xue
 *  Initial version
 *
 */
 
#ifndef __VIEWER_H
#define __VIEWER_H

#include "Datatype.h"
#include "OSUmatrix.h"
#include "volume.h"
#include "tsplat.h"

enum PROJECTION_TYPE {
	PERSPECTIVE,
	PARALLEL
};

// 3D view structure to hold observer information

typedef struct {
  PROJECTION_TYPE type;// projection type, either PERSPECTIVE or PARALLEL
  REAL view_angle;     // angle between line of sight and
                       //   top of view pyramid
  REAL hither;         // distance from eye to near plane
  REAL yon;            // distance from eye to far plane
  REAL image_plane;    // distance from eye to image plane
  REAL aspect_ratio;   // width/height of view pyramid
  REAL head_tilt;      // (CCW) angle of head tilt in degrees
  VECTOR3 eye;	       // eye position
  VECTOR3 coi;	       // center of interest
	
  REAL left;		// for parallel
  REAL right;
  REAL bottom;
  REAL top;
} VIEW3D_INFO;

typedef struct {
  double kernel_radius;
  INT  tsplat_size;
  REAL sigma;
  REAL slice_depth;
} SPLAT_INFO;

class CRect {
 public:
  CRect() {left=right=top=bottom=0;}
  CRect(INT l, INT t, INT r, INT b) {left=l;right=r;top=t;bottom=b;} 
  INT Width()  { return right-left+1; }
  INT Height() { return bottom-top+1; }
  void Set(INT l, INT t, INT r, INT b) {left=l;right=r;top=t;bottom=b;} 

 public:
  INT left;
  INT right;
  INT top;
  INT bottom;
};

class CLight {
  public:
    CLight() {}
    ~CLight() {}

    void SetPosition(VECTOR3 pos) { position = pos; }
    VECTOR3 GetPosition() { return position; }

    void SetIntensity(REAL intens) { intensity = intens; }
    inline REAL GetIntensity(const VECTOR3 & pt_to_illuminate) { return intensity; }

  protected:
    VECTOR3 position;
    REAL intensity;
};


typedef struct _VoxelEntry {
  int index;
  struct _VoxelEntry *next;
}VoxelEntry;

typedef struct _SliceEntry {
  REAL a, b;
  VoxelEntry *voxel;
}SliceEntry;

typedef struct _Vertex{
	GLfloat x, y, z;
	GLfloat r, g, b, a;

	GLfloat index;		// For vertex program, the vertex index on the billboard, from 0 to 3
} Vertex;

class CViewer 
{
 public:
  CViewer();
  ~CViewer();
  void Load(char* file_name);
  void SetDefaultViewInfo();
  void SetViewInfo(VIEW3D_INFO *info);
  void GetViewInfo(VIEW3D_INFO *info);
  void SetViewport(INT l, INT t, INT r, INT b);
  void Clear();
  void Z_MinMax(REAL *min, REAL *max);

	// without nVidia vertex program
	void RealizeImm();		// render immediate mode with presorted voxel list
	void RealizeList();		// render with display list
	void RealizeStream();	// rednder vertex stream 
	void RealizeConvolution();

	// with nVidia vertex program
	void RealizeImm_NV();			// render immediate mode with presorted voxel list by nVidia vertex program ext
	void RealizeList_NV();		// render display list with nVidia vertex program ext
	void RealizeStream_NV();	// rednder vertex stream with nVidia vertex program
	void RealizeConvolution_NV();  
	

	void InitVoxelList();
	void InitConvolutionVoxelList();

	void InitSortedLists();

 private:
	void Sort(VECTOR3 eye, GLuint* index_list);
	void Sort2(int order,  GLuint* index_list, GLuint vertices);
	void InitEyes();

	int width, height;

	
 public:
  GLfloat slice_depth;
  int slice_num;
  SliceEntry* slice;

  Vertex *vertex_list;
  //BB_Vertex *current_vertex_list;
  //C_Vertex *convolution_vertex_list;
  GLuint *current_vertex_indices;
  GLuint vertex_num;

  GLuint *index_lut;

#define SORTED_LIST_NUM  6
  GLuint* vertex_indices[SORTED_LIST_NUM];
  VECTOR3 eyes[SORTED_LIST_NUM];

  VIEW3D_INFO view_info;
  SPLAT_INFO splat_info;
  VECTOR3 background;
  CLight *light;
  REAL ambient;
  CRect viewport;
  CVolume *volume;
  char output_file[1024];
};

#endif /* __VIEWER_H */
