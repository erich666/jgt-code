#include <assert.h>
#include <stdlib.h>
#include <sys/timeb.h>
#include <time.h>

#include <glh_nveb.h>
#include <glh_extensions.h>
#include <glh_genext.h>
#include <glh_linear.h>

#include "viewer.h"
#include "parser.h"
#include "tsplat.h"
#include "defines.h"
#include "pbuffer.h"
#include "HTimer.h"

using namespace glh;


#define MIN_INTENSITY 1

extern void GetWorld2EyeMatrix(GLdouble *m);
extern void OrientateSplat();
extern int GetListOrder();
extern GaussTextureSplat gts;

extern pbf pbuffer;
extern GLuint texPbuffer;

extern GLuint image_width;
extern GLuint image_height;

extern GLfloat rotate_matrix[16];
extern vec3f rotate_center;

extern GLint list_order;

int compare( const void *arg1, const void *arg2 );
void GetWorld2EyeMatrix(GLdouble *m);

typedef struct _IndexDepthStruct {
	GLuint index;
	GLfloat depth;
}IndexDepthStruct;

CHTimer htimer1;

//********************
// CViewer
//********************

CViewer::CViewer()
{
  vertex_list = NULL;
  current_vertex_indices = NULL;
  vertex_list = NULL;

  index_lut = NULL;

  light = NULL;
  strcpy(output_file, "out.ppm");
  SetDefaultViewInfo();
}

CViewer::~CViewer()
{
  Clear();
}

void CViewer::SetDefaultViewInfo()
{
  view_info.eye = VECTOR3(60, 60, 60);
  view_info.coi = VECTOR3(0, 0, 0);
  view_info.hither = 0.1;
  view_info.yon  = 1000;
  view_info.view_angle = 45;
  view_info.head_tilt  = 0;
  view_info.aspect_ratio = 1.0;
  view_info.image_plane = view_info.hither;
}

void CViewer::SetViewInfo(VIEW3D_INFO *info)
{
  view_info.eye = info->eye;
  view_info.coi = info->coi;
  view_info.hither = info->hither;
  view_info.yon  = info->yon;
  view_info.view_angle = info->view_angle;
  view_info.head_tilt  = info->head_tilt;
  view_info.aspect_ratio = info->aspect_ratio;
  view_info.image_plane = info->image_plane;
}

void CViewer::GetViewInfo(VIEW3D_INFO *info)
{
  info->eye = view_info.eye;
  info->coi = view_info.coi;
  info->hither = view_info.hither;
  info->yon  = view_info.yon;
  info->view_angle = view_info.view_angle;
  info->head_tilt  = view_info.head_tilt;
  info->aspect_ratio = view_info.aspect_ratio;
  info->image_plane  = view_info.image_plane;
}

void CViewer::SetViewport(INT l, INT t, INT r, INT b)
{
  viewport.Set(l, t, r, b);
}

void CViewer::Z_MinMax(REAL *min, REAL *max)
{
    volume->Z_MinMax(min, max);
}

void CViewer::Clear()
{
  if (light!=NULL){
    delete light;
    light = NULL;
  }

  if (vertex_list != NULL){
	  delete [] vertex_list;
	  vertex_list = NULL;
  }

  if (current_vertex_indices != NULL){
	  delete [] current_vertex_indices;
	  current_vertex_indices = NULL;
  }

  if (index_lut != NULL){
	  delete [] index_lut;
	  index_lut = NULL;
  }

  
  if (volume != NULL)
	delete volume;
}

void CViewer::Load(char *filename)
{
  CParser parser(this);
  parser.Parse(filename);

  int nx = volume->nx;
  int ny = volume->ny;
  int nz = volume->nz;

  int m = max(nx, max(ny, nz));
	//int m = min(nx, min(ny, nz));
 
  rotate_center[0] = volume->center[0];
  rotate_center[1] = volume->center[1];
  rotate_center[2] = volume->center[2];

  // reset kernel radius as the interval of adjacent voxels 
//  splat_info.kernel_radius = 1.0/(volume->nx);
//  splat_info.kernel_radius = 1.0/(m-1);

}


void CViewer::InitConvolutionVoxelList()
{
	GLint i, x, y, z;
	GLuint nx, ny, nz, voxel_num;
	GLuint index;
	nx = volume->nx;
	ny = volume->ny;
	nz = volume->nz;
	voxel_num = volume->voxel_num;
	GLfloat r = splat_info.kernel_radius;

	// get # of non-empty vertecis
	vertex_num = 0;
	for (z=0; z<nz; z++)
	  for (y=0; y<ny; y++)
		  for (x=0; x<nx; x++){
			i = z*ny*nx + y*nx + x;
			if (volume->voxel[i].a>=MIN_INTENSITY){
				vertex_num ++;
			}
		  }

	cout << "Non-empty voxels: " << vertex_num << endl;

	vertex_list = new Vertex[vertex_num];

	index_lut = new GLuint[nx*ny*nz];

	index = 0;
	for (z=0; z<nz; z++)
	  for (y=0; y<ny; y++)
		  for (x=0; x<nx; x++){
			i = z*ny*nx + y*nx + x;
			if (volume->voxel[i].a< MIN_INTENSITY)  continue;
				vertex_list[index].x = volume->voxel[i].x - rotate_center[0];
				vertex_list[index].y = volume->voxel[i].y - rotate_center[1];
				vertex_list[index].z = volume->voxel[i].z - rotate_center[2];
				vertex_list[index].r    = volume->voxel[i].r/255.0;
				vertex_list[index].g    = volume->voxel[i].g/255.0;
				vertex_list[index].b    = volume->voxel[i].b/255.0;
				vertex_list[index].a    = volume->voxel[i].a/255.0;

				index_lut[i] = index;
				index ++;
		  }
}

void CViewer::InitVoxelList()
{
	GLint i;
	GLint x,y,z;
	GLuint nx, ny, nz, voxel_num;
	GLuint index;
	nx = volume->nx;
	ny = volume->ny;
	nz = volume->nz;
	voxel_num = volume->voxel_num;
	GLfloat r = splat_info.kernel_radius;


	// get # of non-empty voxels
	vertex_num = 0;
	for (z=0; z<nz; z++)
	  for (y=0; y<ny; y++)
		  for (x=0; x<nx; x++){
			i = z*ny*nx + y*nx + x;
			if (volume->voxel[i].a>=MIN_INTENSITY){
				vertex_num ++;
			}
		  }

	cout << "Non-empty voxels: " << vertex_num << endl;

	vertex_list = new Vertex[vertex_num*4];

	index_lut = new GLuint[nx*ny*nz];

	index = 0;
	for (z=0; z<nz; z++)
	  for (y=0; y<ny; y++)
		  for (x=0; x<nx; x++){
			i = z*ny*nx + y*nx + x;
			if (volume->voxel[i].a<MIN_INTENSITY)  continue;
#ifdef __VERTEX_PROGRAM__
			vertex_list[4*index].x = volume->voxel[i].x - rotate_center[0];
			vertex_list[4*index].y = volume->voxel[i].y - rotate_center[1];
			vertex_list[4*index].z = volume->voxel[i].z - rotate_center[2];
			vertex_list[4*index].index = 0;
			vertex_list[4*index].r    = volume->voxel[i].r/255.0;
			vertex_list[4*index].g    = volume->voxel[i].g/255.0;
			vertex_list[4*index].b    = volume->voxel[i].b/255.0;
			vertex_list[4*index].a    = volume->voxel[i].a/255.0;

			vertex_list[4*index+1].x = volume->voxel[i].x - rotate_center[0];
			vertex_list[4*index+1].y = volume->voxel[i].y - rotate_center[1];
			vertex_list[4*index+1].z = volume->voxel[i].z - rotate_center[2];
			vertex_list[4*index+1].index = 1;
			vertex_list[4*index+1].r    = volume->voxel[i].r/255.0;
			vertex_list[4*index+1].g    = volume->voxel[i].g/255.0;
			vertex_list[4*index+1].b    = volume->voxel[i].b/255.0;
			vertex_list[4*index+1].a    = volume->voxel[i].a/255.0;


			vertex_list[4*index+2].x = volume->voxel[i].x - rotate_center[0];
			vertex_list[4*index+2].y = volume->voxel[i].y - rotate_center[1];
			vertex_list[4*index+2].z = volume->voxel[i].z - rotate_center[2];
			vertex_list[4*index+2].index = 2;
			vertex_list[4*index+2].r    = volume->voxel[i].r/255.0;
			vertex_list[4*index+2].g    = volume->voxel[i].g/255.0;
			vertex_list[4*index+2].b    = volume->voxel[i].b/255.0;
			vertex_list[4*index+2].a    = volume->voxel[i].a/255.0;


			vertex_list[4*index+3].x = volume->voxel[i].x - rotate_center[0];
			vertex_list[4*index+3].y = volume->voxel[i].y - rotate_center[1];
			vertex_list[4*index+3].z = volume->voxel[i].z - rotate_center[2];
			vertex_list[4*index+3].index = 3;
			vertex_list[4*index+3].r    = volume->voxel[i].r/255.0;
			vertex_list[4*index+3].g    = volume->voxel[i].g/255.0;
			vertex_list[4*index+3].b    = volume->voxel[i].b/255.0;
			vertex_list[4*index+3].a    = volume->voxel[i].a/255.0;


			index_lut[i] = index;
			index ++;
#endif
		  }
}


void CViewer::InitSortedLists()
{
	int i;
	GLuint vertices;

	InitEyes();

	for (i=0; i<SORTED_LIST_NUM; i++){
#if defined(_GL_VERTEXSTREAM)
		vertices = vertex_num*4;
		vertex_indices[i] = new GLuint[vertices];
#else //  _GL_CONVOLUTION, _GL_IMMEDIATE, _GL_DISPLAYLIST
		vertices = vertex_num;
		vertex_indices[i] = new GLuint[vertices];
#endif
	}

	for (i=0; i<SORTED_LIST_NUM; i++){
		cout << "presorting list " << i <<endl;

		struct _timeb time1, time2;
		long millisec;
		_ftime(&time1);

		Sort2(i, vertex_indices[i], vertices);
		//Sort(eyes[i], vertex_indices[i]);

		_ftime(&time2);
		millisec = (time2.time-time1.time)*1000 + time2.millitm-time1.millitm;
		cout << "draw frame:\t"<< millisec << " milliseconds" << endl<<endl;
	}
}

void CViewer::Sort2(int order,  GLuint* index_list, GLuint vertices)
{
	GLuint index, num;
	int d, i, j, k;
	int nx, ny, nz;
	nx = volume->nx;
	ny = volume->ny;
	nz = volume->nz;

	int degree = nx-1 + ny-1 + nz-1;

	switch (order){
	case 0:
		num = 0;
		for (i=0; i<nz; i++)
			for (j=0; j<ny; j++)
				for (k=0; k<nx; k++){
					index = i*nx*ny + j*nx + k;
					if (volume->voxel[index].a>=MIN_INTENSITY){

						// get index from index lut
						index = index_lut[index];

#						ifdef _GL_VERTEXSTREAM
							index_list[4*num]   = 4*index;
							index_list[4*num+1] = 4*index+1;
							index_list[4*num+2] = 4*index+2;
							index_list[4*num+3] = 4*index+3;
							num ++;
#						else // _GL_CONVOLUTION, _GL_IMMEDIATE, _GL_DISPLAYLIST
							index_list[num] = index;
							num ++;
#						endif
					}
				}
		break;
	case 1:
		for (i=0; i<vertices; i++){
			vertex_indices[1][i] = vertex_indices[0][vertices -1 - i];
		}
		break;

	case 2:
		num = 0;
		for (j=0; j<ny; j++)
			for (i=0; i<nz; i++)
				for (k=0; k<nx; k++){
					index = i*nx*ny + j*nx + k;
					if (volume->voxel[index].a>=MIN_INTENSITY){
						// get index from index lut
						index = index_lut[index];

#						ifdef _GL_VERTEXSTREAM
							index_list[4*num]   = 4*index;
							index_list[4*num+1] = 4*index+1;
							index_list[4*num+2] = 4*index+2;
							index_list[4*num+3] = 4*index+3;
							num ++;
#						else // _GL_CONVOLUTION, _GL_IMMEDIATE, _GL_DISPLAYLIST
							index_list[num] = index;
							num ++;
#						endif
					}
				}
		break;
	case 3:
		for (i=0; i<vertices; i++){
			vertex_indices[3][i] = vertex_indices[2][vertices -1 - i];
		}
		break;

	case 4:
		num = 0;
		for (k=0; k<nx; k++)
			for (i=0; i<nz; i++)
				for (j=0; j<ny; j++){
					index = i*nx*ny + j*nx + k;
					if (volume->voxel[index].a>=MIN_INTENSITY){
						// get index from index lut
						index = index_lut[index];

#						ifdef _GL_VERTEXSTREAM
							index_list[4*num]   = 4*index;
							index_list[4*num+1] = 4*index+1;
							index_list[4*num+2] = 4*index+2;
							index_list[4*num+3] = 4*index+3;
							num ++;
#						else // _GL_CONVOLUTION, _GL_IMMEDIATE, _GL_DISPLAYLIST
							index_list[num] = index;
							num ++;
#						endif

					}
				}
		break;
	case 5:
		for (i=0; i<vertices; i++){
			vertex_indices[5][i] = vertex_indices[4][vertices -1 - i];
		}
		break;

	case 6:
		num = 0;
		for (d=0; d<=degree; d++){
			//cout <<"degree: " << d <<endl;
			for (i=0; i<nz &&i<=d; i++){
				for (j=0; j<ny && j<=d-i; j++){
					for (k=0; k<nx && k<=d-j-i; k++){
						if (i+j+k!=d)  continue;
						//cout<<"("<<k<<" "<<j<<" "<<i<<")"<<endl;
						index = i*nx*ny + j*nx + k;
						if (volume->voxel[index].a>=MIN_INTENSITY){

						// get index from index lut
						index = index_lut[index];

#						ifdef _GL_VERTEXSTREAM
							index_list[4*num]   = 4*index;
							index_list[4*num+1] = 4*index+1;
							index_list[4*num+2] = 4*index+2;
							index_list[4*num+3] = 4*index+3;
							num ++;
#						else // _GL_CONVOLUTION, _GL_IMMEDIATE, _GL_DISPLAYLIST
							index_list[num] = index;
							num ++;
#						endif
						}
					}
				}
			}
		}
		break;

	case 7:  //inverse order with 6
		for (i=0; i<vertices; i++){
			vertex_indices[7][i] = vertex_indices[6][vertices -1 - i];
		}
		break;

	case 8:
		num = 0;
		for (d=0; d<=degree; d++){
			//cout <<"degree: " << d <<endl;
			for (i=0; i<nz &&i<=d; i++){
				for (j=0; j<ny && j<=d-i; j++){
					for (k=nx-1; k>=0 && nx-1-k<=d-j-i; k--){
						if (i+j+nx-1-k!=d)  continue;
						//cout<<"("<<k<<" "<<j<<" "<<i<<")"<<endl;
						index = i*nx*ny + j*nx + k;
						if (volume->voxel[index].a>=MIN_INTENSITY){

						// get index from index lut
						index = index_lut[index];

#						ifdef _GL_VERTEXSTREAM
							index_list[4*num]   = 4*index;
							index_list[4*num+1] = 4*index+1;
							index_list[4*num+2] = 4*index+2;
							index_list[4*num+3] = 4*index+3;
							num ++;
#						else // _GL_CONVOLUTION, _GL_IMMEDIATE, _GL_DISPLAYLIST
							index_list[num] = index;
							num ++;
#						endif
						}
					}
				}
			}
		}
		break;
	case 9:
		for (i=0; i<vertices; i++){
			vertex_indices[9][i] = vertex_indices[8][vertices -1 - i];
		}
		break;

	case 10:
		num = 0;
		for (d=0; d<=degree; d++){
			//cout <<"degree: " << d <<endl;
			for (i=0; i<nz &&i<=d; i++){
				for (k=0; k<nx && k<=d-i; k++){
					for (j=ny-1; j>=0 && ny-1-j<=d-i-k; j--){
						if (i+ny-1-j+k!=d)  continue;
						//cout<<"("<<k<<" "<<j<<" "<<i<<")"<<endl;
						index = i*nx*ny + j*nx + k;
						if (volume->voxel[index].a>=MIN_INTENSITY){

						// get index from index lut
						index = index_lut[index];

#						ifdef _GL_VERTEXSTREAM
							index_list[4*num]   = 4*index;
							index_list[4*num+1] = 4*index+1;
							index_list[4*num+2] = 4*index+2;
							index_list[4*num+3] = 4*index+3;
							num ++;
#						else // _GL_CONVOLUTION, _GL_IMMEDIATE, _GL_DISPLAYLIST
							index_list[num] = index;
							num ++;
#						endif
						}
					}
				}
			}
		}
		break;
	case 11:
		for (i=0; i<vertices; i++){
			vertex_indices[11][i] = vertex_indices[10][vertices -1 - i];
		}
		break;

	case 12:
		num = 0;
		for (d=0; d<=degree; d++){
			//cout <<"degree: " << d <<endl;
			for (j=0; j<ny && j<=d; j++){
				for (k=0; k<nx && k<=d-j; k++){
					for (i=nz-1; i>=0 && nz-1-i<=d-j-k; i--){
						if (nz-1-i+j+k!=d)  continue;
						//cout<<"("<<k<<" "<<j<<" "<<i<<")"<<endl;
						index = i*nx*ny + j*nx + k;
						if (volume->voxel[index].a>=MIN_INTENSITY){

						// get index from index lut
						index = index_lut[index];

#						ifdef _GL_VERTEXSTREAM
							index_list[4*num]   = 4*index;
							index_list[4*num+1] = 4*index+1;
							index_list[4*num+2] = 4*index+2;
							index_list[4*num+3] = 4*index+3;
							num ++;
#						else // _GL_CONVOLUTION, _GL_IMMEDIATE, _GL_DISPLAYLIST
							index_list[num] = index;
							num ++;
#						endif
						}
					}
				}
			}
		}
	case 13:
		for (i=0; i<vertices; i++){
			vertex_indices[13][i] = vertex_indices[12][vertices -1 - i];
		}
		break;
	}
}

void CViewer::InitEyes()
{
	eyes[0]= VECTOR3(0,0,1);    eyes[1]= VECTOR3(0,0,-1);   eyes[2]= VECTOR3(0,1,0);    eyes[3]= VECTOR3(0,-1,0);
	eyes[4]= VECTOR3(1,0,0);    eyes[5]= VECTOR3(-1,0,0);   
/*
	eyes[6]= VECTOR3(1,1,1);    eyes[7]= VECTOR3(-1,-1,-1);	eyes[8]= VECTOR3(-1,1,1);   eyes[9]= VECTOR3(1,-1,-1);  
	eyes[10]= VECTOR3(1,-1,1);  eyes[11]= VECTOR3(-1,1,-1);	eyes[12]= VECTOR3(1,1,-1);  eyes[13]= VECTOR3(-1,-1,1);
*/
}

void CViewer::Sort(VECTOR3 eye, GLuint* index_list)
{
	GLuint voxel_num = volume->voxel_num;
	GLuint index, i;
	GLuint nx = volume->nx;
	GLuint ny = volume->ny;
	GLuint nz = volume->nz;
	GLdouble m[16];

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	gluLookAt(eye[0], eye[1], eye[2], 0, 0, 0, 0, 1, 0);
	glTranslatef(eye[0]+rotate_center[0], eye[1]+rotate_center[1], eye[2]+rotate_center[2]);

	IndexDepthStruct *index_depth = new IndexDepthStruct[vertex_num];
	
	GetWorld2EyeMatrix(m);

	float min = 10000;
	float max = -1000;
	for (i=0; i<vertex_num; i++){
#ifdef _GL_CONVOLUTION
		GLfloat x = vertex_list[i].x;
		GLfloat y = vertex_list[i].y;
		GLfloat z = vertex_list[i].z;
#else
		GLfloat x = vertex_list[4*i].x;
		GLfloat y = vertex_list[4*i].y;
		GLfloat z = vertex_list[4*i].z;
#endif
		index_depth[i].index = i;
		index_depth[i].depth = m[2]*x + m[6]*y + m[10]*z + m[14];

		if (index_depth[i].depth<min)  { min = index_depth[i].depth; }
		if (index_depth[i].depth>max)  { max = index_depth[i].depth; }
	}

	cout << "min: " << min <<"\tmax: " <<max<<endl;

	qsort(index_depth, vertex_num, sizeof(IndexDepthStruct), compare);

	for (i=0; i<vertex_num; i++){
		index = index_depth[i].index;
#ifdef _GL_CONVOLUTION
		index_list[i]   = index;
#else
		index_list[4*i]   = 4*index;
		index_list[4*i+1] = 4*index+1;
		index_list[4*i+2] = 4*index+2;
		index_list[4*i+3] = 4*index+3;
#endif
	}

	delete [] index_depth;

	glPopMatrix();

}


int compare( const void *arg1, const void *arg2 )
{

	GLfloat d1 = ((IndexDepthStruct*)arg1)->depth;
	GLfloat d2 = ((IndexDepthStruct*)arg2)->depth;

	if (d1<d2)
		return -1;
	else if (d1==d2)
		return 0;
	else
		return 1;
}


// immediate mode with presorted voxel list
void CViewer::RealizeImm()
{
	struct _timeb time1, time2;
	_ftime(&time1);
	
	GLuint i;

	glClear(GL_COLOR_BUFFER_BIT);
	glEnable(GL_TEXTURE_2D);

	cout <<"order: " <<list_order <<endl;
	current_vertex_indices	= vertex_indices[list_order];

	Vertex *v;
	GLuint index;
	for (i=0; i<vertex_num; i++){
		index = current_vertex_indices[i];
		v = &vertex_list[index];
		glPushMatrix();
		glTranslated(v->x, v->y, v->z);
		OrientateSplat();
		gts.Render(v->r, v->g, v->b, v->a);
		glPopMatrix();
	}

	glFlush();
	glutSwapBuffers();
	glDisable(GL_TEXTURE_2D);

	_ftime(&time2);
	long millisec = (time2.time-time1.time)*1000 + time2.millitm-time1.millitm;
	cout << "consuming time: "<< millisec << " milliseconds" << endl;
}


void CViewer::RealizeImm_NV()
{

	struct _timeb time1, time2;
	_ftime(&time1);
	
	GLuint i;

	glClear(GL_COLOR_BUFFER_BIT);
	glEnable(GL_TEXTURE_2D);

	cout <<"order: " <<list_order <<endl;
	current_vertex_indices	= vertex_indices[list_order];

	Vertex *v;
	GLuint index;
	for (i=0; i<vertex_num; i++){
		index = current_vertex_indices[i];
		v = &vertex_list[index];
		// Put the translate position in register v[1]
		glVertexAttrib4fNV(1, v->x, v->y, v->z, 1.0);
		gts.RenderNV(v->r, v->g, v->b, v->a);
	}

	glFlush();
	glutSwapBuffers();
	glDisable(GL_TEXTURE_2D);

	_ftime(&time2);
	long millisec = (time2.time-time1.time)*1000 + time2.millitm-time1.millitm;
	cout << "consuming time: "<< millisec << " milliseconds" << endl;
}

// display list
void CViewer::RealizeList()
{

	struct _timeb time1, time2;
	_ftime(&time1);

	glClear(GL_COLOR_BUFFER_BIT);

	gts.RenderList();

	glFlush();
	glutSwapBuffers();

	_ftime(&time2);
	long millisec = (time2.time-time1.time)*1000 + time2.millitm-time1.millitm;
	cout << "consuming time: "<< millisec << " milliseconds" << endl;

}


void CViewer::RealizeList_NV()
{

	struct _timeb time1, time2;
	_ftime(&time1);

	glClear(GL_COLOR_BUFFER_BIT);

	gts.RenderList();

	glFlush();
	glutSwapBuffers();

	_ftime(&time2);
	long millisec = (time2.time-time1.time)*1000 + time2.millitm-time1.millitm;
	cout << "consuming time: "<< millisec << " milliseconds" << endl;

}

void CViewer::RealizeStream_NV()
{

	struct _timeb time1, time2;
	_ftime(&time1);

	glClear(GL_COLOR_BUFFER_BIT);

	cout <<"order: " <<list_order <<endl;
	current_vertex_indices	= vertex_indices[list_order];


	//Turn on our attribute streams!
	glEnableClientState(GL_VERTEX_ATTRIB_ARRAY0_NV);
	glEnableClientState(GL_VERTEX_ATTRIB_ARRAY1_NV);
	glEnableClientState(GL_VERTEX_ATTRIB_ARRAY3_NV);

	glVertexAttribPointerNV(0, 1, GL_FLOAT, sizeof(Vertex), &vertex_list[0].index);
	glVertexAttribPointerNV(1, 3, GL_FLOAT, sizeof(Vertex), &vertex_list[0].x);
	glVertexAttribPointerNV(3, 4, GL_FLOAT, sizeof(Vertex), &vertex_list[0].r);

	glDrawElements(GL_QUADS, 4*vertex_num, GL_UNSIGNED_INT, current_vertex_indices);

	glDisableClientState(GL_VERTEX_ATTRIB_ARRAY0_NV);
	glDisableClientState(GL_VERTEX_ATTRIB_ARRAY1_NV);
	glDisableClientState(GL_VERTEX_ATTRIB_ARRAY3_NV);


	glFlush();
	glutSwapBuffers();
}


void CViewer::RealizeStream()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glFlush();
	glutSwapBuffers();

	cout << "Vertex stream must use vertex program."<< endl;
}

void CViewer::RealizeConvolution_NV()
{
	struct _timeb time1, time2;
	long millisec;
	_ftime(&time1);

	int order = 0;

	cout <<"order: " <<order <<endl;
	current_vertex_indices	= vertex_indices[order];

	htimer1.Mark();

	pbf_makeCurrent(pbuffer);
	glClear(GL_COLOR_BUFFER_BIT);

	glDisable(GL_TEXTURE_2D);

	//Turn on our attribute streams!
	glEnableClientState(GL_VERTEX_ATTRIB_ARRAY0_NV);
	glEnableClientState(GL_VERTEX_ATTRIB_ARRAY3_NV);

	glVertexAttribPointerNV(0, 3, GL_FLOAT, sizeof(Vertex), &vertex_list[0].x);
	glVertexAttribPointerNV(3, 4, GL_FLOAT, sizeof(Vertex), &vertex_list[0].r);

	glDrawElements(GL_POINTS, vertex_num, GL_UNSIGNED_INT, current_vertex_indices);

	glDisableClientState(GL_VERTEX_ATTRIB_ARRAY0_NV);
	glDisableClientState(GL_VERTEX_ATTRIB_ARRAY3_NV);

	LARGE_INTEGER timing = htimer1.Elapse_us();
	cout << "draw buffer micro sec: " << timing.LowPart <<endl;

	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, texPbuffer);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
//	glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 256, 256, 32, 32);
	glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, image_width, image_height);

	pbf_makeGlutWindowCurrent(pbuffer);

	//
	// paste the pbuffer onto the screen
	//

	glClear(GL_COLOR_BUFFER_BIT);
	glColor3f(1.0, 1.0, 1.0);
	glBindTexture(GL_TEXTURE_2D, texPbuffer);

	glBegin(GL_QUADS);	
	glTexCoord2d(0, 0);  glVertex3d(-1.0, -1.0, 0);  
	glTexCoord2d(1, 0);  glVertex3d( 1.0, -1.0, 0);
	glTexCoord2d(1, 1);  glVertex3d( 1.0,  1.0, 0);  
	glTexCoord2d(0, 1);  glVertex3d(-1.0,  1.0, 0);  
	glEnd();

	glFlush();
	glutSwapBuffers();

	_ftime(&time2);
	millisec = (time2.time-time1.time)*1000 + time2.millitm-time1.millitm;
	cout << "draw frame:\t"<< millisec << " milliseconds" << endl<<endl;
}

void CViewer::RealizeConvolution()
{
	struct _timeb time1, time2;
	long millisec;
	_ftime(&time1);

	int order = 0;

	cout <<"order: " <<order <<endl;
	current_vertex_indices	= vertex_indices[order];

	htimer1.Mark();

	pbf_makeCurrent(pbuffer);
	glClear(GL_COLOR_BUFFER_BIT);

	glDisable(GL_TEXTURE_2D);

	//Turn on our attribute streams!
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);


	glVertexPointer(3, GL_FLOAT, sizeof(Vertex), &vertex_list[0].x);
	glColorPointer(4, GL_FLOAT, sizeof(Vertex), &vertex_list[0].r);

	glDrawElements(GL_POINTS, vertex_num, GL_UNSIGNED_INT, current_vertex_indices);

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);


	LARGE_INTEGER timing = htimer1.Elapse_us();
	cout << "draw buffer micro sec: " << timing.LowPart <<endl;

	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, texPbuffer);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
//	glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 256, 256, 32, 32);
	glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, image_width, image_height);


	_ftime(&time2);
	millisec = (time2.time-time1.time)*1000 + time2.millitm-time1.millitm;
	cout << "copy texture:\t"<< millisec << " milliseconds" << endl;

	pbf_makeGlutWindowCurrent(pbuffer);

	_ftime(&time2);
	millisec = (time2.time-time1.time)*1000 + time2.millitm-time1.millitm;
	cout << "switch buffer:\t"<< millisec << " milliseconds" << endl;


	glClear(GL_COLOR_BUFFER_BIT);
	glColor3f(1.0, 1.0, 1.0);
	glBindTexture(GL_TEXTURE_2D, texPbuffer);

	glBegin(GL_QUADS);	
	glTexCoord2d(0, 0);  glVertex3d(-1.0, -1.0, 0);  
	glTexCoord2d(1, 0);  glVertex3d( 1.0, -1.0, 0);
	glTexCoord2d(1, 1);  glVertex3d( 1.0,  1.0, 0);  
	glTexCoord2d(0, 1);  glVertex3d(-1.0,  1.0, 0);  
	glEnd();

	glFlush();
	glutSwapBuffers();

	_ftime(&time2);
	millisec = (time2.time-time1.time)*1000 + time2.millitm-time1.millitm;
	cout << "draw frame:\t"<< millisec << " milliseconds" << endl<<endl;
}
