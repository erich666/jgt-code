#include <stdio.h>

#include <glh_nveb.h>
#include <glh_extensions.h>
#include <glh_genext.h>
#include <glh_linear.h>

#include <math.h>
#include "tsplat.h"
#include "viewer.h"
#include "volume.h"
#include "defines.h"

using namespace glh;

extern vec3f rotate_center;
extern CViewer viewer;


extern void OrientateSplat();

typedef struct _LuminanceAlphaPair{
	GLfloat luminance;
	GLfloat alpha;
}LuminanceAlphaPair;

//*******************
// TextureSplat
//*******************

TextureSplat::TextureSplat()
{
}

TextureSplat::~TextureSplat()
{
}

//*******************
// GaussTextureSplat
//*******************

void GaussTextureSplat::InitSplat(int w, int h, SCALAR s, SCALAR kernel_radius, SPLAT_TYPE type)
{

	width  = w;
	height = h;
	sigma  = s;
	radius = kernel_radius;
	s_type = type;
	LuminanceAlphaPair *image = new LuminanceAlphaPair[w*h];

	int i,j;
	float r, weight;
	float max_weight = 0.;
	float cx = (width-1)/2.0;
	float cy = (height-1)/2.0;

	sigma = s;
	for (i=0; i<height; i++)
		for (j=0; j<width; j++){
			float x, y;
			x = ((float)j)/(width-1.0) - 0.5;
			y = ((float)i)/(height-1.0) - 0.5;
			r = x*x + y*y;
			weight = exp(-r/(sigma*sigma));
			image[i*width+j].luminance  = 1;        // luminance
			image[i*width+j].alpha = weight;		// alpha
		}

	InitSplatTexture((GLfloat*)image);

#ifdef _GL_DISPLAYLIST
#ifdef __VERTEX_PROGRAM__
	InitRenderNV();
#else
	InitRender();
#endif
#endif

	delete [] image;
}

void GaussTextureSplat::InitSplatTexture(GLfloat* image)
{
	glShadeModel(GL_FLAT);

	glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

	glGenTextures(1, &texName);
	glBindTexture(GL_TEXTURE_2D, texName);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_LUMINANCE_ALPHA, GL_FLOAT, image);
//	glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE_ALPHA, width, height, 0, GL_LUMINANCE_ALPHA, GL_FLOAT, image);

	GLfloat priority=1.0;
	glPrioritizeTextures(1, &texName, &priority);
}

void GaussTextureSplat::InitRender()
{

	splatList = glGenLists(1);
	glNewList(splatList, GL_COMPILE);

	GLuint i,j,k;
	GLuint nx, ny, nz, voxel_num;
	CVolume *volume = viewer.volume;

	nx = volume->nx;
	ny = volume->ny;
	nz = volume->nz;
	voxel_num = volume->voxel_num;

	for(i=0; i<nz; i++)
	  for(j=0; j<ny; j++)
		  for(k=0; k<nx; k++){
			VOXEL *vox = &volume->voxel[i*nx*ny + j*nx + k];

			GLdouble vertex[3];
			vertex[0] = vox->x - rotate_center[0];
			vertex[1] = vox->y - rotate_center[1];
			vertex[2] = vox->z - rotate_center[2];

			GLdouble p0_o[3], p1_o[3], p2_o[3], p3_o[3];

			p0_o[0] = vertex[0] - radius; p0_o[1] = vertex[1] - radius; p0_o[2] = vertex[2];
			p1_o[0] = vertex[0] + radius; p1_o[1] = vertex[1] - radius; p1_o[2] = vertex[2];
			p2_o[0] = vertex[0] + radius; p2_o[1] = vertex[1] + radius; p2_o[2] = vertex[2];
			p3_o[0] = vertex[0] - radius; p3_o[1] = vertex[1] + radius; p3_o[2] = vertex[2];

			GLdouble p0_t[2] = {0., 0.};
			GLdouble p1_t[2] = {1., 0.};
			GLdouble p2_t[2] = {1., 1.};
			GLdouble p3_t[2] = {0., 1.};

			GLfloat r = vox->r/255.0;
			GLfloat g = vox->g/255.0;
			GLfloat b = vox->b/255.0;
			GLfloat a = vox->a/255.0;

			glColor4f(r,g,b,a);

			glBegin(GL_TRIANGLE_STRIP);
				glTexCoord2dv(p0_t); glVertex3dv(p0_o);
				glTexCoord2dv(p1_t); glVertex3dv(p1_o);
				glTexCoord2dv(p3_t); glVertex3dv(p3_o);
				glTexCoord2dv(p2_t); glVertex3dv(p2_o);
			glEnd();

		}

	glEndList();
}

void GaussTextureSplat::InitRenderNV()
{

	splatList = glGenLists(1);
	glNewList(splatList, GL_COMPILE);

	GLuint i,j,k;
	GLuint nx, ny, nz, voxel_num;
	CVolume *volume = viewer.volume;

	nx = volume->nx;
	ny = volume->ny;
	nz = volume->nz;
	voxel_num = volume->voxel_num;

	for(i=0; i<nz; i++)
	  for(j=0; j<ny; j++)
		  for(k=0; k<nx; k++){
			VOXEL *vox = &volume->voxel[i*nx*ny+j*nx+k];

			GLdouble p0_o[3], p1_o[3], p2_o[3], p3_o[3];

			GLdouble vertex[3] = { 0, 0, 0 };
			p0_o[0] = 0; p0_o[1] = radius; p0_o[2] = 0;
			p1_o[0] = 1; p1_o[1] = radius; p1_o[2] = 0;
			p2_o[0] = 2; p2_o[1] = radius; p2_o[2] = 0;
			p3_o[0] = 3; p3_o[1] = radius; p3_o[2] = 0;

			GLdouble p0_t[2] = {0., 0.};
			GLdouble p1_t[2] = {1., 0.};
			GLdouble p2_t[2] = {1., 1.};
			GLdouble p3_t[2] = {0., 1.};

			GLfloat r = vox->r/255.0;
			GLfloat g = vox->g/255.0;
			GLfloat b = vox->b/255.0;
			GLfloat a = vox->a/255.0;

			glColor4f(r,g,b,a);


			glVertexAttrib4fNV(1, vox->x-rotate_center[0], vox->y-rotate_center[1], vox->z-rotate_center[2], 1.0);

			glBegin(GL_QUADS);
				glVertex3dv(p0_o);
				glVertex3dv(p1_o);
				glVertex3dv(p2_o);
				glVertex3dv(p3_o);
			glEnd();

		  }

	glEndList();
}

void GaussTextureSplat::RenderList()
{
	if (s_type == CIRCLE){
		glCallList (splatList);
	}
	else if (s_type == HEXAGON){
	}
	else {
		printf("invalid splat type\n");
	}
}

void GaussTextureSplat::Render(GLfloat intens)
{

	    GLdouble p0_o[3], p1_o[3], p2_o[3], p3_o[3];

		GLdouble vertex[3] = { 0, 0, 0 };
		p0_o[0] = vertex[0]-radius; p0_o[1] = vertex[1]-radius; p0_o[2] = vertex[2];
		p1_o[0] = vertex[0]+radius; p1_o[1] = vertex[1]-radius; p1_o[2] = vertex[2];
		p2_o[0] = vertex[0]+radius; p2_o[1] = vertex[1]+radius; p2_o[2] = vertex[2];
		p3_o[0] = vertex[0]-radius; p3_o[1] = vertex[1]+radius; p3_o[2] = vertex[2];

		GLdouble p0_t[2] = {0., 0.};
		GLdouble p1_t[2] = {1., 0.};
		GLdouble p2_t[2] = {1., 1.};
		GLdouble p3_t[2] = {0., 1.};

		GLfloat c = intens;
		glColor4f(c, c, c, c);	

		glBegin(GL_QUADS);
			glTexCoord2dv(p0_t); glVertex3dv(p0_o);
			glTexCoord2dv(p1_t); glVertex3dv(p1_o);
			glTexCoord2dv(p2_t); glVertex3dv(p2_o);
			glTexCoord2dv(p3_t); glVertex3dv(p3_o);
		glEnd();
}

void GaussTextureSplat::Render(GLfloat r, GLfloat g, GLfloat b, GLfloat a)
{

	    GLdouble p0_o[3], p1_o[3], p2_o[3], p3_o[3];

		GLdouble vertex[3] = { 0, 0, 0 };
		p0_o[0] = vertex[0]-radius; p0_o[1] = vertex[1]-radius; p0_o[2] = vertex[2];
		p1_o[0] = vertex[0]+radius; p1_o[1] = vertex[1]-radius; p1_o[2] = vertex[2];
		p2_o[0] = vertex[0]+radius; p2_o[1] = vertex[1]+radius; p2_o[2] = vertex[2];
		p3_o[0] = vertex[0]-radius; p3_o[1] = vertex[1]+radius; p3_o[2] = vertex[2];

		GLdouble p0_t[2] = {0., 0.};
		GLdouble p1_t[2] = {1., 0.};
		GLdouble p2_t[2] = {1., 1.};
		GLdouble p3_t[2] = {0., 1.};

		glColor4f(r, g, b, a);	

		glBegin(GL_QUADS);
			glTexCoord2dv(p0_t); glVertex3dv(p0_o);
			glTexCoord2dv(p1_t); glVertex3dv(p1_o);
			glTexCoord2dv(p2_t); glVertex3dv(p2_o);
			glTexCoord2dv(p3_t); glVertex3dv(p3_o);
		glEnd();
}

void GaussTextureSplat::RenderNV(GLfloat intens)
{

	    GLdouble p0_o[3], p1_o[3], p2_o[3], p3_o[3];

		GLdouble vertex[3] = { 0, 0, 0 };
		p0_o[0] = 0; p0_o[1] = radius; p0_o[2] = 0;
		p1_o[0] = 1; p1_o[1] = radius; p1_o[2] = 0;
		p2_o[0] = 2; p2_o[1] = radius; p2_o[2] = 0;
		p3_o[0] = 3; p3_o[1] = radius; p3_o[2] = 0;

		GLdouble p0_t[2] = {0., 0.};
		GLdouble p1_t[2] = {1., 0.};
		GLdouble p2_t[2] = {1., 1.};
		GLdouble p3_t[2] = {0., 1.};

		GLfloat c = intens;
		glColor4f(c, c, c, c);	

		glBegin(GL_QUADS);
			glTexCoord2dv(p0_t); glVertex3dv(p0_o);
			glTexCoord2dv(p1_t); glVertex3dv(p1_o);
			glTexCoord2dv(p2_t); glVertex3dv(p2_o);
			glTexCoord2dv(p3_t); glVertex3dv(p3_o);

		glEnd();

}

void GaussTextureSplat::RenderNV(GLfloat r, GLfloat g, GLfloat b, GLfloat a)
{

	    GLdouble p0_o[3], p1_o[3], p2_o[3], p3_o[3];

		GLdouble vertex[3] = { 0, 0, 0 };
		p0_o[0] = 0; p0_o[1] = radius; p0_o[2] = 0;
		p1_o[0] = 1; p1_o[1] = radius; p1_o[2] = 0;
		p2_o[0] = 2; p2_o[1] = radius; p2_o[2] = 0;
		p3_o[0] = 3; p3_o[1] = radius; p3_o[2] = 0;

		GLdouble p0_t[2] = {0., 0.};
		GLdouble p1_t[2] = {1., 0.};
		GLdouble p2_t[2] = {1., 1.};
		GLdouble p3_t[2] = {0., 1.};

		glColor4f(r, g, b, a);

		glBegin(GL_QUADS);
			glTexCoord2dv(p0_t); glVertex3dv(p0_o);
			glTexCoord2dv(p1_t); glVertex3dv(p1_o);
			glTexCoord2dv(p2_t); glVertex3dv(p2_o);
			glTexCoord2dv(p3_t); glVertex3dv(p3_o);
		glEnd();
}
