#include <stdio.h>
#include <io.h>
#include <fcntl.h>
#include <time.h>

#define GLH_NVEB_USING_NVPARSE
#define GLH_EXT_SINGLE_FILE

#include <glh_nveb.h>
#include <glh_extensions.h>
#include <glh_glut.h>
#include "nvparse.h"
#include "read_text_file.h"

#include "tsplat.h"
#include "viewer.h"
#include "defines.h"
#include "pbuffer.h"
#include "HTimer.h"

void reshape(int w, int h);
void display(void);
void key(unsigned char key, int x, int y);
void init_splat_texture();
void loadVolume(char* filename);


using namespace glh;

glut_callbacks cb;
glut_simple_mouse_interactor camera, object;

static GLuint texName;
static GLuint vpid;
static GLuint vspid;
static bool vertex_program=false;


CViewer viewer;
GaussTextureSplat gts;

CHTimer htimer;

pbf pbuffer;
GLuint texPbuffer;

GLuint image_width;
GLuint image_height;

vec3f rotate_center;
GLfloat rotate_matrix[16];

GLint list_order;

float NULL_DATA[4] = {0.0f, 0.0f, 0.0f, 0.0f};


//*****************************************
// routines for pre-sorting the voxel lsit
//*****************************************
void initSortedLists()
{
#if  defined(_GL_VERTEXSTREAM) 
	viewer.InitVoxelList();
	viewer.InitSortedLists();
#else
	viewer.InitConvolutionVoxelList();
	viewer.InitSortedLists();
#endif 
}

//****************************
// routines for texture splat
//****************************

void initSplatTexture()
{	
	int w = viewer.splat_info.tsplat_size;
	int h = w;
	float sigma = viewer.splat_info.sigma;
	float r = viewer.splat_info.kernel_radius;

	vec3f color;
	color[0] = viewer.volume->red;
	color[1] = viewer.volume->green;
	color[2] = viewer.volume->blue;

	gts.SetTransferColor(color[0], color[1], color[2]);
	gts.InitSplat(w, h, sigma, r, CIRCLE);
	//gts.InitSplat(w, h, sigma, r, HEXAGON);
}

void GetWorld2EyeMatrix(GLdouble *m)
{
	glGetDoublev(GL_MODELVIEW_MATRIX, m);
}

void OrientateSplat()
{
	object.trackball.apply_inverse_transform();
}

int GetListOrder()
{
	register int order=0;

	glPushMatrix();
	glLoadIdentity();
	object.trackball.apply_inverse_transform();
	GLdouble m[16];
	glGetDoublev(GL_MODELVIEW_MATRIX, m);
	matrix4f tm(m[0], m[4], m[8],  m[12],
			   m[1], m[5], m[9],  m[13],
			   m[2], m[6], m[10], m[14],
			   m[3], m[7], m[11], m[15]);
	
	tm = tm.inverse();
	tm = tm.transpose();
	
	vec3f splat_normal(0, 0, 1);
	tm.mult_matrix_vec(splat_normal);
	splat_normal.normalize();

	float max = -1;
	float dm;
	for (int i=0; i<SORTED_LIST_NUM; i++){
		VECTOR3 s(splat_normal[0], splat_normal[1], splat_normal[2]);
		VECTOR3 t = viewer.eyes[i];
		t.Normalize();
		dm = dot(t, s);

		if (dm>max){
			order = i;
			max = dm;
		}
	}

	glPopMatrix();
	return order;
}


//***************************
// miscellenious routines
//***************************

void loadVolume(char* filename)
{
	viewer.Load(filename);
}

void WriteToPNM()  {
  char *image_buf = new char [image_width*image_height*3];

	glPixelTransferf(GL_RED_BIAS, 0.);
	glPixelTransferf(GL_GREEN_BIAS, 0.);
	glPixelTransferf(GL_BLUE_BIAS, 0.);
  
  glReadPixels(0, 0, image_width, image_height, GL_RGB, GL_UNSIGNED_BYTE, image_buf);

  FILE* fp;
  if (NULL != (fp = fopen("out.pnm", "wb"))){

    // Write the 'header' information
    fprintf(fp, "P6 %d %d 255\n", image_width, image_height);

	_setmode(_fileno(fp), _O_BINARY);

    for (int i = image_height - 1; i >= 0; --i){
          // write binary data
			fwrite(image_buf+3*i*image_width, sizeof(unsigned char), 3*image_width, fp); 
    } 

    fclose(fp);
  }
  delete [] image_buf;
}


int getConvolutionSize()
{
	double r = viewer.splat_info.kernel_radius;
	int w = image_width;
	double ratio = viewer.view_info.right - viewer.view_info.left;
	return ((int)(r*w*2/ratio));
}

//***************************
// GLUT callback functions
//****************************

void reshape(int w, int h)
{	

	image_width  = w;
	image_height = h;

#if defined(_GL_CONVOLUTION) && defined(__PBUFFER__)
	pbf_makeCurrent(pbuffer);
#endif

	glViewport (0, 0, w, h);
	camera.reshape (w, h);
	object.reshape (w, h);


	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	if (viewer.view_info.type == PARALLEL){
		glOrtho(viewer.view_info.left, viewer.view_info.right, viewer.view_info.bottom, viewer.view_info.top, 
			viewer.view_info.hither, viewer.view_info.yon );
	}
	else{ // perspective
		float aspect = viewer.view_info.aspect_ratio * float(w)/float(h);
		float real_fov;
		if ( aspect < 1 ){
		  // fovy is a misnomer.. we really mean the fov applies to the
		  // smaller dimension
		  float fovx, fovy; 
		  fovx = fovy = viewer.view_info.view_angle; 
		  real_fov = to_degrees(2 * atan(tan(to_radians(fovx/2))/aspect));
		}
		else{
			real_fov = viewer.view_info.view_angle;
		}

		gluPerspective(real_fov, 
					   aspect,
					   viewer.view_info.hither,
					   viewer.view_info.yon);

	}
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

#if defined(_GL_CONVOLUTION) && defined(__PBUFFER__)
	pbf_makeGlutWindowCurrent(pbuffer);
#endif

#if defined(_GL_CONVOLUTION) && defined(__PBUFFER__)
	// set GL parameters for on-screen frame buffer
	glViewport (0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-1, 1, -1, 1, -1, 1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
#endif
}

void display(void)
{	

#if defined(_GL_CONVOLUTION) && defined(__PBUFFER__)
	pbf_makeCurrent(pbuffer);
#endif

	// set new model view and projection matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	camera.apply_inverse_transform();
	object.apply_transform();

	glRotatef(1, 1, 0, 0);
	glRotatef(1, 0, 1, 0);

	list_order = GetListOrder();
// for snapshot
/*
	{

		// for hipip
		GLdouble m[16];
		glGetDoublev(GL_MODELVIEW_MATRIX, m);
		printf("m[0]=%10f; m[4]=%10f; m[8]=%10f; m[12]=%10f;\n", m[0], m[4], m[8], m[12]);
		printf("m[1]=%10f; m[5]=%10f; m[9]=%10f; m[13]=%10f;\n", m[1], m[5], m[9], m[13]);
		printf("m[2]=%10f; m[6]=%10f; m[10]=%10f; m[14]=%10f;\n", m[2], m[6], m[10], m[14]);
		printf("m[3]=%10f; m[7]=%10f; m[11]=%10f; m[15]=%10f;\n", m[3], m[7], m[11], m[15]);


		m[0]=  0.050937; m[4]=  0.942415; m[8]= -0.330544; m[12]=  0.000000;
		m[1]=  0.207775; m[5]= -0.333731; m[9]= -0.919485; m[13]=  0.000000;
		m[2]= -0.976849; m[6]= -0.021843; m[10]= -0.212810; m[14]= -5.000000;
		m[3]=  0.000000; m[7]=  0.000000; m[11]=  0.000000; m[15]=  1.000000;

		glLoadMatrixd(m);
		list_order = 5;

	}
*/
#if defined(_GL_CONVOLUTION) && defined(__PBUFFER__)
	pbf_makeGlutWindowCurrent(pbuffer);
#endif


	htimer.Mark();

#ifdef __VERTEX_PROGRAM__ // with nVidia vertex program
		// Enable the vertex program.

#	if defined(_GL_CONVOLUTION) && defined(__PBUFFER__)
	pbf_makeCurrent(pbuffer);
#	endif

		glEnable( GL_VERTEX_PROGRAM_NV );
		glBindProgramNV( GL_VERTEX_PROGRAM_NV, vpid );
		glExecuteProgramNV(GL_VERTEX_STATE_PROGRAM_NV, vspid, NULL_DATA);

#	ifdef _GL_CONVOLUTION
	pbf_makeGlutWindowCurrent(pbuffer);
#	endif


#	if defined( _GL_DISPLAYLIST )
		viewer.RealizeList_NV();

#	elif defined( _GL_IMMEDIATE )
		viewer.RealizeImm_NV();

#	elif defined( _GL_VERTEXSTREAM )
		viewer.RealizeStream_NV();

#	else // _GL_CONVOLUTION
		viewer.RealizeConvolution_NV();
#	endif

#	if defined(_GL_CONVOLUTION) && defined(__PBUFFER__)
		glDisable( GL_VERTEX_PROGRAM_NV );
#	endif

#else // no nVidia vertex program


#	if defined( _GL_DISPLAYLIST )
		viewer.RealizeList();

#	elif defined( _GL_IMMEDIATE )
		viewer.RealizeImm();

#	elif defined( _GL_VERTEXSTREAM )
		viewer.RealizeStream();

#	else // _GL_CONVOLUTION
		viewer.RealizeConvolution();
#	endif

#endif // __VERTEX_PROGRAM__


		LARGE_INTEGER timing = htimer.Elapse_us();

		cout << "micro sec: " << timing.LowPart <<endl;
}


void key(unsigned char key, int x, int y)
{
	switch (key) {
    case 's':
	case 'S':
		WriteToPNM();
		break;
    case 'f':
	case 'F':
		glutPostRedisplay();
		break;
	case '\033':
	case 'q':
	case 'Q':
#ifdef __VERTEX_PROGRAM__
        glDeleteProgramsNV (1, &vpid);
        glDeleteProgramsNV (1, &vspid);
#endif

#if defined(__PBUFFER__)
		pbf_destroy(pbuffer);
#endif
		exit(0);
		break;
	}
}

//******************************************
// initialization code for GL and NV_EXT
//******************************************

void CheckParseErrors()
{
	for (char * const * errors = nvparse_get_errors(); *errors; errors++)
		fprintf(stderr, *errors);
}

void LOAD_VERTEX_PROGRAM( unsigned int _i, char *_n )
{
    char *str = read_text_file( _n );
    glBindProgramNV( GL_VERTEX_PROGRAM_NV, _i );
    nvparse( str );
   	CheckParseErrors();
    delete [] str;
}

void LOAD_VERTEX_STATE_PROGRAM( unsigned int _i, char *_n )
{
    char *str = read_text_file( _n );
	nvparse( str, _i );
	CheckParseErrors();
    delete [] str;
}

void initNV()
{
	if( !glh_init_extensions( "GL_NV_vertex_program" ) ){
		cerr << "Necessary extensions were not supported:" << endl
			 << glh_get_unsupported_extensions() << endl << endl
			 << "Press <enter> to quit." << endl;
		char buff[10];
		cin.getline(buff, 10);
		exit( -1 );
    }


#if defined(_GL_CONVOLUTION) && defined(__PBUFFER__)
	pbf_makeCurrent(pbuffer);
#endif

	// Load the vertex state program
	glGenProgramsNV(1, &vspid);
	LOAD_VERTEX_STATE_PROGRAM(vspid, "tsplat.vsp");

    // Load the vertex program.
    glGenProgramsNV(1, &vpid);
#if defined(_GL_CONVOLUTION)
	LOAD_VERTEX_PROGRAM( vpid, "tsplat_c.vp" );
#else
	LOAD_VERTEX_PROGRAM( vpid, "tsplat.vp" );
#endif

	//Track the Modelview matrix into Vector Registers c[0] to c[3].
	glTrackMatrixNV(GL_VERTEX_PROGRAM_NV, 0, GL_MODELVIEW, GL_IDENTITY_NV);

    // Track the concatenation of the modelview and projection matrix in registers 4-7.
    glTrackMatrixNV( GL_VERTEX_PROGRAM_NV, 4, GL_MODELVIEW_PROJECTION_NV, GL_IDENTITY_NV );

	
	// multiplier for billboard
	glProgramParameter4fNV( GL_VERTEX_PROGRAM_NV, 16,  -1.0f, -1.0f, 1, 0 );
	glProgramParameter4fNV( GL_VERTEX_PROGRAM_NV, 17,  1.0f, -1.0f, 1, 0 );
	glProgramParameter4fNV( GL_VERTEX_PROGRAM_NV, 18,  1.0f, 1.0f, 1, 0 );
	glProgramParameter4fNV( GL_VERTEX_PROGRAM_NV, 19,  -1.0f, 1.0f, 1, 0 );

	glProgramParameter4fNV( GL_VERTEX_PROGRAM_NV, 24,  0, 0, 0, 0 );
	glProgramParameter4fNV( GL_VERTEX_PROGRAM_NV, 25,  1.f, 0, 0, 0 );
	glProgramParameter4fNV( GL_VERTEX_PROGRAM_NV, 26,  1.f, 1.f, 0, 0 );
	glProgramParameter4fNV( GL_VERTEX_PROGRAM_NV, 27,  0, 1.f, 0, 0 );

	float r = viewer.splat_info.kernel_radius;
	glProgramParameter4fNV( GL_VERTEX_PROGRAM_NV, 8, r, r, r, r);


#if defined(_GL_CONVOLUTION) && defined(__PBUFFER__)
	pbf_makeGlutWindowCurrent(pbuffer);
#endif

}

void initConvolution()
{
/*
	if( !glh_init_extensions( "GL_EXT_convolution" ) ){
		cerr << "Necessary extensions were not supported:" << endl
			 << glh_get_unsupported_extensions() << endl << endl
			 << "Press <enter> to quit." << endl;
		char buff[10];
		cin.getline(buff, 10);
		exit( -1 );
    }
*/

	//
	//  GL_EXT_convolution is supported by Gefore3/4 boards. But it is not appeared in extension string.
	//  So we load it mannually.
	//

    GLH_EXT_NAME(glConvolutionFilter1D) = (PFNGLCONVOLUTIONFILTER1DPROC)GLH_EXT_GET_PROC_ADDRESS("glConvolutionFilter1D");
    GLH_EXT_NAME(glConvolutionFilter2D) = (PFNGLCONVOLUTIONFILTER2DPROC)GLH_EXT_GET_PROC_ADDRESS("glConvolutionFilter2D");
    GLH_EXT_NAME(glConvolutionParameterf) = (PFNGLCONVOLUTIONPARAMETERFPROC)GLH_EXT_GET_PROC_ADDRESS("glConvolutionParameterf");
    GLH_EXT_NAME(glConvolutionParameterfv) = (PFNGLCONVOLUTIONPARAMETERFVPROC)GLH_EXT_GET_PROC_ADDRESS("glConvolutionParameterfv");
    GLH_EXT_NAME(glConvolutionParameteri) = (PFNGLCONVOLUTIONPARAMETERIPROC)GLH_EXT_GET_PROC_ADDRESS("glConvolutionParameteri");
    GLH_EXT_NAME(glConvolutionParameteriv) = (PFNGLCONVOLUTIONPARAMETERIVPROC)GLH_EXT_GET_PROC_ADDRESS("glConvolutionParameteriv");
    GLH_EXT_NAME(glCopyConvolutionFilter1D) = (PFNGLCOPYCONVOLUTIONFILTER1DPROC)GLH_EXT_GET_PROC_ADDRESS("glCopyConvolutionFilter1D");
    GLH_EXT_NAME(glCopyConvolutionFilter2D) = (PFNGLCOPYCONVOLUTIONFILTER2DPROC)GLH_EXT_GET_PROC_ADDRESS("glCopyConvolutionFilter2D");
    GLH_EXT_NAME(glGetConvolutionFilter) = (PFNGLGETCONVOLUTIONFILTERPROC)GLH_EXT_GET_PROC_ADDRESS("glGetConvolutionFilter");
    GLH_EXT_NAME(glGetConvolutionParameterfv) = (PFNGLGETCONVOLUTIONPARAMETERFVPROC)GLH_EXT_GET_PROC_ADDRESS("glGetConvolutionParameterfv");
    GLH_EXT_NAME(glGetConvolutionParameteriv) = (PFNGLGETCONVOLUTIONPARAMETERIVPROC)GLH_EXT_GET_PROC_ADDRESS("glGetConvolutionParameteriv");
    GLH_EXT_NAME(glGetSeparableFilter) = (PFNGLGETSEPARABLEFILTERPROC)GLH_EXT_GET_PROC_ADDRESS("glGetSeparableFilter");
    GLH_EXT_NAME(glSeparableFilter2D) = (PFNGLSEPARABLEFILTER2DPROC)GLH_EXT_GET_PROC_ADDRESS("glSeparableFilter2D");

//	GLfloat kernel[3][3] = { {1.0, 0.0, 0.0}, {0.0, .0, 0.0}, {0.0, 0.0, 0.0} };
//	GLfloat kernel[3][3] = { {0.2, 0.2, 0.2}, {0.2, 1.0, 0.2}, {0.2, 0.2, 0.2} };
//	GLfloat kernel[3][3] = { {1, 1, 1}, {1, 1, 1}, {1, 1, 1} };
//	GLfloat kernel[5][5] = { {1.0, 1.0, 1.0, 1.0, 1.0}, {1.0, 1.0, 1.0, 1.0, 1.0}, {1.0, 1.0, 1.0, 1.0, 1.0},{1.0, 1.0, 1.0, 1.0, 1.0}, {1.0, 1.0, 1.0, 1.0, 1.0} };

//	GLfloat kernel[7][7] = { {1,1,1,1,1,1,1},{1,1,1,1,1,1,1},{1,1,1,1,1,1,1},{1,1,1,1,1,1,1},{1,1,1,1,1,1,1},{1,1,1,1,1,1,1},{1,1,1,1,1,1,1}};
//	GLfloat kernel[7][7] = { {0.2,0.2,0.2,0.2,0.2,0.2,0.2},{0.2,0.4,0.4,0.4,0.4,0.4,0.2}, {0.2,0.4,0.8,0.8,0.8,0.4,0.2}, {0.2,0.4,0.8,1,0.8,0.4,0.2}, {0.2,0.4,0.8,0.8,0.8,0.4,0.2}, {0.2,0.4,0.4,0.4,0.4,0.4,0.2}, {0.2,0.2,0.2,0.2,0.2,0.2,0.2} };
	int size;

	//size = getConvolutionSize();
	size = 9;

	cout << "size: " << size <<endl;

	GLfloat *kernel = new GLfloat[size*size];

	int i,j;
	float r, weight;
	float sigma;
	sigma = viewer.splat_info.sigma;

	for (i=0; i<size; i++)
		for (j=0; j<size; j++){
			float x, y;
			x = ((float)j)/(size-1.0) - 0.5;
			y = ((float)i)/(size-1.0) - 0.5;
			r = x*x + y*y;
			weight = exp(-r/(sigma*sigma));
			kernel[i*size+j] = weight;   // luminance
		}

    glEnable(GL_CONVOLUTION_2D);

	if (glGetError()!=GL_NO_ERROR){
		cout << "error-pre" <<endl;
	}

    //glConvolutionFilter2D (GL_CONVOLUTION_2D, GL_LUMINANCE, size, size, GL_LUMINANCE, GL_FLOAT, kernel);
	glConvolutionFilter2D (GL_CONVOLUTION_2D, GL_RGBA, size, size, GL_LUMINANCE, GL_FLOAT, kernel);


	if (glGetError()!=GL_NO_ERROR){
		cout << "error" <<endl;
	}

	delete [] kernel;
}


void initPbufferTexture()
{
	GLfloat *image = new GLfloat[image_width*image_height];

	int w = image_width;//32;
	int h = image_height;//32;

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glGenTextures(1, &texPbuffer);
	glBindTexture(GL_TEXTURE_2D, texPbuffer);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
	//glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, w, h, 0, GL_LUMINANCE, GL_FLOAT, image);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_LUMINANCE, GL_FLOAT, image);

	GLfloat priority=1.0;
	glPrioritizeTextures(1, &texPbuffer, &priority);

	glEnable(GL_TEXTURE_2D);

	delete [] image;
}

void initPbuffer()
{

	if( !glh_init_extensions( "WGL_ARB_pbuffer" ) ){
		cerr << "Necessary extensions were not supported:" << endl
			 << glh_get_unsupported_extensions() << endl << endl
			 << "Press <enter> to quit." << endl;
		char buff[10];
		cin.getline(buff, 10);
		exit( -1 );
    }


	if( !glh_init_extensions( "WGL_ARB_pixel_format" ) ){
		cerr << "Necessary extensions were not supported:" << endl
			 << glh_get_unsupported_extensions() << endl << endl
			 << "Press <enter> to quit." << endl;
		char buff[10];
		cin.getline(buff, 10);
		exit( -1 );
    }

	pbuffer = pbf_create(image_width, image_height,	GLUT_RGBA | GLUT_SINGLE , true);

	if (pbuffer==NULL){
		cerr << "could not create pbuffer" <<endl;
		exit(-1);
	}

	pbf_makeCurrent(pbuffer);

	glShadeModel(GL_FLAT);
	initPbufferTexture();

	//glEnable(GL_POINT_SMOOTH);
	//glPointSize(1.);

#ifdef _GL_CONVOLUTION
	initConvolution();
#endif

	glClearColor(0., 0., 0., 0.);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE);
	pbf_makeGlutWindowCurrent(pbuffer);

}

void initGL()
{
#ifdef __PBUFFER__
	initPbuffer();
#endif

#ifdef __VERTEX_PROGRAM__
	initNV();
#endif

	glEnable(GL_BLEND);
#ifdef __X_RAY_MODEL__
	glBlendFunc(GL_SRC_ALPHA, GL_ONE);
#else
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
#endif

	glEnable(GL_TEXTURE_2D);

	glClearColor(0., 0., 0., 0.);
}

//**********************
// program entry
//**********************

int main(int argc, char **argv)
{
	if (argc!=2){
		printf("%s <input_file\n", argv[0]);
		exit(1);
	}

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);

	loadVolume(argv[1]);
	image_width  = viewer.viewport.Width();
	image_height = viewer.viewport.Height();

	glutInitWindowSize(image_width, image_height); 
	glutCreateWindow("Texture splat");
	glutInitWindowPosition(200, 200);

	initGL();

	initSortedLists();

#if !defined(_GL_CONVOLUTION)
	initSplatTexture();
#endif
	// initilize glut helper
    glut_helpers_initialize();

	cb.keyboard_function = key;
	camera.configure_buttons(1);
	camera.set_camera_mode(true);
	camera.pan.pan = vec3f( 0.0, 0.0, viewer.view_info.eye[2]); 
	object.configure_buttons(1);

	object.dolly.dolly[0] = 0;
	object.dolly.dolly[1] = 0;
	object.dolly.dolly[2] = 0;

	glut_add_interactor(&cb);
	glut_add_interactor(&object);

	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(key);

	glutMainLoop();

	return 0;
}
