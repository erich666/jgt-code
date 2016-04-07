/**
 * @file    glui_mc.cpp
 * @author  Thomas Lewiner <thomas.lewiner@polytechnique.org>
 * @author  Math Dept, PUC-Rio
 * @version 0.3
 * @date    30/05/2006
 *
 * @brief   MarchingCubes Graphical interface: main commands
 */
//________________________________________________


#ifndef WIN32
#pragma implementation "glui_defs.h"
#endif // WIN32

#include <stdio.h>
#include "gl2ps.h"
#include "csg.h"
#include "fparser.h"
#include "glui_defs.h"



//_____________________________________________________________________________
// declarations of this file

// main marching cubes object
MarchingCubes mc ;

// isovalue defining the isosurface
float isoval = 0.0f ;

// original/topological MC switch
int   originalMC = 0 ;

// grid extension
float xmin=-1.0f, xmax=1.0f,  ymin=-1.0f, ymax=1.0f,  zmin=-1.0f, zmax=1.0f ;
// grid size control
int   size_x=50, size_y=50, size_z=50 ;

// implicit formula
char  formula[1024] ;

// implicit functions
char *fun_list[NFUNS] =
{
  "Type Formula",
  "Sphere"      ,
  "Ellipsoid"   ,
  "Hyperboloid" ,
  "Plane"       ,
  "Cubic"       ,
  "Cushin"      ,
  "Cassini"     ,
  "Blooby"      ,
  "Chair"       ,
  "Cyclide"     ,
  "2 Spheres"   ,
  "2 Torii"     ,
  "Heart"       ,
  "Helio"
} ;

// implicit functions
char *fun_def [NFUNS] =
{
  "f(x,y,z, c,i)",
  "x^2+y^2+z^2-0.49",
  "2*x^2+y^2+z^2-0.49",
  "2*x^2-y^2-z^2-0.49",
  "x+y+z",
  "4*y^2-8*x^3+2*x",
  "(1.5*z)^2*(1.5*x)^2 - (1.5*z)^4 - 2*(1.5*z)*(1.5*x)^2 + 2*(1.5*z)^3 + (1.5*x)^2 - (1.5*z)^2 - ((1.5*x)^2 - (1.5*z))*((1.5*x)^2 - (1.5*z)) - (1.5*y)^4 - 2*(1.5*x)^2*(1.5*y)^2 - (1.5*y)^2*(1.5*z)^2 + 2*(1.5*y)^2*(1.5*z) + (1.5*y)^2",
  "((1.7*x)^2 + (1.7*y)^2 + (1.7*z)^2 + 0.45^2)*((1.7*x)^2 + (1.7*y)^2 + (1.7*z)^2 + 0.45^2) - 16*0.45^2*((1.7*x)^2 + (1.7*z)^2) - 0.25",
  "(3*x)^4 - 45*x^2+ (3*y)^4 - 45*y^2 + (3*z)^4 - 45*z^2 + 11.8",
  "((5*x)^2+(5*y)^2+(5*z)^2-0.95*25)*((5*x)^2+(5*y)^2+(5*z)^2-0.95*25)-0.8*(((5*z)-5)^2-2*(5*x)^2)*(((5*z)+5)^2-2*(5*y)^2)",
  "(25 - (6.9)^2)*(25 - (2.9)^2)*((10*x+4)^4+(10*y)^4+(10*z)^4)+ 2*((25 - (6.9)^2 )*(25 - (2.9)^2) * ((10*x+4)^2*(10*y)^2+(10*x+4)^2*(10*z)^2+(10*y)^2*(10*z)^2))+ 18*((21+4.9^2)* (4*(10*x+4)+9))*((10*x+4)^2+(10*y)^2+(10*z)^2)+ 4*3^4*(2*(10*x+4))*(-9+2*(10*x+4))+4*3^4*4.9^2*(10*y)^2+3^8",
  "((x-0.31)^2+(y-0.31)^2+(z-0.31)^2-0.263) * ((x+0.3)^2+(y+0.3)^2+(z+0.3)^2-0.263)",
  "( ( (8*x)^2 + (8*y-2)^2 + (8*z)^2 + 16 - 1.85*1.85 ) * ( (8*x)^2 + (8*y-2)^2 + (8*z)^2 + 16 - 1.85*1.85 ) - 64 * ( (8*x)^2 + (8*y-2)^2 ) ) * ( ( (8*x)^2 + ((8*y-2)+4)*((8*y-2)+4) + (8*z)^2 + 16 - 1.85*1.85 ) * ( (8*x)^2 + ((8*y-2)+4)*((8*y-2)+4) + (8*z)^2 + 16 - 1.85*1.85 ) - 64 * ( ((8*y-2)+4)*((8*y-2)+4) + (8*z)^2 ) ) + 1025",
  "(2*(1.3*x)^2+(1.3*y)^2+(1.3*z)^2-1)^3-(1/10)*(1.3*x)^2*(1.3*z)^3-(1.3*y)^2*(1.3*z)^3",
  "4*y^2-8*x^3+2*x",
} ;

// chosen implicit function
int   curr_string = -1 ;

// cube data
float v[8] ;

// loaded iso grid
FILE        *isofile  ;

// loaded CSG tree
CSG_Node    *csg_root ;


//-----------------------------------------------------------------------------

// run the MC algorithm
bool run() ;

//-----------------------------------------------------------------------------

// switch to export iso grid
int  export_iso = 0 ;

// set file extension of out_filename
int  set_ext( const char ext[3] ) ;

/// EPS export
void export_eps() ;

/// PPM export
void export_ppm() ;

/// TGA export
void export_tga() ;

//_____________________________________________________________________________







//_____________________________________________________________________________
// run the MC algorithm
bool run()
//-----------------------------------------------------------------------------
{
  if( strlen(formula) <= 0 ) return false ;
  if( export_iso && strlen( out_filename->get_text() ) <= 0 ) export_iso = 0 ;

  // Init data
  mc.clean_all() ;
  mc.set_resolution( size_x, size_y, size_z ) ;
  mc.init_all() ;

  // Parse formula
  FunctionParser fparser ;
  fparser.Parse( (const char*)formula, "x,y,z,c,i" ) ;
  if( fparser.EvalError() )
  {
    printf( "parse error\n" ) ;
    return false ;
  }

  // Fills data structure
  int i,j,k ;
  float val[5], w ;
  float rx = (xmax-xmin) / (size_x - 1) ;
  float ry = (ymax-ymin) / (size_y - 1) ;
  float rz = (zmax-zmin) / (size_z - 1) ;
  unsigned char buf[sizeof(float)] ;
  for( i = 0 ; i < size_x ; i++ )
  {
    val[X] = (float)i * rx  + xmin ;
    for( j = 0 ; j < size_y ; j++ )
    {
      val[Y] = (float)j * ry  + ymin ;
      for( k = 0 ; k < size_z ; k++ )
      {
        val[Z] = (float)k * rz  + zmin ;

        if( csg_root )
        {
          val[3] = csg_root->eval( val[X],val[Y],val[Z] ) ;
        }
        if( isofile  )
        {
          fread (buf, sizeof(float), 1, isofile);
          val[4] = * (float*) buf ;
        }

        w = fparser.Eval(val) - isoval ;
        mc.set_data( w, i,j,k ) ;
      }
    }
  }
  if( export_iso ) mc.writeISO( out_filename->get_text() ) ;

/*
  float data1[] = {0,3,1,-3,-3,-4,-1,-2,-4,-1,-3,-5,2,-1,-3,3,1,-1,-3,0,3,-3,1,2,0,-1,-1};
  float data2[] = {5,5,6, 10,12,13, 20,19,17, -4,-3,2, 1,-1,0, 10,8,5, 7,14,14, -2,3,-1, -1,-2,-2 };

  for (int k=km; k < kM; k++)
  {
    for (int j=jm; j < jM; j++)
    {
      for (int i=im; i < iM; i++)
      {
        printf( "%d,%d,%d->%d \t ", i,j,k, (int)data2[ 9*k + 3*j + i ] ) ;
        mc.set_data(data2[ 9*k + 3*j + i ], i-im, j-jm, k-km);
      }
      printf( " \t " ) ;
    }
    printf( "\n" ) ;
  }
*/

  // Run MC
  mc.set_method( originalMC == 1 ) ;
  mc.run() ;
  mc.clean_temps() ;

  // Rescale positions
  for( i = 0 ; i < mc.nverts() ; ++i )
  {
    Vertex &v = mc.vertices()[i] ;
    v.x = rx * v.x + xmin ;
    v.y = ry * v.y + xmin ;
    v.z = rz * v.z + ymin ;
    float nrm = v.nx * v.nx + v.ny * v.ny + v.nz * v.nz ;
    if( nrm != 0 )
    {
      nrm = 1.0 / sqrt(nrm) ;
      v.nx *= nrm ;
      v.ny *= nrm ;
      v.nz *= nrm ;
    }
  }

  if( isofile )
  {
    rewind( isofile ) ;
    fread (buf, sizeof(float), 1, isofile);
    fread (buf, sizeof(float), 1, isofile);
    fread (buf, sizeof(float), 1, isofile);
    fread (buf, sizeof(float), 1, isofile);
    fread (buf, sizeof(float), 1, isofile);
    fread (buf, sizeof(float), 1, isofile);
    fread (buf, sizeof(float), 1, isofile);
    fread (buf, sizeof(float), 1, isofile);
    fread (buf, sizeof(float), 1, isofile);
  }

#if USE_GL_DISPLAY_LIST
  draw() ;
#endif // USE_GL_DISPLAY_LIST

  return true ;
}
//_____________________________________________________________________________



//_____________________________________________________________________________
//_____________________________________________________________________________



//_____________________________________________________________________________
// set file extension of out_filename
int set_ext( const char ext[3] )
//-----------------------------------------------------------------------------
{
  char fn[1024] ;
  strcpy( fn, out_filename->get_text()) ;
  int l = strlen(fn) ;
  if( l == 0 ) return 0 ;
  if( fn[l-4] != '.' )
  {
    strcat( fn, "." ) ;
    strcat( fn, ext ) ;
  }
  else if( strcmp( fn + l-3, ext ) )
  {
    fn[l-3] = ext[0] ;
    fn[l-2] = ext[1] ;
    fn[l-1] = ext[2] ;
  }
  
  out_filename->set_text( fn ) ;
  return l ;
}
//_____________________________________________________________________________



//_____________________________________________________________________________
// EPS export
void export_eps()
//-----------------------------------------------------------------------------
{
  set_ext( "eps" ) ;
  FILE *fp = fopen( out_filename->get_text(), "wb" ) ;
  if( !fp ) return ;

  GLint viewport_gl[4];
  int   viewport[4];
  glutSetWindow(main_window);
  GLUI_Master.get_viewport_area( viewport + 0, viewport + 1, viewport + 2, viewport + 3 );
  viewport_gl[0] = viewport[0] ;
  viewport_gl[1] = viewport[1] ;
  viewport_gl[2] = viewport[2] + viewport[0] ;
  viewport_gl[3] = viewport[3] + viewport[1] ;

  gl2psBeginPage ( out_filename->get_text(), "Topological Marching Cubes (by Thomas Lewiner)", viewport_gl,
                   GL2PS_EPS, GL2PS_SIMPLE_SORT, // GL2PS_BSP_SORT,
                   GL2PS_SIMPLE_LINE_OFFSET | GL2PS_NO_PS3_SHADING | GL2PS_OCCLUSION_CULL,
                   GL_RGBA, 0, NULL, 0, 0, 0,  100*viewport[2]*viewport[3],
                   fp, NULL );
  display();
  gl2psEndPage();

  fclose(fp);
  printf( "exported image to EPS file %s\n", (const char*)out_filename ) ;
}
//_____________________________________________________________________________



//_____________________________________________________________________________
// EPS export
void export_ppm()
//-----------------------------------------------------------------------------
{
  set_ext( "ppm" ) ;
  FILE *fp = fopen( out_filename->get_text(), "w" ) ;
  if( !fp ) return ;

  int tx, ty, tw, th;
  glutSetWindow(main_window);
  GLUI_Master.get_viewport_area( &tx, &ty, &tw, &th );

  // read openGL buffer
  int buf_size = (tw * th * 3);
  unsigned char *buffer = new unsigned char[buf_size] ;
  draw() ;
  glFinish() ;
  glPixelStorei( GL_PACK_ALIGNMENT,  4 ); // Force 4-byte alignment
  glPixelStorei( GL_PACK_ROW_LENGTH, 0 );
  glPixelStorei( GL_PACK_SKIP_ROWS,  0 );
  glPixelStorei( GL_PACK_SKIP_PIXELS,0 );
  glReadPixels(tx, ty, tw, th, GL_RGB, GL_UNSIGNED_BYTE, buffer);

  // write the header
  fprintf(fp, "P6\n%d %d\n", tw, th);
  fprintf(fp, "# %s (Topological Marching Cubes by Thomas Lewiner)\n", (const char*)out_filename ) ;
  fprintf(fp, "255\n");

  // write file
  for (int i = 0; i < buf_size; ++i)
    fputc( buffer[i], fp ) ;

  delete [] buffer ;

  fclose(fp);
  printf( "exported image to PPM file %s\n", (const char*)out_filename ) ;

}
//_____________________________________________________________________________
