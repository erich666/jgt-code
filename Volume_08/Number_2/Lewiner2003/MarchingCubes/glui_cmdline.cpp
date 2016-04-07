/**
 * @file    glui_cmdline.cpp
 * @author  Thomas Lewiner <thomas.lewiner@polytechnique.org>
 * @author  Math Dept, PUC-Rio
 * @version 0.3
 * @date    30/05/2006
 *
 * @brief   MarchingCubes Graphical interface: command line parser
 */
//________________________________________________


#ifndef WIN32
#pragma implementation "glui_defs.h"
#endif // WIN32

#include "glui_defs.h"


//_____________________________________________________________________________
// parse_cmdline
bool parse_cmdline(int argc, char* argv[])
//-----------------------------------------------------------------------------
{
  bool quit = false ;
  for( int i = 1 ; i < argc ; ++i )
  {
    if     ( !strcmp( argv[i], "-i" ) )
    {
      if( ++i != argc )
      {
        in_filename->set_text( argv[i] ) ;
        control_cb( ISO_ID ) ;
      }
    }
    else if( !strcmp( argv[i], "-c" ) )
    {
      if( ++i != argc )
      {
        in_filename->set_text( argv[i] ) ;
        control_cb( CSG_ID ) ;
      }
    }
    if     ( !strcmp( argv[i], "-iply" ) )
    {
      if( ++i != argc )
      {
        in_filename->set_text( argv[i] ) ;
        control_cb( IPLY_ID ) ;
      }
    }
    else if( !strcmp( argv[i], "-f" ) )
    {
      if( ++i != argc ) strcpy( formula, argv[i] ) ;
    }
    else if( !strcmp( argv[i], "-r" ) )
    {
      if( ++i != argc ) { size_x = size_y = size_z = atoi( argv[i] ) ; }
    }
    else if( !strcmp( argv[i], "-rx" ) )
    {
      if( ++i != argc ) size_x = atoi( argv[i] ) ;
    }
    else if( !strcmp( argv[i], "-ry" ) )
    {
      if( ++i != argc ) size_y = atoi( argv[i] ) ;
    }
    else if( !strcmp( argv[i], "-rz" ) )
    {
      if( ++i != argc ) size_z = atoi( argv[i] ) ;
    }
    else if( !strcmp( argv[i], "-iv" ) )
    {
      if( ++i != argc ) { out_filename->set_text( argv[i] ) ; control_cb( IV_ID ) ; }
    }
    else if( !strcmp( argv[i], "-ply" ) )
    {
      if( ++i != argc ) { out_filename->set_text( argv[i] ) ; control_cb( PLY_ID ) ; }
    }
    else if( !strcmp( argv[i], "-R" ) )
    {
      control_cb( RUN_ID ) ;
    }
    else if( !strcmp( argv[i], "-eps" ) )
    {
      if( ++i != argc ) { out_filename->set_text( argv[i] ) ; control_cb( EPS_ID ) ; }
    }
    else if( !strcmp( argv[i], "-ppm" ) )
    {
      if( ++i != argc ) { out_filename->set_text( argv[i] ) ; control_cb( PPM_ID ) ; }
    }
    else if( !strcmp( argv[i], "-q" ) )
    {
      quit = true ;
    }
    else if( !strcmp( argv[i], "-h" ) )
    {
      printf( "usage %s [-i file.iso] [-iply file.ply] [-c file.csg] [-f 'formula'] [-r[xyz] res] -R [-iv file.iv] [-ply file.ply] [-ppm file.ppm] [-eps file.eps] [-q]\n", argv[0] ) ;
    }
  }
  glui_side  ->sync_live() ;
  glui_bottom->sync_live() ;

  return quit ;
}
//_____________________________________________________________________________
