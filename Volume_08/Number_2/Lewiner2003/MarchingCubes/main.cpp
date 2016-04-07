//------------------------------------------------
// MarchingCubes
//------------------------------------------------
//
// MarchingCubes Command Line interface
// Version 0.2 - 12/08/2002
//
// Thomas Lewiner thomas.lewiner@polytechnique.org
// Math Dept, PUC-Rio
//
//________________________________________________


#include <stdio.h>
#include "MarchingCubes.h"

void compute_data( MarchingCubes &mc ) ;

//_____________________________________________________________________________
// main function
int main (int argc, char **argv)
//-----------------------------------------------------------------------------
{
  MarchingCubes mc ;
  mc.set_resolution( 60,60,60 ) ;

  mc.init_all() ;
  compute_data( mc ) ;
  mc.run() ;
  mc.clean_temps() ;

  mc.writePLY("test.ply") ;
  mc.clean_all() ;

  return 0 ;
}
//_____________________________________________________________________________




//_____________________________________________________________________________
// Compute data
void compute_data( MarchingCubes &mc )
//-----------------------------------------------------------------------------
{
  float x,y,z      ;
  float sx,sy,sz   ;
  float tx,ty,tz   ;

  float r,R ;
  r = 1.85f ;
  R = 4 ;

  sx     = (float) mc.size_x() / 16 ;
  sy     = (float) mc.size_y() / 16 ;
  sz     = (float) mc.size_z() / 16 ;
  tx     = (float) mc.size_x() / (2*sx) ;
  ty     = (float) mc.size_y() / (2*sy) + 1.5f ;
  tz     = (float) mc.size_z() / (2*sz) ;

  for( int k = 0 ; k < mc.size_z() ; k++ )
  {
    z = ( (float) k ) / sz  - tz ;

    for( int j = 0 ; j < mc.size_y() ; j++ )
    {
      y = ( (float) j ) / sy  - ty ;

      for( int i = 0 ; i < mc.size_x() ; i++ )
      {
        x = ( (float) i ) / sx - tx ;
        mc.set_data( (float) (
          // cushin
          //            z*z*x*x - z*z*z*z - 2*z*x*x + 2*z*z*z + x*x - z*z - (x*x - z)*(x*x - z) - y*y*y*y - 2*x*x*y*y - y*y*z*z + 2*y*y*z + y*y ,
          // sphere
          //            ( (x-2)*(x-2) + (y-2)*(y-2) + (z-2)*(z-2) - 1 ) * ( (x+2)*(x+2) + (y-2)*(y-2) + (z-2)*(z-2) - 1 ) * ( (x-2)*(x-2) + (y+2)*(y+2) + (z-2)*(z-2) - 1 )) ,
          //  plane
          //            x+y+z -3,
          // cassini
          //            (x*x + y*y + z*z + 0.45f*0.45f)*(x*x + y*y + z*z + 0.45f*0.45f) - 16*0.45f*0.45f*(x*x + z*z) - 0.5f*0.5f ,
          // blooby
          //           x*x*x*x - 5*x*x+ y*y*y*y - 5*y*y + z*z*z*z - 5*z*z + 11.8 ),
          //  chair
          //            x*x+y*y+z*z-0.95f*25)*(x*x+y*y+z*z-0.95f*25)-0.8f*((z-5)*(z-5)-2*x*x)*((z+5)*(z+5)-2*y*y ,
          // cyclide
          //            ( x*x + y*y + z*z + b*b - d*d ) * ( x*x + y*y + z*z + b*b - d*d ) - 4 * ( ( a*x - c*d ) * ( a*x - c*d ) + b*b * y*y ),
          // 2 torus
                      ( ( x*x + y*y + z*z + R*R - r*r ) * ( x*x + y*y + z*z + R*R - r*r ) - 4 * R*R * ( x*x + y*y ) ) *
                      ( ( x*x + (y+R)*(y+R) + z*z + R*R - r*r ) * ( x*x + (y+R)*(y+R) + z*z + R*R - r*r ) - 4 * R*R * ( (y+R)*(y+R) + z*z ) ) ) ,
          // mc case
          // - 26.5298*(1-x)*(1-y)*(1-z) + 81.9199*x*(1-y)*(1-z) - 100.68*x*y*(1-z) + 3.5498*(1-x)*y*(1-z)
          // + 24.1201*(1-x)*(1-y)*  z   - 74.4702*x*(1-y)*  z   + 91.5298*x*y*  z  - 3.22998*(1-x)*y*  z  ),
          // Drip
          //          x*x + y*y - 0.5*( 0.995*z*z + 0.005 - z*z*z ) +0.0025 ),  // -0.0754+0.01, -0.0025 + 0.01, grid 40^3, [-1.5,1.5]

          i,j,k ) ;
      }
    }
  }
}
//_____________________________________________________________________________



