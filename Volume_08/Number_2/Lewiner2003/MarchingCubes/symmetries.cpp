//------------------------------------------------
// Symmetries
//------------------------------------------------
//
// Computes the symmetries from the lookup table
// Version 0.1 - 29/12/2006
//
// Thomas Lewiner thomas.lewiner@polytechnique.org
// Math Dept, PUC-Rio
//
//________________________________________________


#include <stdio.h>
#include "LookUpTable.h"

/**____________________________________________________________________________
 * vertex definition
 *         7 ________ 6 
 *         /|       /|  
 *       /  |     /  |  
 *   4 /_______ /    |  
 *    |     |  |5    |  
 *    |    3|__|_____|2 
 *    |    /   |    /   
 *    |  /     |  /     
 *    |/_______|/       
 *   0          1       
 *---------------------------------------------------------------------------*/

static const char vertex[8][3] = {
  /* vertex 0 */ { -1, -1, -1 },
  /* vertex 1 */ { +1, -1, -1 },
  /* vertex 2 */ { +1, +1, -1 },
  /* vertex 3 */ { -1, +1, -1 },
  /* vertex 4 */ { -1, -1, +1 },
  /* vertex 5 */ { +1, -1, +1 },
  /* vertex 6 */ { +1, +1, +1 },
  /* vertex 7 */ { -1, +1, +1 }
};
//_____________________________________________________________________________



/**____________________________________________________________________________
 * edge definition
 *           _____6__  
 *        7/|       /| 
 *       /  |     /5 | 
 *     /__4____ /    10
 *    |    11  |     | 
 *    |     |__|__2__| 
 *    8   3/   9    /  
 *    |  /     |  /1   
 *    |/___0___|/      
 *---------------------------------------------------------------------------*/

static const char edge[12][2] = {
  /* edge  0 */ { 0, 1 },
  /* edge  1 */ { 1, 2 },
  /* edge  2 */ { 2, 3 },
  /* edge  3 */ { 3, 0 },
  /* edge  4 */ { 4, 5 },
  /* edge  5 */ { 5, 6 },
  /* edge  6 */ { 6, 7 },
  /* edge  7 */ { 7, 4 },
  /* edge  8 */ { 0, 4 },
  /* edge  9 */ { 1, 5 },
  /* edge 10 */ { 2, 6 },
  /* edge 11 */ { 3, 7 }
};
//_____________________________________________________________________________




/**____________________________________________________________________________
 * face definition
 *         ________
 *       /|       /|
 *     /  6     /  |
 *   /_______3/    |
 *  |     |  |   2 |
 *  | 4   |__|_____|
 *  |    /   |    /
 *  |  /     5  /
 *  |/_1_____|/
 *  
 *---------------------------------------------------------------------------*/

static const char face[7][4] = {
  /* face 0 */ { 0, 0, 0, 0 }, /* invalid */
  /* face 1 */ { 0, 1, 5, 4 },
  /* face 2 */ { 1, 2, 6, 5 },
  /* face 3 */ { 2, 3, 7, 6 },
  /* face 4 */ { 3, 0, 4, 7 },
  /* face 5 */ { 0, 1, 2, 3 },
  /* face 6 */ { 4, 5, 6, 7 }
};
//_____________________________________________________________________________




//_____________________________________________________________________________
// configuration that generated the case

static const char confcase6[48][8] = {
  /* case  6 (  0 ):  67 */   { 1, 1, 0, 0, 0, 0, 1, 0 },
  /* case  6 (  1 ): 131 */   { 1, 1, 0, 0, 0, 0, 0, 1 },
  /* case  6 (  2 ):  21 */   { 1, 0, 1, 0, 1, 0, 0, 0 },
  /* case  6 (  3 ):  69 */   { 1, 0, 1, 0, 0, 0, 1, 0 },
  /* case  6 (  4 ):  41 */   { 1, 0, 0, 1, 0, 1, 0, 0 },
  /* case  6 (  5 ):  73 */   { 1, 0, 0, 1, 0, 0, 1, 0 },
  /* case  6 (  6 ):  81 */   { 1, 0, 0, 0, 1, 0, 1, 0 },
  /* case  6 (  7 ):  97 */   { 1, 0, 0, 0, 0, 1, 1, 0 },
  /* case  6 (  8 ): 193 */   { 1, 0, 0, 0, 0, 0, 1, 1 },
  /* case  6 (  9 ):  22 */   { 0, 1, 1, 0, 1, 0, 0, 0 },
  /* case  6 ( 10 ): 134 */   { 0, 1, 1, 0, 0, 0, 0, 1 },
  /* case  6 ( 11 ):  42 */   { 0, 1, 0, 1, 0, 1, 0, 0 },
  /* case  6 ( 12 ): 138 */   { 0, 1, 0, 1, 0, 0, 0, 1 },
  /* case  6 ( 13 ): 146 */   { 0, 1, 0, 0, 1, 0, 0, 1 },
  /* case  6 ( 14 ): 162 */   { 0, 1, 0, 0, 0, 1, 0, 1 },
  /* case  6 ( 15 ): 194 */   { 0, 1, 0, 0, 0, 0, 1, 1 },
  /* case  6 ( 16 ):  28 */   { 0, 0, 1, 1, 1, 0, 0, 0 },
  /* case  6 ( 17 ):  44 */   { 0, 0, 1, 1, 0, 1, 0, 0 },
  /* case  6 ( 18 ):  52 */   { 0, 0, 1, 0, 1, 1, 0, 0 },
  /* case  6 ( 19 ):  84 */   { 0, 0, 1, 0, 1, 0, 1, 0 },
  /* case  6 ( 20 ): 148 */   { 0, 0, 1, 0, 1, 0, 0, 1 },
  /* case  6 ( 21 ):  56 */   { 0, 0, 0, 1, 1, 1, 0, 0 },
  /* case  6 ( 22 ): 104 */   { 0, 0, 0, 1, 0, 1, 1, 0 },
  /* case  6 ( 23 ): 168 */   { 0, 0, 0, 1, 0, 1, 0, 1 },
  /* case  6 ( 24 ):  87 */   { 1, 1, 1, 0, 1, 0, 1, 0 },
  /* case  6 ( 25 ): 151 */   { 1, 1, 1, 0, 1, 0, 0, 1 },
  /* case  6 ( 26 ): 199 */   { 1, 1, 1, 0, 0, 0, 1, 1 },
  /* case  6 ( 27 ): 107 */   { 1, 1, 0, 1, 0, 1, 1, 0 },
  /* case  6 ( 28 ): 171 */   { 1, 1, 0, 1, 0, 1, 0, 1 },
  /* case  6 ( 29 ): 203 */   { 1, 1, 0, 1, 0, 0, 1, 1 },
  /* case  6 ( 30 ): 211 */   { 1, 1, 0, 0, 1, 0, 1, 1 },
  /* case  6 ( 31 ): 227 */   { 1, 1, 0, 0, 0, 1, 1, 1 },
  /* case  6 ( 32 ):  61 */   { 1, 0, 1, 1, 1, 1, 0, 0 },
  /* case  6 ( 33 ):  93 */   { 1, 0, 1, 1, 1, 0, 1, 0 },
  /* case  6 ( 34 ): 109 */   { 1, 0, 1, 1, 0, 1, 1, 0 },
  /* case  6 ( 35 ): 117 */   { 1, 0, 1, 0, 1, 1, 1, 0 },
  /* case  6 ( 36 ): 213 */   { 1, 0, 1, 0, 1, 0, 1, 1 },
  /* case  6 ( 37 ): 121 */   { 1, 0, 0, 1, 1, 1, 1, 0 },
  /* case  6 ( 38 ): 233 */   { 1, 0, 0, 1, 0, 1, 1, 1 },
  /* case  6 ( 39 ):  62 */   { 0, 1, 1, 1, 1, 1, 0, 0 },
  /* case  6 ( 40 ): 158 */   { 0, 1, 1, 1, 1, 0, 0, 1 },
  /* case  6 ( 41 ): 174 */   { 0, 1, 1, 1, 0, 1, 0, 1 },
  /* case  6 ( 42 ): 182 */   { 0, 1, 1, 0, 1, 1, 0, 1 },
  /* case  6 ( 43 ): 214 */   { 0, 1, 1, 0, 1, 0, 1, 1 },
  /* case  6 ( 44 ): 186 */   { 0, 1, 0, 1, 1, 1, 0, 1 },
  /* case  6 ( 45 ): 234 */   { 0, 1, 0, 1, 0, 1, 1, 1 },
  /* case  6 ( 46 ): 124 */   { 0, 0, 1, 1, 1, 1, 1, 0 },
  /* case  6 ( 47 ): 188 */   { 0, 0, 1, 1, 1, 1, 0, 1 }
};
//_____________________________________________________________________________





//_____________________________________________________________________________
//
int main (int argc, char **argv)
//-----------------------------------------------------------------------------
{
  //---------------------------------------------------------------------------
  // generate the cube symmetry group D8
  int n = 0 ;
  int  sym[48][3][3] ;

  for( int e0 = 0 ; e0 < 3 ; ++e0 )
  { // pick up one line entry for the third line
    for( int i0 = -1 ; i0 <= 1 ; i0 += 2 )
    { // choose the entry sign

      for( int e1 = 0 ; e1 < 3 ; ++e1 )
      { // pick up one line entry for the first line
        if( e1 == e0 ) continue ;

        for( int i1 = -1 ; i1 <= 1 ; i1 += 2 )
        { // choose the entry sign

          for( int e2 = 0 ; e2 < 3 ; ++e2 )
          { // pick up one line entry for the second line
            if( e2 == e0 || e2 == e1 ) continue ;

            for( int i2 = -1 ; i2 <= 1 ; i2 += 2 )
            { // choose the entry sign
              sym[n][0][e0] = i0 ;
              sym[n][0][(e0+1)%3] = 0 ;
              sym[n][0][(e0+2)%3] = 0 ;

              sym[n][1][e1] = i1 ;
              sym[n][1][(e1+1)%3] = 0 ;
              sym[n][1][(e1+2)%3] = 0 ;
              sym[n][2][e2] = i2 ;

              sym[n][2][(e2+1)%3] = 0 ;
              sym[n][2][(e2+2)%3] = 0 ;

              ++n ;
            }
          }
        }
      }
    }
  }
  printf( "generated %d symmetries\n", n ) ;


  //---------------------------------------------------------------------------
  // generate symmetry signs
  bool sym_sign[48] ;
  for( n = 0 ; n < 48 ; ++n )
  {
    int det =
      sym[n][0][0] * sym[n][1][1] * sym[n][2][2] +
      sym[n][0][1] * sym[n][1][2] * sym[n][2][0] +
      sym[n][0][2] * sym[n][1][0] * sym[n][2][1] -
      sym[n][0][2] * sym[n][1][1] * sym[n][2][0] -
      sym[n][0][0] * sym[n][1][2] * sym[n][2][1] -
      sym[n][0][1] * sym[n][1][0] * sym[n][2][2] ;
    sym_sign[n] = det > 0 ;
  }
  printf( "generated symmetry signs\n", n ) ;

  //---------------------------------------------------------------------------
  // generate the vertex correspondence from the symmetry
  char vertex_map[48][8] ;

  for( n = 0 ; n < 48 ; ++n )
  {
    for( char v = 0 ; v < 8 ; ++v )
    {
      vertex_map[n][v] = -1 ;

      // original vertex
      int x  = vertex[v][0] ;
      int y  = vertex[v][1] ;
      int z  = vertex[v][2] ;

      // transformed vertex
      int sx = sym[n][0][0] * x + sym[n][0][1] * y + sym[n][0][2] * z ;
      int sy = sym[n][1][0] * x + sym[n][1][1] * y + sym[n][1][2] * z ;
      int sz = sym[n][2][0] * x + sym[n][2][1] * y + sym[n][2][2] * z ;

      // look for the transformed vertex index
      for( char sv = 0 ; sv < 8 ; ++sv )
      {
        if( vertex[sv][0] == sx && vertex[sv][1] == sy && vertex[sv][2] == sz )
        {
          vertex_map[n][v] = sv ;
          break ;
        }
      }

      if( vertex_map[n][v] < 0 || vertex_map[n][v] > 7 )
        printf( "error at vertex mapping of vertex %d under symmetry %d!\n", v, n ) ;
    }
  }
  printf( "generated vertex map\n" ) ;


  //---------------------------------------------------------------------------
  // generate the edge correspondence from the symmetry
  char edge_map[48][13] ;

  for( n = 0 ; n < 48 ; ++n )
  {
    for( char e = 0 ; e < 12 ; ++e )
    {
      edge_map[n][e] = -1 ;

      // original edge
      char v0 = edge[e][0] ;
      char v1 = edge[e][1] ;

      // transformed edge
      char sv0 = vertex_map[n][v0] ;
      char sv1 = vertex_map[n][v1] ;

      // look for the transformed vertex index
      for( char se = 0 ; se < 12 ; ++se )
      {
        if( ( edge[se][0] == sv0 && edge[se][1] == sv1 ) ||
            ( edge[se][0] == sv1 && edge[se][1] == sv0 ) )
        {
          edge_map[n][e] = se ;
          break ;
        }
      }

      if( edge_map[n][e] < 0 || edge_map[n][e] > 11 )
        printf( "error at edge mapping of edge %d under symmetry %d!\n", e, n ) ;
    }
    edge_map[n][12] = 12 ;
  }
  printf( "generated edge map\n" ) ;


  //---------------------------------------------------------------------------
  // generate the face correspondence from the symmetry
  char face_map[48][7] ;

  for( n = 0 ; n < 48 ; ++n )
  {
    face_map[n][0] = 0 ;
    for( char f = 1 ; f < 7 ; ++f )
    {
      face_map[n][f] = 0 ;

      // original face
      char v0 = face[f][0] ;
      char v1 = face[f][1] ;
      char v2 = face[f][2] ;
      char v3 = face[f][3] ;

      // transformed face
      char sv0 = vertex_map[n][v0] ;
      char sv1 = vertex_map[n][v1] ;
      char sv2 = vertex_map[n][v2] ;
      char sv3 = vertex_map[n][v3] ;

      // look for the transformed vertex index
      for( char sf = 1 ; sf < 7 ; ++sf )
      {
        for( int i1 = 0 ; i1 < 4 ; ++i1 )
        {
          for( int i2 = 0 ; i2 < 4 ; ++i2 )
          {
            if( i2 == i1 ) continue ;
            for( int i3 = 0 ; i3 < 4 ; ++i3 )
            {
              if( i3 == i1 || i3 == i2 ) continue ;
              for( int i4 = 0 ; i4 < 4 ; ++i4 )
              {
                if( i4 == i1 || i4 == i2 || i4 == i3 ) continue ;

                if( face[sf][i1] == sv0 && face[sf][i2] == sv1 && face[sf][i3] == sv2 && face[sf][i4] == sv3 )
                {
                  face_map[n][f] = sf ;
                  break ;
                }
              }
              if( face_map[n][f] != 0 ) break ;
            }
            if( face_map[n][f] != 0 ) break ;
          }
          if( face_map[n][f] != 0 ) break ;
        }
        if( face_map[n][f] != 0 ) break ;
      }

      if( face_map[n][f] < 1 || face_map[n][f] > 6 )
        printf( "error at face mapping of face %d under symmetry %d!\n", f, n ) ;
    }
  }
  printf( "generated face map\n" ) ;



  //---------------------------------------------------------------------------
  //---------------------------------------------------------------------------

  printf( "working on case 6.1.2\n" ) ;


  //---------------------------------------------------------------------------
  // find the configuration symmetry
  int nconfig = 48 ;
#define confcase confcase6
  const char  *refconf  = confcase[0]    ;

  int ntrigs  =  7 ;
#define tilecase tiling6_1_2
  const char  *reftile  = tiling6_1_2[0] ;

  int new_ntrigs  =  9 ;
  static const char newtile[27]  = 
  {  1, 12,  3, 12, 10,  3,  6,  3, 10,  3,  6,  8,  5,  8,  6,  8,  5, 12, 12,  9,  8,  1,  9, 12, 12,  5, 10 } ;

  char *tmatch = new char[ntrigs] ;
  for( int c = 1 ; c < nconfig ; ++c )
  {
    const char *conf = confcase[c] ;
    const char *tile = tilecase[c] ;
    for( n = 0 ; n < 48 ; ++n )
    {
      bool sign = sym_sign[n] ;

      //-----------------------------------------------------------------------
      // check if the symmetry maps the configuration case to the reference configuration
      bool match = true ;
      for( int v = 0 ; v < 8 ; ++v )
      {
        if( conf[ vertex_map[n][v] ] != refconf[v] )
        {
          match = false ;
          break ;
        }
      }

      // try complementary configuration
      if( !match )
      {
        match = true ;
        for( int v = 0 ; v < 8 ; ++v )
        {
          if( conf[ vertex_map[n][v] ] == refconf[v] )
          {
            match = false ;
            break ;
          }
        }
        if( match ) sign = !sign ;
      }
      if( !match ) continue ;


      //-----------------------------------------------------------------------
      // check if the lookup table is coherent with the symmetry
      for( int st = 0 ; st < ntrigs ; ++st ) tmatch[st] = -1 ;  // triangles st of conf matching with triangle tmatch[st] of refcong
      for( int  t = 0 ;  t < ntrigs ; ++ t )
      {
        char v0 = edge_map[n][ reftile[ 3*t+0 ] ] ;
        char v1 = edge_map[n][ reftile[ 3*t+1 ] ] ;
        char v2 = edge_map[n][ reftile[ 3*t+2 ] ] ;

        // look for a triangle of conf to match with
        for( int st = 0 ; st < ntrigs ; ++st )
        {
          if( tmatch[st] != -1 ) continue ;

          char sv0 = tile[ 3*st+0 ] ;
          char sv1 = tile[ 3*st+1 ] ;
          char sv2 = tile[ 3*st+2 ] ;

          // check the triangles with coherent orientations
          if( sign )
          {
            if( ( sv0 == v0 && sv1 == v1 && sv2 == v2 ) ||
                ( sv0 == v1 && sv1 == v2 && sv2 == v0 ) ||
                ( sv0 == v2 && sv1 == v0 && sv2 == v1 ) )
            {
              tmatch[st] = t ;
              break ;
            }
          }
          else
          {
            if( ( sv0 == v0 && sv1 == v2 && sv2 == v1 ) ||
                ( sv0 == v1 && sv1 == v0 && sv2 == v2 ) ||
                ( sv0 == v2 && sv1 == v1 && sv2 == v0 ) )
            {
              tmatch[st] = t ;
              break ;
            }
          }
        }
      }

      // result consolidation
      match = true ;
      for( int st = 0 ; st < ntrigs ; ++st )
      {
        if( tmatch[st] == -1 )
        {
          match = false ;
          break ;
        }
      }
      if( !match ) continue ;


      //-----------------------------------------------------------------------
      // writes a one-to-one line of the lookup table
      printf( "/* %2d */ { ", c ) ;
      for( int t = 0 ; t < new_ntrigs ; ++t )
      {
        if( sign )
        {
          for( int v = 0 ; v < 3 ; ++v )
          {
            printf( "%2d, ", edge_map[n][ newtile[3*t+v] ] ) ;
          }
        }
        else
        {
          for( int v = 2 ; v >= 0 ; --v )
          {
            printf( "%2d, ", edge_map[n][ newtile[3*t+v] ] ) ;
          }
        }
        printf( "  " ) ;
      }
      printf( "},\n" ) ;

    }
  }

  delete [] tmatch ;

  return 0;
}
//_____________________________________________________________________________

