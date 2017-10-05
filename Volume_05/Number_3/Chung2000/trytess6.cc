#include "tessellate.h"
#include "tessellate.cc"
#include <stdlib.h>
#include <math.h>

template void Tessellate<float>::tessellate(TriTile<float> &);


class MyTess : public Tessellate<float> {
public:
  float split_edge( float[2], float [2] );
  bool split_cen( float[3][2] );
  void render_final( TriTile<float>& );
};

static float fun0( float x ) { x = 3-x; return 10-(x+7)*(x+7)/17.0;}
//static float fun0( float x ) { return x;}
static float fun1( float y ) { return 18.0-200.0/(y+15.0);}

float MyTess::split_edge( float a[2], float b[2] ) {
  const int steps = 10;
  float d = 0.0;
  for (int i = 0; i<steps; i++) {
    float d0 = fun0((a[0]*(steps-i)+b[0]*i)/steps)
      - fun0((a[0]*(steps-i-1)+b[0]*(i+1))/steps);
    float d1 = fun1((a[1]*(steps-i)+b[1]*i)/steps)
      - fun1((a[1]*(steps-i-1)+b[1]*(i+1))/steps);
    d += sqrt( d0*d0 + d1*d1);
  }
  return d;
}

bool MyTess::split_cen( float v[3][2] ) {
  return false;
}

void MyTess::render_final( TriTile<float>& tile ) {
  cout << "pp 3\n";
  for (int i = 0; i<3; i++) {
    int j0 = (i+1)%3;
    if ((tile[i][0]==tile[j0][0]) && (tile[i][1]==tile[j0][1])) {
      cerr << "degenerate tile\n";
    }
    cout << fun0(tile[i][0]) << ' ' << fun1(tile[i][1]) << ' ' << 0.0 << endl;
  }
}

int main( int argc, char* argv[] ) {
  MyTess the_surf;

  if (argc>1) {
    the_surf.set_threshold(atof(argv[1]));
    if (argc>2) {
      the_surf.set_splitbias(atof(argv[2]));
    }
  }


  TriTile<float> root;
  float v[2];

  v[0] = -7.;
  v[1] = -7.;
  root.set_vtx(0,v);

  v[0] = 10.0;
  v[1] = -7.;
  root.set_vtx(1,v);

  v[0] = -7.;
  v[1] = 10.0;
  root.set_vtx(2,v);

  the_surf.tessellate( root );

  v[0] = 10.;
  v[1] = -7.;
  root.set_vtx(0,v);

  v[0] = 10.;
  v[1] = 10.;
  root.set_vtx(1,v);

  v[0] = -7.;
  v[1] = 10.;
  root.set_vtx(2,v);

  the_surf.tessellate( root );

}
