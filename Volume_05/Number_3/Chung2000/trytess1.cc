#include "tessellate.h"
#include "tessellate.cc"
#include <stdlib.h>


template void Tessellate<float>::tessellate(TriTile<float> &);


class MyTess : public Tessellate<float> {
public:
  float split_edge( float[2], float [2] );
  bool split_cen( float[3][2] );
  void render_final( TriTile<float>& );
};

static float func( float x[2] ) {
  //return 0.;
  return 400./(10.0 + x[0]*x[0] + x[1]*x[1]);
}

float MyTess::split_edge( float a[2], float b[2] ) {
  float d = func(a) - func(b);
  float d0 = a[0] - b[0];
  float d1 = a[1] - b[1];
  return d*d + d0*d0 + d1*d1;
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
    cout << tile[i][0] << ' ' << tile[i][1] << ' ' << func(tile[i]) << endl;
  }
}

int main( int argc, char* argv[] ) {
  MyTess the_surf;


  if (argc>1)
    the_surf.set_splitbias(atof(argv[1]));

  //the_surf.set_threshold(0.2);

  TriTile<float> root;
  float v[2];

  v[0] = -8.;
  v[1] = -9.;
  root.set_vtx(0,v);

  v[0] = 10.0;
  v[1] = -7.;
  root.set_vtx(1,v);

  v[0] = -3.;
  v[1] = 5.;
  root.set_vtx(2,v);

  the_surf.tessellate( root );

  v[0] = -3.;
  v[1] = 5.;
  root.set_vtx(0,v);

  v[0] = 10.;
  v[1] = -7.;
  root.set_vtx(1,v);

  v[0] = 10.;
  v[1] = 10.;
  root.set_vtx(2,v);

  the_surf.tessellate( root );

}
