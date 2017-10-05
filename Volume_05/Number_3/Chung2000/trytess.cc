// In the original archive this file is actually a symbolic link to another file, in this case trytess2.cc
#include "tessellate.h"
#include "tessellate.cc"
#include <stdlib.h>




class MyTess : public TessTrim<float> {
  void _render( TriTile<float>& );
public:
  bool perim;

  float split_edge( float[2], float [2] );
  bool split_cen( float[3][2] );
  void render_final( TriTile<float>& );

  void trim_curve_fn( int, int, float, float[2] );
  void render_trimmed( TriTile<float>& );
};

static float func( float x[2] ) {
  float a = 0.5+20.0/(40.0 + 12*(x[0]+3)*(x[0]+3));
  float b = 1.-20.0/(40.0 + 16*(x[0]-2)*(x[0]-2));
  return a*(2.0-x[1]) + b*x[1];
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


void MyTess::_render( TriTile<float>& tile ) {
  cout << "pp 3\n";
  for (int i = 0; i<3; i++) {
    int j0 = (i+1)%3;
    if ((tile[i][0]==tile[j0][0]) && (tile[i][1]==tile[j0][1])) {
      cerr << "degenerate tile\n";
    }
    cout << tile[i][0] << ' ' << tile[i][1] << ' ' << func(tile[i]) << endl;
  }
}

void MyTess::render_final( TriTile<float>& tile ) {
 if (!perim) _render( tile );
}

void MyTess::render_trimmed( TriTile<float>& tile ) {
  if (perim) _render( tile );
}

static float tcurve[7][3][2] = {{{0.,4.},{0.,.5},{4.,0.}},
			  {{4.,0.},{0.,-.5},{0.,-4.}},
			  {{0.,-4.},{-4.,0.},{0.,4.}},
			  {{-6.,-6.},{0.,-8.},{4.,-4.}},
			  {{4.,-4.},{8.,0.},{4.,4.}},
			  {{4.,4.},{0.,8.},{-6.,6.}},
			  {{-6.,6.},{-2.,0.},{-6.,-6.}}};

void MyTess::trim_curve_fn( int loop, int seg, float u, float v[2] ) {
  float b2 = u*u;
  float b1 = (1.-u);
  float b0 = b1*b1;
  b1 *= 2*u;

  int i = loop*3+seg;
  v[0] = b0*tcurve[i][0][0] + b1*tcurve[i][1][0] + b2*tcurve[i][2][0];
  v[1] = b0*tcurve[i][0][1] + b1*tcurve[i][1][1] + b2*tcurve[i][2][1];
}

int main( int argc, char* argv[] ) {
  MyTess the_surf;

  the_surf.set_num_loops(2);
  the_surf.set_num_segs(0,3);
  the_surf.set_num_segs(1,4);

  if (argc>1)
    the_surf.set_threshold(atof(argv[1]));

  the_surf.perim = (argc>2);

  the_surf.set_splitbias(0.6);

  TriTile<float> root;
  float v[2];

  v[0] = -12.;
  v[1] = -12.;
  root.set_vtx(0,v);

  v[0] = 12.0;
  v[1] = -12.;
  root.set_vtx(1,v);

  v[0] = -12.;
  v[1] = 12.;
  root.set_vtx(2,v);

  the_surf.tessellate( root );

  v[0] = -12.;
  v[1] = 12.;
  root.set_vtx(0,v);

  v[0] = 12.;
  v[1] = -12.;
  root.set_vtx(1,v);

  v[0] = 12.;
  v[1] = 12.;
  root.set_vtx(2,v);

  the_surf.tessellate( root );

}
