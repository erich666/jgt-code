#include <fstream.h>
#include <strstream.h>
#include <my_io.h>
#include <Manifold/manifold.h>

#include "tessellate.h"
#include "tessellate.cc"
#include <stdlib.h>

extern void set_dischack( Real );

class MyTess : public Tessellate<Real> {
public:
  Real split_edge( Real[2], Real [2] );
  bool split_cen( Real[3][2] );
  void render_final( TriTile<Real>& );
};


static manifold* the_man;
static int facetx;
static void func( Real x[2], Real p[3] ) {
  //return 0.;
  Real t0 = x[0];
  Real t1 = x[1];
  the_man->eval( facetx, 0, 0, t0, t1, p );
}


Real MyTess::split_edge( Real a[2], Real b[2] ) {
  const int n = 10;
  Real td = 0.0;
  for( int i = 0; i<n; i++) {
    Real pa[3], pb[3], a1[2], b1[2];
    a1[0] = lerp(a[0],b[0],i*1.0/n);
    a1[1] = lerp(a[1],b[1],i*1.0/n);
    b1[0] = lerp(a[0],b[0],(i+1)*1.0/n);
    b1[1] = lerp(a[1],b[1],(i+1)*1.0/n);
    func(a1, pa);
    func(b1, pb);
    Real d0 = pa[0] - pb[0];
    Real d1 = pa[1] - pb[1];
    Real d2 = pa[2] - pb[2];
    td += d0*d0 + d1*d1 + d2*d2;
  }
  return td;
}

bool MyTess::split_cen( Real v[3][2] ) {
  return false;
}

void MyTess::render_final( TriTile<Real>& tile ) {
  cout << "pp 3\n";
  for (int i = 0; i<3; i++) {
    Real p[4];
    Real x[2];
    x[0] = tile[i][0];
    x[1] = tile[i][1];
    func(x,p);

    the_man->eval( facetx, 0, 1, x[0], x[1], p+3 );


    cout << p[0] << ' ' << p[1] << ' ' << p[2] << ' ' << p[3] << ' ' << p[3] << ' ' << p[3] << endl;
  }
}

int main( int argc, char* argv[] ) {
  MyTess the_surf;
  char* fmesh = "../Occlusion/recv2.tri";
  char* sampfile = "../Raycast/recvsamp.txt";

  Real thres = 0.02;
  Real bias = 0.2;

  int num_iter = 4;

  Real d;
  for(int argi = 1; argi<argc; argi++)
    if (argv[argi][0] == '-')
      switch (argv[argi][1]) {
      case 'f':
	fmesh = argv[++argi];
	break;

      case 'T':
	thres = atof(argv[++argi]);
	break;

      case 'B':
	bias = atof(argv[++argi]);
	break;

      case 'D':
	d = atof(argv[++argi]);
	set_dischack(d);
	break;

      case 'S':
	d = atof(argv[++argi]);
	the_surf.set_skewness( d );
	break;

      case 's':
	sampfile = argv[++argi];
	break;

      case 'n':
	num_iter = atoi(argv[++argi]);
	break;
      }

  the_surf.set_threshold( thres );
  the_surf.set_splitbias( bias );

  TriTile<Real> root;
  Real v[2];

  trimesh themesh;
  themesh.load(fmesh);

  manifold aman( &themesh );

  aman.reset_vtx_domains( 1, 2);
  aman.set_vlen(0,3);
  aman.set_vlen(1,1);
  aman.init_blends(0,0);

  aman.fit_init(0);
  for (int i = 0; i<themesh.num_triangles(); i++) {
    Real pt[3];
    vect3d v0 = themesh.point(i,0);
    vect3d v1 = themesh.point(i,1);
    vect3d v2 = themesh.point(i,2);

    vect3d vp = 0.25*(2*v0+v1+v2);
    pt[0] = vp[0];
    pt[1] = vp[1];
    pt[2] = vp[2];
    aman.fit_point( i, 0, 0.25, 0.25, pt );

    vp = 0.25*(v0+2*v1+v2);
    pt[0] = vp[0];
    pt[1] = vp[1];
    pt[2] = vp[2];
    aman.fit_point( i, 0, 0.5, 0.25, pt );

    vp = 0.25*(v0+v1+2*v2);
    pt[0] = vp[0];
    pt[1] = vp[1];
    pt[2] = vp[2];
    aman.fit_point( i, 0, 0.25, 0.5, pt );

  }
  aman.fit_end(0);


  for(;;) {
    aman.fit_init(1);
    ifstream fin(sampfile);
    if (!fin) {
      cerr << "Failed to open file \"" << sampfile << "\" for reading\n";
      exit(-1);
    }

    for(;;) {
      int ti;
      Real c1, c0, samp[3];
      fin >> ti >> c1 >> c0 >> samp[0] >> samp[1] >> samp[2];
      if (fin.eof()) break;
      
      aman.fit_point( ti, 1, c1, c0, samp );
    }
    fin.close();
    aman.fit_end(1);
    num_iter--;
    if (num_iter<=0) break;
    aman.fit_refine_exbord(1);
  }


  the_man = &aman;
  for( int i = 0; i<themesh.num_triangles(); i++ ) {
    facetx = i;

    v[0] = 0.;
    v[1] = 0.;
    root.set_vtx(0,v);

    v[0] = 1.;
    v[1] = 0.;
    root.set_vtx(1,v);

    v[0] = 0.;
    v[1] = 1.;
    root.set_vtx(2,v);

    the_surf.tessellate( root );
  }


}
