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
  vect3d v[3];
  for (int i = 0; i<3; i++) {
    Real p[3];
    Real x[2];
    x[0] = tile[i][0];
    x[1] = tile[i][1];
    func(x,p);
    v[i] = Vec3( p[0], p[1], p[2] );
  }
  vect3d va = v[1] - v[0];
  vect3d vb = v[2] - v[0];
  va = va^vb;
  va.norm();

  cout << "pp 3\n";
  for (int i = 0; i<3; i++)
    cout << v[i][0] << ' ' << v[i][1] << ' ' << v[i][2] << va <<endl;
}

int main( int argc, char* argv[] ) {
  MyTess the_surf;
  char *fmesh = "../Occlusion/lit.tri";

  Real thres = 0.02;
  Real bias = 0.2;

  Real d;
  for(int argi = 1; argi<argc; argi++)
    if (argv[argi][0] == '-')
      switch (argv[argi][1]) {
      case 'f':
	fmesh = argv[++argi];
	break;

      case 't':
	thres = atof(argv[++argi]);
	break;

      case 'b':
	bias = atof(argv[++argi]);
	break;

      case 'd':
	d = atof(argv[++argi]);
	set_dischack(d);
	break;

      case 's':
	d = atof(argv[++argi]);
	the_surf.set_skewness( d );
	break;
      }

  the_surf.set_threshold( thres );
  the_surf.set_splitbias( bias );

  TriTile<Real> root;
  Real v[2];

  trimesh themesh;
  themesh.load(fmesh);

  manifold aman( &themesh );

  aman.reset_vtx_domains( 1, 1);
  aman.set_vlen(0,3);
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
