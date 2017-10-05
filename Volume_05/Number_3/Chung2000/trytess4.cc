#include <fstream.h>
#include <strstream.h>
#include <my_io.h>
#include <Manifold/manifold.h>

#include "tessellate.h"
#include "tessellate.cc"
#include <stdlib.h>


static bspl_fit* one;
static bspl_fit* two;
static bspl_fit* three;
static trimesh themesh;



void my_begpts( int vi, int xi, int n, int vj,
			       Real& u, Real& v) {

  if (vi%13==12) return;

  Real u0, v0, u1, v1;

  if (vi%13<8) {
    vect3d va = themesh.point(vi);
    u0 = va[0]/2+1;
    v0 = va[1]/2+1;
    if (va[2]>0.5)
      u0 = 4. - u0;

    if (vj%13==12) {
      u1 = u0*2-2;
      v1 = v0*2-2;
    }
    else if ((vi/13) != (vj/13)) {
      u1 = v1 = 2.;
      if ((vi%13)==(vj%13)) {
	u1 -= (2.-v0)/2;
	v1 -= (u0-2.)/2;
      }
      else {
	u1 += (2.-v0)/2;
	v1 += (u0-2.)/2;
      }
    }
    else {
      vect3d vb = themesh.point(vj);
      u1 = vb[0]/2+1;
      v1 = vb[1]/2+1;
      if (vb[2]>0.5)
	u1 = 4. - u1;
    }
  }
  else {
    int i = vi%13-8;
    u0 = (vi>12) ? 2.3 : 2.7;

    if (i>1) i = 5-i;
    v0 = 2 + i;

    int j = vj%13;
    if (j<8) {
      switch(i) {
      case 0:
	v1 = (j==0) ? 2 : ((j==3) ? 1.5 : 2.5);
	break;
      case 1:
	v1 = (j==2) ? 3 : ((j==1) ? 2.5 : 3.5);
	break;
      case 2:
	v1 = (j==7) ? 4 : ((j==4) ? 3.5 : 4.5);
	break;
      default:
	v1 = (j==5) ? 5 : ((j==6) ? 4.5 : 5.5);
      }
      Real x = ((j==1) || (j==3) || (j==4) || (j==6)) ? 0.4 : 1.0;
      u1 = (vj>12) ? (2-x) : (3+x);
    }
    else {
      j -= 8;
      if (j>1) j = 5-j;
      u1 = (vj>12) ? 2.3 : 2.7;
      v1 = 2 + j;
      if ((i==0) && (j==3))
	v1 -= 4;
      if ((i==3) && (j==0))
	v1 += 4;
    }
  }


  u = u1 - u0;
  v = v1 - v0;
  //cerr << vi << ' '  << vj << ' ' << u << ' ' << v << endl;
}


int my_inidat( mf_patchlet *mfp, manifold *man,
			      int vi, int vn ) {
  if (vi%13==12) return 0;

  Real u0, v0, u1, v1;

  if (vi%13<8) {
    vect3d va = themesh.point(vi);
    u0 = va[0]/2+1;
    v0 = va[1]/2+1;
    if (va[2]>0.5)
      u0 = 4. - u0;

    mfp->set_patch((vi<13) ? one : two);
  }
  else {
    int i = vi%13-8;
    u0 = (vi>12) ? 2.3 : 2.7;
    v0 = 2 + ((i>1) ? (5-i) : i);

    mfp->set_patch( three );
  }

  Real map[2][2];
  map[0][0] = map[1][1] = 1.;
  map[1][0] = map[0][1] = 0.;
  mfp->set_trans( map );
  mfp->set_cen(u0,v0);

  return 0;
}

int my_iniwts( mf_patchlet *mfp, manifold *man, int vi ) {
  const int wres = 15;
  const int wsz = wres*2 + 1;
  Real m[2][2];

  bspl_fit *p = new bspl_fit(wsz,wsz,1);
  for (int i = 0; i<wsz; i++)
    for (int j = 0; j<wsz; j++) {
      Real du = j - wres, dv = i - wres;
      p->item(i,j) = 4.0 / (4.+du*du+dv*dv);
    }

  m[0][0] = m[1][1] = wres - .5;
  m[0][1] = m[1][0] = 0.;
  mfp->set_patch(p);
  mfp->set_trans(m);
  mfp->set_cen((Real)wres, (Real)wres);
  return 0;
}






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
  //cerr << x[0] << ' ' << x[1] << endl;
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
  if (length(va)>1e-9)
    va.norm();

  cout << "pp 3\n";
  for (int i = 0; i<3; i++)
    cout << v[i][0] << ' ' << v[i][1] << ' ' << v[i][2] << va <<endl;
}

int main( int argc, char* argv[] ) {
  MyTess the_surf;
  char *fmesh = "/homes/ajc/Grafix/OOC/Manifold/blend.tri";

  Real thres = 0.02;
  Real bias = 0.2;
  Real skew = 0.001;

  for(int argi = 1; argi<argc; argi++)
    if (argv[argi][0] == '-')
      switch (argv[argi][1]) {
      case 't':
	thres = atof(argv[++argi]);
	break;

      case 'b':
	bias = atof(argv[++argi]);
	break;

      case 's':
	skew = atof( argv[++argi] );
	the_surf.set_skewness( skew );
      }

  the_surf.set_threshold( thres );
  the_surf.set_splitbias( bias );

  TriTile<Real> root;
  Real v[2];

  one = new bspl_fit(5,5,3);
  two = new bspl_fit(5,5,3);
  three = new bspl_fit(6,8,3);

  for( int i = 0; i<5; i++)
    for( int j = 0; j<5; j++) {
      one->item(i,j,0) = i-2;
      one->item(i,j,1) = (j==2) ? 0 : ((j<2) ? -1 : 1);
      one->item(i,j,2) = ((j==0) || (j==4)) ? -2 : 0;


      two->item(i,j,0) = 2-i;
      two->item(i,j,1) = j-2;
      two->item(i,j,2) = 1.0;
    }

  for( int i = 0; i<6; i++)
    for( int j = 0; j<8; j++) {
      Real d = (i-2.5);
      d *= d*0.5;
      d += .2;
      if (i==0) d = 4;
      if (i==5) d = 7;
      three->item(i,j,0) = sin((4.5-j)*M_PI/2) * d;
      three->item(i,j,1) = cos((4.5-j)*M_PI/2) * d;
      three->item(i,j,2) = (i<=1) ? 1 : ((i>=4) ? 0.0 : ((2.5-i)*0.8+.5) );
    }

  themesh.load(fmesh);

  manifold theman( &themesh, my_begpts );


  theman.reset_vtx_domains( 1, 1);
  theman.set_vlen(0,3);
  theman.init_blends(my_iniwts,my_inidat);

  the_man = &theman;
  for( int i = 0; i<themesh.num_triangles(); i++ ) {
    if (i%20>=12) continue;
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


