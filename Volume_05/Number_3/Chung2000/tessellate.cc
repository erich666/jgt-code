#include <list>
#include <cmath>
#include "tessellate.h"

/************************************************************
  Class variables
 ************************************************************/
template < class T >
T Tessellate<T>::edge_thresh = 1.0;
template < class T >
T Tessellate<T>::split_bias = 0.7;
template < class T >
T Tessellate<T>::skew_thresh = 0.0001;


/************************************************************
 TriTilef methods
 ************************************************************/
template < class T >
void TriTile<T>::dice( Tessellate<T>* tess ) {
  //pick a central point
  T cv[2];
  cv[0] = (vt[0][0] + vt[1][0] + vt[2][0])/3.0;
  cv[1] = (vt[0][1] + vt[1][1] + vt[2][1])/3.0;

  for( int ed = 0; ed<3; ed++) {
    list<T> divs;
    int e1 = (ed+1)%3;

    T d[2];
    d[0] = vt[e1][0] - vt[ed][0];
    d[1] = vt[e1][1] - vt[ed][1];

    divs.push_front( 1.0 );
    T beg = 0.0;
    while( !divs.empty() ) {
      T a[2],b[2];
      T end = divs.front();
      a[0] = vt[ed][0] + d[0]*beg;
      a[1] = vt[ed][1] + d[1]*beg;
      b[0] = vt[ed][0] + d[0]*end;
      b[1] = vt[ed][1] + d[1]*end;
      if (tess->split_edge( a, b) > tess->edge_thresh) {
	//split edge
	divs.push_front( 0.5*(beg+end) );
      }
      else {
	//render it
	TriTile<T> slice;
	slice.set_vtx(0,cv);
	slice.set_vtx(1,a);
	slice.set_vtx(2,b);
	tess->trim_final( &slice );
	divs.pop_front();
	beg = end;
      }
    }//while
  }//for
}

template < class T >
int TriTile<T>::refine( Tessellate<T>* tess, TriTile<T>* res[4] ) {
  if (tess->tile_culled( this )) return 0;

  /***** Measure facet degeneracy ****/

  //area of triangle
  T area = vt[0][0]*vt[1][1] - vt[1][0]*vt[0][1]
    + vt[1][0]*vt[2][1] - vt[2][0]*vt[1][1]
    + vt[2][0]*vt[0][1] - vt[0][0]*vt[2][1];
  if (area<0) area = -area;

  //calc sum of squares of edge lengths
  T a[2], b[2];
  a[0] = vt[0][0] - vt[1][0];
  a[1] = vt[0][1] - vt[1][1];
  T max_ed = a[0]*a[0] + a[1]*a[1];
  a[0] = vt[1][0] - vt[2][0];
  a[1] = vt[1][1] - vt[2][1];
  max_ed += a[0]*a[0] + a[1]*a[1];
  a[0] = vt[2][0] - vt[0][0];
  a[1] = vt[2][1] - vt[0][1];
  max_ed += a[0]*a[0] + a[1]*a[1];

  area /= max_ed;
  if (area <= tess->skew_thresh) {
    dice(tess);
    return 0;
  }


  //split edges
  T eds[3];

  max_ed = 0.;
  for (int i = 0; i<3; i++) {
    int j0 = (i+1)%3;
    int j1 = (i+2)%3;
    a[0] = vt[j0][0];
    a[1] = vt[j0][1];
    b[0] = vt[j1][0];
    b[1] = vt[j1][1];
    eds[i] = tess->split_edge( a, b );
    if (eds[i]>max_ed) max_ed = eds[i];
    //cerr << i << ' ' << eds[i] << endl;
  }

  max_ed *= tess->split_bias;

  int m[3];
  T mv[3][2];
  int co = 0;
  for (int i = 0; i<3; i++) {
    int j0 = (i+1)%3;
    int j1 = (i+2)%3;

    if ((eds[i]>tess->edge_thresh) && (eds[i] >= max_ed)) {
      co++;
      mv[i][0] = 0.5*(vt[j0][0] + vt[j1][0]);
      mv[i][1] = 0.5*(vt[j0][1] + vt[j1][1]);
      m[i] = i - 3;
    }
    else {
      //move midpt to vertex closer to center
      if (eds[j0]>eds[j1]) {
	mv[i][0] = vt[j0][0];
	mv[i][1] = vt[j0][1];
	m[i] = j0;
      }
      else {
	mv[i][0] = vt[j1][0];
	mv[i][1] = vt[j1][1];
	m[i] = j1;
      }
    }
  }


  if (co) {
    //add one for center tile
    co++;

    //corner tiles
    int j = co;
    for (int i = 0; i<3; i++) {
      int j0 = (i+1)%3;
      int j1 = (i+2)%3;
      if ((m[j1]!=i) && (m[j0]!=i)) {
	res[--j] = new TriTile<T>();
	res[j]->set_vtx( 0, vt[i] );
	res[j]->set_vtx( 1, mv[j1] );
	res[j]->set_vtx( 2, mv[j0] );
      }
    }

    //center tile
    if (j) {
      if ((m[0]==m[1]) || (m[1]==m[2]) || (m[2]==m[0])) {
	cerr << "degenerate center tile\n";
	exit(-1);
      }
      res[0] = new TriTile<T>();
      for( int i = 0; i<3; i++)
	res[0]->set_vtx( i, mv[i] );
    }
    return co;
  }
  else if (tess->split_cen( mv )) {    //no edges split; add vertex to center?
    a[0] = (1./3.)*(vt[0][0] + vt[1][0] + vt[2][0]);
    a[1] = (1./3.)*(vt[0][1] + vt[1][1] + vt[2][1]);

    for( int i = 0; i<3; i++) {
      res[i] = new TriTile<T>();
      res[i]->set_vtx( 0, vt[i] );
      res[i]->set_vtx( 1, vt[(i+1)%2] );
      res[i]->set_vtx( 2, a );
    }
    return 3;
  }

  tess->trim_final(this);
  return 0;
}


template < class T >
class TTPtr {
  TriTile<T> *tp;
public:
  TTPtr( TriTile<T>* x =0) : tp(x) {}
  TTPtr( const TTPtr<T>& x) : tp(x.tp) {};
  TTPtr& operator= ( const TTPtr<T>& x) { tp = x.tp; }
  TriTile<T>* p() { return tp; }
  const TriTile<T>* p() const { return tp; }
};


/************************************************************
 Tessellation methods
 ************************************************************/
template < class T >
void Tessellate<T>::tessellate( TriTile<T>& root ) {
  list< TTPtr<T> > work;

  //make dynamic copy (deleted during refinement)
  TriTile<T>* begin = new TriTile<T>(root);
  work.push_front(TTPtr<T>(begin));

  while( !work.empty() ) {
    TTPtr<T>& top = work.front();
    TriTile<T>* curr = top.p();
    work.pop_front();

    TriTile<T>* piece[4];
    int n = curr->refine( this, piece );

    for( int i = 0; i<n; i++) {
      work.push_front( TTPtr<T>(piece[i]));
    }

    delete curr;
  }
}

/*********************************************************************
 Methods for trimmed tessellations
 *********************************************************************/


/**********
 Default method used to control refinement of straight edged approximation
 of the the trimming curves
 **********/
template < class T >
bool TessTrim<T>::refine_trimcurve( int loop, int seg, T beg, T end ) {
  T pa[2], pb[2];
  trim_curve_fn( loop, seg, beg, pa );
  trim_curve_fn( loop, seg, end, pb );
  return (split_edge( pa, pb ) > edge_thresh);
} 


/**********
 Construct straight edged approximation of the trimming curves
 **********/
template < class T >
void TessTrim<T>::init_trimming_loops() {
  for( int ilp = 0; ilp < num_loops; ilp++) {
    TrimLoop<T>& lp = loop[ilp];
    int nsegs = lp.num_segs;

    if (!lp.pts.empty() || !nsegs)
      continue;

    for( int j = 0; j<nsegs; j++) {
      T v[2];
      trim_curve_fn( ilp, j, 0.0, v);
      lp.pts.push_back( TrimPoint<T>( j, 0.0, v) );

      //check connectivity
      T va[2];
      int j1 = (j + nsegs - 1)%nsegs;
      trim_curve_fn( ilp, j1, 1.0, va );
      if ((v[0] != va[0]) || (v[1] != va[1])) {
	cerr << "trim curve connectivity error: Loop " << ilp
	     << "; Seg " << j << endl;
      }
    }

    list< TrimPoint<T> >::iterator x = lp.pts.begin();
    while( x != lp.pts.end() ) {
      TrimPoint<T>& p0 = *x;
      T u0 = p0.uval;

      list< TrimPoint<T> >::iterator y = x;
      y++;

      T u1;
      if (y != lp.pts.end() ) {
	TrimPoint<T>& p1 = *y;
	if ((p1.seg == p0.seg+1) && (p1.uval == 0.0) )
	  u1 = 1.0;
	else if (p0.seg == p1.seg)
	  u1 = p1.uval;
	else {
	  cerr << "error: point list corrupted interior\n";
	  exit(-1);
	}
      }
      else if (p0.seg == nsegs-1)
	u1 = 1.0;
      else {
	cerr << "error: point list end corrupted\n";
	exit(-1);
      }

      if (refine_trimcurve( ilp, p0.seg, p0.uval, u1)) {
	//split edge
	T v[2];
	T um = 0.5 * (u0 + u1);
	trim_curve_fn( ilp, p0.seg, um, v );
	y = lp.pts.insert( y, TrimPoint<T>( p0.seg, um, v) );
      }
      else // proceed to next edge
	x = y;
    }//foreach edge
  }//foreach loop
}



/**********
 **********/
template < class T >
int TessTrim<T>::inout_test( TriTile<T>* tile ) {
  T a[3][2];
  T ad[3];

  //counters for intersections
  int cl[3]; //less
  int cg[3]; //greater
  int ndx[3];

  for( int i = 0; i<3; i++) {
    T *va = (*tile)[i];
    T *vb = (*tile)[(i+1)%3];

    //make line equation for edge
    a[i][0] = va[1] - vb[1];
    a[i][1] = vb[0] - va[0];
    ad[i] = a[i][0]*va[0] + a[i][1]*va[1];

    cl[i] = cg[i] = 0;
    ndx[i] = (abs(a[i][0]) > abs(a[i][1])) ? 1 : 0;
  }

  for( int ilp = 0; ilp < num_loops; ilp++) {
    TrimLoop<T>& lp = loop[ilp];

    for( list< TrimPoint<T> >::iterator ix = lp.pts.begin();
	 ix!=lp.pts.end(); ix++ ) {
      list< TrimPoint<T> >::iterator iy = ix;
      iy++;
      if (iy==lp.pts.end()) iy = lp.pts.begin();

      TrimPoint<T>& p0 = *ix;
      TrimPoint<T>& p1 = *iy;

      int ni = 0;
      for( int i = 0; i<3; i++) {
	T d0 = p0.x[0]*a[i][0] + p0.x[1]*a[i][1] - ad[i];
	T d1 = p1.x[0]*a[i][0] + p1.x[1]*a[i][1] - ad[i];

	ni += (d0<0.) ? 1 : -1;

	if ((d0<0) ? (d1<0) : (d1>=0))
	  continue;//no intersection

	//find intersection point
	T ip = (p1.x[ndx[i]]*d0 - p0.x[ndx[i]]*d1) / (d0-d1);
	bool ba = ip<(*tile)[i][ndx[i]];
	bool bb = ip<(*tile)[(i+1)%3][ndx[i]];
	if (ba && bb) cl[i]++;
	else if (!(ba || bb)) cg[i]++;
	else return 0;
      }

      //point inside tile
      if ((ni==3) || (ni==-3)) return 0;
    }
  }

  if ((cl[0]&1) && (cl[1]&1) && (cl[2]&1)) return 1;
  if ( !((cl[0]&1) || (cl[1]&1) || (cl[2]&1)) ) return -1;
  return 0;
}




/**********
 return true if tile lies completely outside of trimming region
 **********/
template < class T >
bool TessTrim<T>::tile_culled( TriTile<T>* tile) {
  return (inout_test(tile)<0);
}


/**********
 test if trimming curve passes through this tile and calls 
 render_final() or render_trimmed() as appropriate
 **********/
template < class T >
void TessTrim<T>::trim_final( TriTile<T>* leaf ) {
  if (inout_test(leaf) == 0)
    render_trimmed( *leaf );
  else
    render_final( *leaf );
}
