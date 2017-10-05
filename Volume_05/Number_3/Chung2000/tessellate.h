#ifndef _TESSELLATE_
#define _TESSELLATE_

#include <list>
#include <iostream.h>

template < class T > class Tessellate;
template < class T > class TessTrim;


/***************************************************************
 Triangle tile that makes up tessellation
 ***************************************************************/
template< class T >
class TriTile {
  T vt[3][2];


  void dice( class Tessellate<T>* );

public:
  TriTile() { }

  void set_vtx( int i , T v[2] ) {
    vt[i][0] = v[0];
    vt[i][1] = v[1];
  }

  T* operator[]( int i ) {
    if ((i<0) || (i>=3)) {
      cerr << "TriTile::[] subscript out of range\n";
      exit(-1);
    }
    return vt[i];
  }

  //can tile be split; if so, return resulting tiles
  int refine( class Tessellate<T>*, TriTile<T>*[4] );

};



/************************************************************
 Abstract base class for surface definition
  The tessellation object
 ************************************************************/
template < class T >
class Tessellate {

  static T edge_thresh; //for deciding when an edge short enough

  static T split_bias;  /**********
			     for deciding which edges to split;
			     0 to 1.0
			     Low values cause all edges that exceed the
			     threshold to be split.
			     High values cause only the longer edges to be
			     split
			     **********/

  static T skew_thresh; /* for deciding when a facet is approaching
			   degeneracy.  To avoid infinite recursion
			   facet is diced non-recursively */

public:

  /*
   User function that decides if an edge should be split
   Return value gives a measure on which to prioritise edge splitting
   (e.g. edge length) and values below the set threshold will not be
   split; must be comutative
   */
  virtual T split_edge( T[2], T[2] ) =0;

  /*
   To avoid missing high frequency detail (e.g. spikes in functions)
   should all three edges fall below the split threshold, this function
   is called to determine if the triangle should be split by adding
   a vertex to its center.  An interval bound on the function over the
   domain of the triangle is one possible immplementation.
   */
  virtual bool split_cen( T[3][2] ) =0;

  /*
   User supplied function typically for rendering the refined tile
   once all edges match the acceptance criteria.
   */
  virtual void render_final( TriTile<T>& ) =0;

  static void set_threshold( T t ) { if (t>0.) edge_thresh = t; }
  static void set_splitbias( T t ) {
    if ((t>=0.) && (t<1.0))
      split_bias = t;
  }
  static void set_skewness( T t ) { if (t>0.) skew_thresh = t; }


  void tessellate( TriTile<T>& );

protected:
  virtual void trim_final( TriTile<T>* leaf ) { render_final(*leaf); }
  virtual bool tile_culled( TriTile<T>* ) { return false; } //do nothing;


  friend class TriTile<T>;
};



/************************************************************
 Auxiliary objects for Trimming
 ************************************************************/
template < class T >
class TrimPoint {
  int seg;
  T uval;
  T x[2];
public:
  TrimPoint( int sg, T u, T y[2] ) : seg(sg), uval(u) {
    x[0] = y[0]; x[1] = y[1];
  }

  TrimPoint() {}

  friend class TessTrim<T>;
};


template < class T >
class TrimLoop {
  list< TrimPoint<T> > pts;
  int num_segs;
public:
  TrimLoop() : num_segs(0) {}

  friend class TessTrim<T>;
};

/************************************************************
 Abstract base class for trimmed surface tesselation
 ************************************************************/
template < class T >
class TessTrim<T> : public Tessellate<T> {
  int num_loops;
  TrimLoop<T>* loop;

  void init_trimming_loops();
  int inout_test( TriTile<T>* );

public:
  TessTrim() : num_loops(0), loop(0) {}

  ~TessTrim() { if (num_loops) delete[] loop; }

  set_num_loops( int n ) {
    if (!num_loops && (n>num_loops)) {
      num_loops = n;
      loop = new TrimLoop<T>[n];
    }
  }

  set_num_segs( int i, int n ) {
    if ((i>=0) && (i<num_loops) && (n>0)) {
      loop[i].num_segs = n;
    }
  }

  /*
   User supplied function defining trim curves
   The trim curves must define one or more close loops in uv space.
   Each loop consists of one or more linked segments. Each segment is
   individually parameterised [0.0,1.0] such that seg = k, u = 0.0
   defines the same point as seg = k-1, u = 1.0; plus seg = 0, u = 0.0
   and seg = N-1, u = 1.0 likewise
   */
  virtual void trim_curve_fn( int loop, int seg, T u, T v[2] ) =0;

  /*
    Function to decide if straight edge approximation of the trim curve
    should be refined more.  A default function using 
    Tessellationf::split_edge() is provided but can be overided by the user
   */
  virtual bool refine_trimcurve( int, int, T, T );

  /*
   User function for rendering partially trimmed tiles
   */
  virtual void render_trimmed( TriTile<T>& ) =0;

  bool tile_culled( TriTile<T>* );
  void trim_final( TriTile<T>* leaf );

  void tessellate( TriTile<T>& root ) {
    init_trimming_loops();
    Tessellate<T>::tessellate( root );
  };

};

#endif
