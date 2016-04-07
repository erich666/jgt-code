//------------------------------------------------
// MarchingCubes
//------------------------------------------------
//
// CSG parser
// Version 0.2 - 12/08/2002
//
// Antonio Wilson Vieira: awilson@mat.puc-rio.br
// Thomas Lewiner thomas.lewiner@polytechnique.org
// Math Dept, PUC-Rio
//
//________________________________________________

#ifndef _CSG_H_
#define _CSG_H_

#include <math.h>
#include <float.h>

enum Axis      { X=0, Y=1, Z=2 };
enum Primitive { none, sphere, cylinder, block, torus, cone, heart, tangle } ;
enum Operation { None, Union, Inter, Diff } ;

//_____________________________________________________________________________
//
class CSG_Node
//-----------------------------------------------------------------------------
{
//-----------------------------------------------------------------------------
// Constructors
public :
  CSG_Node() : op(None), prim(none), r(0), R(0), axe(X), left(NULL), right(NULL) { med[X] = med[Y] = med[Z] = min[X] = min[Y] = min[Z] = max[X] = max[Y] = max[Z] = 0 ; }
  ~CSG_Node() { delete left  ; left  = NULL ; delete right ; right = NULL ; }
/*
  void operator = ( const CSG_Node * node )
  {
  op     = node->op     ;
  prim   = node->prim   ;
  min[X] = node->min[X] ;   min[Y] = node->min[Y] ;   min[Z] = node->min[Z] ;
  med[X] = node->med[X] ;   med[Y] = node->med[Y] ;   med[Z] = node->med[Z] ;
  max[X] = node->max[X] ;   max[Y] = node->max[Y] ;   max[Z] = node->max[Z] ;
  r      = node->r      ;
  R      = node->R      ;
  axe    = node->axe    ;
  left   = node->left   ;
  right  = node->right  ;
  }
*/
//-----------------------------------------------------------------------------
// Operations
public :
  static CSG_Node *parse( FILE *fp )
    {
      CSG_Node *node = new CSG_Node() ;

      char s;
      float a,b,c,d,e,f;
      fscanf(fp," %c ",&s);
      switch(s)
      {
// Operations
      case 'U'  : node->op = Union ; node->left = parse(fp) ; node->right = parse(fp) ; break ;
      case 'I'  : node->op = Inter ; node->left = parse(fp) ; node->right = parse(fp) ; break ;
      case '/'  : node->op = Diff  ; node->left = parse(fp) ; node->right = parse(fp) ; break ;
      case '\\' : node->op = Diff  ; node->left = parse(fp) ; node->right = parse(fp) ; break ;

// Primitives
      case 't' :
        fscanf(fp," %f %f %f %f %f %f ",&a,&b,&c,&d, &e, &f);
        node->prim   = torus ;
        node->med[X] = a ;
        node->med[Y] = b ;
        node->med[Z] = c ;
        node->r      = d ;
        node->R      = e ;
        node->axe    = f ;
        break ;

      case 's':
        fscanf(fp,"%f %f %f %f",&a,&b,&c,&d);
        node->prim   = sphere ;
        node->med[X] = a ;
        node->med[Y] = b ;
        node->med[Z] = c ;
        node->r      = d ;
        break;

      case 'c':
        fscanf(fp,"%f %f %f %f %f %f",&a,&b,&c,&d,&e,&f);
        node->prim   = cylinder ;
        node->med[X] = a ;
        node->med[Y] = b ;
        node->med[Z] = c ;
        node->r      = d ;
        node->min[Z] = e ;
        node->max[Z] = f ;
        break;

      case 'b':
        fscanf(fp,"%f %f %f %f %f %f",&a,&b,&c,&d,&e,&f);
        node->prim   = block ;
        node->min[X] = a ;
        node->min[Y] = b ;
        node->min[Z] = c ;
        node->max[X] = d ;
        node->max[Y] = e ;
        node->max[Z] = f ;
        break;

      case 'n':
        fscanf(fp,"%f %f %f %f",&a,&b,&c,&d);
        node->prim   = cone ;
        node->min[Z] = a ;
        node->max[Z] = b ;
        node->r      = c ;
        node->R      = d ;
        break;

      case 'h':
        node->prim   = heart ;
        break;

      case 'g':
        node->prim   = tangle ;
        break;

      default :
        printf( "CSG_Node::parse warning : unknown code %c\n", s ) ;
      }

      return node ;
    }


  static inline float MIN(float a, float b ) { if(a<b) return a; return b; }
  static inline float MAX(float a, float b ) { if(a>b) return a; return b; }

  const float eval( float x, float y, float z ) const
    {
      float i=0,j=0,k=0, t, res=0 ;
      if( ((prim == none) && (op == None)) || ((prim != none) && (op != None))  )
      {
        printf( "CSG_Node::eval warning : inconsistend node\n" ) ;
        return 0 ;
      }

      switch(prim)
      {
      case sphere:
        res = ( (x-med[X])*(x-med[X]) + (y-med[Y])*(y-med[Y]) + (z-med[Z])*(z-med[Z]) - r*r );
        break ;

      case heart:
        res = (2*x*x+y*y+z*z-1) * (2*x*x+y*y+z*z-1) * (2*x*x+y*y+z*z-1) - (1/10)*x*x*z*z*z - y*y*z*z*z;
        break ;

      case tangle:
        res = x*x*x*x - 5*x*x + y*y*y*y - 5*y*y+z*z*z*z - 5*z*z + 11.8f;
        break ;

      case torus:
        if (axe==X) {i=x; j=y; k=z;}
        if (axe==Y) {i=y; j=z; k=x;}
        if (axe==Z) {i=z; j=x; k=y;}
        t = sqrt( i*i - 2*i*med[X] + med[X]*med[X] + k*k - 2*k*med[Z] + med[Z]*med[Z] ) ;
        res = ( t * (med[X]*med[X] - 2*i*med[X] + med[Z]*med[Z] + k*k + med[Y]*med[Y] - 2*k*med[Z] + j*j - 2*j*med[Y] + r*r + i*i) +
                (-2*r*med[X]*med[X] + 4*r*i*med[X] + 4*r*k*med[Z] - 2*r*med[Z]*med[Z] - 2*r*i*i - 2*r*k*k ) - t*R*R);
        break ;

      case cylinder:
        if (z>max[Z] || z<min[Z]) res = 1;
        else res = ( (x-med[X])*(x-med[X]) + (y-med[Y])*(y-med[Y]) - r*r );
        break ;

      case cone:
        if (z>max[Z] || z<min[Z]) res = 1;
        else res = ( (float)hypot(x,y) - ( r+(z-min[Z])/(max[Z]-min[Z])*(R-r) ) );
        break ;

      case block:
        if (x<min[X] || y<min[Y] || z<min[Z] || x>max[X] || y>max[Y] || z>max[Z]) res = 1;
        else res = -1;
        break ;

      default : break ;
      }

      switch(op)
      {
      case Union:
        res = MIN( left->eval(x,y,z),  right->eval(x,y,z)  );
        break ;

      case Inter:
        res = MAX( left->eval(x,y,z),  right->eval(x,y,z)  );
        break ;

      case Diff:
        res = MAX( left->eval(x,y,z), - right->eval(x,y,z) );
        break ;

      default : break ;
      }

#if defined(WIN32) && !defined(__CYGWIN__)
      if( _isnan(res) )
#else  // WIN32
        if( isnan(res) )
#endif // WIN32
        {
          printf( "CSG_Node::eval warning : invalid calculus\n" ) ;
          return 0 ;
        }
      return res ;
    }

//-----------------------------------------------------------------------------
// Elements
private :
  Operation op   ;
  Primitive prim ;
  float     min[3], med[3], max[3] ;
  float     r, R, axe;
  CSG_Node *left, *right;
};
//_____________________________________________________________________________



#endif // _CSG_H_
