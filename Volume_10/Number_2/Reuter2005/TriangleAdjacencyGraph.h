#ifndef TriangleAdjacencyGraph_CLASS_DECLARATION
#define TriangleAdjacencyGraph_CLASS_DECLARATION

/*---------------------------------------------------------------------------*\
 *                                OpenSG                                     *
 *                                                                           *
 *                                                                           *
 *             Copyright (C) 2000-2002 by the OpenSG Forum                   *
 *                                                                           *
 *                            www.opensg.org                                 *
 *                                                                           *
 *   contact: dirk@opensg.org, gerrit.voss@vossg.org, jbehr@zgdv.de          *
 *                                                                           *
 \*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*\
 *                                License                                    *
 *                                                                           *
 * This library is free software; you can redistribute it and/or modify it   *
 * under the terms of the GNU Library General Public License as published    *
 * by the Free Software Foundation, version 2.                               *
 *                                                                           *
 * This library is distributed in the hope that it will be useful, but       *
 * WITHOUT ANY WARRANTY; without even the implied warranty of                *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU         *
 * Library General Public License for more details.                          *
 *                                                                           *
 * You should have received a copy of the GNU Library General Public         *
 * License along with this library; if not, write to the Free Software       *
 * Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.                 *
 *                                                                           *
 \*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*\
 *                                Changes                                    *
 *                                                                           *
 *                                                                           *
 *                                                                           *
 *                                                                           *
 *                                                                           *
 *                                                                           *
 \*---------------------------------------------------------------------------*/

#include <list>
#include <vector>
#include <map>  
#include <iterator>

/** .
 *
 *
 * @author jbehr, Tue Feb 15 18:16:59 2000
 */

using namespace std;

typedef unsigned int Index;

class TriangleAdjacencyGraph {

  private:

    enum TriangleState { INVALID = -30, FAN_PART = -20, STRIP_PART = -10,
      DEGREE_0 = 0, DEGREE_1 = 1, 
      DEGREE_2 = 2, DEGREE_3 = 3 };

    enum WalkCase { START, LEFT, RIGHT, FINISH };

    class Triangle;

    class HalfEdge {
      Index _vertexIndex;
      public:
      Triangle *triangle;
      HalfEdge *twin;
      HalfEdge *next;

      inline void setVertex (Index startVertexIndex, Index endVertexIndex) { _vertexIndex = startVertexIndex; }
      inline Index vertexStart ( void) { return _vertexIndex; }
      inline Index vertexEnd   ( void) { return next->_vertexIndex; }
    };                           

    class Triangle {
      public:
	int state;
	Triangle *next;
	Triangle *prev;
	HalfEdge halfEdgeVec[3];
	inline void init (void) {
	  state = DEGREE_0;
	  next = prev = 0;
	  halfEdgeVec[0].next = &(halfEdgeVec[1]);
	  halfEdgeVec[0].triangle = this;
	  halfEdgeVec[1].next = &(halfEdgeVec[2]);
	  halfEdgeVec[1].triangle = this;
	  halfEdgeVec[2].next = &(halfEdgeVec[0]);
	  halfEdgeVec[2].triangle = this;
	}
	inline bool valid (void) {
	  return (state >= DEGREE_0) ? true : false; 
	}
	inline void resetDegreeState (const int type) {
	  state = 
	    ( halfEdgeVec[0].twin && 
	      (halfEdgeVec[0].twin->triangle->state >= type) ? 1 : 0) +
	    ( halfEdgeVec[1].twin && 
	      (halfEdgeVec[1].twin->triangle->state >= type) ? 1 : 0) +
	    ( halfEdgeVec[2].twin && 
	      (halfEdgeVec[2].twin->triangle->state >= type) ? 1 : 0);
	}
	inline void drop(void) {
	  if ( halfEdgeVec[0].twin && (halfEdgeVec[0].twin->triangle->state > 0) )
	    halfEdgeVec[0].twin->triangle->state--;
	  if ( halfEdgeVec[1].twin && (halfEdgeVec[1].twin->triangle->state > 0) )
	    halfEdgeVec[1].twin->triangle->state--;
	  if ( halfEdgeVec[2].twin && (halfEdgeVec[2].twin->triangle->state > 0) )
	    halfEdgeVec[2].twin->triangle->state--;
	}
	bool verify (void);
    };

    class TriangleList {
      public:
	Triangle *first;
	Triangle *last;
        
	TriangleList (void)
	  : first(0), last(0) {;}
	inline void reset(void) 
	{ first = 0; last = 0; }
	inline bool empty(void) { return first ? false : true; }
	inline unsigned countElem(void) {
	  unsigned count = 0;
	  for (Triangle *f = first; f; f = f->next) count++;
	  return count;
	}
	inline void release(Triangle &node) {
	  if (node.next) {
	    if (node.prev) {
	      node.next->prev = node.prev;
	      node.prev->next = node.next;
	    }
	    else {
	      node.next->prev = 0;
	      this->first = node.next;
	    }
	  }
	  else {
	    if (node.prev) {
	      node.prev->next = 0;
	      this->last = node.prev;
	    }
	    else {
	      this->first = 0;
	      this->last = 0;
	    }
	  }
	  node.next = node.prev = 0;
	}
	inline void add (Triangle &triangle) {
	  if (last) {
	    last->next = &triangle;
	    triangle.prev = last;
	    last = &triangle;
	  }
	  else {
	    last = &triangle;
	    first = &triangle;
	  }
	}
	inline void paste (TriangleList &list) {
	  if (&list != this) {
	    if (empty()) {
	      first = list.first;
	      last  = list.last;
	    }
	    else 
	      if (list.first) {
		last->next = list.first;
		list.first->prev = last;
		last = list.last;
	      }
	    list.first = 0;
	    list.last  = 0;
	  }
	}
    };

    inline void dropOutTriangle ( Triangle &triangle, 
	TriangleList *degreeBag ) {
      HalfEdge *twin;
      degreeBag[triangle.state].release(triangle);
      triangle.state = STRIP_PART;

      if ( (twin = triangle.halfEdgeVec[0].twin) && (twin->triangle->state > 0)) {
	degreeBag[twin->triangle->state--].release(*twin->triangle);
	degreeBag[twin->triangle->state].add(*twin->triangle);
      }
      if ( (twin = triangle.halfEdgeVec[1].twin) && (twin->triangle->state > 0)) {
	degreeBag[twin->triangle->state--].release(*twin->triangle);
	degreeBag[twin->triangle->state].add(*twin->triangle);
      }
      if ((twin = triangle.halfEdgeVec[2].twin) && (twin->triangle->state > 0)) {
	degreeBag[twin->triangle->state--].release(*twin->triangle);
	degreeBag[twin->triangle->state].add(*twin->triangle);
      }
    }

    class TrianglePool {
      enum { DEFAULT_CHUNK_SIZE = 2048 };

      struct Chunk { 
	const unsigned _size;
	unsigned _freeElem;      
	Chunk    *_next;
	Triangle   *_data;
	inline Chunk  (const unsigned size) 
	  : _size(size), _freeElem(size), _next(0) 
	  { _data = new Triangle[size]; }
	inline ~Chunk (void) { delete [] _data; delete _next; }
	inline unsigned countElem(void) {
	  return ((_size - _freeElem) + (_next ? _next->countElem() : 0));
	}
      };

      unsigned _defaultChunkSize;
      Chunk *_first;
      Chunk *_last;   

      public:
      inline TrianglePool (unsigned chunkSize = DEFAULT_CHUNK_SIZE) 
	: _defaultChunkSize(chunkSize), _first(0), _last(0) {;}
      inline ~TrianglePool(void) { clear(); }
      inline Triangle *createTriangle(void) {
	if (!_first)
	  _first = _last = new Chunk(_defaultChunkSize);
	else
	  if (_last->_freeElem == 0) 
	    _last = _last->_next = new Chunk(_defaultChunkSize);
	return &(_last->_data[_last->_size - _last->_freeElem--]);
      }
      inline void clear(void) { 
	delete _first;
	_first = _last = 0;
      }
      inline unsigned countElem (void) {
	return (_first ? _first->countElem() : 0);
      }
      inline void setChunkSize(unsigned chunkSize = DEFAULT_CHUNK_SIZE) {
	_defaultChunkSize = chunkSize;
      }

    };

    // temporary vector data structure

    typedef vector < std::pair<unsigned,HalfEdge *> > HalfEdgeLink;
    vector<HalfEdgeLink> _temporaryVector;

    // Triangle Data Pool

    TrianglePool _trianglePool;

    // Input

    TriangleList _validTriangleBag;
    TriangleList _invalidTriangleBag;

    // Output

    typedef std::pair<Index,TriangleList*> Primitive;
    vector<Primitive> _stripBag;
    vector<Primitive> _fanBag;
    vector<Primitive> _triBag;

  protected:

    inline HalfEdge * getHalfEdge (unsigned startVertexIndex, unsigned endVertexIndex) {
      unsigned i, n = _temporaryVector.size();
      const HalfEdgeLink *edgeLink((startVertexIndex < n) ? &_temporaryVector[startVertexIndex] : 0);
      HalfEdge *halfEdge = 0;

      if (edgeLink && (n = edgeLink->size()))
	for (i = 0; i < n; i++)
	  if ((*edgeLink)[i].first == endVertexIndex) {
	    halfEdge = (*edgeLink)[i].second;
	    break;
	  }

      return halfEdge;
    }

    inline void addHalfEdge (HalfEdge &halfEdge, unsigned startVertexIndex, unsigned endVertexIndex) {
      unsigned n(_temporaryVector.size());
      bool     validIndex(startVertexIndex < n);
      HalfEdge   *twin(validIndex ? getHalfEdge(endVertexIndex, startVertexIndex) : 0);

      halfEdge.setVertex(startVertexIndex,endVertexIndex);

      if (validIndex == false)
	_temporaryVector.resize(startVertexIndex * 2);


      _temporaryVector[startVertexIndex].push_back(std::pair<Index,HalfEdge*>(endVertexIndex,&halfEdge));

      if ((halfEdge.twin = twin)) {
	twin->twin = &halfEdge;
	halfEdge.triangle->state++;
	twin->triangle->state++;
      }
    } 

    inline HalfEdge *findGateEdge( Triangle *triangleOut, 
	Triangle *triangleIn ) {
      HalfEdge *halfEdge = 0;

      if (triangleOut && triangleIn)
	if ( !(halfEdge = triangleOut->halfEdgeVec[0].twin) || 
	    (halfEdge->triangle != triangleIn))
	  if (!(halfEdge = triangleOut->halfEdgeVec[1].twin) || 
	      (halfEdge->triangle != triangleIn))
	    if (!(halfEdge = triangleOut->halfEdgeVec[2].twin) || 
		(halfEdge->triangle != triangleIn))
	      halfEdge = 0;

      return halfEdge ? halfEdge->twin : 0;
    }

    int calcStripCost ( TriangleList &strip, bool reverse );

    int fillIndexFromFan   ( vector<Index> &indexVec, HalfEdge &firstEdge);
    int fillIndexFromStrip ( vector<Index> &indexVec, TriangleList &strip, 
                             bool reverse );

 public:

    /** Default Constructor */
    TriangleAdjacencyGraph (void);

    /** Copy Constructor */
    TriangleAdjacencyGraph (const TriangleAdjacencyGraph &obj);

    /** Destructor */
    virtual ~TriangleAdjacencyGraph (void);

    /**  */
    void reserve ( unsigned vertexNum, 
	unsigned triangleNum, 
	unsigned reserveEdges = 8 );

    /**  */
    inline void addTriangle (Index v0, Index v1, Index v2 )
    {
      Triangle *triangle = 0;

      if ((v0 != v1) && (v0 != v2) && (v2 != v1)) {

	// create new triangle

	triangle = _trianglePool.createTriangle();
	triangle->init();

	// reg edges

	if (!getHalfEdge(v0,v1) && !getHalfEdge(v1,v2) && !getHalfEdge(v2,v0)) {
	  addHalfEdge(triangle->halfEdgeVec[0],v0,v1);
	  addHalfEdge(triangle->halfEdgeVec[1],v1,v2);
	  addHalfEdge(triangle->halfEdgeVec[2],v2,v0);
	  _validTriangleBag.add(*triangle);
	}
	else {
	  triangle->halfEdgeVec[0].setVertex(v0,v1);
	  triangle->halfEdgeVec[1].setVertex(v1,v2);
	  triangle->halfEdgeVec[2].setVertex(v2,v0);
	  triangle->state = INVALID;
	  _invalidTriangleBag.add(*triangle);
	}
      }
    }

    inline unsigned triangleCount(void) { return _trianglePool.countElem(); }

    bool verify (void);

    unsigned int calcOptPrim ( unsigned iteration = 1,
	bool doStrip = true, bool doFan = true, 
	unsigned minFanTriangleCount = 16 );

    unsigned primitiveCount (void) {
      return ( _stripBag.size() + _fanBag.size() + _triBag.size() );
    }

    int getPrimitive ( vector<Index> & indexVec, int type = 0 );

    int calcEgdeLines ( vector<Index> & indexVec, bool codeBorder = false );

    void clear(void);

};

typedef TriangleAdjacencyGraph* TriangleAdjacencyGraphP;

#endif // TriangleAdjacencyGraph_CLASS_DECLARATION
