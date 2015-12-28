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

// System declarations

#include <iostream>
#include <stdlib.h>
#include <math.h>

#include <GL/gl.h>

// Application declarations



// Class declarations

#include "TriangleAdjacencyGraph.h"


// Static Class Varible implementations: 


//----------------------------------------------------------------------

// Method: TriangleAdjacencyGraph

// Author: jbehr

// Date:   Tue Feb 15 18:16:59 2000

// Description:

//         Default Constructor

//----------------------------------------------------------------------

TriangleAdjacencyGraph::TriangleAdjacencyGraph (void)
{
  ;
}          

//----------------------------------------------------------------------

// Method: TriangleAdjacencyGraph

// Author: jbehr

// Date:   Tue Feb 15 18:16:59 2000

// Description:

//         Copy Constructor

//----------------------------------------------------------------------

TriangleAdjacencyGraph::TriangleAdjacencyGraph (const TriangleAdjacencyGraph &obj )
{
  cerr << "Run TriangleAdjacencyGraph copy constructor; not impl.\n" << endl;
}

//----------------------------------------------------------------------

// Method: ~TriangleAdjacencyGraph

// Author: jbehr

// Date:   Tue Feb 15 18:16:59 2000

// Description:

//         Destructor

//----------------------------------------------------------------------

TriangleAdjacencyGraph::~TriangleAdjacencyGraph (void )
{
  clear();
}

//----------------------------------------------------------------------

// Method: ~TriangleAdjacencyGraph

// Author: jbehr

// Date:   Tue Feb 15 18:16:59 2000

// Description:

//         Destructor

//----------------------------------------------------------------------

bool TriangleAdjacencyGraph::Triangle::verify(void)
{
  bool retCode = true;
  Triangle *neighbor[3];

  neighbor[0] = halfEdgeVec[0].twin ? halfEdgeVec[0].twin->triangle : 0;
  neighbor[1] = halfEdgeVec[1].twin ? halfEdgeVec[1].twin->triangle : 0;
  neighbor[2] = halfEdgeVec[2].twin ? halfEdgeVec[2].twin->triangle : 0;

  if ( ( neighbor[0] &&
	( (neighbor[0] == neighbor[1]) ||
	  (neighbor[0] == neighbor[2]) ) ) ||
      ( neighbor[1] &&
	( (neighbor[1] == neighbor[0]) ||
	  (neighbor[1] == neighbor[2]) ) ) ||
      ( neighbor[2] &&
	( (neighbor[2] == neighbor[0]) )   ||
	(neighbor[2] == neighbor[1]) ) ) {
    cerr << "Neighbor linked more than once" << endl;
    retCode = false;
  }

  if ( (halfEdgeVec[0].vertexStart() == halfEdgeVec[1].vertexStart()) ||
      (halfEdgeVec[0].vertexStart() == halfEdgeVec[2].vertexStart()) ||
      (halfEdgeVec[1].vertexStart() == halfEdgeVec[2].vertexStart()) ) {
    cerr << "Invalid collapsed Triangle" << endl;
    retCode = false;
  }

  if ( (halfEdgeVec[0].triangle != this) ||
      (halfEdgeVec[1].triangle != this) ||
      (halfEdgeVec[2].triangle != this) ) {
    cerr << "Invalid halfEdge->triangle pointer" << endl;
    retCode = false;
  }

  if ( (halfEdgeVec[0].next != &halfEdgeVec[1]) ||
      (halfEdgeVec[1].next != &halfEdgeVec[2]) ||
      (halfEdgeVec[2].next != &halfEdgeVec[0]) ) {
    cerr << "Edge next link error" << endl;
    retCode = false;
  }

  return retCode;
}

//----------------------------------------------------------------------

// Method: reserve

// Author: jbehr

// Date:   Tue Feb 15 18:16:59 2000

// Description:

//         

//----------------------------------------------------------------------

void TriangleAdjacencyGraph::reserve ( unsigned vertexNum, 
    unsigned triangleNum, 
    unsigned reserveEdges )
{
  unsigned i;

  _trianglePool.setChunkSize(triangleNum);
  _temporaryVector.resize(vertexNum); 

  if (reserveEdges > 0)
    for (i = 0; i < vertexNum; i++)
      _temporaryVector[i].reserve(reserveEdges);
}

//----------------------------------------------------------------------

// Method: verify

// Author: jbehr

// Date:   Tue Feb 15 18:16:59 2000

// Description:

//         

//----------------------------------------------------------------------

bool TriangleAdjacencyGraph::verify (void )
{
  bool retCode = true;
  unsigned int i, n;
  Triangle *triangle;
  int triangleState[4];
  int invalidTriangles = 0;
  int halfEdgeCount = 0;
  map< int, int > connectionMap;
  map< int, int >::iterator connectionI;
  int connectionCount;

  for (i = 0; i < 4; i++)
    triangleState[i] = 0;

  for (triangle = _validTriangleBag.first; triangle; 
      triangle = triangle->next) {
    if ( triangle->verify() && 
	(triangle->state >= 0) || (triangle->state <= 3) )
      triangleState[triangle->state]++;
    else
      invalidTriangles++;
  }

  cout << "nonmanifold split: " << _invalidTriangleBag.countElem() << endl;

  cout << invalidTriangles << endl;

  cout << "TriangleState: ";
  for (i = 0; i < 4; i++)
    cout << triangleState[i] << " ";
  cout << endl;

  if (invalidTriangles) {
    cout << "######################################################\n";
    cout << "invalid: " << invalidTriangles << endl;
    cout << "######################################################\n";
  }

  n = _temporaryVector.size();
  for (i = 0; i < n; i++) {
    connectionCount = _temporaryVector[i].size();

    halfEdgeCount += connectionCount;
    if (connectionMap.find(connectionCount) == connectionMap.end())
      connectionMap[connectionCount] = 1;
    else
      connectionMap[connectionCount]++;
  }
  for ( connectionI = connectionMap.begin();
      connectionI != connectionMap.end(); ++connectionI ) 
    cout << connectionI->first << '/' << connectionI->second << ' ';
  cout << endl;

  cout << "HalfEdgeCount: " << halfEdgeCount << endl;

  return retCode;
}

//----------------------------------------------------------------------

// Method: calcOptPrim

// Author: jbehr

// Date:   Tue Feb 15 18:16:59 2000

// Description:

//         

//----------------------------------------------------------------------

unsigned int TriangleAdjacencyGraph::calcOptPrim ( unsigned extIteration,
    bool doStrip, bool doFan, 
    unsigned minFanTriangles )
{
  int iteration = extIteration;
  bool sample = iteration > 1 ? true : false;
  bool checkRevOrder = sample;
  TriangleList degreeBag[4];
  TriangleList *fList = 0;
  int cost = 0, sampleCost = 0;
  int stripCost = 0, revCost = 0, fanCost = 0, triCost = 0;
  int bestCost = 0, worstCost = 0, lowDegree;
  unsigned int i, n;
  WalkCase walkCase = START;
  Triangle *triangle, *next;
  HalfEdge *twin = 0, *gateEdge = 0, *halfEdge = 0;
  bool doMainLoop = true;
  unsigned int seed = 1, bestSeed = 1;
  int mostDegree = 3;
  unsigned triangleLeft = _trianglePool.countElem();
  srand(1);

  if (doFan) {
    n = _temporaryVector.size();
    fanCost = 0;

    // find fans 

    for (i = 0; i < n; i++) 
      if ( (_temporaryVector[i].size() >= minFanTriangles) &&
	  (gateEdge = _temporaryVector[i][0].second) &&
	  (gateEdge->triangle->valid()) ) {
	for ( halfEdge = gateEdge->next->next->twin;
	    (halfEdge && halfEdge->triangle->valid() && (halfEdge != gateEdge));
	    halfEdge = halfEdge->next->next->twin )
	  ;
	if (halfEdge == gateEdge) {
	  // fan is closed; mark every triangle          

	  triangle = 0;
	  fList = new TriangleList;
	  for ( halfEdge = gateEdge;
	      !triangle || (halfEdge != gateEdge);
	      halfEdge = halfEdge->next->next->twin ) {
	    triangle = halfEdge->triangle;
	    _validTriangleBag.release(*triangle);
	    triangle->drop();
	    triangle->state = FAN_PART;
	    fList->add(*triangle);
	  }
	  _fanBag.push_back(Primitive(i,fList));
	  fanCost += (_temporaryVector[i].size() + 2);
	  triangleLeft -= _temporaryVector[i].size();
	}
      }
  }

  if (doStrip && iteration) {

    // push every triangle into the according degree bag

    degreeBag[mostDegree].paste(_validTriangleBag);
    for (triangle = degreeBag[mostDegree].first; triangle; triangle = next) {
      next = triangle->next;
      if (triangle->valid()) {
	if (triangle->state != mostDegree) {
	  degreeBag[mostDegree].release(*triangle);
	  _validTriangleBag.release(*triangle);
	  degreeBag[triangle->state].add( *triangle);
	}
      }
      else {
	cerr << "INVALID TRIANGLE IN VALID TRIANGLE BAG\n" << endl;
      }
    }

    for (iteration--; iteration >= 0; iteration--) {

      seed = iteration ? rand() : bestSeed;
      srand (seed);

      fList = 0;
      cost = 0;
      doMainLoop = true;
      walkCase = START;

      // run the main loop

      while (doMainLoop) {

	switch (walkCase) {
	  case START:      

	    stripCost = 0;
	    triangle = 0;

	    for (lowDegree = 1; lowDegree < 4; lowDegree++)
	      if ((degreeBag[lowDegree].empty() == false)) {
		if (sample) {
		  // pick a random triangle

		  n = degreeBag[lowDegree].countElem() - 1;
		  i = int(float(n) * rand()/float(RAND_MAX));
		  triangle = degreeBag[lowDegree].first;
                  while (i--) 
                     triangle = triangle->next;
		}
		else {
		  // pick the first triangle

		  triangle = degreeBag[lowDegree].first;
		}              
		break;
	      }

	    if (triangle) {

	      // create the new list

	      fList = new TriangleList;

	      // find the best neighbour

	      gateEdge = 0;
	      for (i = 0; i < 3; i++) 
		if ( (twin = triangle->halfEdgeVec[i].twin) && 
		    (twin->triangle->state > 0) ) {
		  if ( twin->next->next->twin &&
		      (twin->next->next->twin->triangle->state > 0) ) {
		    gateEdge = &triangle->halfEdgeVec[i];
		    break;
		  }
		  else {
		    if ( twin->next->twin &&
			(twin->next->twin->triangle->state > 0) )
		      gateEdge = &triangle->halfEdgeVec[i];
		    else {
		      if ((twin->triangle->state > 0))
			gateEdge = &triangle->halfEdgeVec[i];
		    }
		  }
		}

	      // release and store the first triangle

	      dropOutTriangle (*triangle,degreeBag);
	      fList->add(*triangle);
	      stripCost += 3;

	      // set the next step

	      if (gateEdge) {          
		walkCase = LEFT;
		stripCost++;
	      }
	      else 
		walkCase = FINISH;
	    }
	    else
	      doMainLoop = false;      
	    break;

	  case LEFT:
	    gateEdge = gateEdge->twin;
	    triangle = gateEdge->triangle;

	    // find the next gate

	    if (triangle->state == DEGREE_0) {
	      gateEdge = 0;
	      walkCase = FINISH;
	    }
	    else
	      if ( (twin = gateEdge->next->next->twin) && 
		  (twin->triangle->state > 0) ){
		gateEdge = gateEdge->next->next;
		stripCost++;
		walkCase = RIGHT;
	      }
	      else {
		gateEdge = gateEdge->next;
		stripCost += 2;
		walkCase = LEFT;
	      }

	    // store the current triangle

	    dropOutTriangle (*triangle,degreeBag);
	    fList->add(*triangle);
	    break;

	  case RIGHT:      
	    gateEdge = gateEdge->twin;
	    triangle = gateEdge->triangle;

	    // find the next gate

	    if (triangle->state == DEGREE_0) {
	      gateEdge = 0;
	      walkCase = FINISH;
	    }
	    else
	      if ( (twin = gateEdge->next->twin) && 
		  (twin->triangle->state > 0) ) {
		gateEdge = gateEdge->next;
		stripCost++;
		walkCase = LEFT;
	      }
	      else {
		gateEdge = gateEdge->next->next;
		stripCost += 2;
		walkCase = RIGHT;
	      }

	    // store the current triangle

	    dropOutTriangle (*triangle,degreeBag);
	    fList->add(*triangle);
	    break;

	  case FINISH:      
	    // try to reverse the strip

	    if ( checkRevOrder &&
		(revCost = calcStripCost(*fList,true)) &&
		(revCost < stripCost) ) {
	      _stripBag.push_back(Primitive(1,fList));
	      cost += revCost;
	    }
	    else {
	      _stripBag.push_back(Primitive(0,fList));
	      cost += stripCost;
	    }
	    walkCase = START;
	    fList = 0;
	    break;
	}  
      }

      if (sample) {
	sampleCost = cost + (degreeBag[0].countElem() * 3) + fanCost;
	if (!bestCost || (sampleCost < bestCost)) {
	  bestCost = sampleCost;
	  bestSeed = seed;
	}
	if (sampleCost > worstCost)
	  worstCost = sampleCost;

	cout << " cost/best/worst: " 
	  << sampleCost << '/' << bestCost << '/' << worstCost
	  << endl;
      }

      if (iteration) {
	// reinit the four degree bags

	degreeBag[mostDegree].paste(degreeBag[0]);
	n = _stripBag.size();
	for (i = 0; i < n; i++) {
	  degreeBag[mostDegree].paste(*_stripBag[i].second);
	  delete _stripBag[i].second;
	}
	_stripBag.clear();
	for ( triangle = degreeBag[mostDegree].first; triangle; 
	    triangle = next) {
	  next = triangle->next;
	  triangle->resetDegreeState(STRIP_PART);
	  if (triangle->valid()) {
	    if (triangle->state != mostDegree) {
	      degreeBag[mostDegree].release(*triangle);
	      degreeBag[triangle->state].add(*triangle);
	    }
	  }
	  else {
	    cerr << "INVALID TRIANGLE IN REINIT\n" << endl;
	    cerr << triangle->state << endl;
	  }
	}
      }
    }
  }
  else {    
    // push every valid triangle in degree 0; we don't strip anything

    degreeBag[0].paste(_validTriangleBag);
  }

  if (sample) {
    cerr << "range: " 
      << bestCost << '/' << worstCost << ' '
      << float(100 * (worstCost-bestCost))/float(bestCost) << '%'
      << endl;
  }

  // collect isolated triangles  

  degreeBag[0].paste(_invalidTriangleBag);  
  triCost = degreeBag[0].countElem() * 3;
  if (triCost) {
    fList = new TriangleList;  
    fList->paste(degreeBag[0]);
    _triBag.push_back(Primitive(0,fList));
  }

  return (cost + fanCost + triCost);
}                                                  

//----------------------------------------------------------------------

// Method: calcStripCost

// Author: jbehr

// Date:   Tue Feb 15 18:16:59 2000

// Description:

//         

//----------------------------------------------------------------------

int TriangleAdjacencyGraph::calcStripCost ( TriangleList &strip, bool rev )
{
  Triangle *triangle = rev ? strip.last : strip.first, *nextTriangle;
  HalfEdge *firstEdge, *halfEdge, *gate;
  WalkCase walkCase;
  int cost = 0;

  if (triangle) {
    cost = 3;
    if ((nextTriangle = rev ? triangle->prev : triangle->next)) {
      gate = findGateEdge(triangle,nextTriangle);
      firstEdge = gate->next->next;
      cost++;
      walkCase = LEFT;
      for ( triangle = nextTriangle; 
	  (nextTriangle = (rev ? triangle->prev : triangle->next));
	  triangle = nextTriangle ) {
	halfEdge = gate->twin;
	gate = findGateEdge(triangle,nextTriangle); 
	if (walkCase == RIGHT)
	  // RIGHT

	  if (halfEdge->next == gate) {
	    cost++;
	    walkCase = LEFT;
	  }
	  else {
	    // swap; walkCase stays RIGHT;

	    cost += 2;
	  }
	else
	  // LEFT

	  if (halfEdge->next->next == gate) {
	    cost++;
	    walkCase = RIGHT;
	  }
	  else {
	    // swap; walkCase stays LEFT;

	    cost += 2;
	  }
      }
    }
  }

  return cost;
}

//----------------------------------------------------------------------

// Method: fillPrimFromStrip

// Author: jbehr

// Date:   Tue Feb 15 18:16:59 2000

// Description:

//         

//----------------------------------------------------------------------

int TriangleAdjacencyGraph::fillIndexFromStrip ( std::vector<Index> &indexVec,
    TriangleList &strip, bool rev )
{
  Triangle *triangle = rev ? strip.last : strip.first, *nextTriangle;
  HalfEdge *firstEdge, *halfEdge, *gate;
  WalkCase walkCase;
  Index vertex;
  int cost = 0;

  if (triangle) {
    cost = 3;
    indexVec.reserve(32); // find better value

    indexVec.resize(3);
    if ((nextTriangle = (rev ? triangle->prev : triangle->next))) {
      cost++;
      gate = findGateEdge(triangle,nextTriangle);
      firstEdge = gate->next->next;
      indexVec.push_back(vertex = gate->twin->next->vertexEnd());

      walkCase = LEFT;
      for ( triangle = nextTriangle; 
	  (nextTriangle = (rev ? triangle->prev : triangle->next));
	  triangle = nextTriangle ) {
	halfEdge = gate->twin;
	gate = findGateEdge(triangle,nextTriangle); 
	if (walkCase == RIGHT)
	  // RIGHT

	  if (halfEdge->next == gate) {
	    indexVec.push_back(vertex = gate->twin->next->vertexEnd());
	    walkCase = LEFT;
	    cost++;
	  }
	  else {
	    // swap; walkCase stays RIGHT;

	    indexVec.back() = gate->vertexEnd();
	    indexVec.push_back(gate->vertexStart());              
	    indexVec.push_back(vertex = gate->twin->next->vertexEnd());
	    cost += 2;
	  }
	else
	  // LEFT

	  if (halfEdge->next->next == gate) {
	    indexVec.push_back(vertex = gate->twin->next->vertexEnd());
	    walkCase = RIGHT;
	    cost++;
	  }
	  else {
	    // swap; walkCase stays LEFT;

	    indexVec.back() = gate->vertexStart();
	    indexVec.push_back(gate->vertexEnd());
	    indexVec.push_back(vertex = gate->twin->next->vertexEnd());
	    cost += 2;
	  }
      }
    }
    else 
      firstEdge = &triangle->halfEdgeVec[0];      
    indexVec[0] = vertex = firstEdge->vertexStart();
    indexVec[1] = vertex = firstEdge->next->vertexStart();      
    indexVec[2] = vertex = firstEdge->next->next->vertexStart();
  }

  return cost;
}

//----------------------------------------------------------------------

// Method: fillPrimFromFan

// Author: jbehr

// Date:   Tue Feb 15 18:16:59 2000

// Description:

//         

//----------------------------------------------------------------------

int TriangleAdjacencyGraph::fillIndexFromFan( std::vector<Index> &indexVec,
    HalfEdge &firstEdge )
{
  int count = 0;
  HalfEdge *halfEdge(&firstEdge);
  HalfEdge *gateEdge = 0;

  if (halfEdge) {
    count = 3;
    indexVec.resize(2);
    indexVec[0] = halfEdge->vertexStart();
    indexVec[1] = halfEdge->vertexEnd();
    for ( gateEdge = halfEdge->next->next->twin;
	gateEdge != halfEdge;
	gateEdge = gateEdge->next->next->twin ) {
      indexVec.push_back(gateEdge->vertexEnd());
      count++;
    }
    indexVec.push_back(halfEdge->vertexEnd());
  }
  else {
    cerr << "Invalid fac in fillIndexFromFan()" << endl;
  }

  return count;
}

//----------------------------------------------------------------------

// Method: calcEdgeLines

// Author: jbehr

// Date:   Tue Feb 15 18:16:59 2000

// Description:

//         

//----------------------------------------------------------------------

int TriangleAdjacencyGraph::getPrimitive ( vector<Index> & indexVec, int type  )
{
  unsigned i = 0, n = 0;
  Triangle *triangle;
  std::vector<Primitive> *bag = 0;

  indexVec.clear();

  // fan

  if ( !bag && 
      (type == GL_TRIANGLE_FAN) || (!type && (n = _fanBag.size()))) {
    i = n - 1;
    bag = &_fanBag;
    fillIndexFromFan ( indexVec, 
	*_temporaryVector[_fanBag[i].first][0].second );
    type = GL_TRIANGLE_FAN;
  }

  // strip

  if ( !bag &&
      (type == GL_TRIANGLE_STRIP) || (!type && (n = _stripBag.size()))) {
    i = n - 1;
    bag = &_stripBag;
    fillIndexFromStrip ( indexVec,
	*_stripBag[i].second,
	_stripBag[i].first ? true : false );
    type = GL_TRIANGLE_STRIP;
  }

  // tri

  if (!bag &&
      (type == GL_TRIANGLES) || (!type && (n = _triBag.size()))) {
    bag = &_triBag;
    if (_triBag[0].second->empty() == false) {
      n = _triBag[0].second->countElem() * 3;
      indexVec.resize(n);
      i = 0;
      for ( triangle = _triBag[0].second->first; triangle; 
	  triangle = triangle->next ) {
	indexVec[i++] = triangle->halfEdgeVec[0].vertexStart();
	indexVec[i++] = triangle->halfEdgeVec[1].vertexStart();
	indexVec[i++] = triangle->halfEdgeVec[2].vertexStart();
      }
    }
    type = GL_TRIANGLES;
    i = 0;
  }

  if (bag) {
    _invalidTriangleBag.paste(*((*bag)[i].second));
    delete (*bag)[i].second;
    if (i)
      bag->resize(i);    
    else
      bag->clear();
  }
  else
    type = 0;

  return type;
}

//----------------------------------------------------------------------

// Method: calcEdgeLines

// Author: jbehr

// Date:   Tue Feb 15 18:16:59 2000

// Description:

//         

//----------------------------------------------------------------------

int TriangleAdjacencyGraph::calcEgdeLines ( vector<Index> & indexVec, bool codeBorder )
{
  unsigned int i, nN, j, nE, halfEdgeCount = 0;
  Index startVertexIndex, endVertexIndex;
  HalfEdge *halfEdge;
  bool isBorder;

  indexVec.clear();
  nN = _temporaryVector.size();
  for (i = 0; i < nN; i++) {
    nE = _temporaryVector[i].size();
    for ( j = 0; j < nE; j++) {      
      halfEdge = _temporaryVector[i][j].second;
      startVertexIndex = halfEdge->vertexStart();
      endVertexIndex = halfEdge->vertexEnd();

      if ((isBorder = (halfEdge->twin == 0)) || (startVertexIndex <
	    endVertexIndex)) {
	indexVec.push_back(startVertexIndex);
	indexVec.push_back(endVertexIndex);
	if (codeBorder)
	  indexVec.push_back(isBorder ? 0 : 1);
	halfEdgeCount++;
      }
    }
  }

  return halfEdgeCount;
}

//----------------------------------------------------------------------

// Method: 

// Author: jbehr

// Date:   Tue Feb 15 18:16:59 2000

// Description:

//         

//----------------------------------------------------------------------

void TriangleAdjacencyGraph::clear(void)
{
  unsigned int i,n;

  _temporaryVector.clear();
  _trianglePool.clear();

  n = _stripBag.size();
  for (i = 0; i < n; i++)
    delete _stripBag[i].second;
  _stripBag.clear();

  n = _fanBag.size();
  for (i = 0; i < n; i++)
    delete _fanBag[i].second;
  _fanBag.clear();

  n = _triBag.size();
  for (i = 0; i < n; i++)
    delete _triBag[i].second;
  _triBag.clear();
}