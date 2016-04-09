#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include "kl_convexhull2d.h"
#include "kl_minquad.h"


#define DET_THRESHOLD		0.0000001
#define INTERSECT_THRESHOLD	0.00001
#define FLATNESS_THRESHOLD	0.0000001


static 
void *checked_malloc( size_t size )
	// same as malloc(), but checks for out-of-memory.
{
	void *p = malloc( size );
	if ( p == NULL )
	{
		fprintf( stderr, "Cannot allocate memory\n." );
		exit( 1 );
	}
	return p;
}


static
double fsqr( double f )
    // returns the square of f.
{
    return ( f * f );
}


static
double SqrDist( double v1[2], double v2[2] )
	// returns square of the distance between 2 points
{
	return fsqr( v1[0] - v2[0] ) + fsqr( v1[1] - v2[1] );
}



static
double TriangleArea( const double p1[2], const double p2[2], const double p3[2] )
{
	return fabs( 0.5 * ( (p2[0] - p1[0]) * (p3[1] - p1[1]) - 
		                 (p3[0] - p1[0]) * (p2[1] - p1[1]) ) ); 
}



static
bool EdgeForwardExtIntersection( const double edgeA[2][2], const double edgeB[2][2],
					             double intersection[2] )
{
	double vecA[2] = { edgeA[1][0] - edgeA[0][0], edgeA[1][1] - edgeA[0][1] };
	double vecB[2] = { edgeB[1][0] - edgeB[0][0], edgeB[1][1] - edgeB[0][1] };

	double a = vecA[0];
	double b = -vecB[0];
	double c = vecA[1];
	double d = -vecB[1];

	double det = a * d - b * c;

	if ( fabs( det ) >= DET_THRESHOLD )
	{
		double E[2] = { edgeB[0][0] - edgeA[0][0], edgeB[0][1] - edgeA[0][1] };
		double tA = ( d * E[0] - b * E[1] ) / det;
		// return true iff intersection is at the edges' forward extensions
		if ( tA > INTERSECT_THRESHOLD )
		{		
			intersection[0] = edgeA[0][0] + tA * vecA[0];
			intersection[1] = edgeA[0][1] + tA * vecA[1];
			return true;
		}
		else 
			return false;
	}
	else
	{
		if ( TriangleArea( edgeA[0], edgeA[1], edgeB[0] ) > FLATNESS_THRESHOLD  &&
			 TriangleArea( edgeA[0], edgeA[1], edgeB[1] ) > FLATNESS_THRESHOLD )
		{
			// CASE 1: Parallel edges, and edges' extensions do not intersect	
			return false;
		}
		else
		{
			// CASE 2: Parallel edges, and edges' extensions intersect
			// Use midpoint of edgeA[1] and edgeB[1]
			intersection[0] = 0.5 * ( edgeA[1][0] + edgeB[1][0] );
			intersection[1] = 0.5 * ( edgeA[1][1] + edgeB[1][1] );
			return true;
		}
	}
}



typedef struct EdgeNode
{
	double p1[2];
	double p2[2];
	bool hasIntersection;
	double intersection[2];
	double area;
	EdgeNode *prev;
	EdgeNode *next;
	int heapIndex;
}
EdgeNode;



// ******************** HEAP (BEGIN) ********************************************

typedef struct Heap
{
	EdgeNode **A;	// array of pointers to EdgeNode
	int heapSize;
	int arraySize;
}
Heap;

static int HeapParent( int i ) { return (i-1)/2; }
static int HeapLeft( int i ) { return 2*i+1; }
static int HeapRight( int i ) { return 2*i+2; }

static void Heapify( Heap *h, int i )
{
	int L = HeapLeft( i );
	int R = HeapRight( i );
	int smallest = i;
	if ( L < h->heapSize && h->A[L]->area < h->A[i]->area ) smallest = L;
	if ( R < h->heapSize && h->A[R]->area < h->A[smallest]->area ) smallest = R;
	if ( smallest != i )
	{
		EdgeNode *tmp = h->A[i];
		h->A[i] = h->A[smallest];
		h->A[smallest] = tmp;
		h->A[i]->heapIndex = i;
		h->A[smallest]->heapIndex = smallest;
		Heapify( h, smallest );
	}
}

static Heap *BuildHeap( EdgeNode *listHead, int listSize )
{
	Heap *heap = (Heap *) checked_malloc( sizeof(Heap) );
	heap->A = (EdgeNode **) checked_malloc( sizeof(EdgeNode *) * listSize );
	heap->arraySize = listSize;
	heap->heapSize = listSize;

	EdgeNode *node = listHead;
	int i;

	for ( i = 0; i < listSize; i++, node = node->next )
	{
		heap->A[i] = node;
		node->heapIndex = i;
	}

	for ( i = HeapParent( listSize - 1 ); i >= 0; i-- ) Heapify( heap, i );

	return heap;
}

static void DeleteHeap( Heap *heap )
{
	free( heap->A );
	free( heap );
}

static EdgeNode *HeapExtractMinAreaNode( Heap *heap )
{
	if ( heap->heapSize <= 0 ) return NULL;
	EdgeNode *minAreaNode = heap->A[0];
	heap->A[0] = heap->A[ heap->heapSize - 1 ];
	heap->A[0]->heapIndex = 0;
	(heap->heapSize)--;
	Heapify( heap, 0 );
	return minAreaNode;
}

static void UpdateHeap( Heap *h, int i )
{
	int k = i;
	while ( k > 0  &&  h->A[k]->area < h->A[ HeapParent(k) ]->area )
	{
		int ParentIndex = HeapParent(k);
		EdgeNode *tmp = h->A[k];
		h->A[k] = h->A[ ParentIndex ];
		h->A[ ParentIndex ] = tmp;
		h->A[k]->heapIndex = k;
		h->A[ ParentIndex ]->heapIndex = ParentIndex;
		k = ParentIndex;
	}

	if ( k == i ) Heapify( h, i );
}

// ******************** HEAP (END) ********************************************


static
void FindMinQuad( const double inPoints2D[], const int hull[], int hullSize, 
				  double outPoints2D[8] )
{
	EdgeNode *edgeBuf = (EdgeNode *) checked_malloc( sizeof(EdgeNode) * hullSize );
	int i;

	for ( i = 0; i < hullSize; i++ )
	{
		edgeBuf[i].p1[0] = inPoints2D[ 2*hull[i] ];
		edgeBuf[i].p1[1] = inPoints2D[ 2*hull[i]+1 ];
		edgeBuf[i].p2[0] = inPoints2D[ 2*hull[(i+1)%hullSize] ];
		edgeBuf[i].p2[1] = inPoints2D[ 2*hull[(i+1)%hullSize]+1 ];
		edgeBuf[i].prev = &edgeBuf[ (i-1+hullSize)%hullSize ];
		edgeBuf[i].next = &edgeBuf[ (i+1)%hullSize ];
	}

	for ( i = 0; i < hullSize; i++ )
	{
		double edgeA[2][2] = { { edgeBuf[i].prev->p1[0], edgeBuf[i].prev->p1[1] },
								{ edgeBuf[i].prev->p2[0], edgeBuf[i].prev->p2[1] } };
		double edgeB[2][2] = { { edgeBuf[i].next->p2[0], edgeBuf[i].next->p2[1] },
								{ edgeBuf[i].next->p1[0], edgeBuf[i].next->p1[1] } };

		edgeBuf[i].hasIntersection = 
			EdgeForwardExtIntersection( edgeA, edgeB, edgeBuf[i].intersection );
		
		if ( edgeBuf[i].hasIntersection )
			edgeBuf[i].area = TriangleArea( edgeBuf[i].p1, edgeBuf[i].intersection, edgeBuf[i].p2 );
		else
			edgeBuf[i].area = DBL_MAX;
	}


	// REDUCE CONVEX HULL TO A QUAD
	
	EdgeNode *listHead = edgeBuf;
	int numSides = hullSize;

	// Build Heap
	Heap *heap = BuildHeap( listHead, numSides );

	while ( numSides > 4 )
	{
		// find edge node with minimum triangle area
		EdgeNode *minAreaNode = HeapExtractMinAreaNode( heap );

		// delete the edge node from circular list
		EdgeNode *prev = minAreaNode->prev;
		EdgeNode *next = minAreaNode->next;
		if ( listHead == minAreaNode ) listHead = next;
		prev->p2[0] = minAreaNode->intersection[0];
		prev->p2[1] = minAreaNode->intersection[1];
		prev->next = next;
		next->p1[0] = minAreaNode->intersection[0];
		next->p1[1] = minAreaNode->intersection[1];
		next->prev = prev;

		{   // update previous node
			double edgeA[2][2] = { { prev->prev->p1[0], prev->prev->p1[1] },
									{ prev->prev->p2[0], prev->prev->p2[1] } };
			double edgeB[2][2] = { { prev->next->p2[0], prev->next->p2[1] },
									{ prev->next->p1[0], prev->next->p1[1] } };

			prev->hasIntersection = EdgeForwardExtIntersection( edgeA, edgeB, prev->intersection );
			
			if ( prev->hasIntersection )
				prev->area = TriangleArea( prev->p1, prev->intersection, prev->p2 );
			else
				prev->area = DBL_MAX;

			// Update heap, since area in previous node has changed
			UpdateHeap( heap, prev->heapIndex );
		}

		{   // update next node
			double edgeA[2][2] = { { next->prev->p1[0], next->prev->p1[1] },
									{ next->prev->p2[0], next->prev->p2[1] } };
			double edgeB[2][2] = { { next->next->p2[0], next->next->p2[1] },
									{ next->next->p1[0], next->next->p1[1] } };

			next->hasIntersection = EdgeForwardExtIntersection( edgeA, edgeB, next->intersection );
			
			if ( next->hasIntersection )
				next->area = TriangleArea( next->p1, next->intersection, next->p2 );
			else
				next->area = DBL_MAX;

			// Update heap, since area in next node has changed
			UpdateHeap( heap, next->heapIndex );
		}
		
		numSides--;
	}


	// copy result to output
	EdgeNode *node = listHead;
	for ( i = 0; i < 4; i++, node = node->next )
	{
		outPoints2D[ 2*i ] = node->p1[0];
		outPoints2D[ 2*i+1 ] = node->p1[1];
	}

	DeleteHeap( heap );
	free( edgeBuf );
}



bool kl_MinQuad( const double inPoints2D[], const int convexhull[], int hullSize, double outPoints2D[8] )
{
	if ( hullSize < 3 )
	{
		return false;
	}

	else if ( hullSize == 3 )
	{
		// create a parallelogram with the longest edge of triangle as a diagonal

		double v[3][2];
		for ( int i = 0; i < 3; i++ )
		{
			v[i][0] = inPoints2D[ 2 * convexhull[i] ];
			v[i][1] = inPoints2D[ 2 * convexhull[i] + 1];
		}

		// find longest triangle edge
		double sqrLen[3];
		sqrLen[0] = SqrDist( v[0], v[1] );
		sqrLen[1] = SqrDist( v[1], v[2] );
		sqrLen[2] = SqrDist( v[2], v[0] );

		if ( sqrLen[0] >= sqrLen[1] && sqrLen[0] >= sqrLen[2] )
		{
			// edge 0 is longest
			outPoints2D[2*0+0] = v[0][0];
			outPoints2D[2*0+1] = v[0][1];
			outPoints2D[2*1+0] = v[1][0] - v[2][0] + v[0][0];
			outPoints2D[2*1+1] = v[1][1] - v[2][1] + v[0][1];
			outPoints2D[2*2+0] = v[1][0];
			outPoints2D[2*2+1] = v[1][1];
			outPoints2D[2*3+0] = v[2][0];
			outPoints2D[2*3+1] = v[2][1];
		}
		else if ( sqrLen[1] >= sqrLen[2] )
		{
			// edge 1 is longest
			outPoints2D[2*0+0] = v[0][0];
			outPoints2D[2*0+1] = v[0][1];
			outPoints2D[2*1+0] = v[1][0];
			outPoints2D[2*1+1] = v[1][1];
			outPoints2D[2*2+0] = v[2][0] - v[0][0] + v[1][0];
			outPoints2D[2*2+1] = v[2][1] - v[0][1] + v[1][1];
			outPoints2D[2*3+0] = v[2][0];
			outPoints2D[2*3+1] = v[2][1];
		}
		else
		{
			// edge 2 is longest
			outPoints2D[2*0+0] = v[0][0];
			outPoints2D[2*0+1] = v[0][1];
			outPoints2D[2*1+0] = v[1][0];
			outPoints2D[2*1+1] = v[1][1];
			outPoints2D[2*2+0] = v[2][0];
			outPoints2D[2*2+1] = v[2][1];
			outPoints2D[2*3+0] = v[0][0] - v[1][0] + v[2][0];
			outPoints2D[2*3+1] = v[0][1] - v[1][1] + v[2][1];
		}
		return true;
	}

	else if ( hullSize == 4 )
	{
		for ( int i = 0; i < 4; i++ )
		{
			outPoints2D[2*i] = inPoints2D[ 2 * convexhull[i] ];
			outPoints2D[2*i+1] = inPoints2D[ 2 * convexhull[i] + 1];
		}
		return true;
	}

	else
	{
		FindMinQuad( inPoints2D, convexhull, hullSize, outPoints2D );
		return true;
	}
}



bool kl_MinQuad( const double inPoints2D[], int numInPoints2D, double outPoints2D[8] )
{
	int *convexhull;
	int hullSize;
	kl_ConvexHull2D( inPoints2D, numInPoints2D, &convexhull, &hullSize );
	bool result = kl_MinQuad( inPoints2D, convexhull, hullSize, outPoints2D );
	free( convexhull );
	return result;
}





// ********************* ANOTHER MINQUAD HEURISTICS. EXPERIMENTAL *********************

static
void FindMinQuad2( const double inPoints2D[], const int hull[], int hullSize, 
				  double outPoints2D[8] )
{
	EdgeNode *edgeBuf = (EdgeNode *) checked_malloc( sizeof(EdgeNode) * hullSize );
	int i;

	for ( i = 0; i < hullSize; i++ )
	{
		edgeBuf[i].p1[0] = inPoints2D[ 2*hull[i] ];
		edgeBuf[i].p1[1] = inPoints2D[ 2*hull[i]+1 ];
		edgeBuf[i].p2[0] = inPoints2D[ 2*hull[(i+1)%hullSize] ];
		edgeBuf[i].p2[1] = inPoints2D[ 2*hull[(i+1)%hullSize]+1 ];
		edgeBuf[i].prev = &edgeBuf[ (i-1+hullSize)%hullSize ];
		edgeBuf[i].next = &edgeBuf[ (i+1)%hullSize ];
	}

	for ( i = 0; i < hullSize; i++ )
	{
		double edgeA[2][2] = { { edgeBuf[i].prev->p1[0], edgeBuf[i].prev->p1[1] },
								{ edgeBuf[i].prev->p2[0], edgeBuf[i].prev->p2[1] } };
		double edgeB[2][2] = { { edgeBuf[i].next->p2[0], edgeBuf[i].next->p2[1] },
								{ edgeBuf[i].next->p1[0], edgeBuf[i].next->p1[1] } };

		edgeBuf[i].hasIntersection = 
			EdgeForwardExtIntersection( edgeA, edgeB, edgeBuf[i].intersection );
		
		if ( edgeBuf[i].hasIntersection )
			edgeBuf[i].area = TriangleArea( edgeBuf[i].p1, edgeBuf[i].intersection, edgeBuf[i].p2 );
		else
			edgeBuf[i].area = DBL_MAX;
	}

	// REDUCE CONVEX HULL TO A QUAD
	
	EdgeNode *listHead = edgeBuf;
	int numSides = hullSize;

	while ( numSides > 4 )
	{
		// find edge node with minimum additional squared edge length
		double minSqrLen = DBL_MAX;
		EdgeNode *minSqrLenEdgeNode = NULL;
		EdgeNode *node = listHead;

		for ( int i = 0; i < numSides; i++, node = node->next )
		{
			if ( !node->hasIntersection ) continue;
			double oldPrevSqrLen = fsqr( node->prev->p2[0] - node->prev->p1[0] ) + 
								   fsqr( node->prev->p2[1] - node->prev->p1[1] );
			double newPrevSqrLen = fsqr( node->intersection[0] - node->prev->p1[0] ) + 
								   fsqr( node->intersection[1] - node->prev->p1[1] );
			double oldNextSqrLen = fsqr( node->next->p2[0] - node->next->p1[0] ) + 
								   fsqr( node->next->p2[1] - node->next->p1[1] );
			double newNextSqrLen = fsqr( node->next->p2[0] - node->intersection[0] ) + 
								   fsqr( node->next->p2[1] - node->intersection[1] );
			double addSqrLen = newPrevSqrLen - oldPrevSqrLen + newNextSqrLen - oldNextSqrLen;

			if ( addSqrLen < minSqrLen )
			{
				minSqrLen = addSqrLen;
				minSqrLenEdgeNode = node;
			}
		}

		// delete the edge node from circular list
		EdgeNode *prev = minSqrLenEdgeNode->prev;
		EdgeNode *next = minSqrLenEdgeNode->next;
		if ( listHead == minSqrLenEdgeNode ) listHead = next;
		prev->p2[0] = minSqrLenEdgeNode->intersection[0];
		prev->p2[1] = minSqrLenEdgeNode->intersection[1];
		prev->next = next;
		next->p1[0] = minSqrLenEdgeNode->intersection[0];
		next->p1[1] = minSqrLenEdgeNode->intersection[1];
		next->prev = prev;
		

		{   // update previous node

			double edgeA[2][2] = { { prev->prev->p1[0], prev->prev->p1[1] },
									{ prev->prev->p2[0], prev->prev->p2[1] } };
			double edgeB[2][2] = { { prev->next->p2[0], prev->next->p2[1] },
									{ prev->next->p1[0], prev->next->p1[1] } };

			prev->hasIntersection = EdgeForwardExtIntersection( edgeA, edgeB, prev->intersection );
			
			if ( prev->hasIntersection )
				prev->area = TriangleArea( prev->p1, prev->intersection, prev->p2 );
			else
				prev->area = DBL_MAX;
		}

		{   // update next node

			double edgeA[2][2] = { { next->prev->p1[0], next->prev->p1[1] },
									{ next->prev->p2[0], next->prev->p2[1] } };
			double edgeB[2][2] = { { next->next->p2[0], next->next->p2[1] },
									{ next->next->p1[0], next->next->p1[1] } };

			next->hasIntersection = EdgeForwardExtIntersection( edgeA, edgeB, next->intersection );
			
			if ( next->hasIntersection )
				next->area = TriangleArea( next->p1, next->intersection, next->p2 );
			else
				next->area = DBL_MAX;
		}
		
		numSides--;
	}


	// copy result to output
	EdgeNode *node = listHead;
	for ( i = 0; i < 4; i++, node = node->next )
	{
		outPoints2D[ 2*i ] = node->p1[0];
		outPoints2D[ 2*i+1 ] = node->p1[1];
	}

	free( edgeBuf );
}



bool kl_MinQuad2( const double inPoints2D[], const int convexhull[], int hullSize, double outPoints2D[8] )
{
	if ( hullSize < 3 )
	{
		return false;
	}

	else if ( hullSize == 3 )
	{
		// create a parallelogram with the longest edge of triangle as a diagonal

		double v[3][2];
		for ( int i = 0; i < 3; i++ )
		{
			v[i][0] = inPoints2D[ 2 * convexhull[i] ];
			v[i][1] = inPoints2D[ 2 * convexhull[i] + 1];
		}

		// find longest triangle edge
		double sqrLen[3];
		sqrLen[0] = SqrDist( v[0], v[1] );
		sqrLen[1] = SqrDist( v[1], v[2] );
		sqrLen[2] = SqrDist( v[2], v[0] );

		if ( sqrLen[0] >= sqrLen[1] && sqrLen[0] >= sqrLen[2] )
		{
			// edge 0 is longest
			outPoints2D[2*0+0] = v[0][0];
			outPoints2D[2*0+1] = v[0][1];
			outPoints2D[2*1+0] = v[1][0] - v[2][0] + v[0][0];
			outPoints2D[2*1+1] = v[1][1] - v[2][1] + v[0][1];
			outPoints2D[2*2+0] = v[1][0];
			outPoints2D[2*2+1] = v[1][1];
			outPoints2D[2*3+0] = v[2][0];
			outPoints2D[2*3+1] = v[2][1];
		}
		else if ( sqrLen[1] >= sqrLen[2] )
		{
			// edge 1 is longest
			outPoints2D[2*0+0] = v[0][0];
			outPoints2D[2*0+1] = v[0][1];
			outPoints2D[2*1+0] = v[1][0];
			outPoints2D[2*1+1] = v[1][1];
			outPoints2D[2*2+0] = v[2][0] - v[0][0] + v[1][0];
			outPoints2D[2*2+1] = v[2][1] - v[0][1] + v[1][1];
			outPoints2D[2*3+0] = v[2][0];
			outPoints2D[2*3+1] = v[2][1];
		}
		else
		{
			// edge 2 is longest
			outPoints2D[2*0+0] = v[0][0];
			outPoints2D[2*0+1] = v[0][1];
			outPoints2D[2*1+0] = v[1][0];
			outPoints2D[2*1+1] = v[1][1];
			outPoints2D[2*2+0] = v[2][0];
			outPoints2D[2*2+1] = v[2][1];
			outPoints2D[2*3+0] = v[0][0] - v[1][0] + v[2][0];
			outPoints2D[2*3+1] = v[0][1] - v[1][1] + v[2][1];
		}
		return true;
	}

	else if ( hullSize == 4 )
	{
		for ( int i = 0; i < 4; i++ )
		{
			outPoints2D[2*i] = inPoints2D[ 2 * convexhull[i] ];
			outPoints2D[2*i+1] = inPoints2D[ 2 * convexhull[i] + 1];
		}
		return true;
	}

	else
	{
		FindMinQuad2( inPoints2D, convexhull, hullSize, outPoints2D );
		return true;
	}
}



bool kl_MinQuad2( const double inPoints2D[], int numInPoints2D, double outPoints2D[8] )
{
	int *convexhull;
	int hullSize;
	kl_ConvexHull2D( inPoints2D, numInPoints2D, &convexhull, &hullSize );
	bool result = kl_MinQuad2( inPoints2D, convexhull, hullSize, outPoints2D );
	free( convexhull );
	return result;
}


#undef DET_THRESHOLD
#undef INTERSECT_THRESHOLD
#undef FLATNESS_THRESHOLD

