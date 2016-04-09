#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include "kl_convexhull2d.h"

#define SAME_POINT_THRESHOLD		0.000001


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
bool SamePoint( const double p1[2], const double p2[2] )
{
	return ( fabs( p1[0] - p2[0] ) <= SAME_POINT_THRESHOLD ) &&
		   ( fabs( p1[1] - p2[1] ) <= SAME_POINT_THRESHOLD );
}



static 
void PartialHull( const double inPoints2D[], const int subsetID[], int subsetIDSize, 
				  int point1ID, int point2ID, int point3ID, int **partialHull, int *partialHullSize )
{
	if ( subsetIDSize <= 0 )
	{
		*partialHullSize = 2;
		*partialHull = (int *) checked_malloc( sizeof(int) * 2 );
		(*partialHull)[0] = point1ID;
		(*partialHull)[1] = point2ID;
		return;
	}
	else
	{
		// FIND POINTS THAT ARE ON THE OUTSIDE OF 
		// THE POINT1-POINT2 LINE SEGMENT OR THE POINT2-POINT3 LINE SEGMENT

		double NA[2]; // vector perpendicular to point1-point2 line segment (pointing outside)
		NA[0] = inPoints2D[ 2*point2ID+1 ] - inPoints2D[ 2*point1ID+1 ];
		NA[1] = inPoints2D[ 2*point1ID ] - inPoints2D[ 2*point2ID ];

		double NB[2]; // vector perpendicular to point2-point3 line segment (pointing outside)
		NB[0] = inPoints2D[ 2*point3ID+1 ] - inPoints2D[ 2*point2ID+1 ];
		NB[1] = inPoints2D[ 2*point2ID ] - inPoints2D[ 2*point3ID ];

		double midpointA[2]; // midpoint between point1 & point2
		midpointA[0] = 0.5 * ( inPoints2D[ 2*point1ID ] + inPoints2D[ 2*point2ID ] );
		midpointA[1] = 0.5 * ( inPoints2D[ 2*point1ID+1 ] + inPoints2D[ 2*point2ID+1] );

		double midpointB[2]; // midpoint between point2 & point3
		midpointB[0] = 0.5 * ( inPoints2D[ 2*point2ID ] + inPoints2D[ 2*point3ID ] );
		midpointB[1] = 0.5 * ( inPoints2D[ 2*point2ID+1 ] + inPoints2D[ 2*point3ID+1] );

		// front of array are points on the outside of the point1-point2 line segment;
		// back of array are points on the outside of the point2-point3 line segment.
		int *outsidePoints = (int *) checked_malloc( sizeof(int) * subsetIDSize );
		int numOutsidePointsA = 0;
		int numOutsidePointsB = 0;

		// outside point furthest from point1-point2 line segment
		double maxDistA = 0.0;
		int maxDistIDA = 0; 
		int maxDistIDPosInOutsidePointsA = 0;

		// outside point furthest from point2-point3 line segment
		double maxDistB = 0.0;
		int maxDistIDB = 0; 
		int maxDistIDPosInOutsidePointsB = 0;


		// check whether each point is outside 
		// point1-point2 line segment or point2-point3 line segment

		for ( int i = 0; i < subsetIDSize; i++ )
		{
			if ( SamePoint( &inPoints2D[ 2*subsetID[i] ], &inPoints2D[ 2*point2ID ] ) ) continue;

			double distA = ( inPoints2D[ 2*subsetID[i] ] - midpointA[0] ) * NA[0] + 
				           ( inPoints2D[ 2*subsetID[i]+1 ] - midpointA[1] ) * NA[1];

			if ( distA > 0.0 )
			{
				if ( distA > maxDistA )
				{
					maxDistA = distA;
					maxDistIDA = subsetID[i];
					maxDistIDPosInOutsidePointsA = numOutsidePointsA;
				}

				outsidePoints[ numOutsidePointsA ] = subsetID[i];
				numOutsidePointsA++;
			}
			else
			{
				double distB = ( inPoints2D[ 2*subsetID[i] ] - midpointB[0] ) * NB[0] + 
							   ( inPoints2D[ 2*subsetID[i]+1 ] - midpointB[1] ) * NB[1];

				if ( distB > 0.0 )
				{
					if ( distB > maxDistB )
					{
						maxDistB = distB;
						maxDistIDB = subsetID[i];
						maxDistIDPosInOutsidePointsB = subsetIDSize - 1 - numOutsidePointsB;
					}

					outsidePoints[ subsetIDSize - 1 - numOutsidePointsB ] = subsetID[i];
					numOutsidePointsB++;
				}
			}
		}


		// when no points are outside both segments

		if ( numOutsidePointsA + numOutsidePointsB == 0 )
		{
			*partialHullSize = 2;
			*partialHull = (int *) checked_malloc( sizeof(int) * 2 );
			(*partialHull)[0] = point1ID;
			(*partialHull)[1] = point2ID;
			free( outsidePoints );
			return;
		}

		
		int *partialHullA;
		int partialHullASize;

		if ( numOutsidePointsA == 0 )
		{
			// when no points are outside point1-point2 line segment
			partialHullASize = 1;
			partialHullA = (int *) checked_malloc( sizeof(int) * 1 );
			partialHullA[0] = point1ID;
		}
		else
		{
			// when some points are outside point1-point2 line segment
			outsidePoints[ maxDistIDPosInOutsidePointsA ] = outsidePoints[ numOutsidePointsA - 1 ];
			numOutsidePointsA--;
			PartialHull( inPoints2D, outsidePoints, numOutsidePointsA, point1ID, maxDistIDA, point2ID,
				         &partialHullA, &partialHullASize );
		}


		int *partialHullB;
		int partialHullBSize;

		if ( numOutsidePointsB == 0 )
		{
			// when no points are outside point2-point3 line segment
			partialHullBSize = 1;
			partialHullB = (int *) checked_malloc( sizeof(int) * 1 );
			partialHullB[0] = point2ID;
		}
		else
		{
			// when some points are outside point2-point3 line segment
			outsidePoints[ maxDistIDPosInOutsidePointsB ] = outsidePoints[ subsetIDSize - numOutsidePointsB ];
			numOutsidePointsB--;
			PartialHull( inPoints2D, &outsidePoints[ subsetIDSize - numOutsidePointsB ], 
				         numOutsidePointsB, point2ID, maxDistIDB, point3ID,
				         &partialHullB, &partialHullBSize );
		}

		// combine 2 partial hulls
		*partialHullSize = partialHullASize + partialHullBSize;
		*partialHull = (int *) checked_malloc( sizeof(int) * (*partialHullSize) );
		memcpy( *partialHull, partialHullA, sizeof(int) * partialHullASize );
		memcpy( (*partialHull) + partialHullASize, partialHullB, sizeof(int) * partialHullBSize );

		free( partialHullA );
		free( partialHullB );
		free( outsidePoints );
		return;
	}
}




void kl_ConvexHull2D( const double inPoints2D[], int numInPoints2D, int **hull, int *hullSize )
{
	if ( numInPoints2D <= 0 )
	{
		*hullSize = 0;
		*hull = NULL;
		return;
	}
	else if ( numInPoints2D == 1 )
	{
		*hullSize = 1;
		*hull = (int *) checked_malloc( sizeof(int) * 1 );
		(*hull)[0] = 0;
		return;
	}
	else if ( numInPoints2D == 2 )
	{
		*hullSize = 2;
		*hull = (int *) checked_malloc( sizeof(int) * 2 );
		(*hull)[0] = 0;
		(*hull)[1] = 1;
		return;
	}
	else
	{
		int i;

		// FIND LEFTMOST AND RIGHTMOST POINTS
		int leftID = 0, rightID = 0;
		double leftX = inPoints2D[0], rightX = inPoints2D[0];

		for ( i = 1; i < numInPoints2D; i++ )
		{
			if ( inPoints2D[2*i] < leftX )
			{
				leftID = i;
				leftX = inPoints2D[2*i]; 
			}
			else if ( inPoints2D[2*i] > rightX )
			{
				rightID = i;
				rightX = inPoints2D[2*i]; 
			}
		}


		double N[2]; // vector perpendicular to left-right line segment (pointing down)
		N[0] = inPoints2D[ 2*rightID+1 ] - inPoints2D[ 2*leftID+1 ];
		N[1] = inPoints2D[ 2*leftID ] - inPoints2D[ 2*rightID ];

		double midpoint[2]; // midpoint between leftmost & rightmost points
		midpoint[0] = 0.5 * ( inPoints2D[ 2*leftID ] + inPoints2D[ 2*rightID ] );
		midpoint[1] = 0.5 * ( inPoints2D[ 2*leftID+1 ] + inPoints2D[ 2*rightID+1] );

		// front of array are points below the left-right line segment;
		// back of array are points above the left-right line segment;
		int *outsidePoints = (int *) checked_malloc( sizeof(int) * numInPoints2D );
		int numOutsidePointsA = 0;
		int numOutsidePointsB = 0;

		// point furthest below the left-right line segment
		double maxDistA = 0.0;
		int maxDistIDA = 0; 
		int maxDistIDPosInOutsidePointsA = 0;

		// point furthest above the left-right line segment
		double maxDistB = 0.0;
		int maxDistIDB = 0; 
		int maxDistIDPosInOutsidePointsB = 0;


		// check whether each point is below or above the left-right line segment 

		for ( i = 0; i < numInPoints2D; i++ )
		{
			if ( i == leftID || i == rightID ) continue;
			if ( SamePoint( &inPoints2D[ 2*i ], &inPoints2D[ 2*leftID ] ) ||
				 SamePoint( &inPoints2D[ 2*i ], &inPoints2D[ 2*rightID ] ) ) continue;

			double dist = ( inPoints2D[ 2*i ] - midpoint[0] ) * N[0] + 
				          ( inPoints2D[ 2*i+1 ] - midpoint[1] ) * N[1];

			if ( dist > 0.0 )
			{
				// point is below left-right line segment
				if ( dist > maxDistA )
				{
					maxDistA = dist;
					maxDistIDA = i;
					maxDistIDPosInOutsidePointsA = numOutsidePointsA;
				}

				outsidePoints[ numOutsidePointsA ] = i;
				numOutsidePointsA++;
			}
			else if ( dist < 0.0 )
			{
				dist = -dist;

				// point is above left-right line segment
				if ( dist > maxDistB )
				{
					maxDistB = dist;
					maxDistIDB = i;
					maxDistIDPosInOutsidePointsB = numInPoints2D - 1 - numOutsidePointsB;
				}

				outsidePoints[ numInPoints2D - 1 - numOutsidePointsB ] = i;
				numOutsidePointsB++;
			}
		}


		// when no points are outside both segments

		if ( numOutsidePointsA + numOutsidePointsB == 0 )
		{
			*hullSize = 2;
			*hull = (int *) checked_malloc( sizeof(int) * 2 );
			(*hull)[0] = leftID;
			(*hull)[1] = rightID;
			free( outsidePoints );
			return;
		}

		
		int *partialHullA;
		int partialHullASize;

		if ( numOutsidePointsA == 0 )
		{
			// when no points are below the left-right line segment
			partialHullASize = 1;
			partialHullA = (int *) checked_malloc( sizeof(int) * 1 );
			partialHullA[0] = leftID;
		}
		else
		{
			// when some points are below the left-right line segment
			outsidePoints[ maxDistIDPosInOutsidePointsA ] = outsidePoints[ numOutsidePointsA - 1 ];
			numOutsidePointsA--;
			PartialHull( inPoints2D, outsidePoints, numOutsidePointsA, leftID, maxDistIDA, rightID,
				         &partialHullA, &partialHullASize );
		}


		int *partialHullB;
		int partialHullBSize;

		if ( numOutsidePointsB == 0 )
		{
			// when no points are above the left-right line segment
			partialHullBSize = 1;
			partialHullB = (int *) checked_malloc( sizeof(int) * 1 );
			partialHullB[0] = rightID;
		}
		else
		{
			// when some points are above the left-right line segment
			outsidePoints[ maxDistIDPosInOutsidePointsB ] = outsidePoints[ numInPoints2D - numOutsidePointsB ];
			numOutsidePointsB--;
			PartialHull( inPoints2D, &outsidePoints[ numInPoints2D - numOutsidePointsB ], 
				         numOutsidePointsB, rightID, maxDistIDB, leftID,
				         &partialHullB, &partialHullBSize );
		}

		// combine 2 partial hulls
		*hullSize = partialHullASize + partialHullBSize;
		*hull = (int *) checked_malloc( sizeof(int) * (*hullSize) );
		memcpy( *hull, partialHullA, sizeof(int) * partialHullASize );
		memcpy( (*hull) + partialHullASize, partialHullB, sizeof(int) * partialHullBSize );

		free( partialHullA );
		free( partialHullB );
		free( outsidePoints );
		return;
	}
}




void kl_FreeHull2D( int *hull )
{
	free( hull );
}




#undef SAME_POINT_THRESHOLD
