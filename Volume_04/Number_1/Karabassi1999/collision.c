#include <math.h>
#include <stdio.h>
#include <float.h>

#define MAXTRIANGLES 100000
#define X 0
#define Y 1
#define Z 2

typedef double DBL;
typedef DBL VECTOR[3];

// The blob structure definition
typedef struct BLOBSTRUCT
{
	VECTOR Centre;
	VECTOR PreviousCentre;
	DBL Radius;
} BLOB;

// The triangle structure and triangle mesh definitions
typedef struct
{ VECTOR P1,     //
         P2,     // triangle vertices
		 P3,     //
		 N,      // triangle surface normal 
		 C;      // bounding sphere centre
   DBL   r;      // bounding sphere radius
} TRIANGLE;


typedef struct 
{ int       Faces; // Number of triangles in mesh 
  TRIANGLE  Face[MAXTRIANGLES]; 
} MESH;

//Elementary functions

#define min(a,b)  (((a) < (b)) ? (a) : (b)) 
#define max(a,b)  (((a) > (b)) ? (a) : (b))
#define CopyVector(A,B)    { A[0]=B[0]; A[1]=B[1]; A[2]=B[2]; }
#define CopyTriangle(A,B) { CopyVector(A.P1,B.P1);CopyVector(A.P2,B.P2);CopyVector(A.P3,B.P3);CopyVector(A.N,B.N);CopyVector(A.C,B.C);A.r=B.r; }
#define SetVector(A,x,y,z) { A[0]=x;    A[1]=y;    A[2]=z;    }
#define Product(a,b,c) { a = b[X]*c[X] + b[Y]*c[Y] + b[Z]*c[Z]; }
#define Subtract(s,a,b) { s[X]=a[X]-b[X]; s[Y]=a[Y]-b[Y]; s[Z]=a[Z]-b[Z]; }
#define Add(s,a,b) { s[X]=a[X]+b[X]; s[Y]=a[Y]+b[Y]; s[Z]=a[Z]+b[Z]; }
#define Multiply(m,a,b) { m[X]=a[X]*b[X]; m[Y]=a[Y]*b[Y]; m[Z]=a[Z]*b[Z]; }
#define Divide(d,a,b) { d[X]=a[X]/b[X]; d[Y]=a[Y]/b[Y]; d[Z]=a[Z]/b[Z]; }
#define LinearV(l,a,b,t) { l[X]=(float)a[X]*t + (float)b[X]*(1.0-t); l[Y]=(float)a[Y]*t + (float)b[Y]*(1.0-t); l[Z]=(float)a[Z]*t + (float)b[Z]*(1.0-t); }
#define LinearS(l,a,b,t) { l=(float)a*t + (float)b*(1.0-t); l=(float)a*t + (float)b*(1.0-t); l=(float)a*t + (float)b*(1.0-t); }
#define OuterProduct(p,a,b) { p[X]=a[Y]*b[Z]-a[Z]*b[Y];p[Y]=a[Z]*b[X]-a[X]*b[Z];p[Z]=a[X]*b[Y]-a[Y]*b[X]; }
#define ScaleS(m,a,b) { m[X]=a[X]*b; m[Y]=a[Y]*b; m[Z]=a[Z]*b; }
#define IsZero(a) (a[X]==0.0)&&(a[Y]==0.0)&&(a[Z]==0.0) 
#define InBoundaries( x, a, b ) ( ( x >= a ) && ( x <= b ) )


//------------------------------------------------------------------------------


void CopyBlob( BLOB blob1, BLOB blob2 )
{ 
	CopyVector( blob1.Centre, blob2.Centre );
	CopyVector( blob1.PreviousCentre, blob2.PreviousCentre );
	blob1.Radius = blob2.Radius;
	
} //CopyBlob


//------------------------------------------------------------------------------


int IsEqual(const VECTOR a, const VECTOR b)
// Checks if 2 vectors are equal.
{ 
	return ( (a[X]==b[X]) && (a[Y]==b[Y]) && (a[Z]==b[Z]) );
}//IsEqual


//------------------------------------------------------------------------------


DBL Distance( VECTOR a,VECTOR b )
// Calculates the distance between two vectors 
{ if (IsEqual(a,b)) return( 0.0 );
  else  return( sqrt( (a[X]-b[X])*(a[X]-b[X]) + (a[Y]-b[Y])*(a[Y]-b[Y]) +
		  (a[Z]-b[Z])*(a[Z]-b[Z]) ) );
}//Distance


//------------------------------------------------------------------------------


DBL SqrDistance( VECTOR a,VECTOR b )
// Calculates the distance^2 between two vectors 
{ if (IsEqual(a,b)) return( 0.0 );
  else  return( (a[X]-b[X])*(a[X]-b[X]) + (a[Y]-b[Y])*(a[Y]-b[Y]) +
		  (a[Z]-b[Z])*(a[Z]-b[Z]) );
}//SqrDistance


//------------------------------------------------------------------------------


DBL Norm( VECTOR V )
{ DBL P;
  Product(P,V,V);
  return (P>0)?(sqrt(P)):0.0;
}//Norm


//------------------------------------------------------------------------------


void ReduceToUnit(VECTOR vector)
// Reduces a normal vector specified as a set of three coordinates,
// to a unit normal vector of length one.
{
	float length;
	
	// Calculate the length of the vector		
	length = (float)sqrt((vector[0]*vector[0]) + 
						(vector[1]*vector[1]) +
						(vector[2]*vector[2]));

	// Keep the program from blowing up by providing an exceptable
	// value for vectors that may calculated too close to zero.
	if(length == 0.0f)
		length = 1.0f;

	// Dividing each element by the length will result in a
	// unit normal vector.
	vector[0] /= length;
	vector[1] /= length;
	vector[2] /= length;
}//ReduceToUnit


//------------------------------------------------------------------------------


DBL PlaneValue( VECTOR N, VECTOR P0, VECTOR P)
// Calculates the equation of plane with normal N that contains 
// point P0 and calculates the outcome for point P. 
{
	DBL d, tmp;
	
	//Check that the plane is valid
	
	Product( tmp, N, N ); 
	if ( tmp == 0.0 )
	  return( -1.0 );

	Product( d, N, P0 );
	Product( tmp, N, P );

	return( tmp - d );
}//PlaneValue 


//------------------------------------------------------------------------------


int NewCoordinateSystem( VECTOR N, VECTOR A, VECTOR B )
// Sets a new coordinate system, with one axis defined by normal vector N. 
{	VECTOR i, j, k;
    DBL tmp;

	SetVector( i, 1, 0, 0 );
	SetVector( j, 0, 1, 0 );
	SetVector( k, 0, 0, 1 );


	if ( ( N[X]==0.0 ) && ( N[Y]==0.0 ) )    // N||z
	  { CopyVector(A,j);
	    CopyVector(B,i);
		return (0);
	  }
	
	if ( ( N[X]==0.0 ) && ( N[Z]==0.0 ) )    // N||y
	  { CopyVector(A,k);
	    CopyVector(B,i);
		return(0);
	  }
	
	if ( ( N[Y]==0.0 ) && ( N[Z]==0.0 ) )    // N||x
	  { CopyVector(A,j);
	    CopyVector(B,k);
		return(0);
	  }
	
	A[X] = 0;
	A[Y] = N[Z] / sqrt( N[Y]*N[Y] + N[Z]*N[Z] );
	A[Z] = -N[Y] / sqrt( N[Y]*N[Y] + N[Z]*N[Z] );

	OuterProduct( B, A, N );

	return(1);
}//NewCoordinateSystem


//------------------------------------------------------------------------------


void ChangeCoordinates( VECTOR P, VECTOR Q, VECTOR N, VECTOR A, VECTOR B )
// Using as a new base the vectors N,A,B, the function converts vector
// P from cartesian coordinates to the new base. 
{
	DBL tmp, tmp2;
	
	Product( tmp, P, N );
	Product( tmp2, N, N );
	Q[X] = tmp/tmp2;

	Product( tmp, P, A );	
	Product( tmp2, A, A );
	Q[Y] = tmp/tmp2;

	Product( tmp, P, B );	
	Product( tmp2, B, B );
	Q[Z] = tmp/tmp2;

}//ChangeCoordinates


//------------------------------------------------------------------------------


DBL PointFromPlaneDistance( TRIANGLE Tr, VECTOR P )
// Calculates the distance between a point and a plane, defined by a triangle. 
{
	DBL d, tmp;
	
	//Check that the plane is valid
	Product( tmp, Tr.N, Tr.N ); 
	if ( tmp == 0.0 )
	  return( -1.0 );

	Product( d, Tr.N, Tr.P1 );
	Product( tmp, Tr.N, P );

	return(  fabs( tmp - d )  );
} // PointFromPlaneDistance


//------------------------------------------------------------------------------


int SphereEdgeIntersection( VECTOR P1, VECTOR P2, BLOB B )
// Returns 1 if an edge defined by two points intersects with a sphere.  
{
	DBL d, r;
	DBL t,           // t is for the line Parametric equation ( x = x1 + t*Nx , etc )
		a,           // xN * x1 + yN * y1 + zN * z1
		b,           // xN * xP + yN * yP + zN * zP
		c,           // xN^2 + yN^2 + zN^2
		r2,
		i, j, k, 
		t1, t2;

	DBL n_i, n_j, n_k, q_i, q_j, q_k;
	
	n_i = P2[X] - P1[X];
	n_j = P2[Y] - P1[Y];
	n_k = P2[Z] - P1[Z];
	
	a = n_i*P1[X] + n_j*P1[Y] + n_k*P1[Z];
	b = n_i*B.Centre[X] + n_j*B.Centre[Y] + n_k*B.Centre[Z];
	c = n_i*n_i + n_j*n_j + n_k*n_k;
	
	// d = (b-a)/sqrt(c) is the distance between C1 and the plane defined by vector N and point P.
	// t = d/sqrt(c) defines the normalized distance between C1 and the plane.
	t = (b-a)/c;
	
	r2 = B.Radius*B.Radius; 
	
	q_i = P1[X] + t*n_i;
	q_j = P1[Y] + t*n_j;
	q_k = P1[Z] + t*n_k;
	
	i = B.Centre[X]-q_i;
	j = B.Centre[Y]-q_j;
	k = B.Centre[Z]-q_k;
    d = i*i + j*j + k*k;
	if ( d > r2 ) return ( 0 );
	
	if ( d == r2 ) 
	{
	  if ( ( t >= 0 ) && ( t <= 1 ) )
		return( 1 );
	  else return(0);
	}

	r = sqrt( (r2-d)/c);
	t1 = t - r;
	t2 = t + r;

	if ( ( (t1<0) && (t2<0) )
		||( (t1>1) && (t2>1) ) )
  	  return( 0 );

	return(1); 

}//SphereEdgeIntersection


//------------------------------------------------------------------------------


int PointInTriangle( TRIANGLE Tr, VECTOR C )
// Returns 1 if point C is inside triangle Tr, else 0.
// C and Tr are coplanar.
{
	VECTOR i;
	DBL a1, a2, a3,   //a1,a2,a3 are used to check if plane's normal vector N is parallel to any axis.
		g1, g2, g3;   //g1,g2,g3 are the edge equation values for point C.

    // if N _|_ Z-axis
	if ( Tr.N[Z] == 0.0 )
	   { if ( Tr.N[X] == 0.0 )
	        // use (x,z) plane
	        { g1 = C[Z]*(Tr.P2[X]-Tr.P1[X]) + C[X]*(Tr.P1[Z]-Tr.P2[Z]) + (Tr.P2[Z]*Tr.P1[X]-Tr.P1[Z]*Tr.P2[X]);
		      g2 = C[Z]*(Tr.P3[X]-Tr.P2[X]) + C[X]*(Tr.P2[Z]-Tr.P3[Z]) + (Tr.P3[Z]*Tr.P2[X]-Tr.P2[Z]*Tr.P3[X]);
		      g3 = C[Z]*(Tr.P1[X]-Tr.P3[X]) + C[X]*(Tr.P3[Z]-Tr.P1[Z]) + (Tr.P1[Z]*Tr.P3[X]-Tr.P3[Z]*Tr.P1[X]);
		      if ( ( g1*g2>0 ) && ( g1*g3>0 ) )	
		         return(1);
		      return(0);
	        }
	     else 
            // use (y,z) plane
			{ g1 = C[Z]*(Tr.P2[Y]-Tr.P1[Y]) + C[Y]*(Tr.P1[Z]-Tr.P2[Z]) + (Tr.P2[Z]*Tr.P1[Y]-Tr.P1[Z]*Tr.P2[Y]);
		      g2 = C[Z]*(Tr.P3[Y]-Tr.P2[Y]) + C[Y]*(Tr.P2[Z]-Tr.P3[Z]) + (Tr.P3[Z]*Tr.P2[Y]-Tr.P2[Z]*Tr.P3[Y]);
		      g3 = C[Z]*(Tr.P1[Y]-Tr.P3[Y]) + C[Y]*(Tr.P3[Z]-Tr.P1[Z]) + (Tr.P1[Z]*Tr.P3[Y]-Tr.P3[Z]*Tr.P1[Y]);
		      if ( ( g1*g2>0 ) && ( g1*g3>0 ) )	
		        return(1);
		      return(0);
	        }
	   }

    // if N _|_ X-axis 
	if ( Tr.N[X] == 0.0 )
	   { if ( Tr.N[Y] == 0.0 )
	        // use (x,y) plane
	        { g1 = C[Y]*(Tr.P2[X]-Tr.P1[X]) + C[X]*(Tr.P1[Y]-Tr.P2[Y]) + (Tr.P2[Y]*Tr.P1[X]-Tr.P1[Y]*Tr.P2[X]);
	          g2 = C[Y]*(Tr.P3[X]-Tr.P2[X]) + C[X]*(Tr.P2[Y]-Tr.P3[Y]) + (Tr.P3[Y]*Tr.P2[X]-Tr.P2[Y]*Tr.P3[X]);
	          g3 = C[Y]*(Tr.P1[X]-Tr.P3[X]) + C[X]*(Tr.P3[Y]-Tr.P1[Y]) + (Tr.P1[Y]*Tr.P3[X]-Tr.P3[Y]*Tr.P1[X]);
	          if ( ( g1*g2>0 ) && ( g1*g3>0 ) )	
	             return(1);
	  	      return(0);
	        }
	     else
            // use (x,z) plane
	        { g1 = C[Z]*(Tr.P2[X]-Tr.P1[X]) + C[X]*(Tr.P1[Z]-Tr.P2[Z]) + (Tr.P2[Z]*Tr.P1[X]-Tr.P1[Z]*Tr.P2[X]);
		      g2 = C[Z]*(Tr.P3[X]-Tr.P2[X]) + C[X]*(Tr.P2[Z]-Tr.P3[Z]) + (Tr.P3[Z]*Tr.P2[X]-Tr.P2[Z]*Tr.P3[X]);
		      g3 = C[Z]*(Tr.P1[X]-Tr.P3[X]) + C[X]*(Tr.P3[Z]-Tr.P1[Z]) + (Tr.P1[Z]*Tr.P3[X]-Tr.P3[Z]*Tr.P1[X]);
		      if ( ( g1*g2>0 ) && ( g1*g3>0 ) )	
		         return(1);
		      return(0);
	        }
	   }


      // otherwise
	g1 = C[Z]*(Tr.P2[Y]-Tr.P1[Y]) + C[Y]*(Tr.P1[Z]-Tr.P2[Z]) + (Tr.P2[Z]*Tr.P1[Y]-Tr.P1[Z]*Tr.P2[Y]);
	g2 = C[Z]*(Tr.P3[Y]-Tr.P2[Y]) + C[Y]*(Tr.P2[Z]-Tr.P3[Z]) + (Tr.P3[Z]*Tr.P2[Y]-Tr.P2[Z]*Tr.P3[Y]);
	g3 = C[Z]*(Tr.P1[Y]-Tr.P3[Y]) + C[Y]*(Tr.P3[Z]-Tr.P1[Z]) + (Tr.P1[Z]*Tr.P3[Y]-Tr.P3[Z]*Tr.P1[Y]);
		
	if ( ( g1*g2>0 ) && ( g1*g3>0 ) )	
	  return(1);

	return(0);
}//PointInTriangle


//------------------------------------------------------------------------------


int TriangleSphereTest( TRIANGLE Tr, BLOB B, DBL d )
{
	DBL p;          //Shows the position of the sphere's centre with respect to the plane
		VECTOR A,   //N*d
		C;          //Circle's centre

	// Project the blob's centre on the triangle plane.
	p = PlaneValue( Tr.N, Tr.P1, B.Centre );
	ScaleS( A, Tr.N, d );
	if ( p >= 0 )
      Subtract( C, B.Centre, A )
	else Add( C, B.Centre, A );
				
	//Check if the circle's centre lies inside the triangle.
	if ( PointInTriangle( Tr, C ) )
	   return( 1 );

	//Check if the blob intersects with any of the triangle edges.
	if ( SphereEdgeIntersection( Tr.P1, Tr.P2, B ) )
      return ( 1 );
	if ( SphereEdgeIntersection( Tr.P2, Tr.P3, B ) )
      return ( 1 );
	if ( SphereEdgeIntersection( Tr.P3, Tr.P1, B ) )
      return ( 1 );
	
	return( 0 );

}

//------------------------------------------------------------------------------


int SphereTriangleIntersectionTest( TRIANGLE Tr, BLOB B, VECTOR Q )
// Returns 1 if Triangle Tr and Blob B intersect, -1 on error, 0 otherwise.
{	
	DBL d;
	DBL sqr_R;

	CopyVector( Q, B.Centre );
		
	// Check if the plane defined by the triangle intersects with the sphere.
	if ( ( d = PointFromPlaneDistance( Tr, B.Centre ) ) < 0 )
	{
		printf("Error in intersection test");
		return ( -1 );
	}

	if ( d > B.Radius )
	  return ( 0 );
	
	if ( d <= B.Radius )
	{
		
		// Check if any of the triangle vertices is inside the sphere.
		sqr_R = B.Radius * B.Radius;
		if ( SqrDistance( Tr.P1, B.Centre ) <= sqr_R )
	      return ( 1 );
		if ( SqrDistance( Tr.P2, B.Centre ) <= sqr_R )
          return ( 1 );
		if ( SqrDistance( Tr.P3, B.Centre ) <= sqr_R )
          return ( 1 );

		// Check if the triangle contains the sphere. 
		if ( TriangleSphereTest( Tr, B, d ) )
	      return( 1 );
	
	}
	return ( 0 );

} //SphereTriangleIntersectionTest

//------------------------------------------------------------------------------


int PointFromLineDistance( VECTOR P, VECTOR C1, VECTOR C2, DBL *d )
// Calculates the distance between a point P and the line defined by two points C1 and C2. 
// Returns 1 if the intersection point Q is found between P1 and P2, else 0. 
//       C1 o
//			|
//			|	
//			|     
//			|     
//		  Q o------o P
//			|
//			|
//			|
//		 C2 o
{
	VECTOR Plane_N,  // This is a vector || to the line defined by C1 and C2
		             // and NOT a normal vector. (N = C2 - C1)
		   Q;        // Intersection point between the line
	                 // and the distance between the line and point P. 
	DBL t,           // For the line Parametric equation ( x = x1 + t*Nx , etc )
		a,           // xN * x1 + yN * y1 + zN * z1
		b,           // xN * xP + yN * yP + zN * zP
		c;           // xN^2 + yN^2 + zN^2

	Subtract( Plane_N, C2, C1 );
    Product( a, Plane_N, C1 );
	Product( b, Plane_N, P );
	Product( c, Plane_N, Plane_N );

	// d = (b-a)/sqrt(c) is the distance between C1 and the plane defined by vector N and point P.
	// t = d/sqrt(c) defines the normalized distance between C1 and the plane.
	t = (b-a)/c;

	Q[X] = C1[X] + t*Plane_N[X];
	Q[Y] = C1[Y] + t*Plane_N[Y];
	Q[Z] = C1[Z] + t*Plane_N[Z];
	
	*d = Distance( P, Q );

	if ( ( t < 0 ) || ( t > 1 ) )
      return( 0 );
	return( 1 );

}//PointFromLineDistance


//------------------------------------------------------------------------------


int PointInCylinder( VECTOR P, BLOB B )
{
	DBL d;

	if ( PointFromLineDistance( P, B.Centre, B.PreviousCentre, &d ) )
	  if ( d <= B.Radius )
        return( 1 );

	return( 0 );

}//PointInCylinder 


//------------------------------------------------------------------------------


int LinePlaneIntersection( VECTOR N, VECTOR P, VECTOR C1, VECTOR C2, VECTOR Q )
// Checks if a plane and a line intersect. Returns 1 if the plane intersects within C1C2, 0 if the intersection
// is outside C1C2 and -1 else. Q is the intersection point.
{
	VECTOR A; //C2-C1
		    
	DBL p, // (C2-C1)*N
		d, // d in plane equation
		a, // N*C1
		t; // t in line parametric equation : x = x1 +t*(x2-x1) etc

	Subtract( A, C2, C1 );
	Product( p, A, N );
	
	//Check if line is parallel to plane
	if ( p == 0 )
		return( -1 );
	
	// The intersection point Q(x,y,z) must satisfy the following equations:
	// x = x1 + t(x2-x1), y = y1 +t(y2-y1), z = z1 + t(z2-z1)
	// with 0<=t<=1
	// and ax+by+cz+d = 0.
	// That leads to :
	// t = - ( N*C1-N*P ) / N*(C2-C1)
	Product( d, N, P );	
	Product( a, N, C1 );
	a = a - d;
	t = -a/p;
	
	if ( ( t < 0 )  || ( t > 1 ) )
		return( 0 );

	SetVector( Q, C1[X]+t*A[X], C1[Y]+t*A[Y], C1[Z]+t*A[Z] );
	
	return( 1 );

}//LinePlaneIntersection


//------------------------------------------------------------------------------


int CircleEdgeIntersection( VECTOR P1, VECTOR P2, VECTOR C, DBL r )
// Returns 1 if an edge defined by two points intersects with a circle.  
{
	DBL d;

	if ( PointFromLineDistance( C, P1, P2, &d ) )  
      if ( d <= r )
		return( 1 );
	return( 0 );

}//CircleEdgeIntersection


//------------------------------------------------------------------------------


int TriangleCircleIntersection( TRIANGLE Tr, VECTOR C, DBL r )
// Checks if a triangle and a circular disk ( coplanar ) intersect. 
// We assume that no triangle vertex is inside the disk.
{
	if ( CircleEdgeIntersection( Tr.P1, Tr.P2, C, r ) )
      return( 1 );
	if ( CircleEdgeIntersection( Tr.P2, Tr.P3, C, r ) )
      return( 1 );
	if ( CircleEdgeIntersection( Tr.P3, Tr.P1, C, r ) )
      return( 1 );

	if ( PointInTriangle( Tr, C ) )
	  return( 1 );
	
	return( 0 );

}//TriangleCircleIntersection


//------------------------------------------------------------------------------


int LineLineIntersection( VECTOR P1, VECTOR P2, VECTOR C1, VECTOR C2, VECTOR Q )
// Finds the intersection between two coplanar linear segments. Only Y, Z 
// coordinates are  considered.
{
	DBL D, Dp, Dc;
	DBL tp, tc;

	D = (P2[Y]-P1[Y])*(C1[Z]-C2[Z]) - (C1[Y]-C2[Y])*(P2[Z]-P1[Z]);
	
	if ( D == 0 )
      return( -1 );

	Dp = (C1[Y]-P1[Y])*(C1[Z]-C2[Z]) - (C1[Y]-C2[Y])*(C1[Z]-P1[Z]);
	Dc = (P2[Y]-P1[Y])*(C1[Z]-P1[Z]) - (C1[Y]-P1[Y])*(P2[Z]-P1[Z]);

	tp = Dp/D;
	tc = Dc/D;
	
	Q[X] = P1[X];
	Q[Y] = P1[Y] + tp*(P2[Y]-P1[Y]);
	Q[Z] = P1[Z] + tp*(P2[Z]-P1[Z]);


	if( ( tc>=0 ) && ( tc<=1 ) && ( tp>=0) && ( tp<=1 ) )
		return( 1 );
	
	return( 0 );

}//LineLineIntersection


//------------------------------------------------------------------------------


int CylinderTriangleIntersection( TRIANGLE Tr, BLOB B, VECTOR Q )
// Checks if a triangle and a cylinder, defined by a blob's previous and current position, intersect.
// Also returns the approximate "intersection" point Q between the cylinder axis and the triangle.
{
	int col;  //collision flag
	VECTOR A, // Line "normal".
		   D;
	DBL p, 
		p1a, p1b, // Plane values for each triangle vertex, in regard to caps. 
		p2a, p2b,
		p3a, p3b,
		d;

	TRIANGLE Tproj;

	//Check triangle plane position relevantly to cylinder axis.
	col = LinePlaneIntersection( Tr.N, Tr.P1, B.Centre, B.PreviousCentre, Q);
	
	// Axis-plane intersection lies outside cylinder caps. Intersection should then be 
	// treated with appropriate sphere.
	if ( col == 0 )
	   return( 0 );

	// Do the following if the tranjectory is parallel to the triangle plane 
	if (  col == -1 ) 
	{   
		//        B.Centre 
		//    ----o----    
		//    |	  |   | 
		//    |   |   |   |
		//    |   |   |   | Tr
		//    |   |   |   |
		//    |   |   |   |
		//    |   |   |   |
		//    |   |   |   
		//    |   |   |
		//    |   |   |
		//    ---------
	
		DBL distance;
		
		distance = PointFromPlaneDistance( Tr, B.Centre );
		
		if ( distance  > B.Radius )
			return( 0 );

		
	   //If any of the triangle vertices lies inside the cylinder, set the intersection point
	   //as the projection on the tranjectory of the vertex closer to the tranjectory origin.
	   // (B.PreviousCentre)
	
		VECTOR classify = {-DBL_MAX, -DBL_MAX, -DBL_MAX};
		int inside = 0;
		d = 0.0;

		Subtract( A, B.PreviousCentre, B.Centre );
		ReduceToUnit( A );

		if ( PointInCylinder( Tr.P1,B ))
		  { classify[0] = PlaneValue( A, B.PreviousCentre, Tr.P1 );
		    inside ++;
		  }
	    if ( PointInCylinder( Tr.P2,B ))
		  { classify[1] = PlaneValue( A, B.PreviousCentre, Tr.P2 );
		    inside ++;
		  }
		if ( PointInCylinder( Tr.P3,B ))
		  { classify[2] = PlaneValue( A, B.PreviousCentre, Tr.P3 );
		    inside++;
		  }
		
		if ( inside )
		  {	d = max( max( classify[0], classify[1] ), classify[2] );
		    ScaleS( Q, A, d); 
			Add( Q, Q, B.PreviousCentre );
			return( 1 );
		}
				
		
		// Project the cylinder axis on the triangle plane. Calculate the projected line's 
		// intersections with the triangle ( if any ) and find the intersection T closer 
		// to the tranjectory's begining. Set as intersection point Q the "unprojected" T.
		VECTOR ProjCentre, ProjPreviousCentre,
			   U, V,
			   Transformed_C, Transformed_PC,
			   Transformed_P1, Transformed_P2, Transformed_P3,
			   T,
			   i, j, k, new_i, new_j, new_k;
		
		DBL QDistance = DBL_MAX, 
			intersections = 0;

		//Project the cylinder axis on the triangle plane.
		p = PlaneValue( Tr.N, Tr.P1, B.Centre );
		ScaleS( D, Tr.N, -p );
		Add( ProjCentre, B.Centre, D );

		p = PlaneValue( Tr.N, Tr.P1, B.PreviousCentre );
		ScaleS( D, Tr.N, -p );
		Add( ProjPreviousCentre, B.PreviousCentre, D );

		NewCoordinateSystem( Tr.N, U, V );
	   
		//Project the two centres in the new coordinate system.
	    ChangeCoordinates( ProjCentre, Transformed_C, Tr.N, U, V );		
		ChangeCoordinates( ProjPreviousCentre, Transformed_PC, Tr.N, U, V );	
	    ChangeCoordinates( Tr.P1, Transformed_P1, Tr.N, U, V );
		ChangeCoordinates( Tr.P2, Transformed_P2, Tr.N, U, V );
		ChangeCoordinates( Tr.P3, Transformed_P3, Tr.N, U, V );

		
		if ( LineLineIntersection( Transformed_PC, Transformed_C, 
			 Transformed_P1, Transformed_P2, Q ) == 1 )
		{  QDistance = Distance( Transformed_PC, Q ); 
		   CopyVector( T, Q );
		   intersections ++;
		}
		
		if ( LineLineIntersection( Transformed_PC, Transformed_C, 
			 Transformed_P2, Transformed_P3, Q ) == 1 )
		{ intersections ++; 
		  if ( Distance ( Transformed_PC, Q ) < QDistance )
		  {  QDistance = Distance( Transformed_PC, Q ); 
		     CopyVector( T, Q );
		  }
		}

		if ( LineLineIntersection( Transformed_PC, Transformed_C, 
			 Transformed_P3, Transformed_P1, Q ) == 1 )
		{ intersections ++;
		  if ( Distance ( Transformed_PC, Q ) < QDistance )
		  {  QDistance = Distance( Transformed_PC, Q ); 
		     CopyVector( T, Q );
		  }
		}

		if ( intersections == 0 )
			return( 0 );
		
		// Initialize coordinate system
		SetVector( i, 1, 0, 0 );
		SetVector( j, 0, 1, 0 );
		SetVector( k, 0, 0, 1 );
		
		ChangeCoordinates( i, new_i, Tr.N, U, V );
		ChangeCoordinates( j, new_j, Tr.N, U, V );
		ChangeCoordinates( k, new_k, Tr.N, U, V );
		ChangeCoordinates( T, Q, new_i, new_j, new_k );
		return( 1 );
	}

	// Do the following if the trajectory intersects with the triangle plane
	if ( col == 1 )
    {
		//If any of the triangle vertices lies inside the cylinder, set the intersection point
	   //as the projection on the trajectory of the vertex closer to the trajectory origin.
	   // (B.PreviousCentre)

		Subtract( A, B.PreviousCentre, B.Centre );
		ReduceToUnit( A );

		VECTOR classify = {-DBL_MAX, -DBL_MAX, -DBL_MAX};
		int inside = 0;
		DBL d = 0.0;
	
		if ( PointInCylinder( Tr.P1,B ))
		  { classify[0] = PlaneValue( A, B.PreviousCentre, Tr.P1 );
		    inside ++;
		  }
	    if ( PointInCylinder( Tr.P2,B ))
		  { classify[1] = PlaneValue( A, B.PreviousCentre, Tr.P2 );
		    inside ++;
		  }
		if ( PointInCylinder( Tr.P3,B ))
		  { classify[2] = PlaneValue( A, B.PreviousCentre, Tr.P3 );
		    inside++;
		  }
		
		if ( inside )
		  {	d = max( max( classify[0], classify[1] ), classify[2] );
		    ScaleS( Q, A, d); 
			Add( Q, Q, B.PreviousCentre );
			return( 1 );
 	       }

        //----------------------------------------------------------------------
	     	

		//Calculate the plane values for each vertex with regard to both plane caps. 
		//If all the vertices are outside the cylinder and on the same side, then there 
		//is no intersection point.

		p1a = PlaneValue( A, B.PreviousCentre, Tr.P1 );
		p1b = PlaneValue( A, B.Centre, Tr.P1 );
		p2a = PlaneValue( A, B.PreviousCentre, Tr.P2 );
		p2b = PlaneValue( A, B.Centre, Tr.P2 );
	    p3a = PlaneValue( A, B.PreviousCentre, Tr.P3 );
		p3b = PlaneValue( A, B.Centre, Tr.P3 );
		
		int flag = 0;

		if ( p1a*p1b > 0 ) 
		  { if ( p1a>0 ) 
		      flag++;
		   else flag--;
		  }
		if ( p2a*p2b > 0 )
		 { if ( p2a>0 ) 
		     flag++;
		   else flag--;
		 }
		if ( p3a*p3b > 0 )
		 { if ( p3a>0 ) 
		     flag++;
		   else flag--;
		 }
              
		if ( abs(flag)==3 ) return (0);   
		   
		
		//Calculate the plane values for the triangle vertices with regard to the 
		//plane which is perpendicular to the cylinder axis and crosses Q.
		//Project the triangle on this plane.
				
		p1a = PlaneValue( A, Q, Tr.P1 );
		ScaleS( D, A, -p1a );
		Add( Tproj.P1, Tr.P1, D );

		p2a = PlaneValue( A, Q, Tr.P2 );
		ScaleS( D, A, -p2a );
		Add( Tproj.P2, Tr.P2, D );

		p3a = PlaneValue( A, Q, Tr.P3 );
		ScaleS( D, A, -p3a );
		Add( Tproj.P3, Tr.P3, D );

		CopyVector( Tproj.N, A );
	 
		if ( TriangleCircleIntersection( Tproj, Q, B.Radius ) ) 
		{
		   // If the triangle vertices are in both sides of the plane that crosses Q
		   // then use Q as the intersection point.
			int flag = 0;

		   if ( p1a >= 0 )
			   flag ++;
		   else
			   flag --;
		   if ( p2a >= 0 )
			   flag ++;
		   else
			   flag --;
		   if ( p3a >= 0 )
			   flag ++;
		   else
			   flag --;

		   if ( abs(flag) < 3 ) return (1);	
			
		   // If all vertices are on one side of the plane that crosses Q then return as
		   // intersection point the projection on the trajectory of the vertex that is
		   // closer to the trajectory begining.

		   VECTOR Q1, classify = {DBL_MAX, DBL_MAX, DBL_MAX};
		   DBL d = 0.0;
	       
		   classify[0] = abs( PlaneValue( A, Q, Tr.P1 ) );	
		   classify[1] = abs( PlaneValue( A, Q, Tr.P2 ) );
		   classify[2] = abs( PlaneValue( A, Q, Tr.P3 ) );

	       d = min( min( classify[0], classify[1] ), classify[2] );
		   if ( PlaneValue( A, Q, Tr.P1 ) < 0 )
			 d = -d;
		   ScaleS( Q1, A, d); 
		   Add( Q, Q1, Q );
			   return( 1 );
		}
		
	  return( 0 ) ;
		
	} // if i = 1  

return( 0 );	

} //CylinderTriangleIntersection



//------------------------------------------------------------------------------

int LineTriangleIntersection( TRIANGLE Tr, VECTOR C1, VECTOR C2, VECTOR Q )
{ 
	
	if ( LinePlaneIntersection( Tr.N, Tr.P1, C1, C2, Q ) == 1 ) 
	if ( PointInTriangle( Tr, Q ) )
	  return( 1);
	return( 0 );

} //LineTriangleIntersection 


//------------------------------------------------------------------------------


int IntersectionTest( TRIANGLE Tr, BLOB B, VECTOR Q )
// Returns 1 if an intersection is detected between a triangle and a sphere, or a triangle and the spheres 
// path from it's last position; else returns 0. Also calculates the position where the intersection was found

{
	if ( SphereTriangleIntersectionTest( Tr, B, Q ) )
	   return( 1 );
	if ( !IsEqual( B.Centre, B.PreviousCentre ) )
	  if ( CylinderTriangleIntersection( Tr, B, Q ) )
	 	return( 1 );
	return( 0 ); 

}//IntersectionTest


//------------------------------------------------------------------------------


int CollisionDetection( MESH *M, BLOB B, VECTOR Q )
{
	int i,
		collision_num = -1;
	VECTOR A,              //Normalized vector defined by current 
		                   //and previous position of the blob.
		   U, V,           //(A, U, V) define a new coordinate system
		   Transformed_C,  //Blob's centre in the new ccordinate system.
		   Transformed_PC, //Blob's previous centre -//- .
		   Transformed_Q,  //Last intersection point detected -//-.
		   I,              //Previous intersection point.
		   Transformed_I;  
	BLOB PrevBlob;
	DBL d;

	//Define a new coordinate system. The system is defined by the vector A, which
	//is the normalized vector between the blob's current and previous centre.
	Subtract( A, B.PreviousCentre, B.Centre );
	ReduceToUnit( A );	    	
	NewCoordinateSystem( A, U, V );
	//Project the two centres in the new coordinate system.
	ChangeCoordinates( B.PreviousCentre, Transformed_PC, A, U, V );
	ChangeCoordinates( B.Centre, Transformed_C, A, U, V );
	
	//Initialize last known intersection vector, as the sphere's current position.
	//In this way if another intersection point exists, it is certain that it will
	//be closer to the old sphere's position.
	CopyVector( I, B.Centre );
	CopyBlob( PrevBlob, B );
	CopyVector( PrevBlob.Centre, PrevBlob.PreviousCentre );

	
	//Check for intersection points between the blob's trajectory 
	//and the triangle mesh.
	for ( i = 0; i < M->Faces; i++ )  
	  if (  ( Distance( B.Centre, M->Face[i].C ) <= ( B.Radius + M->Face[i].r ) )
	      ||( Distance( B.PreviousCentre, M->Face[i].C ) <= ( B.Radius + M->Face[i].r ) )
		  ||( ( PointFromLineDistance( M->Face[i].C, B.PreviousCentre, B.Centre, &d ) )
		    &&( d <= B.Radius + M->Face[i].r ) 
			)
		  )
	{

	   if (   (   ( !IsEqual( B.Centre, B.PreviousCentre ) )
	           && ( CylinderTriangleIntersection( M->Face[i], B, Q ) )
		      )
			  || ( SphereTriangleIntersectionTest( M->Face[i], B, Q ) ) 
          )
		  {	
			// Compare new intersection point to last one, 
		    // to find  which one is the first the blob meets.			
			ChangeCoordinates( Q, Transformed_Q, A, U, V );
			ChangeCoordinates( I, Transformed_I, A, U, V );
			//"A" coordinate is the"X" coordinate.
			if ( Transformed_Q[X] > Transformed_I[X] )
			{  CopyVector( I, Q );	
			   collision_num = i;
			}
	      }
	    
	}

	CopyVector( Q, I );
	return( collision_num );
} // CollisionDetection